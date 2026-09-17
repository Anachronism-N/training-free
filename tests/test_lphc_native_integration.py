from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from lifecycle_kv.history_interventions import apply_history_rope
from lifecycle_kv.lphc import LPHCCallContext, LPHCConfig, LPHCController


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CAUSAL_MODEL = (
    _REPO_ROOT / "third_party" / "Self-Forcing" / "wan" / "modules" / "causal_model.py"
)


def _native_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / math.sqrt(query.shape[-1])
    probabilities = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value)


@pytest.fixture(scope="module")
def native_module():
    """Load the production attention module without its optional model stack."""
    saved = {name: sys.modules.get(name) for name in (
        "wan",
        "wan.modules",
        "wan.modules.attention",
        "wan.modules.model",
        "diffusers",
        "diffusers.configuration_utils",
        "diffusers.models",
        "diffusers.models.modeling_utils",
    )}
    original_compile = torch.compile
    try:
        wan = types.ModuleType("wan")
        wan_modules = types.ModuleType("wan.modules")
        attention_module = types.ModuleType("wan.modules.attention")
        attention_module.attention = _native_attention
        model_module = types.ModuleType("wan.modules.model")
        model_module.WanRMSNorm = nn.Identity
        model_module.WanLayerNorm = nn.LayerNorm
        model_module.WAN_CROSSATTENTION_CLASSES = {}
        model_module.rope_apply = lambda x, *_args, **_kwargs: x
        model_module.rope_params = lambda *_args, **_kwargs: None
        model_module.MLPProj = nn.Identity
        model_module.sinusoidal_embedding_1d = lambda *_args, **_kwargs: None

        diffusers = types.ModuleType("diffusers")
        configuration = types.ModuleType("diffusers.configuration_utils")
        configuration.ConfigMixin = type("ConfigMixin", (), {})
        configuration.register_to_config = lambda function: function
        diffusers_models = types.ModuleType("diffusers.models")
        modeling = types.ModuleType("diffusers.models.modeling_utils")
        modeling.ModelMixin = type("ModelMixin", (nn.Module,), {})

        replacements = {
            "wan": wan,
            "wan.modules": wan_modules,
            "wan.modules.attention": attention_module,
            "wan.modules.model": model_module,
            "diffusers": diffusers,
            "diffusers.configuration_utils": configuration,
            "diffusers.models": diffusers_models,
            "diffusers.models.modeling_utils": modeling,
        }
        sys.modules.update(replacements)
        torch.compile = lambda function, **_kwargs: function
        spec = importlib.util.spec_from_file_location("_lphc_native_causal_model", _CAUSAL_MODEL)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        torch.compile = original_compile
        sys.modules.pop("_lphc_native_causal_model", None)
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _frequency_table(rows: int = 64, complex_dim: int = 3) -> torch.Tensor:
    positions = torch.arange(rows, dtype=torch.float64).unsqueeze(1)
    channels = torch.arange(complex_dim, dtype=torch.float64).unsqueeze(0)
    angles = positions * (channels + 1) * 0.017
    return torch.polar(torch.ones_like(angles), angles)


def _attention(native_module, *, capacity_frames: int = 4, sink_size: int = 0):
    layer = native_module.CausalWanSelfAttention(
        dim=12,
        num_heads=2,
        local_attn_size=capacity_frames,
        sink_size=sink_size,
        qk_norm=False,
    )
    with torch.no_grad():
        identity = torch.eye(12)
        for projection in (layer.q, layer.k, layer.v, layer.o):
            projection.weight.copy_(identity)
            projection.bias.zero_()
    return layer


def _cache(capacity_frames: int, *, local_end: int = 0, global_end: int = 0):
    return {
        "k": torch.zeros(1, capacity_frames, 2, 6),
        "v": torch.zeros(1, capacity_frames, 2, 6),
        "global_end_index": torch.tensor([global_end], dtype=torch.long),
        "local_end_index": torch.tensor([local_end], dtype=torch.long),
    }


def _context(*, kind: str = "noisy", phase: int = 0, block: int = 0):
    return LPHCCallContext(
        call_kind=kind,
        phase_index=phase,
        block_id=block,
        source_index=17,
        local_frame_ids=(),
        trajectory_id="native-integration",
    )


def _forward(layer, x, cache, *, start: int, controller=None, context=None):
    frames = x.shape[1]
    return layer(
        x,
        torch.tensor([frames]),
        torch.tensor([[frames, 1, 1]]),
        _frequency_table(),
        None,
        kv_cache=cache,
        current_start=start,
        lphc_controller=controller,
        lphc_context=context,
    )


def _clone_cache(cache):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in cache.items()}


def test_native_attention_captures_absolute_local_ids_before_capacity_and_after_fifo_roll(native_module):
    layer = _attention(native_module, capacity_frames=3)
    controller = LPHCController(LPHCConfig(alpha=0.0), layer_idx=0)
    cache = _cache(3)

    _forward(
        layer,
        torch.arange(24, dtype=torch.float32).reshape(1, 2, 12),
        cache,
        start=0,
        controller=controller,
        context=_context(block=0),
    )
    assert controller.last_local_frame_ids == (0, 1)
    assert (cache["global_end_index"].item(), cache["local_end_index"].item()) == (2, 2)

    _forward(
        layer,
        torch.arange(24, 48, dtype=torch.float32).reshape(1, 2, 12),
        cache,
        start=2,
        controller=controller,
        context=_context(block=1),
    )
    assert controller.last_local_frame_ids == (1, 2, 3)
    assert (cache["global_end_index"].item(), cache["local_end_index"].item()) == (4, 3)


def test_v211_sink1_recent20_reports_exact_ids_before_and_after_rollover(native_module):
    layer = _attention(native_module, capacity_frames=21, sink_size=1)
    controller = LPHCController(
        LPHCConfig(
            alpha=0.0,
            protocol="v211",
            local_policy="sink1_recent20",
            history_budget=1,
        ),
        layer_idx=0,
    )
    cache = _cache(21)

    _forward(
        layer,
        torch.arange(24, dtype=torch.float32).reshape(1, 2, 12),
        cache,
        start=0,
        controller=controller,
        context=_context(block=0),
    )
    assert controller.last_local_frame_ids == (0, 1)

    _forward(
        layer,
        torch.arange(24, 264, dtype=torch.float32).reshape(1, 20, 12),
        cache,
        start=2,
        controller=controller,
        context=_context(block=1),
    )
    assert controller.last_local_frame_ids == (0, *range(2, 22))
    assert len(controller.last_local_frame_ids) == 21
    assert len(set(controller.last_local_frame_ids)) == 21
    assert (cache["global_end_index"].item(), cache["local_end_index"].item()) == (22, 21)


def test_invalid_protocol_and_sink_policy_combinations_fail_closed(native_module):
    with pytest.raises(ValueError, match="v210.*fifo21"):
        LPHCConfig(
            alpha=0.0,
            protocol="v210",
            local_policy="sink1_recent20",
        )

    x = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    sink_policy = LPHCController(
        LPHCConfig(
            alpha=0.0,
            protocol="v211",
            local_policy="sink1_recent20",
            history_budget=1,
        ),
        layer_idx=0,
    )
    with pytest.raises(RuntimeError, match="sink1_recent20"):
        _forward(
            _attention(native_module, capacity_frames=21, sink_size=0),
            x,
            _cache(21),
            start=0,
            controller=sink_policy,
            context=_context(),
        )

    fifo_policy = LPHCController(
        LPHCConfig(alpha=0.0, protocol="v211", local_policy="fifo21"),
        layer_idx=0,
    )
    with pytest.raises(RuntimeError, match="fifo21"):
        _forward(
            _attention(native_module, capacity_frames=21, sink_size=1),
            x,
            _cache(21),
            start=0,
            controller=fifo_policy,
            context=_context(),
        )


def test_clean_tail_commit_preserves_sentinel_provenance(native_module):
    layer = _attention(native_module)
    controller = LPHCController(LPHCConfig(alpha=0.0), layer_idx=0)
    cache = _cache(4, local_end=2, global_end=2)
    cache["k"][:, :2].fill_(-101.0)
    cache["v"][:, :2].fill_(-202.0)
    clean = torch.arange(24, dtype=torch.float32).reshape(1, 2, 12) + 50.0

    _forward(
        layer,
        clean,
        cache,
        start=2,
        controller=controller,
        context=_context(kind="clean", phase=4, block=0),
    )
    local_end = int(cache["local_end_index"].item())
    block_start = local_end - clean.shape[1]
    tail_k = cache["k_pre_rope"][:, block_start:local_end]
    tail_v = cache["v"][:, block_start:local_end]
    controller.commit_clean(tail_k, tail_v, (2, 3), context=_context(kind="clean", phase=4))

    expected = clean.reshape(2, 1, 2, 6)
    assert torch.equal(controller.archive.k, expected)
    assert torch.equal(controller.archive.v, expected)
    assert controller.archive.frame_ids.tolist() == [2, 3]
    assert not torch.any(controller.archive.k == -101.0)
    assert not torch.any(controller.archive.v == -202.0)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    ((torch.float32, 2e-6, 2e-6), (torch.float64, 1e-10, 1e-10)),
)
def test_sparse_absolute_history_rope_matches_native_position_semantics(
    native_module, dtype, atol, rtol
):
    generator = torch.Generator().manual_seed(210)
    positions = torch.tensor([1, 7, 19], dtype=torch.long)
    raw = torch.randn(1, 12, 2, 6, generator=generator, dtype=dtype)
    grid = torch.tensor([[3, 2, 2]])
    freqs = _frequency_table()

    history = apply_history_rope(
        raw,
        grid_sizes=grid,
        freqs=freqs,
        temporal_positions=positions,
    )
    native = native_module.causal_rope_apply_pos(raw, grid, freqs, positions)
    torch.testing.assert_close(history, native, atol=atol, rtol=rtol)



def test_alpha_zero_matches_native_output_rng_cache_and_index_trajectory(native_module):
    layer = _attention(native_module)
    config = LPHCConfig(alpha=0.0, mode="random")
    controller = LPHCController(config, layer_idx=0)
    native_cache = _cache(4)
    enabled_cache = _cache(4)
    generator = torch.Generator().manual_seed(45)
    inputs = [torch.randn(1, 2, 12, generator=generator) for _ in range(3)]
    native_indices = []
    enabled_indices = []

    initial_rng = torch.random.get_rng_state().clone()
    native_outputs = []
    for block, x in enumerate(inputs):
        native_outputs.append(_forward(layer, x, native_cache, start=2 * block))
        native_indices.append(
            (native_cache["global_end_index"].item(), native_cache["local_end_index"].item())
        )
    native_rng = torch.random.get_rng_state().clone()

    torch.random.set_rng_state(initial_rng)
    enabled_outputs = []
    for block, x in enumerate(inputs):
        enabled_outputs.append(
            _forward(
                layer,
                x,
                enabled_cache,
                start=2 * block,
                controller=controller,
                context=_context(block=block),
            )
        )
        enabled_indices.append(
            (enabled_cache["global_end_index"].item(), enabled_cache["local_end_index"].item())
        )
    enabled_rng = torch.random.get_rng_state().clone()

    assert all(torch.equal(native, enabled) for native, enabled in zip(native_outputs, enabled_outputs))
    assert torch.equal(native_rng, enabled_rng)
    assert torch.equal(native_cache["k"], enabled_cache["k"])
    assert torch.equal(native_cache["v"], enabled_cache["v"])
    assert native_indices == enabled_indices == [(2, 2), (4, 4), (6, 4)]
    assert controller.counters["provider_calls"] == 0
    assert controller.counters["second_attention_calls"] == 0


def test_nonzero_noisy_correction_is_temporary_and_clean_refresh_overwrites_cache(native_module):
    layer = _attention(native_module)
    config = LPHCConfig(alpha=0.5, schedule="e1")
    controller = LPHCController(config, layer_idx=0)
    history_k = torch.ones(1, 1, 2, 6)
    history_v = torch.full_like(history_k, 100.0)
    controller.commit_clean(history_k, history_v, (0,), context=_context(kind="clean", phase=4))
    controller.finish_clean_block(history_v, (0,), context=_context(kind="clean", phase=4))

    initial = _cache(4, local_end=2, global_end=4)
    initial["k"][:, :2].zero_()
    initial["v"][:, :2].fill_(1.0)
    native_cache = _clone_cache(initial)
    corrected_cache = _clone_cache(initial)
    noisy = torch.ones(1, 2, 12)

    native_output = _forward(layer, noisy, native_cache, start=4)
    corrected_output = _forward(
        layer,
        noisy,
        corrected_cache,
        start=4,
        controller=controller,
        context=_context(block=2),
    )
    assert not torch.equal(corrected_output, native_output)
    assert controller.counters["second_attention_calls"] == 1

    clean = torch.full((1, 2, 12), -7.0)
    _forward(
        layer,
        clean,
        corrected_cache,
        start=4,
        controller=controller,
        context=_context(kind="clean", phase=4, block=2),
    )
    local_end = int(corrected_cache["local_end_index"].item())
    tail = slice(local_end - 2, local_end)
    expected = clean.reshape(1, 2, 2, 6)
    assert torch.equal(corrected_cache["k_pre_rope"][:, tail], expected)
    assert torch.equal(corrected_cache["v"][:, tail], expected)
    assert not torch.equal(corrected_cache["v"][:, tail], noisy.reshape(1, 2, 2, 6))
