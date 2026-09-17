import hashlib

import pytest
import torch

from lifecycle_kv.lphc import (
    LPHCCallContext,
    LPHCConfig,
    LPHCController,
    LPHCHistory,
    apply_lphc_attention,
)


def _context(
    *,
    kind="noisy",
    phase=0,
    block=1,
    source=17,
    local=(),
    trajectory="prompt-a",
):
    return LPHCCallContext(kind, phase, block, source, tuple(local), trajectory)


def _kv(frame_ids, *, spatial=2, heads=2, dim=3, offset=0.0):
    values = []
    for frame_id in frame_ids:
        base = torch.arange(dim, dtype=torch.float32) + float(frame_id) + offset
        values.append(base.view(1, 1, dim).expand(spatial, heads, dim))
    tensor = torch.stack(values).reshape(1, -1, heads, dim)
    return tensor.clone(), (tensor + 0.25).clone()


def _ready_controller(config=None, frame_ids=range(8)):
    config = config or LPHCConfig(alpha=0.1)
    controller = LPHCController(config, layer_idx=3)
    k, v = _kv(frame_ids)
    controller.commit_clean(k, v, tuple(frame_ids), context=_context(kind="clean"))
    _, descriptor_v = _kv((100, 101))
    controller.finish_clean_block(
        descriptor_v, (100, 101), context=_context(kind="clean", block=0)
    )
    return controller


def _attention_inputs(batch=1):
    y = torch.randn(batch, 3, 2, 4)
    q = torch.randn(batch, 3, 2, 4)
    k = torch.randn(batch, 5, 2, 4)
    v = torch.randn(batch, 5, 2, 4)
    history = LPHCHistory(
        torch.randn(1, 2, 2, 4),
        torch.randn(1, 2, 2, 4),
        torch.tensor([2]),
        key_is_roped=True,
    )
    return y, q, k, v, history


def test_frozen_config_validation_and_schedules():
    assert LPHCConfig.from_env({"LPHC_ENABLE": "0"}) is None
    env_config = LPHCConfig.from_env(
        {
            "LPHC_ENABLE": "1",
            "LPHC_ALPHA": "0.2",
            "LPHC_PHASE": "e2",
            "LPHC_RETRIEVAL_MODE": "random",
            "LPHC_ARCHIVE_FRAMES": "12",
            "LPHC_HISTORY_FRAMES": "4",
            "LPHC_CONTROL_SEED": "9",
        }
    )
    assert env_config == LPHCConfig(
        alpha=0.2, mode="random", schedule="e2", control_seed=9
    )
    assert LPHCConfig(alpha=0.1, schedule="e1").enabled_phases == {0}
    assert LPHCConfig(alpha=0.1, schedule="e2").enabled_phases == {0, 1}
    assert LPHCConfig(alpha=0.1, schedule="full").enabled_phases == {0, 1, 2, 3}
    for kwargs in (
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": 0.1, "mode": "recent"},
        {"alpha": 0.1, "schedule": "late"},
        {"alpha": 0.1, "archive_capacity": 11},
        {"alpha": 0.1, "history_budget": 3},
        {"alpha": 0.1, "history_budget": 1},
        {"alpha": 0.1, "protocol": "v210", "local_policy": "sink1_recent20"},
    ):
        with pytest.raises(ValueError):
            LPHCConfig(**kwargs)

    v211 = LPHCConfig(
        alpha=0.1,
        protocol="v211",
        local_policy="sink1_recent20",
        history_budget=1,
    )
    assert v211.history_budget == 1
    assert v211.protocol == "v211"


def test_config_from_env_uses_frozen_v210_names_and_rejects_partial_enable():
    assert LPHCConfig.from_env({}) is None
    assert LPHCConfig.from_env({"LPHC_ENABLE": "off"}) is None
    config = LPHCConfig.from_env(
        {
            "LPHC_ENABLE": "1",
            "LPHC_ALPHA": "0.05",
            "LPHC_PHASE": "e2",
            "LPHC_RETRIEVAL_MODE": "random",
            "LPHC_ARCHIVE_FRAMES": "12",
            "LPHC_HISTORY_FRAMES": "4",
            "LPHC_CONTROL_SEED": "21017",
        }
    )
    assert config == LPHCConfig(
        alpha=0.05, mode="random", schedule="e2", control_seed=21017
    )
    with pytest.raises(ValueError, match="LPHC_ALPHA"):
        LPHCConfig.from_env({"LPHC_ENABLE": "true"})
    with pytest.raises(ValueError, match="LPHC_ENABLE"):
        LPHCConfig.from_env({"LPHC_ENABLE": "sometimes"})

    v211 = LPHCConfig.from_env(
        {
            "LPHC_ENABLE": "1",
            "LPHC_ALPHA": "0.1",
            "LPHC_PROTOCOL": "v211",
            "LPHC_LOCAL_POLICY": "sink1_recent20",
            "LPHC_HISTORY_FRAMES": "1",
        }
    )
    assert v211 == LPHCConfig(
        alpha=0.1,
        protocol="v211",
        local_policy="sink1_recent20",
        history_budget=1,
    )


def test_alpha_zero_is_exact_identity_without_provider_attention_or_rng_change():
    config = LPHCConfig(alpha=0.0, mode="random")
    controller = LPHCController(config, layer_idx=0)
    y, q, k, v, _ = _attention_inputs()
    calls = {"provider": 0, "attention": 0}

    def provider(_):
        calls["provider"] += 1
        raise AssertionError("provider must not run")

    def attention(*_):
        calls["attention"] += 1
        raise AssertionError("second attention must not run")

    rng_before = torch.random.get_rng_state().clone()
    result = apply_lphc_attention(
        y,
        q,
        k,
        v,
        history_provider=provider,
        attention_fn=attention,
        context=_context(local=(7, 8, 9)),
        config=config,
        controller=controller,
    )
    assert result is y
    assert calls == {"provider": 0, "attention": 0}
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert controller.counters["provider_calls"] == 0
    assert controller.last_local_frame_ids == (7, 8, 9)

    local_context = _context(local=(20, 21))
    assert apply_lphc_attention(
        y,
        q,
        k,
        v,
        history_provider=provider,
        attention_fn=attention,
        context=local_context,
        config=config,
        controller=controller,
    ) is y
    assert controller.last_local_frame_ids == (20, 21)
    assert calls == {"provider": 0, "attention": 0}
    assert controller.diagnostics()["local_frame_indices"] == (20, 21)


def test_call_scoped_diagnostics_clear_after_enabled_call():
    config = LPHCConfig(alpha=0.1, schedule="e1")
    controller = LPHCController(config, layer_idx=0)
    y = torch.ones(1, 2, 1, 2)
    q = torch.ones_like(y)
    k = torch.ones_like(y)
    v = torch.ones_like(y)
    history = LPHCHistory(
        torch.ones(1, 2, 1, 2),
        torch.ones(1, 2, 1, 2) * 2,
        torch.tensor([7]),
        key_is_roped=True,
    )

    apply_lphc_attention(
        y, q, k, v,
        history_provider=lambda _: history,
        attention_fn=lambda query, key, value: torch.ones_like(query) * 2,
        context=_context(kind="noisy", phase=0, local=(8,)),
        config=config,
        controller=controller,
    )
    assert controller.last_correction_ratio_max > 0

    output = apply_lphc_attention(
        y, q, k, v,
        history_provider=lambda _: pytest.fail("disabled phase retrieved history"),
        attention_fn=lambda *_: pytest.fail("disabled phase ran attention"),
        context=_context(kind="noisy", phase=1, local=(9,)),
        config=config,
        controller=controller,
    )
    assert output is y
    assert controller.last_local_frame_ids == (9,)
    assert controller.last_selected_frame_ids == ()
    assert controller.last_eligible_frame_ids == ()
    assert controller.last_correction_ratio_max == 0.0


def test_absent_clean_phase_disabled_and_empty_paths_are_exact_identity():
    config = LPHCConfig(alpha=0.2, schedule="e1")
    controller = LPHCController(config, layer_idx=0)
    y, q, k, v, _ = _attention_inputs()
    provider_calls = 0
    attention_calls = 0

    def provider(_):
        nonlocal provider_calls
        provider_calls += 1
        return LPHCHistory(
            torch.empty(1, 0, 0, 0),
            torch.empty(1, 0, 0, 0),
            torch.empty(0, dtype=torch.long),
        )

    def attention(*_):
        nonlocal attention_calls
        attention_calls += 1
        return y

    common = dict(history_provider=provider, attention_fn=attention)
    assert apply_lphc_attention(y, q, k, v, context=_context(), config=None, controller=controller, **common) is y
    assert apply_lphc_attention(y, q, k, v, context=_context(), config=config, controller=None, **common) is y
    assert apply_lphc_attention(y, q, k, v, context=_context(kind="clean", local=(3, 4)), config=config, controller=controller, **common) is y
    assert controller.diagnostics()["local_frame_indices"] == (3, 4)
    assert apply_lphc_attention(y, q, k, v, context=_context(phase=1, local=(5, 6)), config=config, controller=controller, **common) is y
    assert controller.diagnostics()["local_frame_indices"] == (5, 6)
    assert provider_calls == 0
    assert attention_calls == 0

    assert apply_lphc_attention(y, q, k, v, context=_context(), config=config, controller=controller, **common) is y
    assert provider_calls == 1
    assert attention_calls == 0


def test_archive_keeps_exact_full_frames_with_stable_content_independent_overflow():
    ids = tuple(range(20))
    first = _ready_controller(frame_ids=ids)
    second = LPHCController(first.config, layer_idx=3)
    k2, v2 = _kv(ids, offset=1000.0)
    second.commit_clean(k2, v2, ids, context=_context(kind="clean"))

    kept = first.archive.frame_ids.tolist()
    assert len(kept) == 12
    assert kept == sorted(kept)
    assert kept[0] == 0 and kept[-1] == 19
    assert kept == second.archive.frame_ids.tolist()
    source_k, source_v = _kv(ids)
    for slot, frame_id in enumerate(kept):
        start = frame_id * 2
        assert torch.equal(first.archive.k[slot], source_k[0, start : start + 2])
        assert torch.equal(first.archive.v[slot], source_v[0, start : start + 2])

    priorities = sorted(
        range(1, 19), key=lambda value: (hashlib.sha256(str(value).encode("ascii")).digest(), value)
    )[:10]
    assert kept == sorted([0, 19, *priorities])


def test_archive_rejects_partial_batch_duplicate_and_nonmonotonic_frames():
    controller = LPHCController(LPHCConfig(alpha=0.1), layer_idx=0)
    k, v = _kv((0, 1))
    with pytest.raises(ValueError, match="batch-one|shape"):
        controller.commit_clean(k.expand(2, -1, -1, -1), v.expand(2, -1, -1, -1), (0, 1))
    with pytest.raises(ValueError, match="complete"):
        controller.commit_clean(k[:, :-1], v[:, :-1], (0, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        controller.commit_clean(k, v, (1, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        controller.commit_clean(k, v, (1, 0))

    controller.commit_clean(k, v, (0, 1))
    one_k, one_v = _kv((2,))
    with pytest.raises(ValueError, match="across clean commits"):
        controller.commit_clean(one_k, one_v, (1,))
    with pytest.raises(ValueError, match="clean commits only"):
        controller.commit_clean(one_k, one_v, (2,), context=_context(kind="noisy"))


def test_correct_and_random_use_same_pool_count_and_private_deterministic_rng():
    correct = _ready_controller(LPHCConfig(alpha=0.1, mode="correct"))
    random_a = _ready_controller(LPHCConfig(alpha=0.1, mode="random", control_seed=91))
    random_b = _ready_controller(LPHCConfig(alpha=0.1, mode="random", control_seed=91))
    context = _context(local=(1, 4, 7))

    state = torch.random.get_rng_state().clone()
    correct_history = correct.retrieve(context)
    random_history_a = random_a.retrieve(context)
    random_history_b = random_b.retrieve(context)

    assert correct.last_eligible_frame_ids == random_a.last_eligible_frame_ids
    assert len(correct_history.frame_ids) == len(random_history_a.frame_ids) == 4
    assert random_history_a.frame_ids.tolist() == random_history_b.frame_ids.tolist()
    assert random_history_a.frame_ids.tolist() == sorted(random_history_a.frame_ids.tolist())
    assert torch.equal(torch.random.get_rng_state(), state)


def test_v211_budget_one_selects_one_deterministically_and_freezes_block():
    config = LPHCConfig(alpha=0.1, protocol="v211", history_budget=1)
    first = _ready_controller(config)
    second = _ready_controller(config)
    context = _context(local=(1, 4, 7), block=9)

    first_history = first.retrieve(context)
    second_history = second.retrieve(context)
    assert first_history.frame_ids.numel() == 1
    assert torch.equal(first_history.frame_ids, second_history.frame_ids)
    frozen_id = first_history.frame_ids.item()

    first.previous_clean_descriptor = -first.previous_clean_descriptor
    assert first.retrieve(context).frame_ids.tolist() == [frozen_id]
    assert first.diagnostics()["selected_count"] == 1


def test_correct_ties_choose_lower_ids_then_return_chronological_order():
    controller = LPHCController(LPHCConfig(alpha=0.1), layer_idx=0)
    ids = tuple(range(6))
    k = torch.zeros(1, 12, 1, 2)
    v = torch.ones_like(k)
    controller.commit_clean(k, v, ids)
    controller.finish_clean_block(torch.ones(1, 2, 1, 2), (10,))

    history = controller.retrieve(_context(local=()))
    assert history.frame_ids.tolist() == [0, 1, 2, 3]


def test_selection_is_frozen_per_block_and_overlap_is_excluded():
    controller = _ready_controller()
    context = _context(local=(0, 1, 2), block=4)
    first = controller.retrieve(context)
    controller.previous_clean_descriptor = -controller.previous_clean_descriptor
    second = controller.retrieve(context)

    assert first.frame_ids.tolist() == second.frame_ids.tolist()
    assert not set(first.frame_ids.tolist()) & {0, 1, 2}
    assert controller.diagnostics()["frozen_blocks"] == 1


def test_finish_clean_block_uses_previous_completed_descriptor_and_invalidates_selection():
    controller = _ready_controller()
    context = _context(block=8)
    first = controller.retrieve(context)
    _, new_clean = _kv((500, 501), offset=-1000.0)
    descriptor = controller.finish_clean_block(new_clean, (500, 501))

    assert torch.equal(descriptor, controller.previous_clean_descriptor)
    assert controller.diagnostics()["frozen_blocks"] == 0
    second = controller.retrieve(context)
    assert controller.diagnostics()["frozen_blocks"] == 1
    assert first.frame_ids.numel() == second.frame_ids.numel() == 4


def test_integration_controller_api_names():
    config = LPHCConfig(alpha=0.1)
    controller = LPHCController(config, layer_idx=2)
    controller.reset("trajectory", 17, 21017)
    assert controller.enabled and controller.alpha == 0.1
    assert controller.source_index == 17 and controller.generation_seed == 21017
    k, v = _kv((0, 1))
    controller.commit_clean_layer(2, k, v, (0, 1), frame_seq_length=2)
    controller.finish_clean_block(0)
    history = controller.history_for_attention(
        2, _context(local=(), trajectory="trajectory"), ()
    )
    assert history is not None
    assert history.raw_key.shape == history.value.shape
    assert history.frame_ids.tolist() == [0, 1]


def test_diagnostics_expose_trace_audit_aliases():
    controller = _ready_controller()
    controller.retrieve(_context(local=(0, 2)))
    controller.last_correction_ratio_max = 0.075
    diagnostics = controller.diagnostics()

    assert diagnostics["clean_history_reads"] == 0
    assert diagnostics["archive_frame_indices"] == diagnostics["archive_frame_ids"]
    assert diagnostics["eligible_frame_indices"] == controller.last_eligible_frame_ids
    assert diagnostics["local_frame_indices"] == (0, 2)
    assert diagnostics["selected_history_frames"] == controller.last_selected_frame_ids
    assert diagnostics["selected_count"] == len(controller.last_selected_frame_ids)
    assert diagnostics["correction_ratio"] == 0.075
    assert diagnostics["lookup_count"] == controller.counters["retrieval_calls"]
    assert diagnostics["random_count"] == controller.counters["random_selections"]
    assert diagnostics["second_attention_count"] == controller.counters["second_attention_calls"]


def test_reset_clears_payload_descriptor_selection_counters_and_trajectory():
    controller = _ready_controller()
    controller.retrieve(_context())
    assert controller.archive.frame_ids is not None
    assert controller.trajectory_id == "prompt-a"
    controller.reset()

    assert controller.archive.frame_ids is None
    assert controller.previous_clean_descriptor is None
    assert controller.trajectory_id is None
    assert controller.diagnostics()["frozen_blocks"] == 0
    assert controller.diagnostics()["max_correction_ratio"] == 0.0
    assert all(value == 0 for key, value in controller.counters.items())


def test_enabled_path_uses_rope_and_exactly_one_second_attention():
    config = LPHCConfig(alpha=0.25)
    controller = LPHCController(config, layer_idx=0)
    y, q, k, v, history = _attention_inputs()
    raw_history = LPHCHistory(history.k, history.v, history.frame_ids, key_is_roped=False)
    calls = {"rope": 0, "attention": 0}

    def rope(raw_k, frame_ids):
        calls["rope"] += 1
        assert frame_ids.tolist() == [2]
        return raw_k + 1

    def attention(query, augmented_k, augmented_v):
        calls["attention"] += 1
        assert augmented_k.shape[1] == k.shape[1] + history.k.shape[1]
        assert augmented_v.shape[1] == v.shape[1] + history.v.shape[1]
        return y + 0.5

    result = apply_lphc_attention(
        y,
        q,
        k,
        v,
        history_provider=lambda _: raw_history,
        attention_fn=attention,
        context=_context(),
        config=config,
        controller=controller,
        history_rope_fn=rope,
    )
    assert result is not y
    assert calls == {"rope": 1, "attention": 1}
    assert controller.counters["second_attention_calls"] == 1
    diagnostics = controller.diagnostics()
    assert 0.0 <= diagnostics["correction_ratio"] <= config.alpha
    assert diagnostics["max_correction_ratio"] == diagnostics["correction_ratio"]


def test_rms_correction_bound_is_per_batch_and_head():
    config = LPHCConfig(alpha=0.3)
    controller = LPHCController(config, layer_idx=0)
    y, q, k, v, history = _attention_inputs(batch=2)
    delta = torch.randn_like(y) * torch.tensor([1.0, 100.0]).view(1, 1, 2, 1)

    result = apply_lphc_attention(
        y,
        q,
        k,
        v,
        history_provider=lambda _: history,
        attention_fn=lambda *_: y + delta,
        context=_context(),
        config=config,
        controller=controller,
    )
    correction_rms = (result.float() - y.float()).square().mean(dim=(1, 3)).sqrt()
    local_rms = y.float().square().mean(dim=(1, 3)).sqrt()
    assert torch.all(correction_rms <= config.alpha * local_rms + 1e-6)


def test_zero_delta_and_zero_local_are_finite():
    config = LPHCConfig(alpha=1.0)
    controller = LPHCController(config, layer_idx=0)
    y, q, k, v, history = _attention_inputs(batch=2)
    y.zero_()
    result = apply_lphc_attention(
        y,
        q,
        k,
        v,
        history_provider=lambda _: history,
        attention_fn=lambda *_: y.clone(),
        context=_context(),
        config=config,
        controller=controller,
    )
    assert torch.equal(result, y)
    assert torch.isfinite(result).all()


def test_nonfinite_augmented_attention_is_rejected():
    config = LPHCConfig(alpha=0.1)
    controller = LPHCController(config, layer_idx=0)
    y, q, k, v, history = _attention_inputs()
    with pytest.raises(FloatingPointError, match="nonfinite"):
        apply_lphc_attention(
            y,
            q,
            k,
            v,
            history_provider=lambda _: history,
            attention_fn=lambda *_: torch.full_like(y, float("nan")),
            context=_context(),
            config=config,
            controller=controller,
        )


def test_inputs_and_provider_payload_are_not_mutated_and_batch_one_history_expands():
    config = LPHCConfig(alpha=0.1)
    controller = LPHCController(config, layer_idx=0)
    y, q, k, v, history = _attention_inputs(batch=3)
    tensors = (y, q, k, v, history.k, history.v, history.frame_ids)
    snapshots = tuple(tensor.clone() for tensor in tensors)

    result = apply_lphc_attention(
        y,
        q,
        k,
        v,
        history_provider=lambda _: history,
        attention_fn=lambda query, augmented_k, augmented_v: torch.zeros_like(query),
        context=_context(),
        config=config,
        controller=controller,
    )
    assert result.shape == y.shape
    assert all(torch.equal(tensor, snapshot) for tensor, snapshot in zip(tensors, snapshots))
