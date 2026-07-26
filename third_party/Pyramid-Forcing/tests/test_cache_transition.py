from __future__ import annotations

import json

import pytest
import torch

from pyramidkv.adaptive_cache import AdaptiveKVCache
from pyramidkv.base import HeadComposition
from pyramidkv.config import PyramidKVConfig
from pyramidkv.cyclic import CyclicStrategy
from pyramidkv.merge import MergeStrategy
from pyramidkv.transition import CacheTransitionConfig, CacheTransitionController


def _controller(**overrides) -> CacheTransitionController:
    values = {
        "enabled": True,
        "mode": "full",
        "min_reliability": 0.55,
        "min_novelty": 0.0,
        "shock_weight": 1.0,
        "denoise_weight": 2.0,
        "min_interval_blocks": 1,
        "max_age_blocks": 3,
        "warmup_blocks": 0,
        "max_commit_fraction": 0.5,
        "stagger_period": 2,
    }
    values.update(overrides)
    return CacheTransitionController(
        CacheTransitionConfig(**values),
        batch_size=1,
        num_heads=4,
        layer_idx=0,
        head_labels=(-1, 1, 2, 1),
    )


def _kv(values) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.tensor(values, dtype=torch.float32).reshape(4, 1, 2)
    return tensor, tensor.clone()


def test_first_clean_block_initialization_respects_commit_budget():
    controller = _controller()
    k, v = _kv([
        1, 0,
        0, 1,
        1, 1,
        -1, 1,
    ])

    decision = controller.decide_clean(k, v, block_id=0, branch="cond")

    assert decision.commit_mask == (False, False, True, True)
    assert decision.reasons == (
        "initialize_budget_deferred",
        "initialize_budget_deferred",
        "initialize",
        "initialize",
    )

    second = controller.decide_clean(k, v, block_id=1, branch="cond")
    assert second.commit_mask == (True, True, False, False)
    assert second.reasons[:2] == ("initialize", "initialize")


def test_warmup_bypasses_quality_gates_but_not_commit_budget():
    controller = _controller(
        mode="gate",
        warmup_blocks=3,
        max_commit_fraction=0.5,
    )
    k, v = _kv([
        1, 0,
        0, 1,
        1, 1,
        -1, 1,
    ])
    controller.decide_clean(k, v, block_id=0, branch="cond")
    controller.decide_clean(k, v, block_id=1, branch="cond")

    decision = controller.decide_clean(k, v, block_id=2, branch="cond")

    assert sum(decision.commit_mask) == 2
    assert decision.reasons.count("warmup") == 2
    assert decision.reasons.count("warmup_budget_deferred") == 2


def test_infeasible_commit_fraction_fails_instead_of_exceeding_budget():
    with pytest.raises(ValueError, match="strict per-batch budget would allow zero"):
        CacheTransitionController(
            CacheTransitionConfig(
                enabled=True,
                mode="gate",
                max_commit_fraction=0.5,
            ),
            batch_size=1,
            num_heads=1,
            layer_idx=0,
        )


def test_denoise_disagreement_rejects_unreliable_candidate():
    controller = CacheTransitionController(
        CacheTransitionConfig(
            enabled=True,
            mode="gate",
            min_reliability=0.8,
            min_novelty=0.0,
            min_interval_blocks=1,
            max_age_blocks=4,
            warmup_blocks=0,
            max_commit_fraction=1.0,
        ),
        batch_size=1,
        num_heads=1,
        layer_idx=0,
    )
    first = torch.tensor([[[1.0, 0.0]]])
    controller.decide_clean(first, first, block_id=0, branch="cond")

    # A stable clean block keeps the active descriptor unchanged.
    controller.decide_clean(first, first, block_id=1, branch="cond")
    noisy = torch.tensor([[[1.0, 0.0]]])
    clean = torch.tensor([[[0.0, 1.0]]])
    controller.observe_noisy(noisy, noisy, block_id=2, branch="cond")
    decision = controller.decide_clean(clean, clean, block_id=2, branch="cond")

    assert decision.commit_mask == (False,)
    assert decision.reasons == ("low_reliability",)
    assert decision.denoise_disagreement[0] == pytest.approx(1.0)
    assert decision.reliability[0] < 0.8


def test_stagger_mode_limits_synchronous_head_updates():
    controller = _controller(
        mode="stagger",
        max_commit_fraction=0.5,
        stagger_period=2,
    )
    first_k, first_v = _kv([
        1, 0,
        0, 1,
        1, 1,
        -1, 1,
    ])
    controller.decide_clean(first_k, first_v, block_id=0, branch="cond")
    controller.decide_clean(first_k, first_v, block_id=1, branch="cond")
    changed_k, changed_v = _kv([
        0, 1,
        1, 0,
        -1, 1,
        1, 1,
    ])

    decision = controller.decide_clean(
        changed_k, changed_v, block_id=2, branch="cond"
    )

    assert sum(decision.commit_mask) == 2
    assert set(decision.reasons) == {"accepted", "stagger_phase"}


def test_max_age_forces_eventual_commit():
    controller = CacheTransitionController(
        CacheTransitionConfig(
            enabled=True,
            mode="gate",
            min_reliability=1.0,
            min_novelty=2.0,
            min_interval_blocks=1,
            max_age_blocks=2,
            warmup_blocks=0,
            max_commit_fraction=1.0,
        ),
        batch_size=1,
        num_heads=1,
        layer_idx=0,
    )
    first = torch.tensor([[[1.0, 0.0]]])
    controller.decide_clean(first, first, block_id=0, branch="cond")
    controller.decide_clean(first, first, block_id=1, branch="cond")
    decision = controller.decide_clean(first, first, block_id=2, branch="cond")

    assert decision.commit_mask == (True,)
    assert decision.reasons == ("forced_max_age",)


def test_forced_refresh_still_respects_transition_budget():
    controller = _controller(
        mode="gate",
        min_reliability=1.0,
        min_novelty=2.0,
        max_age_blocks=2,
        max_commit_fraction=0.5,
    )
    first_k, first_v = _kv([
        1, 0,
        0, 1,
        1, 1,
        -1, 1,
    ])
    controller.decide_clean(first_k, first_v, block_id=0, branch="cond")
    controller.decide_clean(first_k, first_v, block_id=1, branch="cond")
    controller._ages = [1] * controller.num_seq

    decision = controller.decide_clean(
        first_k, first_v, block_id=2, branch="cond"
    )

    assert sum(decision.commit_mask) == 2
    assert decision.reasons.count("forced_max_age") == 2
    assert decision.reasons.count("forced_budget_deferred") == 2


def test_role_conditioning_refreshes_reactive_heads_sooner():
    controller = CacheTransitionController(
        CacheTransitionConfig(
            enabled=True,
            mode="gate",
            min_novelty=0.01,
            min_interval_blocks=1,
            max_age_blocks=6,
            warmup_blocks=0,
            max_commit_fraction=1.0,
            role_conditioning_enabled=True,
            persistent_max_age_blocks=4,
            reactive_max_age_blocks=2,
        ),
        batch_size=1,
        num_heads=2,
        layer_idx=0,
        head_labels=(1, -1),
    )
    tensor = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    controller.decide_clean(tensor, tensor, block_id=0, branch="cond")
    controller.decide_clean(tensor, tensor, block_id=1, branch="cond")

    decision = controller.decide_clean(
        tensor, tensor, block_id=2, branch="cond"
    )

    assert decision.head_roles == ("persistent", "reactive")
    assert decision.effective_max_age == (4, 2)
    assert decision.commit_mask == (False, True)
    assert decision.reasons == ("low_novelty", "forced_max_age")


def test_reactive_utility_bias_controls_budget_tie():
    controller = CacheTransitionController(
        CacheTransitionConfig(
            enabled=True,
            mode="full",
            min_reliability=0.0,
            min_novelty=0.0,
            min_interval_blocks=1,
            max_age_blocks=6,
            warmup_blocks=0,
            max_commit_fraction=0.5,
            stagger_period=1,
            role_conditioning_enabled=True,
            persistent_min_novelty_scale=1.0,
            reactive_min_novelty_scale=1.0,
            persistent_max_age_blocks=6,
            reactive_max_age_blocks=6,
            reactive_utility_bias=0.2,
        ),
        batch_size=1,
        num_heads=2,
        layer_idx=0,
        head_labels=(-1, 1),
    )
    first = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    changed = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]])
    controller.decide_clean(first, first, block_id=0, branch="cond")
    controller.decide_clean(first, first, block_id=1, branch="cond")

    decision = controller.decide_clean(
        changed, changed, block_id=2, branch="cond"
    )

    assert decision.commit_mask == (True, False)
    assert decision.utility[0] > decision.utility[1]
    assert decision.reasons == ("accepted", "budget_deferred")


def test_trace_contains_head_level_reasons(tmp_path):
    trace = tmp_path / "transition.jsonl"
    controller = CacheTransitionController(
        CacheTransitionConfig(
            enabled=True,
            mode="audit",
            trace_path=str(trace),
        ),
        batch_size=1,
        num_heads=1,
        layer_idx=7,
        head_labels=(2,),
    )
    tensor = torch.tensor([[[1.0, 0.0]]])

    controller.decide_clean(tensor, tensor, block_id=9, branch="cond")

    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["event"] == "cache_transition"
    assert payload["layer"] == 7
    assert payload["block_id"] == 9
    assert payload["head_labels"] == [2]
    assert payload["head_roles"] == ["uniform"]
    assert payload["role_conditioning_active"] is False
    assert payload["reasons"] == ["audit_passthrough"]


def test_invalid_transition_budget_is_rejected():
    with pytest.raises(ValueError, match="max_commit_fraction"):
        CacheTransitionConfig(enabled=True, max_commit_fraction=0.0).validate()


def test_transition_role_csv_is_independent_from_pf_labels(tmp_path):
    pf_labels = tmp_path / "pf.csv"
    transition_labels = tmp_path / "transition.csv"
    pf_labels.write_text("2,2,2,2\n", encoding="utf-8")
    transition_labels.write_text("1,-1,1,-1\n", encoding="utf-8")

    config = PyramidKVConfig(
        str(pf_labels),
        num_layers=1,
        num_heads=4,
        default_capacity=16,
        transition_head_type_csv_path=str(transition_labels),
    )

    assert config.get_layer_labels(0) == [2, 2, 2, 2]
    assert config.get_layer_transition_labels(0) == [1, -1, 1, -1]


def test_transition_role_csv_shape_is_strict(tmp_path):
    transition_labels = tmp_path / "transition.csv"
    transition_labels.write_text("1,-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 4 labels"):
        PyramidKVConfig(
            None,
            num_layers=1,
            num_heads=4,
            transition_head_type_csv_path=str(transition_labels),
        )


def test_adaptive_cache_rejected_clean_block_does_not_update_middle():
    config = PyramidKVConfig(
        None,
        num_layers=1,
        num_heads=1,
        default_capacity=16,
        frame_seq_length=2,
    )
    cyclic = CyclicStrategy(period=1, bucket_cap=4)
    config.compositions = [[HeadComposition(
        name="L0_H0_test",
        label=-1,
        sink_frames=0,
        recent_frames=1,
        middle_strategies=[cyclic],
    )]]
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=2,
        layer_idx=0,
        sink_len=0,
        tail_len=16,
        use_osc_frame_mode=True,
        phase_period=1,
        phase_bucket_capacity_frames=4,
        local_tail_frames=1,
        cache_transition_enabled=True,
        cache_transition_mode="gate",
        cache_transition_min_reliability=0.8,
        cache_transition_min_novelty=0.0,
        cache_transition_warmup_blocks=0,
        cache_transition_max_commit_fraction=1.0,
    )
    grid = torch.tensor([[1, 1, 2]], dtype=torch.long)
    first = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    cache.update(
        first,
        first,
        current_start=0,
        grid_sizes=grid,
        cache_update_mode="clean",
    )

    noisy = first.clone()
    clean = torch.tensor([[[[0.0, 1.0]], [[0.0, 1.0]]]])
    cache.update(
        noisy,
        noisy,
        current_start=2,
        grid_sizes=grid,
        cache_update_mode="noisy",
    )
    cache.update(
        clean,
        clean,
        current_start=2,
        grid_sizes=grid,
        cache_update_mode="clean",
    )

    assert [anchor.t for anchor in cyclic._buckets[0][0]] == [0]
    assert cache.dynamic_k[0] is not None
    torch.testing.assert_close(cache.dynamic_k[0][-2:], clean[0, :, 0])


def test_adaptive_cache_rejection_invalidates_partial_merge_blocks():
    config = PyramidKVConfig(
        None,
        num_layers=1,
        num_heads=1,
        default_capacity=32,
        frame_seq_length=2,
    )
    merge = MergeStrategy(patch_size=2, capacity=4)
    config.compositions = [[HeadComposition(
        name="L0_H0_merge",
        label=11,
        sink_frames=0,
        recent_frames=1,
        middle_strategies=[merge],
        policy_type="merge",
    )]]
    cache = AdaptiveKVCache(
        config=config,
        batch_size=1,
        num_heads=1,
        head_dim=2,
        layer_idx=0,
        sink_len=0,
        tail_len=32,
        use_osc_frame_mode=True,
        local_tail_frames=1,
        cache_transition_enabled=True,
        cache_transition_mode="gate",
        cache_transition_min_reliability=0.8,
        cache_transition_min_novelty=0.0,
        cache_transition_warmup_blocks=0,
        cache_transition_max_commit_fraction=1.0,
    )
    grid = torch.tensor([[3, 1, 2]], dtype=torch.long)
    first = torch.zeros((1, 6, 1, 2), dtype=torch.float32)
    first[..., 0] = 1.0
    cache.update(
        first,
        first,
        current_start=0,
        grid_sizes=grid,
        cache_update_mode="clean",
    )
    assert list(merge._blocks[0]) == [0]

    noisy = first.clone()
    clean = torch.zeros_like(first)
    clean[..., 1] = 1.0
    cache.update(
        noisy,
        noisy,
        current_start=6,
        grid_sizes=grid,
        cache_update_mode="noisy",
    )
    cache.update(
        clean,
        clean,
        current_start=6,
        grid_sizes=grid,
        cache_update_mode="clean",
    )

    assert 0 not in merge._blocks[0]
    assert merge._invalid_block_ids[0] == {1}
