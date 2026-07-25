from __future__ import annotations

import csv

import torch

from pyramidkv.cyclic import CyclicStrategy
from pyramidkv.factory import build_compositions
from pyramidkv.merge import MergeStrategy
from pyramidkv.policy_overrides import binary_responsive_policy_overrides
from pyramidkv.stride import StrideStrategy
from wan.modules.attention.head_profile import HeadQKProfileRecorder


def test_binary_merge_override_maps_responsive_to_veil_policy():
    values = binary_responsive_policy_overrides("merge")

    assert values["pyramidkv_label_phase_bucket_map"]["-1"] == 0
    assert values["pyramidkv_label_merge_enabled_map"]["-1"]
    assert not values["pyramidkv_label_stride_enabled_map"]["-1"]
    assert values["pyramidkv_label_sink_frames_map"]["-1"] == 3


def test_binary_cyclic_override_remains_available_as_fallback():
    values = binary_responsive_policy_overrides("cyclic")

    assert values["pyramidkv_label_phase_bucket_map"]["-1"] == 4
    assert not values["pyramidkv_label_merge_enabled_map"]["-1"]
    assert values["pyramidkv_label_sink_frames_map"]["-1"] == 1


def test_binary_policy_builds_requested_middle_strategy(tmp_path):
    labels = tmp_path / "labels.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([-1, 1])

    for policy, responsive_type in (
        ("merge", MergeStrategy),
        ("cyclic", CyclicStrategy),
        ("recent", None),
    ):
        overrides = binary_responsive_policy_overrides(policy)
        kwargs = {
            key.removeprefix("pyramidkv_"): value
            for key, value in overrides.items()
        }
        compositions = build_compositions(
            1,
            2,
            [[32760, 32760]],
            csv_path=str(labels),
            cyclic_enabled=True,
            cyclic_period=6,
            cyclic_bucket_cap=4,
            stride_enabled=True,
            stride_capacity=4,
            merge_enabled=True,
            **kwargs,
        )
        responsive, stable = compositions[0]

        assert any(
            isinstance(strategy, StrideStrategy)
            for strategy in stable.middle_strategies
        )
        if responsive_type is None:
            assert not responsive.middle_strategies
        else:
            assert len(responsive.middle_strategies) == 1
            assert isinstance(
                responsive.middle_strategies[0], responsive_type
            )


def test_qk_recorder_keeps_only_strict_history():
    recorder = HeadQKProfileRecorder()
    recorder.reset(
        max_calls_per_location=2,
        max_records_per_layer_branch=8,
        update_modes=("noisy",),
        branches=("cond",),
    )
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
    probabilities = torch.softmax(logits, dim=-1)

    recorder(
        layer_idx=3,
        frame_attn_logits=logits,
        frame_attn_prob=probabilities,
        q_frame_indices=[5, 6],
        k_frame_indices=[0, 1, 2, 5, 6, 7],
        current_start=10,
        cache_update_mode="noisy",
        cfg_branch="cond",
    )

    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record["layer"] == 3
    assert record["query_frame"] == 6
    assert record["key_frames"].tolist() == [0, 1, 2, 5]
    assert tuple(record["logits"].shape) == (2, 4)
