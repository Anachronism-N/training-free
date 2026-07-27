from __future__ import annotations

import csv
import hashlib
import importlib
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).parents[1]
PF = ROOT / "third_party" / "Pyramid-Forcing"
SCRIPTS = ROOT / "scripts"
for path in (PF, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pyramidkv.cyclic import CyclicStrategy
from pyramidkv.factory import build_compositions
from pyramidkv.merge import MergeStrategy
from pyramidkv.motion_event import MotionEventStrategy
from pyramidkv.policy_overrides import (
    HISTORY_SUPPORT_LABEL,
    HISTORY_SUPPRESS_LABEL,
    binary_head_policy_overrides,
    history_polarity_policy_overrides,
    pf_class_extended_recent_overrides,
)
from pyramidkv.stride import StrideStrategy

trace_summary = importlib.import_module("summarize_v97_policy_traces")


def _write_labels(path: Path, rows: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _factory_kwargs(overrides: dict[str, object]) -> dict[str, object]:
    return {
        key.removeprefix("pyramidkv_"): value
        for key, value in overrides.items()
        if key
        not in {
            "pyramidkv_code_map",
            "pyramidkv_composition_owns_dynamic",
        }
    }


def test_hybrid_support_and_merge_responsive_are_budget_matched(tmp_path):
    labels = tmp_path / "binary.csv"
    _write_labels(labels, [[1, -1]])
    overrides = binary_head_policy_overrides("hybrid", "merge")
    compositions = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        cyclic_period=6,
        cyclic_bucket_cap=4,
        stride_enabled=True,
        stride_interval=6,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )
    support, responsive = compositions[0]

    assert [type(strategy) for strategy in support.middle_strategies] == [
        CyclicStrategy,
        StrideStrategy,
    ]
    assert support.middle_strategies[0].bucket_cap == 2
    assert support.middle_strategies[1].capacity == 2
    assert support.sink_frames == 3
    assert support.recent_frames == 4
    assert [type(strategy) for strategy in responsive.middle_strategies] == [
        MergeStrategy
    ]
    assert responsive.middle_strategies[0].capacity == 4
    assert responsive.sink_frames == 3
    assert responsive.recent_frames == 4


def test_history_polarity_uses_neutral_labels_and_explicit_routes(tmp_path):
    labels = tmp_path / "history_polarity.csv"
    _write_labels(
        labels,
        [[HISTORY_SUPPORT_LABEL, HISTORY_SUPPRESS_LABEL]],
    )
    overrides = history_polarity_policy_overrides("hybrid", "merge")
    compositions = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        cyclic_period=6,
        cyclic_bucket_cap=4,
        stride_enabled=True,
        stride_interval=6,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )[0]
    supportive, suppressive = compositions

    assert supportive.label == 10
    assert [type(strategy) for strategy in supportive.middle_strategies] == [
        CyclicStrategy,
        StrideStrategy,
    ]
    assert supportive.middle_strategies[0].bucket_cap == 2
    assert supportive.middle_strategies[1].capacity == 2
    assert supportive.sink_frames == 3
    assert supportive.recent_frames == 4

    assert suppressive.label == 11
    assert [type(strategy) for strategy in suppressive.middle_strategies] == [
        MergeStrategy
    ]
    assert suppressive.middle_strategies[0].capacity == 4
    assert suppressive.sink_frames == 3
    assert suppressive.recent_frames == 4
    assert overrides["pyramidkv_code_map"] == {
        "10": 32760,
        "11": 32760,
    }


def test_history_polarity_cyclic_route_restores_wave_sink_contract(tmp_path):
    labels = tmp_path / "history_polarity_cyclic.csv"
    _write_labels(
        labels,
        [[HISTORY_SUPPORT_LABEL, HISTORY_SUPPRESS_LABEL]],
    )
    overrides = history_polarity_policy_overrides("stride", "cyclic")
    compositions = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        cyclic_period=6,
        cyclic_bucket_cap=4,
        stride_enabled=True,
        stride_interval=6,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )[0]
    supportive, suppressive = compositions

    assert [type(strategy) for strategy in supportive.middle_strategies] == [
        StrideStrategy
    ]
    assert supportive.sink_frames == 3
    assert supportive.recent_frames == 4
    assert [type(strategy) for strategy in suppressive.middle_strategies] == [
        CyclicStrategy
    ]
    assert suppressive.middle_strategies[0].bucket_cap == 4
    assert suppressive.sink_frames == 1
    assert suppressive.recent_frames == 4


def test_history_polarity_support_cyclic_builds_uniform_safe_carrier(tmp_path):
    labels = tmp_path / "history_polarity_all_cyclic.csv"
    _write_labels(
        labels,
        [[HISTORY_SUPPORT_LABEL, HISTORY_SUPPRESS_LABEL]],
    )
    overrides = history_polarity_policy_overrides("cyclic", "cyclic")
    compositions = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        cyclic_period=6,
        cyclic_bucket_cap=4,
        stride_enabled=True,
        stride_interval=6,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )[0]

    for composition in compositions:
        assert [
            type(strategy) for strategy in composition.middle_strategies
        ] == [CyclicStrategy]
        assert composition.middle_strategies[0].bucket_cap == 4
        assert composition.sink_frames == 1
        assert composition.recent_frames == 4
    assert overrides["pyramidkv_label_stride_enabled_map"] == {
        "10": False,
        "11": False,
    }


def test_history_polarity_motion_cyclic_has_sink3_and_two_plus_two_budget(
    tmp_path,
):
    labels = tmp_path / "history_polarity_motion.csv"
    _write_labels(
        labels,
        [[HISTORY_SUPPORT_LABEL, HISTORY_SUPPRESS_LABEL]],
    )
    overrides = history_polarity_policy_overrides(
        "stride", "motion_cyclic"
    )
    compositions = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        cyclic_period=6,
        cyclic_bucket_cap=4,
        stride_enabled=True,
        stride_interval=6,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )[0]
    supportive, suppressive = compositions

    assert [type(strategy) for strategy in supportive.middle_strategies] == [
        StrideStrategy
    ]
    assert [type(strategy) for strategy in suppressive.middle_strategies] == [
        CyclicStrategy,
        MotionEventStrategy,
    ]
    assert suppressive.middle_strategies[0].bucket_cap == 2
    assert suppressive.middle_strategies[1].capacity == 2
    assert suppressive.sink_frames == 3
    assert suppressive.recent_frames == 4


def test_history_polarity_cyclic_motion1_preserves_wave_cache_and_adds_event(
    tmp_path,
):
    labels = tmp_path / "history_polarity_cyclic_motion1.csv"
    _write_labels(
        labels,
        [[HISTORY_SUPPORT_LABEL, HISTORY_SUPPRESS_LABEL]],
    )
    overrides = history_polarity_policy_overrides(
        "stride", "cyclic_motion1"
    )
    compositions = build_compositions(
        1,
        2,
        torch.full((1, 2), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        cyclic_period=6,
        cyclic_bucket_cap=4,
        stride_enabled=True,
        stride_interval=6,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )[0]
    supportive, suppressive = compositions

    assert [type(strategy) for strategy in supportive.middle_strategies] == [
        StrideStrategy
    ]
    assert [type(strategy) for strategy in suppressive.middle_strategies] == [
        CyclicStrategy,
        MotionEventStrategy,
    ]
    assert suppressive.middle_strategies[0].bucket_cap == 4
    assert suppressive.middle_strategies[1].capacity == 1
    assert suppressive.sink_frames == 1
    assert suppressive.recent_frames == 4


def test_history_polarity_cyclic_sink3_isolated_from_wave_sink_contract():
    overrides = history_polarity_policy_overrides(
        "stride", "cyclic_sink3"
    )

    assert overrides["pyramidkv_label_phase_bucket_map"]["11"] == 4
    assert overrides["pyramidkv_label_sink_frames_map"]["11"] == 3


def test_history_polarity_recent8_matches_four_frame_middle_budget():
    overrides = history_polarity_policy_overrides("stride", "recent8")

    assert overrides["pyramidkv_label_phase_bucket_map"]["11"] == 0
    assert overrides["pyramidkv_label_merge_enabled_map"]["11"] is False
    assert (
        overrides["pyramidkv_label_motion_event_enabled_map"]["11"]
        is False
    )
    assert overrides["pyramidkv_label_recent_frames_map"]["11"] == 8


def test_history_polarity_recent_controls_expose_sink_and_budget():
    recent5 = history_polarity_policy_overrides("cyclic", "recent5")
    matched = history_polarity_policy_overrides(
        "cyclic", "recent8_sink1"
    )

    assert recent5["pyramidkv_label_sink_frames_map"]["11"] == 3
    assert recent5["pyramidkv_label_recent_frames_map"]["11"] == 5
    assert matched["pyramidkv_label_sink_frames_map"]["11"] == 1
    assert matched["pyramidkv_label_recent_frames_map"]["11"] == 8
    assert matched["pyramidkv_label_phase_bucket_map"]["11"] == 0


def test_history_polarity_rejects_pf_reserved_labels():
    with pytest.raises(ValueError, match="must not reuse PF labels"):
        history_polarity_policy_overrides(
            support_label=1,
            suppress_label=11,
        )


def test_pf_extended_recent_replaces_only_selected_class(tmp_path):
    labels = tmp_path / "pf_ablation.csv"
    _write_labels(labels, [[-1, 1, 2, 3]])
    overrides = pf_class_extended_recent_overrides("veil")
    compositions = build_compositions(
        1,
        4,
        torch.full((1, 4), 32760, dtype=torch.int32),
        csv_path=str(labels),
        cyclic_enabled=True,
        cyclic_period=6,
        cyclic_bucket_cap=4,
        stride_enabled=True,
        stride_interval=6,
        merge_enabled=True,
        **_factory_kwargs(overrides),
    )[0]

    assert isinstance(compositions[0].middle_strategies[0], CyclicStrategy)
    assert isinstance(compositions[1].middle_strategies[0], StrideStrategy)
    assert isinstance(compositions[2].middle_strategies[0], MergeStrategy)
    assert compositions[3].middle_strategies == []
    assert compositions[3].sink_frames == 3
    assert compositions[3].recent_frames == 5


def test_trace_audit_matches_frozen_binary_policy(tmp_path):
    labels = tmp_path / "labels.csv"
    _write_labels(labels, [[1, -1]])
    label_hash = hashlib.sha256(labels.read_bytes()).hexdigest()
    config = tmp_path / "method.env"
    config.write_text(
        "\n".join(
            [
                "name=method",
                "mode=binary",
                f"labels={labels}",
                f"label_sha256={label_hash}",
                "score_sha256=score-hash",
                "stable_policy=stride",
                "responsive_policy=merge",
                "pf_extended_recent_ablation=none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace = tmp_path / "method.policy.jsonl"
    events = [
        {
            "event": "middle_selection",
            "layer": 0,
            "head": 0,
            "seq": 0,
            "branch": "cond",
            "sync_t": 12,
            "label": 1,
            "sink_frames": 3,
            "recent_frames": 4,
            "policy_type": "stride",
            "strategies": [
                {
                    "name": "StrideStrategy",
                    "frame_ids": [6],
                    "token_count": 16,
                }
            ],
            "union_frame_ids": [6],
            "union_frame_count": 1,
            "union_token_count": 16,
        },
        {
            "event": "middle_selection",
            "layer": 0,
            "head": 1,
            "seq": 1,
            "branch": "cond",
            "sync_t": 12,
            "label": -1,
            "sink_frames": 3,
            "recent_frames": 4,
            "policy_type": "merge",
            "strategies": [
                {
                    "name": "MergeStrategy",
                    "frame_ids": [5],
                    "token_count": 4,
                }
            ],
            "union_frame_ids": [5],
            "union_frame_count": 1,
            "union_token_count": 4,
        },
    ]
    import json

    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    result = trace_summary.summarize_method(
        "method",
        trace,
        config,
        expected_layers={0},
        num_layers=1,
        num_heads=2,
    )

    assert result["status"] == "nominal"
    assert result["events"] == 2
    assert result["strategy_events"] == {
        "MergeStrategy": 1,
        "StrideStrategy": 1,
    }
