from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy_overrides = _load(
    "v99_policy_overrides",
    PF_ROOT / "pyramidkv" / "policy_overrides.py",
)
pipeline_config = _load(
    "v99_pipeline_config",
    PF_ROOT / "pipeline" / "pyramidkv_config.py",
)
runner = _load(
    "v99_recovery_runner",
    ROOT / "scripts" / "run_v99_binary_cache_recovery_4node_32gpu.py",
)


def test_neutral_binary_route_enables_exclusive_cache_ownership():
    overrides = policy_overrides.history_polarity_policy_overrides(
        "stride",
        "cyclic",
    )

    assert overrides["pyramidkv_composition_owns_dynamic"] is True
    assert overrides["pyramidkv_label_sink_frames_map"] == {
        "10": 3,
        "11": 1,
    }
    assert overrides["pyramidkv_label_recent_frames_map"] == {
        "10": 4,
        "11": 4,
    }


def test_native_pipeline_default_preserves_legacy_dynamic_behavior():
    default = pipeline_config.PyramidKVPipelineConfig()
    enabled = pipeline_config.PyramidKVPipelineConfig.from_args(
        SimpleNamespace(pyramidkv_composition_owns_dynamic=True)
    )

    assert default.pyramidkv_composition_owns_dynamic is False
    assert enabled.pyramidkv_composition_owns_dynamic is True


def test_smoke_mode_generates_exactly_one_selected_pf_video():
    cells = runner.cells_for_mode("smoke1")

    assert [cell.name for cell in cells] == [
        "pf_ar_neutral_stride_cyclic",
    ]
    assert all(cell.engine == "pf" for cell in cells)
    assert all(cell.route == "history_stride_cyclic" for cell in cells)
    assert [
        cell.name
        for cell in runner.cells_for_mode(
            "smoke1",
            "history-polarity",
        )
    ] == ["history_polarity_stride_cyclic"]
    assert [
        cell.name
        for cell in runner.cells_for_mode(
            "smoke1",
            "history-polarity-stride-merge",
        )
    ] == ["history_polarity_stride_merge_fixed"]
    assert [
        cell.name
        for cell in runner.cells_for_mode(
            "smoke1",
            "pf-aw-stride-merge",
        )
    ] == ["pf_aw_neutral_stride_merge"]
    assert [
        cell.name
        for cell in runner.cells_for_mode(
            "smoke1",
            "pf-aw-stride-cyclic",
        )
    ] == ["pf_aw_neutral_stride_cyclic"]
    assert [
        cell.name
        for cell in runner.cells_for_mode(
            "smoke1",
            "history-polarity-random-stride-merge",
        )
    ] == ["history_polarity_random_stride_merge"]
    assert runner.EXCLUSIVE_CACHE_CONTRACT["supportive_label_10"] == {
        "sink_frames": 3,
        "middle": "stride",
        "middle_capacity_frames": 4,
        "stride_interval": 6,
        "recent_frames": 4,
    }
    assert runner.EXCLUSIVE_CACHE_CONTRACT["responsive_label_11"] == {
        "sink_frames": 1,
        "middle": "cyclic",
        "middle_capacity_frames": 4,
        "cyclic_period": 6,
        "recent_frames": 4,
    }
    assert runner.exclusive_cache_contract(
        "history_stride_merge"
    )["suppressive_label_11"] == {
        "sink_frames": 3,
        "middle": "merge",
        "middle_capacity_frames": 4,
        "merge_patch_size": 2,
        "merge_block_frames": 4,
        "recent_frames": 4,
    }


def test_runner_accepts_documented_path_environment(monkeypatch, tmp_path):
    out_root = tmp_path / "out"
    score_root = tmp_path / "scores"
    monkeypatch.setenv("OUT_ROOT", str(out_root))
    monkeypatch.setenv("SCORE_ROOT", str(score_root))
    monkeypatch.setattr(sys, "argv", ["runner", "smoke1"])

    args = runner.parse_args()

    assert args.out_root == out_root.resolve()
    assert args.score_root == score_root.resolve()


def test_reused_baseline_manifest_records_requested_video_count(
    monkeypatch,
    tmp_path,
):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("prompt 0\nprompt 1\n", encoding="utf-8")
    pf_dir = tmp_path / "pf"
    binary_dir = tmp_path / "binary"
    pf_dir.mkdir()
    binary_dir.mkdir()
    out_root = tmp_path / "out"
    (out_root / "diagnostics").mkdir(parents=True)
    args = SimpleNamespace(
        out_root=out_root,
        reuse_pf_dir=pf_dir,
        reuse_pf_binary_dir=binary_dir,
        reuse_sf_dir=None,
        node_rank=0,
        mode="smoke1",
        prompts=prompt_file,
        map_wait_seconds=1,
    )

    def fake_audit(
        _args,
        *,
        output,
        start,
        end,
        report,
        log,
    ):
        del _args, output, start, end, log
        report.write_text('{"ok":true}\n', encoding="utf-8")
        return {"ok": True, "input_fingerprint": report.stem}

    monkeypatch.setattr(runner, "audit_videos", fake_audit)

    result = runner.ensure_reused_baselines(args, start=1, end=2)
    manifest = json.loads(
        (out_root / "reused_baselines.json").read_text(encoding="utf-8")
    )

    assert result["sources"]["pf_native"]["video_count"] == 1
    assert manifest["sources"]["pf_binary_read_reference"][
        "video_count"
    ] == 1


def test_nonzero_node_waits_for_identical_frozen_contract(tmp_path):
    path = tmp_path / "contract.json"
    payload = {"version": 1, "cells": ["a", "b"]}
    expected = runner.write_frozen(path, payload)

    assert runner.wait_for_frozen(
        path,
        payload,
        timeout_seconds=1,
    ) == expected

    with pytest.raises(RuntimeError, match="mixed frozen artifact"):
        runner.wait_for_frozen(
            path,
            {"version": 2},
            timeout_seconds=1,
        )


def test_full_recovery_matrix_keeps_random_inversion_and_threshold_controls():
    names = {cell.name for cell in runner.cells_for_mode("screen32")}

    assert names == {
        "pf_ar_neutral_stride_cyclic",
        "history_polarity_stride_cyclic",
        "history_polarity_stride_cyclic_v78",
        "history_polarity_random_stride_cyclic",
        "history_polarity_inverted_stride_cyclic",
        "history_polarity_tau_m0p1_stride_cyclic",
        "history_polarity_tau_p0p1_stride_cyclic",
        "history_polarity_stride_merge_fixed",
    }


def test_candidate_matrix_excludes_merge_and_main128_is_minimal():
    candidate = runner.cells_for_mode("candidate32")
    main = runner.cells_for_mode("main128")

    assert [cell.name for cell in candidate] == [
        "pf_ar_neutral_stride_cyclic",
        "pf_aw_neutral_stride_cyclic",
        "history_polarity_stride_cyclic",
        "history_polarity_random_stride_cyclic",
        "history_polarity_inverted_stride_cyclic",
        "history_polarity_tau_m0p1_stride_cyclic",
        "history_polarity_tau_p0p1_stride_cyclic",
        "history_polarity_stride_cyclic_v78",
    ]
    assert all(cell.route == "history_stride_cyclic" for cell in candidate)
    assert [cell.name for cell in main] == [
        "pf_ar_neutral_stride_cyclic",
        "history_polarity_stride_cyclic",
        "history_polarity_stride_cyclic_v78",
    ]


def test_causal_matrix_is_the_four_requested_head_route_controls():
    cells = runner.cells_for_mode("causal32")

    assert [cell.name for cell in cells] == [
        "pf_aw_neutral_stride_merge",
        "history_polarity_stride_merge_fixed",
        "history_polarity_stride_cyclic",
        "history_polarity_random_stride_merge",
    ]
    assert [cell.route for cell in cells] == [
        "history_stride_merge",
        "history_stride_merge",
        "history_stride_cyclic",
        "history_stride_merge",
    ]
    assert [cell.map_key for cell in cells] == [
        "pf_aw_binary_control",
        "history_polarity_zero",
        "history_polarity_zero",
        "history_polarity_zero_random",
    ]


def test_binary_map_statistics_are_internally_consistent():
    pf_matrix = [
        [-1] * 4 + [1] * 6 + [2] * 2
        for _ in range(30)
    ]
    binary_matrix = [
        [10] + [11] * 11
        for _ in range(30)
    ]

    assert runner.binary_map_statistics(binary_matrix, pf_matrix) == {
        "label_counts": {"10": 30, "11": 330},
        "pf_cross_tab": {
            "wave": {
                "pf_label": -1,
                "heads": 120,
                "history_supportive": 30,
                "history_suppressive": 90,
            },
            "anchor": {
                "pf_label": 1,
                "heads": 180,
                "history_supportive": 0,
                "history_suppressive": 180,
            },
            "veil": {
                "pf_label": 2,
                "heads": 60,
                "history_supportive": 0,
                "history_suppressive": 60,
            },
        },
    }


def test_binary_map_statistics_include_zero_count_role():
    pf_matrix = [
        [-1] * 4 + [1] * 6 + [2] * 2
        for _ in range(30)
    ]
    binary_matrix = [[11] * 12 for _ in range(30)]

    statistics = runner.binary_map_statistics(binary_matrix, pf_matrix)

    assert statistics["label_counts"] == {"10": 0, "11": 360}


def test_map_manifest_rejects_documented_counts_that_do_not_match_csv(
    tmp_path,
):
    pf_path = tmp_path / "pf.csv"
    map_path = tmp_path / "binary.csv"
    pf_matrix = [
        [-1] * 4 + [1] * 6 + [2] * 2
        for _ in range(30)
    ]
    binary_matrix = [
        [10] + [11] * 11
        for _ in range(30)
    ]
    for path, matrix in ((pf_path, pf_matrix), (map_path, binary_matrix)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(matrix)
    statistics = runner.binary_map_statistics(binary_matrix, pf_matrix)
    required = (
        "history_polarity_zero",
        "history_polarity_zero_random",
        "history_polarity_zero_inverted",
        "history_polarity_m0p1",
        "history_polarity_0p1",
        "pf_ar_binary_control",
        "pf_aw_binary_control",
    )
    maps = {
        name: {
            "path": map_path.name,
            "sha256": runner.sha256(map_path),
            **json.loads(json.dumps(statistics)),
        }
        for name in required
    }
    manifest_path = tmp_path / "history_polarity_manifest.json"
    manifest = {
        "pf_labels": str(pf_path.resolve()),
        "pf_labels_sha256": runner.sha256(pf_path),
        "maps": maps,
    }
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    resolved = runner.validate_map_manifest(
        manifest_path,
        pf_labels=pf_path,
    )
    assert resolved["history_polarity_zero"]["label_counts"] == {
        "10": 30,
        "11": 330,
    }

    manifest["maps"]["history_polarity_zero"]["label_counts"]["10"] = 31
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest label_counts differs"):
        runner.validate_map_manifest(
            manifest_path,
            pf_labels=pf_path,
        )


def test_video_audit_uses_the_auditor_ok_field(monkeypatch, tmp_path):
    report = tmp_path / "audit.json"
    report.write_text(
        '{"ok":true,"input_fingerprint":"verified"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "run_checked", lambda *args, **kwargs: None)
    args = SimpleNamespace(repo_root=ROOT)

    payload = runner.audit_videos(
        args,
        output=tmp_path,
        start=0,
        end=1,
        report=report,
        log=tmp_path / "audit.log",
    )

    assert payload["ok"] is True


def test_v99_trace_audit_rejects_duplicate_dynamic_tokens(tmp_path):
    head_map = tmp_path / "labels.csv"
    rows = [
        [10 if head % 2 == 0 else 11 for head in range(12)]
        for _ in range(30)
    ]
    with head_map.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    events = []
    for layer in runner.TRACE_LAYERS:
        for head in range(12):
            label = rows[layer][head]
            sink_frames = 3 if label == 10 else 1
            events.append(
                {
                    "event": "middle_selection",
                    "layer": layer,
                    "head": head,
                    "prompt_id": 0,
                    "label": label,
                    "branch": "cond",
                    "strategies": [
                        {
                            "name": (
                                "StrideStrategy"
                                if label == 10
                                else "CyclicStrategy"
                            )
                        }
                    ],
                    "sink_frames": sink_frames,
                    "recent_frames": 4,
                    "policy_type": "stride" if label == 10 else "osc",
                    "frame_seqlen": 2,
                    "sink_frame_count": sink_frames,
                    "sink_token_count": sink_frames * 2,
                    "recent_frame_count": 4,
                    "recent_token_count": 8,
                    "union_token_count": 0,
                    "cache_contract_pass": True,
                    "cache_contract_violations": [],
                    "middle_sink_overlap": [],
                    "middle_recent_overlap": [],
                    "composition_present": True,
                    "dynamic_policy_owner": "composition_recent",
                    "explicit_composition_owns_dynamic": True,
                }
            )
    events[0]["recent_token_count"] = 10
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="dynamic token budget exceeded"):
        runner.audit_trace(
            trace,
            map_path=head_map,
            route="history_stride_cyclic",
            prompt_count=1,
            output_path=tmp_path / "audit.json",
        )
