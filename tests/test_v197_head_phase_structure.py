from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def write_fixture(module, root: Path) -> tuple[Path, Path]:
    manifest = write_json(root / "v189/inputs/manifest.json", {"frozen": True})
    rows = []
    operators = {}
    for operator_index, operator in enumerate(("landmark", "retrieval")):
        masks = [
            [[False for _ in range(module.HEADS)] for _ in range(module.LAYERS)]
            for _ in range(module.CALLS)
        ]
        operator_scale = 1.0 - 0.05 * operator_index
        for call in range(module.CALLS):
            for layer in range(module.LAYERS):
                for head in range(module.HEADS):
                    stable_head = 0.060 if head < 3 else -0.012
                    phase_signal = 0.050 if call == head % module.CALLS else 0.0
                    layer_signal = 0.004 if 8 <= layer <= 20 else 0.0
                    discovery = operator_scale * (
                        stable_head + phase_signal + layer_signal
                    )
                    validation = 0.92 * discovery + 0.0002 * ((layer + head) % 3 - 1)
                    win = 0.78 if validation > 0 else 0.35
                    ci_lower = validation - (0.004 if validation > 0 else 0.020)
                    compatible = bool(
                        discovery >= module.PRIMARY_GAIN_THRESHOLD
                        and validation >= module.VALIDATION_GAIN_THRESHOLD
                        and ci_lower >= module.VALIDATION_CI_LOWER
                        and win >= module.PRIMARY_WIN_THRESHOLD
                    )
                    masks[call][layer][head] = compatible
                    rows.append(
                        {
                            "operator": operator,
                            "call_index": call,
                            "layer": layer,
                            "head": head,
                            "discovery_gain": discovery,
                            "validation_gain": validation,
                            "validation_ci_lower": ci_lower,
                            "validation_win_fraction": win,
                            "full_budget_fraction": 1.0,
                            "relative_reference_energy": 1.0,
                            "compatible": compatible,
                        }
                    )
        map_path = write_json(
            root / f"v189/analysis/maps/{operator}_compatible.json",
            {
                "version": 1,
                "map_id": f"{operator}-compatible",
                "coverage_operator": operator,
                "classification": "compatible",
                "call_count": module.CALLS,
                "layer_count": module.LAYERS,
                "head_count": module.HEADS,
                "coverage_masks": masks,
                "coverage_count_by_call": [
                    sum(value for layer in masks[call] for value in layer)
                    for call in range(module.CALLS)
                ],
            },
        )
        operators[operator] = {
            "maps": {
                "compatible": {
                    "path": str(map_path),
                    "sha256": module.sha256(map_path),
                    "map_id": f"{operator}-compatible",
                }
            }
        }
    score_path = root / "v189/analysis/cell_scores.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analysis_path = write_json(
        root / "v189/analysis/analysis.json",
        {
            "version": 1,
            "experiment": module.SOURCE_EXPERIMENT,
            "input_manifest": str(manifest),
            "input_manifest_sha256": module.sha256(manifest),
            "operators": operators,
            "generation_candidates": ["landmark", "retrieval"],
            "recommendation": "advance_head_phase_maps_to_causal_screen",
            "manual_review_required": False,
        },
    )
    return analysis_path, score_path


def test_v197_reports_cross_split_joint_structure_without_changing_map(
    tmp_path: Path,
) -> None:
    module = load_module("v197_joint", SCRIPTS / "analyze_v197_head_phase_structure.py")
    analysis, scores = write_fixture(module, tmp_path)
    report = module.analyze(analysis, scores, tmp_path / "v197", draws=300)
    assert report["diagnostic_only"] is True
    assert report["changes_v189_frozen_map"] is False
    assert report["generation_gate"] == "v190_only"
    assert report["manual_review_required"] is False
    for operator in ("landmark", "retrieval"):
        row = report["operators"][operator]
        assert row["diagnostic_structure_level"] == "joint_head_phase_structure"
        assert row["continuous_reproducibility"]["spearman_all_cells"] > 0.95
        assert (
            row["continuous_reproducibility"]["head_phase_interaction_correlation"]
            > 0.95
        )
        assert (
            row["crossfit"]["head_identity_topk"][1]["one_sided_permutation_p"] <= 0.05
        )
        assert row["crossfit"]["phase_identity_top1"]["one_sided_permutation_p"] <= 0.05
        assert row["primary_map_topology"]["phase_varying_heads"] > 0
    assert (tmp_path / "v197/analysis.json").is_file()
    assert (tmp_path / "v197/threshold_grid.csv").is_file()
    assert (tmp_path / "v197/crossfit_tests.csv").is_file()
    assert (tmp_path / "v197/analysis.md").is_file()


def test_v197_rejects_csv_membership_that_disagrees_with_frozen_map(
    tmp_path: Path,
) -> None:
    module = load_module(
        "v197_map_drift", SCRIPTS / "analyze_v197_head_phase_structure.py"
    )
    analysis, scores = write_fixture(module, tmp_path)
    with scores.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["compatible"] = (
        "False" if rows[0]["compatible"].lower() == "true" else "True"
    )
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CSV/map membership disagreement"):
        module.analyze(analysis, scores, tmp_path / "v197", draws=100)


def test_v197_rejects_incomplete_cell_tensor(tmp_path: Path) -> None:
    module = load_module(
        "v197_incomplete", SCRIPTS / "analyze_v197_head_phase_structure.py"
    )
    analysis, scores = write_fixture(module, tmp_path)
    lines = scores.read_text(encoding="utf-8").splitlines()
    scores.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete v189 score tensor"):
        module.analyze(analysis, scores, tmp_path / "v197", draws=100)


def test_v197_runner_is_zero_gpu_and_packages_only_analysis() -> None:
    runner = (SCRIPTS / "run_v197_head_phase_structure.sh").read_text(encoding="utf-8")
    assert "analyze|show|package" in runner
    assert "CUDA_VISIBLE_DEVICES" not in runner
    assert 'tar -czf "$archive" -C "$OUT_ROOT" analysis' in runner
    assert "tar -czf" in runner
    assert "manual" not in runner.lower()
