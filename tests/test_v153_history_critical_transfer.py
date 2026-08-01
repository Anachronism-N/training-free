from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v152_one_sided_history_critical as analysis  # noqa: E402
import run_v153_history_critical_transfer_1video as runner  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


RESULT_ROOT = ROOT / "docs" / "results" / "v152_online_policy_profile" / "core"
PF_LABELS = (
    ROOT
    / "third_party"
    / "Pyramid-Forcing"
    / "configs"
    / "head_configs"
    / "best_labels.csv"
)
MAP_DIR = ROOT / "configs" / "head_maps"


def test_v152_one_sided_gate_does_not_relabel_symmetric_failure() -> None:
    result = analysis.analyze_one_sided_gates(RESULT_ROOT)

    assert result["one_sided_transfer_candidate"] is True
    assert result["qk_high_uniform_qualifying_contexts"] == [
        "noisy_f117_t1000",
        "noisy_f117_t750",
        "noisy_f117_t500",
    ]
    assert result["qk_high_beats_random_qualifying_contexts"] == [
        "noisy_f117_t750",
        "noisy_f117_t500",
        "noisy_f117_t250",
    ]
    assert result["qk_low_recent_qualifying_contexts"] == []
    assert result["original_symmetric_gates"]["g2_qk_policy_choice"] is False


def test_committed_maps_are_reproducible_from_discovery_seed(tmp_path: Path) -> None:
    payloads, manifest = analysis.build_artifacts(
        result_root=RESULT_ROOT,
        pf_labels=PF_LABELS,
        output_dir=tmp_path,
    )

    for filename, payload in payloads.items():
        assert (MAP_DIR / filename).read_bytes() == payload
    recurrence = manifest["discovery_validation_recurrence"]
    assert recurrence["overlap_heads"] == 112
    assert recurrence["total_selected_heads"] == 120
    assert recurrence["exact_match_layers"] == 23
    assert recurrence["median_layer_jaccard"] == 1.0


def test_v153_maps_are_count_matched_and_pf_distinct() -> None:
    args = SimpleNamespace(
        map_dir=MAP_DIR,
        legacy_map=MAP_DIR / "legacy_v98_absolute_sign_304_56.csv",
        pf_labels=PF_LABELS,
    )
    manifest, head_maps, audits = runner.load_and_audit_maps(args)

    assert set(head_maps) == {
        "qk_top4",
        "qk_bottom4_control",
        "random4_control",
        "legacy",
    }
    for name in analysis.MAP_FILENAMES:
        assert audits[name]["counts"] == {"10": 120, "11": 240}
        assert audits[name]["label10_per_layer"] == [4] * 30
    assert audits["qk_top4"]["pf_cross_tab"] == {
        "wave": {"10": 77, "11": 79},
        "anchor": {"10": 24, "11": 148},
        "veil": {"10": 19, "11": 13},
    }
    assert manifest["classifier"]["discovery_seed_replicate"] == 0


def test_v153_cells_isolate_membership_and_route_controls() -> None:
    assert len(runner.cells_for_mode("membership")) == 4
    assert len(runner.cells_for_mode("controls")) == 2
    assert len(runner.cells_for_mode("reference")) == 1
    assert all(not cell.native for cell in runner.CELLS)

    primary = runner.CELLS[0]
    assert expected_policy(primary, 10) == (
        ("TemporalPrototypeStrategy",),
        1,
        4,
        "temporal_prototype",
    )
    assert expected_policy(primary, 11) == ((), 1, 8, "stride")

    all_recent = next(
        cell for cell in runner.CELLS if cell.name.endswith("all_recent8_control")
    )
    assert expected_policy(all_recent, 10) == ((), 1, 8, "stride")
    assert expected_policy(all_recent, 11) == ((), 1, 8, "stride")

    all_prototype = next(
        cell
        for cell in runner.CELLS
        if cell.name.endswith("all_prototype4_control")
    )
    assert expected_policy(all_prototype, 10) == expected_policy(all_prototype, 11)


def test_manifest_is_valid_json_and_tracks_source_hashes() -> None:
    manifest_path = MAP_DIR / analysis.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["gate_reanalysis"]["one_sided_transfer_candidate"] is True
    assert set(manifest["source"]["files"]) == {
        "policy_pair_summary.csv",
        "random_control_summary.csv",
        "selector_alignment_summary.csv",
        "selector_snapshots.csv.gz",
        "report.json",
    }
