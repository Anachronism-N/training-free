from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_random_controls_separate_head_identity_from_structure() -> None:
    module = load_module(
        "v195_analysis_random",
        SCRIPTS / "analyze_v195_cross_checkpoint_profile.py",
    )
    values = np.full((4, 30, 12), -0.02, dtype=np.float64)
    selected = np.zeros_like(values, dtype=np.bool_)
    selected[:, :, 0] = True
    values[selected] = 0.10
    call_control = module._random_control(
        values,
        selected,
        preserve_call_layer=False,
        draws=500,
        seed=1,
    )
    head_control = module._random_control(
        values,
        selected,
        preserve_call_layer=True,
        draws=500,
        seed=2,
    )
    assert call_control["one_sided_empirical_p"] < 0.01
    assert head_control["one_sided_empirical_p"] < 0.01
    assert head_control["selected_cell_count"] == 120


def test_structural_correlations_report_multiple_resolutions() -> None:
    module = load_module(
        "v195_analysis_corr",
        SCRIPTS / "analyze_v195_cross_checkpoint_profile.py",
    )
    sf = np.arange(4 * 30 * 12, dtype=np.float64).reshape(4, 30, 12)
    cf = sf * 2.0 + 3.0
    report = module._structural_correlations(sf, cf)
    assert set(report) == {
        "exact_call_layer_head",
        "phase_layer_mean_over_heads",
        "layer_head_mean_over_calls",
        "layer_mean_over_calls_heads",
        "phase_mean_over_layers_heads",
    }
    assert report["exact_call_layer_head"]["pearson"] > 0.999
    assert report["phase_layer_mean_over_heads"]["spearman"] > 0.999


def test_analysis_uses_untouched_holdout_and_count_matched_controls(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module(
        "v195_analysis_e2e",
        SCRIPTS / "analyze_v195_cross_checkpoint_profile.py",
    )
    monkeypatch.setattr(module, "RANDOM_DRAWS", 300)
    operator = "landmark"
    mask = np.zeros((4, 30, 12), dtype=np.bool_)
    mask[:, :, 0] = True
    route = tmp_path / "route.json"
    write_json(
        route,
        {
            "map_id": "frozen-sf-route",
            "coverage_masks": mask.tolist(),
            "coverage_count_by_call": [30, 30, 30, 30],
        },
    )
    scores = tmp_path / "cell_scores.csv"
    fieldnames = [
        "operator",
        "call_index",
        "layer",
        "head",
        "discovery_gain",
        "validation_gain",
        "compatible",
    ]
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for call in range(4):
            for layer in range(30):
                for head in range(12):
                    selected = head == 0
                    writer.writerow(
                        {
                            "operator": operator,
                            "call_index": call,
                            "layer": layer,
                            "head": head,
                            "discovery_gain": 0.1 if selected else -0.01,
                            "validation_gain": 0.1 if selected else -0.01,
                            "compatible": selected,
                        }
                    )
    split = {
        "discovery": list(range(64)),
        "validation": list(range(64, 96)),
        "generation_holdout": list(range(96, 128)),
    }
    manifest = {
        "operator": operator,
        "prompt_split": split,
        "expected_record_count": 184320,
        "frozen_sf_route": {"path": str(route), "map_id": "frozen-sf-route"},
        "v189_provenance": {"cell_scores": str(scores)},
        "v194_provenance": {"cross_checkpoint_transfer_confirmed": True},
        "checkpoint": {"sha256": "checkpoint-sha"},
        "claim_boundary": "diagnostic only",
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    audit_path = tmp_path / "audit.json"
    write_json(
        audit_path,
        {
            "experiment": "v195_cross_checkpoint_head_phase_profile_audit",
            "ok": True,
            "input_manifest_sha256": module.sha256(manifest_path),
            "operator": operator,
            "record_count": 184320,
        },
    )
    gain = np.full((128, 4, 30, 12), -0.01, dtype=np.float64)
    gain[:, :, :, 0] = 0.10
    aggregate = {
        "gain": gain,
        "energy": np.ones_like(gain),
        "full_budget": np.ones_like(gain),
    }

    def fake_rows(*_args, **_kwargs):
        return [
            {
                "call_index": call,
                "layer": layer,
                "head": head,
                "compatible": head == 0,
                "validation_win_fraction": 1.0 if head == 0 else 0.0,
                "full_budget_fraction": 1.0,
                "relative_reference_energy": 1.0,
            }
            for call in range(4)
            for layer in range(30)
            for head in range(12)
        ]

    monkeypatch.setattr(module, "verify", lambda _path: manifest)
    monkeypatch.setattr(
        module,
        "aggregate_operator",
        lambda _root, _operator: (aggregate, {"record_count": 184320}),
    )
    monkeypatch.setattr(module, "_cell_rows", fake_rows)
    output = tmp_path / "analysis"
    report = module.analyze(manifest_path, tmp_path / "profiles", audit_path, output)
    assert report["primary_evaluation_split"] == "generation_holdout"
    assert report["mechanism_support_level"] == "exact_head_identity"
    assert report["exact_head_identity_transfer_supported"] is True
    assert (
        report["recommendation"]
        == "freeze_route_with_cross_checkpoint_mechanistic_support"
    )
    assert report["manual_review_required"] is False
    assert (output / "cell_transfer.csv").is_file()
    assert (output / "holdout_prompt_effects.csv").is_file()


def test_runtime_is_strictly_causal_checkpoint_bound() -> None:
    runner = (SCRIPTS / "run_v195_cross_checkpoint_profile_32gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "--checkpoint_state_key generator" in runner
    assert "--model_local_attn_size 21" in runner
    assert "--cache_compat_profile_contract v189" in runner
    assert "--skip_video_decode" in runner
    assert "--use_ema" not in runner
    assert "profile128 is frozen to 4 nodes x 8 GPUs" in runner


def test_profile_metadata_records_checkpoint_contract() -> None:
    inference = (ROOT / "third_party/Pyramid-Forcing/inference.py").read_text(
        encoding="utf-8"
    )
    for field in (
        '"checkpoint_path": os.path.abspath(args.checkpoint_path)',
        '"checkpoint_state_key": args.checkpoint_state_key or "auto"',
        '"use_ema": bool(args.use_ema)',
        '"model_local_attn_size": (',
    ):
        assert field in inference
