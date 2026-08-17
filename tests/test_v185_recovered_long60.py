from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v185_recovered_long60 as analysis
import prepare_v185_recovered_long60_comparison as prepare


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_map(path: Path, coverage_heads: int) -> Path:
    rows = [[20] * 12 for _ in range(30)]
    for index in range(coverage_heads):
        rows[index // 12][index % 12] = 21
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return path


def _synthetic_recovery(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    run_root = repo / "runs" / "v181_rccp_long_stress"
    prompts = run_root / "inputs" / "prompts" / "long60_seed0.txt"
    prompts.parent.mkdir(parents=True)
    prompts.write_text(
        "".join(f"fresh long prompt {index}\n" for index in range(128)),
        encoding="utf-8",
    )
    maps = {}
    for method, coverage in (("rccp_matched", 5), ("all_recent", 0)):
        path = _write_map(run_root / "inputs" / "maps" / f"{method}.csv", coverage)
        maps[method] = {
            "path": str(path.resolve()),
            "sha256": prepare.sha256(path),
            "counts": {"20": 360 - coverage, "21": coverage, "22": 0},
        }
    manifest = {
        "experiment": "v181_rccp_long_stress_inputs",
        "methods": list(prepare.METHODS),
        "evaluation_prompts_used_for_membership": False,
        "upstream_decision": "pass",
        "maps": maps,
        "scopes": [
            {
                "key": prepare.SCOPE,
                "prompt_count": 128,
                "num_output_frames": 240,
                "seed": 0,
                "prompt_source_indices": list(prepare.SOURCE_INDICES),
                "prompt_file_sha256": prepare.sha256(prompts),
                "decoded_video_contract": prepare.VIDEO_CONTRACT,
            }
        ],
    }
    _write_json(run_root / "inputs" / "manifest.json", manifest)

    _write_json(
        repo
        / "runs"
        / "v178_rccp_holdout_generation"
        / "analysis"
        / "v178_paired_metrics.json",
        {"decision": "pass", "methods": ["matched", "all_recent"]},
    )
    scope_root = run_root / "scopes" / prepare.SCOPE
    runtime_methods = {}
    for method in prepare.METHODS:
        route = prepare.EXPECTED_ROUTES.get(method)
        log_dir = scope_root / "logs" / method
        log_dir.mkdir(parents=True)
        route_line = (
            ""
            if route is None
            else (
                "[CacheCompatibilityPolicy] "
                f"recent=20:{route[0]} coverage=21:{route[1]} "
                f"episode=22:{route[2]} budget=9FFE\n"
            )
        )
        (log_dir / "shard00.log").write_text(
            route_line + "successful invocation\n",
            encoding="utf-8",
        )
        parsed = {"shard00.log": [] if route is None else [list(route)]}
        failures = {}
        if method == "rccp_matched":
            (log_dir / "shard15.log").write_text(
                route_line + "Traceback (most recent call last)\n",
                encoding="utf-8",
            )
            parsed["shard15.log"] = [list(route)]
            failures["shard15.log"] = ["runtime_failure_pattern"]
        runtime_methods[method] = {
            "parsed_route_counts": parsed,
            "failures": failures,
        }
    _write_json(
        scope_root / "audits" / "runtime_logs.json",
        {"ok": False, "methods": runtime_methods},
    )

    for method_index, method in enumerate(prepare.METHODS):
        videos = []
        raw = scope_root / "raw" / method
        raw.mkdir(parents=True)
        for prompt in range(128):
            path = raw / f"{prompt}-0_ema.mp4"
            path.write_bytes(f"video:{method_index}:{prompt}".encode())
            videos.append(
                {
                    "file": path.name,
                    "prompt_idx": prompt,
                    "sample_idx": 0,
                    "sha256": prepare.sha256(path),
                    "metadata": {
                        **prepare.VIDEO_CONTRACT,
                        "fully_decoded": True,
                    },
                }
            )
        _write_json(
            scope_root / "audits" / f"{method}.json",
            {
                "ok": True,
                "expected": 128,
                "found": 128,
                "missing": [],
                "empty": [],
                "malformed": [],
                "media_errors": {},
                "input_fingerprint": f"fingerprint-{method}",
                "videos": videos,
            },
        )
    _write_json(
        scope_root / "audits" / "exact_video_duplicates.json",
        {
            "ok": True,
            "map_route_appears_globally_ignored": False,
            "pairwise_exact_duplicates": {
                "sf_native__rccp_matched": {"count": 0, "indices": []},
                "sf_native__all_recent": {"count": 0, "indices": []},
                "rccp_matched__all_recent": {"count": 0, "indices": []},
            },
        },
    )
    return run_root


def test_v185_preparer_locks_recovered_media_and_limitations(tmp_path: Path) -> None:
    run_root = _synthetic_recovery(tmp_path)
    output = tmp_path / "comparison"
    report = prepare.prepare(run_root, output)
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert report["videos"] == 384
    assert manifest["evidence_grade"] == "exploratory_recovered"
    assert manifest["formal_classifier_claim_eligible"] is False
    assert manifest["recovery_audit"]["upstream"]["status"] == "invalid_placeholder"
    assert (
        manifest["recovery_audit"]["runtime"][
            "route_configuration_consistent_in_observed_logs"
        ]
        is True
    )
    assert len(manifest["limitations"]) == 3
    for method in prepare.METHODS:
        assert len(list((output / "published" / method).glob("*.mp4"))) == 128


def test_v185_preparer_rejects_cross_method_duplicate_audit(tmp_path: Path) -> None:
    run_root = _synthetic_recovery(tmp_path)
    path = (
        run_root
        / "scopes"
        / prepare.SCOPE
        / "audits"
        / "exact_video_duplicates.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pairwise_exact_duplicates"]["rccp_matched__all_recent"] = {
        "count": 1,
        "indices": [0],
    }
    _write_json(path, payload)
    with pytest.raises(ValueError, match="exact-duplicate"):
        prepare.prepare(run_root, tmp_path / "comparison")


def _rows() -> dict:
    values = {
        "sf_native": (80.0, 0.960, 0.400),
        "all_recent": (80.2, 0.961, 0.420),
        "rccp_matched": (80.5, 0.962, 0.430),
    }
    result = {}
    for method, (quality, identity, dynamic) in values.items():
        for prompt in range(128):
            result[(method, prompt)] = {
                "official_quality_score": quality,
                "identity_background": identity,
                "dynamic_degree": dynamic,
                "temporal_mechanics": 0.98,
                "semantic_alignment": 0.24,
                "visual_quality": 0.65,
            }
    return result


def test_v185_analyzer_emits_exploratory_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    monkeypatch.setattr(analysis, "_load_window_rows", lambda *args, **kwargs: rows)
    manifest = {
        "experiment": prepare.EXPERIMENT,
        "evidence_grade": "exploratory_recovered",
        "formal_classifier_claim_eligible": False,
        "prompt_count": 128,
        "num_output_frames": 240,
        "vbench_long_dimensions": list(analysis.DIMENSIONS),
        "methods": [
            {"key": method, "video_dir": str(tmp_path / method)}
            for method in prepare.METHODS
        ],
        "prompt_items": [
            {"index": index, "source_index": 256 + index, "text": f"prompt {index}"}
            for index in range(128)
        ],
    }
    summary = {
        "methods": {method: {} for method in prepare.METHODS},
        "dimensions": list(analysis.DIMENSIONS),
        "missing": [],
    }
    report = analysis.analyze(manifest, summary, tmp_path)
    assert report["verdict"] == "static_five_long60_promising_exploratory"
    assert report["formal_classifier_claim_eligible"] is False
    assert report["manual_review_required_for_verdict"] is False
    assert len(report["targeted_review"]) == 4


def test_v181_formal_path_remains_fail_fast_and_v185_runner_is_exploratory() -> None:
    formal_inputs = (SCRIPTS / "prepare_v181_rccp_long_stress.py").read_text(
        encoding="utf-8"
    )
    formal_audit = (SCRIPTS / "audit_v181_rccp_long_stress.py").read_text(
        encoding="utf-8"
    )
    runner = (SCRIPTS / "run_v185_recovered_long60_vbench.sh").read_text(
        encoding="utf-8"
    )
    assert '!= "advance_rccp_membership_to_broader_generation"' in formal_inputs
    assert "_verify_runtime_contract" in formal_inputs
    assert "WARNING: v181" not in formal_audit
    assert "formal_classifier_claim_eligible=false" in runner
    assert "prepare_v185_recovered_long60_comparison.py" in runner
