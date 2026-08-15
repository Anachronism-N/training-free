from __future__ import annotations

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
    spec.loader.exec_module(module)
    return module


def test_v184_schedule_routes_noisy_calls_and_keeps_clean_recent() -> None:
    module = load_module(
        "v184_schedule",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "denoise_schedule.py",
    )
    expected = {
        "recent": ("recent", "recent", "recent", "recent"),
        "coverage": ("coverage", "coverage", "coverage", "coverage"),
        "early1": ("coverage", "recent", "recent", "recent"),
        "early2": ("coverage", "coverage", "recent", "recent"),
        "late2": ("recent", "recent", "coverage", "coverage"),
        "late1": ("recent", "recent", "recent", "coverage"),
    }
    for schedule, policies in expected.items():
        observed = tuple(
            module.resolve_cache_compatibility_policy(
                schedule,
                call_index=index,
                call_count=4,
                update_mode="noisy",
            )
            for index in range(4)
        )
        assert observed == policies
        assert (
            module.resolve_cache_compatibility_policy(
                schedule,
                call_index=None,
                call_count=4,
                update_mode="clean",
            )
            == "recent"
        )
    with pytest.raises(ValueError):
        module.resolve_cache_compatibility_policy(
            "early2",
            call_index=4,
            call_count=4,
            update_mode="noisy",
        )


def test_v184_preparer_freezes_systematic32_and_shared_bank_map(tmp_path: Path) -> None:
    module = load_module(
        "v184_prepare", ROOT / "scripts" / "prepare_v184_denoise_phase_screen.py"
    )
    source = tmp_path / "moviegen128.txt"
    source.write_text(
        "\n".join(f"Qwen rewritten prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "inputs"
    payload = module.prepare(source, output)
    verified = module.verify(output / "manifest.json")
    assert payload == verified
    assert payload["source_indices"] == list(range(2, 128, 4))
    assert payload["prompt_count"] == 32
    assert payload["methods"]["coverage_early2"]["coverage_noisy_calls"] == [0, 1]
    assert payload["methods"]["coverage_late2"]["coverage_noisy_calls"] == [2, 3]
    map_path = Path(payload["methods"]["all_recent"]["head_map"])
    rows = map_path.read_text(encoding="ascii").splitlines()
    assert len(rows) == 30
    assert all(row.split(",") == ["10"] * 12 for row in rows)


def test_v184_trace_audit_checks_phase_and_equal_budget(tmp_path: Path) -> None:
    module = load_module(
        "v184_audit", ROOT / "scripts" / "audit_v184_denoise_phase_screen.py"
    )
    trace_dir = tmp_path / "traces" / "coverage_early2"
    trace_dir.mkdir(parents=True)
    rows = []
    for layer in (0, 10, 20, 29):
        for call_index in range(4):
            policy = "coverage" if call_index in {0, 1} else "recent"
            rows.append(
                {
                    "version": 1,
                    "event": "schedule",
                    "prompt_id": 0,
                    "layer": layer,
                    "schedule": "early2",
                    "effective_policy": policy,
                    "call_index": call_index,
                    "call_count": 4,
                    "update_mode": "noisy",
                    "current_start": 0,
                    "current_frame": 0,
                    "clean_policy_is_recent": True,
                }
            )
            counts = (
                {"static": 1, "dynamic": 4, "anchor": 4}
                if policy == "coverage"
                else {"static": 1, "dynamic": 8, "anchor": 0}
            )
            rows.append(
                {
                    "version": 1,
                    "event": "readout",
                    "prompt_id": 0,
                    "layer": layer,
                    "schedule": "early2",
                    "effective_policy": policy,
                    "call_index": call_index,
                    "call_count": 4,
                    "update_mode": "noisy",
                    "current_frame": 12,
                    "selected_heads": [
                        {
                            "head": 0,
                            "counts": counts,
                            "total_frame_equivalents": 9,
                            "segments": [],
                        }
                    ],
                    "max_total_frame_equivalents": 9,
                    "budget_pass": True,
                }
            )
        rows.append(
            {
                "version": 1,
                "event": "schedule",
                "prompt_id": 0,
                "layer": layer,
                "schedule": "early2",
                "effective_policy": "recent",
                "call_index": None,
                "call_count": 4,
                "update_mode": "clean",
                "current_start": 0,
                "current_frame": 0,
                "clean_policy_is_recent": True,
            }
        )
    (trace_dir / "shard00.schedule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = module.audit_schedule_traces(
        tmp_path,
        "coverage_early2",
        {"schedule": "early2", "coverage_noisy_calls": [0, 1]},
    )
    assert report["ok"] is True
    assert report["max_total_frame_equivalents"] == 9
    assert report["noisy_policies"] == {
        "0": ["coverage"],
        "1": ["coverage"],
        "2": ["recent"],
        "3": ["recent"],
    }


def test_v184_runtime_exposes_shared_banks_phase_trace_and_core9() -> None:
    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    pipeline = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pipeline"
        / "causal_inference.py"
    ).read_text(encoding="utf-8")
    cache = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "adaptive_cache.py"
    ).read_text(encoding="utf-8")
    assert "--pyramidkv_cache_compatibility_denoise_schedule" in inference
    assert "clean_readout=recent" in inference
    assert "set_cache_compatibility_denoise_state" in pipeline
    assert "set_cache_compatibility_active_policy" in cache
    assert "Coverage schedule exceeded its 4+4 budget" in cache

    runner = (
        ROOT / "scripts" / "run_v184_denoise_phase_screen_32gpu.sh"
    ).read_text(encoding="utf-8")
    for method in (
        "all_recent",
        "coverage_early1",
        "coverage_early2",
        "coverage_late2",
        "all_coverage_noisy",
    ):
        assert method in runner
    assert "PYRAMIDKV_DENOISE_SCHEDULE_TRACE_PATH" in runner
    evaluation = (ROOT / "scripts" / "run_v184_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "prepare_v184_vbench_comparison.py" in evaluation
    assert "analyze_v184_denoise_phase_screen.py" in evaluation


def test_v184_analyzer_promotes_only_pareto_phase_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_module(
        "v184_analyze", ROOT / "scripts" / "analyze_v184_denoise_phase_screen.py"
    )
    manifest = {
        "experiment": "v184_denoise_phase_coverage_vbench_screen32",
        "prompt_count": 32,
        "methods": [
            {"key": method, "video_dir": str(tmp_path / method)}
            for method in module.METHODS
        ],
        "prompt_items": [
            {"source_index": 2 + 4 * index, "text": f"prompt {index}"}
            for index in range(32)
        ],
    }
    summary = {
        "methods": {method: {} for method in module.METHODS},
        "missing": [],
    }
    values = {
        "all_recent": (80.0, 0.9700, 0.9800, 0.400),
        "coverage_early1": (80.1, 0.9695, 0.9790, 0.430),
        "coverage_early2": (80.2, 0.9695, 0.9790, 0.440),
        "coverage_late2": (79.9, 0.9690, 0.9790, 0.430),
        "all_coverage_noisy": (80.5, 0.9650, 0.9750, 0.600),
    }
    rows = {}
    for method, (quality, identity, temporal, dynamic) in values.items():
        for prompt in range(32):
            rows[(method, prompt)] = {
                "official_quality_score": quality,
                "identity_background": identity,
                "temporal_mechanics": temporal,
                "semantic_alignment": 0.24,
                "visual_quality": 0.65,
                "dynamic_degree": dynamic,
            }
    monkeypatch.setattr(module.base, "load_prompt_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(module.base, "derived_rows", lambda *args, **kwargs: rows)

    report = module.analyze(manifest, summary, tmp_path)

    assert report["promoted_to_operator_screen"] == ["coverage_early2"]
    assert report["recommendation"] == "advance_phase_schedule_to_operator_screen"
    assert report["manual_review_required_for_recommendation"] is False
