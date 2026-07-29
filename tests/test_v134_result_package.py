import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "package_v134_head_discovery_results.py"
)
SPEC = importlib.util.spec_from_file_location("v134_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_package_excludes_raw_profiles_and_summarizes_logs(tmp_path):
    run_root = tmp_path / "run"
    analysis = run_root / "analysis"
    analysis.mkdir(parents=True)
    for name in MODULE.ANALYSIS_FILES:
        payload = "{}\n"
        if name == "classification_report.json":
            payload = json.dumps(
                {
                    "acceptance_gates": {"accepted": True},
                    "label_counts": {
                        "prompt_conditional": 200,
                        "prompt_invariant": 160,
                    },
                }
            )
        (analysis / name).write_text(payload, encoding="utf-8")
    for stage in ("observational", "counterfactual"):
        profile_dir = run_root / "profiles" / stage
        video_dir = run_root / "videos" / stage
        log_dir = run_root / "logs" / stage
        profile_dir.mkdir(parents=True)
        video_dir.mkdir(parents=True)
        log_dir.mkdir(parents=True)
        (profile_dir / "raw.pt").write_bytes(b"raw")
        (video_dir / "video.mp4").write_bytes(b"video")
        (log_dir / "worker.log").write_text(
            "[HeadProfile] begin\n[HeadProfile] end\n",
            encoding="utf-8",
        )
    output = tmp_path / "bundle"
    inventory = MODULE.package_results(run_root, output)
    assert inventory["profile_counts"] == {
        "observational": 1,
        "counterfactual": 1,
    }
    assert not (output / "raw.pt").exists()
    assert (output / "worker_log_summary.json").is_file()
    assert (output / "review_bundle.md").is_file()
