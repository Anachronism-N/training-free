#!/usr/bin/env python3
"""Materialize the frozen v211 screen as source-bound VBench inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from audit_v211_lphc_trace import audit_trace
from prepare_v211_lphc import (
    AUTHORIZED_NODES,
    BASE_SEED,
    EXPERIMENT as SOURCE_EXPERIMENT,
    GATE0_METHODS,
    GATE0_SOURCE_INDICES,
    METHOD_SPECS,
    METHODS,
    SCREEN8_SOURCE_INDICES,
    SCREEN_FRAMES,
    SMOKE_METHODS,
    V211_RUNTIME_FILES,
    effective_seed,
    sha256,
)
from run_v211_worker import done_matches, make_stamp
from v210_vbench_fingerprint import vbench_checkout_fingerprint

EXPERIMENT = "v211_lphc_vbench_screen8"
SOURCE_INDICES = SCREEN8_SOURCE_INDICES
FROZEN_GENERATION_COMMIT = os.environ.get("V211_GENERATION_COMMIT", "")
DEFAULT_VBENCH_ROOT = Path(
    "/apdcephfs_gy2/share_303214315/cedricnie/develop/research_sprint/"
    "bench_baselines/VBench"
)
MEDIA_CONTRACT = {"width": 832, "height": 480, "fps": 16.0, "frames": 477}
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "overall_consistency",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_style",
)
EVALUATION_RUNTIME_FILES = (
    "scripts/prepare_v211_vbench_comparison.py",
    "scripts/run_v211_vbench.py",
    "scripts/analyze_v211_lphc_screen.py",
    "scripts/v210_vbench_fingerprint.py",
    "scripts/prepare_v174_vbench_splits.py",
    "scripts/prepare_v154_vbench_splits.py",
    "scripts/run_v211_postprocess.sh",
    "scripts/run_v154_vbench_long.py",
    "scripts/eval_vbench_long_prompt_aware.py",
    "scripts/vbench_long_split_cache.py",
    "scripts/vbench_quality_contract.py",
    "scripts/analyze_v165_final_decision.py",
    "scripts/analyze_v174_paired_metrics.py",
    "scripts/analyze_v190_head_phase_causal_screen.py",
    "scripts/compute_temporal_jump_diagnostic.py",
    "scripts/bind_temporal_diagnostics.py",
)


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def require_clean_checkout(root: Path) -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
    )
    if status:
        raise ValueError("v211 evaluation checkout must be exactly clean")


def evaluation_runtime_sha256(root: Path) -> dict[str, str]:
    result = {}
    for name in EVALUATION_RUNTIME_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing v211 evaluation runtime file: {path}")
        result[name] = sha256(path)
    return result


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing v211 {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid v211 {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid v211 {label}: {path}")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _reject_legacy_root(path: Path, label: str) -> None:
    resolved = path.expanduser().resolve()
    lowered = "/".join(resolved.parts[-2:]).lower()
    if "v209" in lowered or "v210" in lowered:
        raise ValueError(f"v211 {label} must not reference a v209/v210 root: {path}")


def _reject_legacy_artifacts(root: Path, label: str) -> None:
    if not root.exists():
        return
    legacy = [
        path
        for path in root.rglob("*")
        if "v209" in path.name.lower() or "v210" in path.name.lower()
    ]
    if legacy:
        raise ValueError(f"v211 {label} contains legacy artifact: {legacy[0]}")


def _require_separate_roots(run_root: Path, comparison_root: Path) -> None:
    _reject_legacy_root(run_root, "generation root")
    _reject_legacy_root(comparison_root, "comparison root")
    _reject_legacy_artifacts(run_root, "generation root")
    _reject_legacy_artifacts(comparison_root, "comparison root")
    source = run_root.resolve()
    target = comparison_root.resolve()
    if target == source or _inside(target, source) or _inside(source, target):
        raise ValueError("v211 comparison root must be separate from the frozen generation root")


def _require_path(path: Path, expected: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"missing v211 {label}: {path}") from error
    if resolved != expected_resolved:
        raise ValueError(f"v211 {label} path drift: {path}")
    return resolved


def _valid_sha256(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_manifest(
    run_root: Path, manifest_path: Path, manifest: dict, generation_commit: str
) -> None:
    runtime_paths = manifest.get("runtime_paths")
    if (
        manifest.get("version") != 1
        or manifest.get("experiment") != SOURCE_EXPERIMENT
        or manifest.get("source_commit") != generation_commit
        or manifest.get("git_commit") != generation_commit
        or manifest.get("methods") != list(METHODS)
        or manifest.get("source_indices") != list(SOURCE_INDICES)
        or int(manifest.get("prompt_count", -1)) != len(SOURCE_INDICES)
        or int(manifest.get("base_seed", -1)) != BASE_SEED
        or int(manifest.get("screen_frames", -1)) != SCREEN_FRAMES
        or manifest.get("method_specs") != METHOD_SPECS
        or not isinstance(runtime_paths, dict)
        or not set(V211_RUNTIME_FILES).issubset(runtime_paths)
        or any(not _valid_sha256(value) for value in runtime_paths.values())
        or tuple(manifest.get("authorized_nodes") or ()) != AUTHORIZED_NODES
        or manifest.get("gate0", {}).get("source_indices")
        != list(GATE0_SOURCE_INDICES)
        or manifest.get("gate0", {}).get("modes") != GATE0_METHODS
        or manifest.get("gate0", {}).get("attention") != "production"
        or manifest.get("gate0", {}).get("same_gpu_sequential") is not True
        or manifest.get("smoke", {}).get("source_index") != SOURCE_INDICES[0]
        or tuple(manifest.get("smoke", {}).get("methods") or ()) != SMOKE_METHODS
        or int(manifest.get("smoke", {}).get("frames", -1)) != SCREEN_FRAMES
        or manifest.get("smoke", {}).get("required_before_screen8") is not True
    ):
        raise ValueError("invalid or mixed frozen v211 generation manifest")

    prompt_items = manifest.get("prompt_items") or ()
    if len(prompt_items) != len(SOURCE_INDICES):
        raise ValueError("v211 prompt item count drift")
    for source_index, item in zip(SOURCE_INDICES, prompt_items):
        expected_path = run_root / "inputs" / "prompts" / f"source_{source_index:03d}.txt"
        prompt_path = _require_path(Path(item.get("path", "")), expected_path, "prompt")
        if (
            item.get("source_index") != source_index
            or item.get("effective_seed") != effective_seed(source_index)
            or item.get("sha256") != sha256(prompt_path)
            or prompt_path.read_text(encoding="utf-8") != f"{item.get('text')}\n"
        ):
            raise ValueError(f"v211 prompt provenance drift: source {source_index}")

    if set(manifest.get("configs") or {}) != set(METHODS):
        raise ValueError("v211 config membership drift")
    for method in METHODS:
        row = manifest["configs"][method]
        expected_path = run_root / "inputs" / "configs" / f"{method}.yaml"
        config_path = _require_path(Path(row.get("path", "")), expected_path, "config")
        if row.get("sha256") != sha256(config_path):
            raise ValueError(f"v211 config hash drift: {method}")

    _require_path(
        manifest_path, run_root / "inputs" / "manifest.json", "manifest"
    )


def _validate_decision(
    path: Path,
    stage: str,
    manifest_sha256: str,
    manifest: dict,
) -> dict:
    decision = _read_json(path, f"{stage} decision")
    if (
        decision.get("version") != 1
        or decision.get("stage") != stage
        or decision.get("pass") is not True
        or decision.get("errors") != []
        or decision.get("input_manifest_sha256") != manifest_sha256
        or decision.get("source_commit") != manifest["source_commit"]
    ):
        raise ValueError(f"v211 {stage} decision is failed, stale, or mixed")
    if stage == "gate0":
        expected_sequence = [
            {"source_index": source_index, "mode": mode}
            for source_index in GATE0_SOURCE_INDICES
            for mode in GATE0_METHODS
        ]
        sequence = decision.get("sequence") or ()
        observed_sequence = [
            {"source_index": row.get("source_index"), "mode": row.get("mode")}
            for row in sequence
        ]
        pairs = decision.get("pairs") or ()
        if (
            decision.get("attention") != "production"
            or decision.get("same_gpu_sequential") is not True
            or observed_sequence != expected_sequence
            or [row.get("source_index") for row in pairs]
            != list(GATE0_SOURCE_INDICES)
            or any(row.get("tensor_comparison", {}).get("pass") is not True for row in pairs)
            or any(
                row.get("alpha0_audit", {}).get("pass") is not True
                for row in pairs
            )
            or any(
                row.get("alpha0_audit", {}).get("totals", {}).get("random") != 0
                for row in pairs
            )
        ):
            raise ValueError("v211 gate0 decision contract drift")
    else:
        audits = decision.get("audits") or {}
        completions = decision.get("completions") or {}
        source_index = SOURCE_INDICES[0]
        generation_manifest = path.parent.parent / "inputs" / "manifest.json"
        if (
            decision.get("source_index") != source_index
            or decision.get("latent_frames") != SCREEN_FRAMES
            or tuple(audits) != SMOKE_METHODS
            or tuple(completions) != SMOKE_METHODS
            or any(audits[method].get("pass") is not True for method in SMOKE_METHODS)
            or any(
                audits[method].get("alpha")
                != float(manifest["method_specs"][method]["alpha"])
                or audits[method].get("phase")
                != str(manifest["method_specs"][method]["phase"])
                or audits[method].get("expected_history_budget")
                != int(manifest["method_specs"][method]["history_frames"])
                or audits[method].get("totals", {}).get("random") != 0
                or not done_matches(
                    completions[method],
                    make_stamp(
                        manifest,
                        generation_manifest,
                        "smoke",
                        method,
                        source_index,
                    ),
                )
                for method in SMOKE_METHODS
            )
        ):
            raise ValueError("v211 smoke decision contract drift")
    return decision


def _screen_membership(run_root: Path) -> None:
    screen_root = run_root / "jobs" / "screen8"
    if not screen_root.is_dir():
        raise ValueError("missing v211 screen8 jobs")
    observed_methods = {path.name for path in screen_root.iterdir() if path.is_dir()}
    if observed_methods != set(METHODS):
        raise ValueError("v211 screen8 method membership drift")
    expected_jobs = {f"source_{source_index:03d}" for source_index in SOURCE_INDICES}
    for method in METHODS:
        observed_jobs = {
            path.name for path in (screen_root / method).iterdir() if path.is_dir()
        }
        if observed_jobs != expected_jobs:
            raise ValueError(f"v211 screen8 job membership drift: {method}")


def _elapsed_seconds(done: dict, label: str) -> float:
    try:
        raw = done["elapsed_seconds"]
    except KeyError as error:
        raise ValueError(f"invalid v211 elapsed time: {label}") from error
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"invalid v211 elapsed time: {label}")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid v211 elapsed time: {label}")
    return value


def _generation_seconds(method_jobs: list[dict]) -> float:
    if len(method_jobs) != len(SOURCE_INDICES):
        raise ValueError("v211 method generation timing requires exactly eight jobs")
    values = [_elapsed_seconds(row, str(row.get("source_index"))) for row in method_jobs]
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ValueError("invalid v211 aggregate generation time") from error
    if not math.isfinite(total):
        raise ValueError("invalid v211 aggregate generation time")
    return total


def _dense_prompt_items(manifest: dict) -> list[dict]:
    return [
        {
            "index": index,
            "source_index": item["source_index"],
            "effective_seed": item["effective_seed"],
            "text": item["text"],
            "source_path": item["path"],
            "source_sha256": item["sha256"],
        }
        for index, item in enumerate(manifest["prompt_items"])
    ]


def _validate_job(
    run_root: Path,
    manifest_path: Path,
    manifest: dict,
    method: str,
    source_index: int,
) -> dict:
    job = run_root / "jobs" / "screen8" / method / f"source_{source_index:03d}"
    done_path = job / "done.json"
    done = _read_json(done_path, "screen8 completion marker")
    stamp = make_stamp(manifest, manifest_path, "screen8", method, source_index)
    media_path = _require_path(
        Path((done.get("media") or {}).get("path", "")),
        job / "media" / "0-0_ema.mp4",
        "screen8 media",
    )
    if not done_matches(done, stamp, media_path):
        raise ValueError(f"invalid v211 completion marker: {method}/source_{source_index:03d}")
    media = done["media"]
    validation = media.get("validation") or {}
    elapsed_seconds = _elapsed_seconds(
        done, f"{method}/source_{source_index:03d}"
    )
    job_position = METHODS.index(method) * len(SOURCE_INDICES) + SOURCE_INDICES.index(source_index)
    expected_node = AUTHORIZED_NODES[job_position % len(AUTHORIZED_NODES)]
    if (
        done.get("returncode") != 0
        or done.get("contract_sha256") != sha256(manifest_path)
        or done.get("node_address") != expected_node
        or not done.get("hostname")
        or not (done.get("gpu_uuid") or done.get("cuda_visible_devices"))
        or media.get("sha256") != sha256(media_path)
        or validation.get("valid") is not True
        or validation.get("validation_level") != "decoded_stream_contract"
        or validation.get("expected") != MEDIA_CONTRACT
        or validation.get("ffprobe") != MEDIA_CONTRACT
        or validation.get("bytes") != media_path.stat().st_size
    ):
        raise ValueError(f"v211 media/provenance drift: {method}/source_{source_index:03d}")

    trace_path = None
    trace_sha256 = None
    trace_audit = None
    spec = manifest["method_specs"][method]
    trace = done.get("trace") or {}
    if spec.get("lphc"):
        trace_path = _require_path(
            Path(trace.get("path", "")), job / "trace.jsonl", "LPHC trace"
        )
        if trace.get("present") is not True or trace.get("sha256") != sha256(trace_path):
            raise ValueError(f"v211 LPHC trace hash drift: {method}/source_{source_index:03d}")
        trace_audit = audit_trace(
            trace_path,
            float(spec["alpha"]),
            expect_second_attention=True,
            phase=str(spec["phase"]),
            expected_history_budget=int(spec["history_frames"]),
            expected_blocks=SCREEN_FRAMES // 3,
            expected_layers=30,
        )
        if not trace_audit["pass"] or trace_audit["totals"]["random"] != 0:
            raise ValueError(
                f"v211 correct-retrieval trace failed: "
                f"{method}/source_{source_index:03d}"
            )
        trace_sha256 = trace_audit["trace_sha256"]
    elif trace.get("present") is not False or trace.get("sha256") is not None:
        raise ValueError(f"unexpected v211 baseline trace: {method}/source_{source_index:03d}")

    return {
        "method": method,
        "prompt_index": SOURCE_INDICES.index(source_index),
        "source_index": source_index,
        "effective_seed": effective_seed(source_index),
        "elapsed_seconds": elapsed_seconds,
        "done": str(done_path.resolve()),
        "done_sha256": sha256(done_path),
        "media": str(media_path),
        "media_sha256": media["sha256"],
        "trace": str(trace_path) if trace_path else None,
        "trace_sha256": trace_sha256,
        "trace_audit": (
            {
                "pass": True,
                "row_count": trace_audit["row_count"],
                "totals": trace_audit["totals"],
                "maximums": trace_audit["maximums"],
            }
            if trace_audit
            else None
        ),
    }


def link_or_validate(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink():
            raise RuntimeError(f"v211 VBench input must be a symlink: {target}")
        try:
            matches = target.resolve(strict=True) == source.resolve(strict=True)
        except OSError:
            matches = False
        if not matches:
            raise RuntimeError(f"refusing mixed v211 VBench input: {target}")
        return "existing"
    target.symlink_to(source.resolve(strict=True))
    return "symlink"


def write_frozen(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"frozen v211 VBench manifest differs: {path}")
    else:
        path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def prepare(
    run_root: Path,
    comparison_root: Path,
    repo_root: Path,
    vbench_root: Path = DEFAULT_VBENCH_ROOT,
    generation_commit: str | None = None,
) -> dict:
    generation_commit = generation_commit or FROZEN_GENERATION_COMMIT
    if not _valid_sha256(generation_commit, 40):
        raise ValueError(
            "V211 generation commit must be supplied as an explicit full SHA"
        )
    run_root = run_root.expanduser().resolve()
    comparison_root = comparison_root.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    vbench_root = vbench_root.expanduser().resolve()
    _require_separate_roots(run_root, comparison_root)

    manifest_path = run_root / "inputs" / "manifest.json"
    manifest = _read_json(manifest_path, "generation manifest")
    _validate_manifest(run_root, manifest_path, manifest, generation_commit)
    manifest_digest = sha256(manifest_path)
    gate_path = run_root / "decisions" / "gate0.json"
    smoke_path = run_root / "decisions" / "smoke.json"
    _validate_decision(gate_path, "gate0", manifest_digest, manifest)
    _validate_decision(smoke_path, "smoke", manifest_digest, manifest)
    _screen_membership(run_root)

    jobs = [
        _validate_job(run_root, manifest_path, manifest, method, source_index)
        for method in METHODS
        for source_index in SOURCE_INDICES
    ]
    expected_jobs = len(METHODS) * len(SOURCE_INDICES)
    expected_traces = len(SMOKE_METHODS) * len(SOURCE_INDICES)
    if len(jobs) != expected_jobs or sum(
        row["trace"] is not None for row in jobs
    ) != expected_traces:
        raise ValueError(
            "v211 screen8 requires exactly 32 jobs and 16 LPHC traces"
        )

    require_clean_checkout(repo_root)
    evaluation_commit = git_commit(repo_root)
    if len(evaluation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in evaluation_commit
    ):
        raise ValueError("invalid evaluation source commit")
    vbench_fingerprint = vbench_checkout_fingerprint(vbench_root)
    if len(str(vbench_fingerprint.get("head", ""))) != 40:
        raise ValueError("invalid VBench checkout fingerprint")
    evaluation_hashes = evaluation_runtime_sha256(repo_root)

    published_root = comparison_root / "published"
    if published_root.is_dir():
        observed_methods = {
            path.name for path in published_root.iterdir() if path.is_dir()
        }
        if not observed_methods.issubset(METHODS):
            raise RuntimeError(
                f"refusing mixed v211 VBench methods: {sorted(observed_methods - set(METHODS))}"
            )
    link_counts = {"existing": 0, "symlink": 0}
    comparison_methods = []
    for method in METHODS:
        target_dir = comparison_root / "published" / method
        expected_targets = {f"{index:06d}-0.mp4" for index in range(len(SOURCE_INDICES))}
        if target_dir.exists() and not target_dir.is_dir():
            raise RuntimeError(f"invalid v211 VBench method target: {target_dir}")
        if target_dir.is_dir():
            observed_targets = {path.name for path in target_dir.glob("*.mp4")}
            if observed_targets - expected_targets:
                raise RuntimeError(
                    f"refusing stale v211 VBench videos for {method}: "
                    f"{sorted(observed_targets - expected_targets)}"
                )
        method_jobs = [row for row in jobs if row["method"] == method]
        for row in method_jobs:
            target = target_dir / f"{row['prompt_index']:06d}-0.mp4"
            link_counts[link_or_validate(Path(row["media"]), target)] += 1
        if {path.name for path in target_dir.glob("*.mp4")} != expected_targets:
            raise RuntimeError(f"incomplete v211 VBench target set: {method}")
        comparison_methods.append(
            {
                "key": method,
                "role": (
                    "primary_control" if method == "sf_fifo21"
                    else "equal_budget_control" if method == "sf_sink1_21"
                    else "candidate"
                ),
                "method_spec": manifest["method_specs"][method],
                "generation_seconds": _generation_seconds(method_jobs),
                "provenance_complete": True,
                "source_video_dir": str(
                    (run_root / "jobs" / "screen8" / method).resolve()
                ),
                "video_dir": str(target_dir.resolve()),
            }
        )

    prompt_items = _dense_prompt_items(manifest)
    payload = {
        "version": 1,
        "experiment": EXPERIMENT,
        "development_only": True,
        "provenance_complete": True,
        "evaluation_commit": evaluation_commit,
        "evaluation_runtime_sha256": evaluation_hashes,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_generation_commit": manifest["source_commit"],
        "source_generation_runtime_sha256": manifest["runtime_paths"],
        "prompt_count": len(SOURCE_INDICES),
        "source_indices": list(SOURCE_INDICES),
        "prompt_items": prompt_items,
        "seed_policy": manifest["seed_policy"],
        "num_output_frames": SCREEN_FRAMES,
        "decoded_video_contract": MEDIA_CONTRACT,
        "primary_baseline": "sf_fifo21",
        "equal_budget_control": "sf_sink1_21",
        "methods": comparison_methods,
        "jobs": jobs,
        "vbench_long_dimensions": list(DIMENSIONS),
        "vbench_root": str(vbench_root),
        "vbench_checkout_fingerprint": vbench_fingerprint,
        "source": {
            "generation_root": str(run_root),
            "generation_manifest": str(manifest_path.resolve()),
            "generation_manifest_sha256": manifest_digest,
            "gate0_decision": str(gate_path.resolve()),
            "gate0_decision_sha256": sha256(gate_path),
            "smoke_decision": str(smoke_path.resolve()),
            "smoke_decision_sha256": sha256(smoke_path),
        },
        "materialization": {
            "mode": "read_only_symlink",
            "generation_root_modified": False,
        },
        "claim_boundary": (
            "Development-only source-bound screen on 8 preselected MovieGen prompts. "
            "Results support candidate screening only, not significance or paper claims. "
            "Dynamic Degree is descriptive and excluded from selection."
        ),
    }
    manifest_output = comparison_root / "comparison_manifest.json"
    digest = write_frozen(manifest_output, payload)
    return {
        "manifest": str(manifest_output.resolve()),
        "manifest_sha256": digest,
        "evaluation_commit": evaluation_commit,
        "methods": len(METHODS),
        "videos": len(jobs),
        "traces": sum(row["trace"] is not None for row in jobs),
        "link_counts": link_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument(
        "--generation-commit",
        default=os.environ.get("V211_GENERATION_COMMIT"),
        help="exact full SHA of the frozen v211 generation checkout",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--vbench-root", type=Path, default=DEFAULT_VBENCH_ROOT)
    args = parser.parse_args()
    report = prepare(
        args.run_root,
        args.comparison_root,
        args.repo_root,
        args.vbench_root,
        args.generation_commit,
    )
    print(
        "[v211-vbench-prepare] "
        f"methods={report['methods']} videos={report['videos']} "
        f"traces={report['traces']} links={report['link_counts']} "
        f"manifest={report['manifest']}"
    )


if __name__ == "__main__":
    main()
