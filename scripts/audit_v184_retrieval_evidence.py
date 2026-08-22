#!/usr/bin/env python3
"""Audit the provenance and metric validity of the uploaded v184 result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
PROMPTS = 128
CLIPS_PER_PROMPT = 15
METHOD = "all_coverage_retrieval"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_payload(root: Path, dimension: str) -> tuple[Path, float, list[dict]]:
    paths = sorted((root / dimension).glob("*_eval_results.json"))
    if len(paths) != 1:
        raise ValueError(
            f"v184 requires one result for {dimension}, found {len(paths)}"
        )
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    value = payload.get(dimension)
    if (
        not isinstance(value, list)
        or len(value) < 2
        or not isinstance(value[1], list)
    ):
        raise ValueError(f"invalid v184 metric payload: {paths[0]}")
    return paths[0], float(value[0]), list(value[1])


def _is_degenerate(rows: list[dict]) -> tuple[bool, object | None]:
    values = [row.get("video_results") for row in rows]
    if not values:
        return True, None
    first = values[0]
    return all(value == first for value in values), first


def audit(run_root: Path) -> dict:
    comparison_root = run_root / "vbench_comparison"
    manifest_path = comparison_root / "comparison_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("missing v184 comparison manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = manifest.get("methods") or ()
    prompt_items = manifest.get("prompt_items") or ()
    if (
        manifest.get("experiment") != "v184_retrieval_128_vbench"
        or int(manifest.get("prompt_count", -1)) != PROMPTS
        or len(methods) != 1
        or methods[0].get("key") != METHOD
        or len(prompt_items) != PROMPTS
        or [int(row.get("index", -1)) for row in prompt_items]
        != list(range(PROMPTS))
    ):
        raise ValueError("v184 comparison manifest contract drift")

    video_dir = Path(methods[0]["video_dir"])
    if not video_dir.is_absolute():
        video_dir = Path.cwd() / video_dir
    expected_videos = {f"{index:06d}-0.mp4" for index in range(PROMPTS)}
    observed_videos = {path.name for path in video_dir.glob("*.mp4")}
    if observed_videos != expected_videos:
        raise ValueError(
            "v184 published video set is incomplete: "
            f"missing={sorted(expected_videos - observed_videos)[:10]} "
            f"extra={sorted(observed_videos - expected_videos)[:10]}"
        )

    log_paths = sorted((run_root / "logs").glob("shard*.log"))
    if not log_paths:
        raise ValueError("v184 has no generation logs")
    log_errors = []
    log_rows = []
    for path in log_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        required = {
            "all_coverage": "recent=20:0 coverage=21:360 episode=22:0" in text,
            "retrieval": "coverage_policy=retrieval" in text,
            "prompt_count": "Number of prompts: 128" in text,
        }
        failures = [
            marker
            for marker in (
                "Traceback (most recent call last)",
                "CUDA out of memory",
                "OutOfMemoryError",
            )
            if marker in text
        ]
        if not all(required.values()) or failures:
            log_errors.append(
                {"path": str(path), "required": required, "failures": failures}
            )
        log_rows.append({"path": str(path.resolve()), "sha256": sha256(path)})
    if log_errors:
        raise ValueError(f"v184 generation log contract failed: {log_errors}")

    metric_root = run_root / "metrics" / "vbench_long_parts" / METHOD
    metrics = {}
    invalid_dimensions = []
    for dimension in DIMENSIONS:
        path, score, rows = _metric_payload(metric_root, dimension)
        if len(rows) != PROMPTS * CLIPS_PER_PROMPT:
            raise ValueError(
                f"v184 {dimension} has {len(rows)} clip rows, expected "
                f"{PROMPTS * CLIPS_PER_PROMPT}"
            )
        degenerate, constant = _is_degenerate(rows)
        if degenerate:
            invalid_dimensions.append(dimension)
        metrics[dimension] = {
            "score": score,
            "clip_rows": len(rows),
            "degenerate": degenerate,
            "constant_video_result": constant if degenerate else None,
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }

    duplicate_score_pairs = []
    for left_index, left in enumerate(DIMENSIONS):
        for right in DIMENSIONS[left_index + 1 :]:
            if metrics[left]["score"] == metrics[right]["score"]:
                duplicate_score_pairs.append([left, right])

    prompt_text = "\n".join(str(row["text"]).strip() for row in prompt_items) + "\n"
    return {
        "version": 1,
        "experiment": "v184_retrieval_evidence_audit",
        "ok": True,
        "method": METHOD,
        "prompt_count": PROMPTS,
        "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "video_count": len(observed_videos),
        "log_count": len(log_paths),
        "runtime_contract": "360-head retrieval Coverage, seed 0, 120 latent frames",
        "metrics": metrics,
        "invalid_dimensions": invalid_dimensions,
        "duplicate_score_pairs": duplicate_score_pairs,
        "comparative_evidence_available": False,
        "comparative_blockers": [
            "comparison manifest contains only one method",
            "no same-prompt same-seed baseline is bound to this artifact",
            "degenerate dimensions cannot be used for effect estimates",
        ],
        "decision": "operator_stability_only_reprofile_before_comparative_claim",
        "claim_boundary": (
            "The artifact proves that all-head retrieval Coverage generated 128 "
            "auditable videos. It does not estimate a treatment effect against "
            "all-Recent, Self-Forcing, or Pyramid-Forcing."
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "logs": log_rows,
    }


def render(report: dict) -> str:
    lines = [
        "# v184 Retrieval Evidence Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Videos: `{report['video_count']}`",
        f"- Generation logs: `{report['log_count']}`",
        f"- Comparative evidence: `{report['comparative_evidence_available']}`",
        f"- Invalid dimensions: `{report['invalid_dimensions']}`",
        "",
        "| Dimension | Score | Clip rows | Degenerate |",
        "|---|---:|---:|---:|",
    ]
    for dimension in DIMENSIONS:
        row = report["metrics"][dimension]
        lines.append(
            f"| {dimension} | {row['score']:.8f} | {row['clip_rows']} | "
            f"{row['degenerate']} |"
        )
    lines.extend(["", "The manifest contains only one method, so no reported score is a paired method comparison."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v184-evidence-audit] PASS "
        f"videos={report['video_count']} invalid={report['invalid_dimensions']} "
        f"decision={report['decision']}"
    )


if __name__ == "__main__":
    main()
