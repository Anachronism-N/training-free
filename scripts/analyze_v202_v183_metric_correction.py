#!/usr/bin/env python3
"""Recompute v183 paired evidence after the corrected RAFT Dynamic Degree run."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import analyze_v174_paired_metrics as base
import numpy as np
from audit_v183_v180_recovery import METHODS, PROMPT_COUNT
from prepare_v178_rccp_holdout import sha256
from prepare_v183_v180_recovery_vbench import EXPERIMENT as SOURCE_EXPERIMENT
from vbench_quality_contract import exclusive_scores, official_quality_score

EXPERIMENT = "v202_v183_corrected_dynamic_degree_evidence"
CLIPS_PER_VIDEO = 15
CLIP_PROMPT_PATTERN = re.compile(r"(?:^|[/\\])(\d{6})-0(?:[/\\]|_)")
PAIRS = (
    ("rccp_matched", "sf_native", "end_to_end"),
    ("all_recent", "sf_native", "host_cache_vs_sf"),
    ("all_coverage", "sf_native", "dense_coverage_vs_sf"),
    ("rccp_matched", "all_recent", "strict5_increment"),
    ("all_coverage", "all_recent", "dense_coverage_increment"),
)
PRIMARY_METRICS = (
    "official_quality_score",
    "identity_background",
    "temporal_mechanics",
    "semantic_alignment",
    "visual_quality",
)


def _finite(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric value for {label}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite value for {label}: {value!r}")
    return result


def _corrected_dynamic_file(parts_root: Path, method: str) -> Path:
    root = parts_root / method / "dynamic_degree"
    candidates = sorted(root.glob("*_eval_results.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"{method}: expected one corrected Dynamic Degree result, got "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def load_corrected_dynamic(parts_root: Path) -> tuple[dict, dict]:
    per_prompt = {}
    provenance = {}
    for method in METHODS:
        path = _corrected_dynamic_file(parts_root, method)
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("dynamic_degree")
        if not isinstance(value, list) or len(value) not in {2, 3}:
            raise ValueError(f"{method}: malformed corrected Dynamic Degree payload")
        aggregate = _finite(value[0], label=f"{method}:aggregate")
        # VBench's torchvision-RAFT path appends continuous per-clip records
        # after its legacy boolean records. Prefer those records when present.
        details = value[-1]
        if not isinstance(details, list):
            raise TypeError(f"{method}: corrected Dynamic Degree details are absent")
        by_prompt: dict[int, list[float]] = {
            prompt: [] for prompt in range(PROMPT_COUNT)
        }
        for row in details:
            if not isinstance(row, dict):
                raise TypeError(f"{method}: malformed corrected clip result")
            match = CLIP_PROMPT_PATTERN.search(str(row.get("video_path", "")))
            if match is None:
                raise ValueError(
                    f"{method}: cannot recover prompt index from clip path"
                )
            prompt = int(match.group(1))
            if not 0 <= prompt < PROMPT_COUNT:
                raise ValueError(f"{method}: corrected prompt index out of range")
            by_prompt[prompt].append(
                _finite(
                    row.get("video_results"),
                    label=f"{method}:prompt{prompt}:clip",
                )
            )
        counts = Counter(len(values) for values in by_prompt.values())
        if counts != Counter({CLIPS_PER_VIDEO: PROMPT_COUNT}):
            raise ValueError(f"{method}: corrected clip grid is incomplete: {counts}")
        prompt_values = {
            prompt: float(np.mean(values)) for prompt, values in by_prompt.items()
        }
        observed = float(np.mean(list(prompt_values.values())))
        if abs(observed - aggregate) > 1e-12:
            raise ValueError(f"{method}: corrected aggregate disagrees with clips")
        per_prompt[method] = prompt_values
        provenance[method] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "aggregate": aggregate,
            "clip_count": len(details),
            "prompt_count": len(prompt_values),
            "all_one": all(
                abs(value - 1.0) <= 1e-12 for value in prompt_values.values()
            ),
        }
    return per_prompt, provenance


def corrected_rows(
    parts_root: Path,
    summary: dict,
    dynamic: dict[str, dict[int, float]],
) -> dict:
    raw = base.load_prompt_rows(parts_root, summary, METHODS, PROMPT_COUNT)
    for method in METHODS:
        for prompt in range(PROMPT_COUNT):
            raw[(method, prompt)]["dynamic_degree"] = dynamic[method][prompt]
    return base.derived_rows(raw, METHODS, PROMPT_COUNT)


def _contrast(
    rows: dict,
    *,
    candidate: str,
    control: str,
    role: str,
    metric: str,
    seed: int,
) -> dict:
    values = np.asarray(
        [
            rows[(candidate, prompt)][metric] - rows[(control, prompt)][metric]
            for prompt in range(PROMPT_COUNT)
        ],
        dtype=np.float64,
    )
    return {
        "candidate": candidate,
        "control": control,
        "comparison": f"{candidate}_minus_{control}",
        "comparison_role": role,
        "metric": metric,
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "win_fraction": float(np.mean(values > 0.0)),
        "bootstrap_ci95": base.bootstrap_ci(values.tolist(), seed=seed),
        "p_value": base.sign_p(values.tolist()),
        "per_prompt_delta": values.tolist(),
    }


def _aggregate_from_summary(summary: dict, *, dynamic_value: float | None) -> dict:
    result = {}
    for method in METHODS:
        row = dict(summary["methods"][method])
        if dynamic_value is not None:
            row["dynamic_degree"] = float(dynamic_value)
        result[method] = {
            "official_quality_score": official_quality_score(row),
            **exclusive_scores(row),
        }
    return result


def analyze(
    manifest: dict,
    summary: dict,
    rows: dict,
    *,
    dynamic_provenance: dict,
) -> dict:
    methods = tuple(str(row.get("key")) for row in manifest.get("methods") or ())
    if (
        manifest.get("experiment") != SOURCE_EXPERIMENT
        or methods != METHODS
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(summary.get("methods") or ()) != METHODS
        or summary.get("missing")
        or set(rows)
        != {(method, prompt) for method in METHODS for prompt in range(PROMPT_COUNT)}
    ):
        raise ValueError("v202 received incomplete or mismatched v183 evidence")
    if not all(row.get("all_one") is True for row in dynamic_provenance.values()):
        raise ValueError("v202 expects the corrected all-one Dynamic Degree rerun")

    comparisons = []
    for pair_index, (candidate, control, role) in enumerate(PAIRS):
        for metric_index, metric in enumerate(base.METRICS):
            comparisons.append(
                _contrast(
                    rows,
                    candidate=candidate,
                    control=control,
                    role=role,
                    metric=metric,
                    seed=2020000 + pair_index * 100 + metric_index,
                )
            )
    primary = [
        row
        for row in comparisons
        if row["control"] == "sf_native" and row["metric"] in PRIMARY_METRICS
    ]
    base.bh(primary)
    primary_ids = {id(row) for row in primary}
    for row in comparisons:
        if id(row) in primary_ids:
            row["inferential_role"] = "corrected_exploratory_sf_comparison"
        else:
            row["q_value"] = None
            row["inferential_role"] = "corrected_descriptive_context"

    candidate_status = {}
    for candidate in ("rccp_matched", "all_recent", "all_coverage"):
        paired = {
            row["metric"]: row
            for row in comparisons
            if row["candidate"] == candidate and row["control"] == "sf_native"
        }
        quality = paired["official_quality_score"]
        identity = paired["identity_background"]
        temporal = paired["temporal_mechanics"]
        significant_quality_gain = bool(
            quality["bootstrap_ci95"][0] > 0.0 and quality["q_value"] <= 0.10
        )
        nonmotion_noninferior = bool(
            identity["bootstrap_ci95"][0] >= -0.001
            and temporal["bootstrap_ci95"][0] >= -0.002
        )
        candidate_status[candidate] = {
            "significant_quality_gain_vs_sf": significant_quality_gain,
            "identity_temporal_noninferior_vs_sf": nonmotion_noninferior,
            "paper_efficacy_signal": significant_quality_gain and nonmotion_noninferior,
            "quality_mean_delta": quality["mean_delta"],
            "quality_ci95": quality["bootstrap_ci95"],
            "identity_mean_delta": identity["mean_delta"],
            "temporal_mean_delta": temporal["mean_delta"],
        }
    passing = [
        method
        for method, status in candidate_status.items()
        if status["paper_efficacy_signal"]
    ]
    recommendation = (
        "retain_v183_candidate_with_corrected_sf_gain"
        if passing
        else "no_v183_method_improves_sf_after_dynamic_correction"
    )
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "source_experiment": SOURCE_EXPERIMENT,
        "prompt_count": PROMPT_COUNT,
        "corrected_dynamic_degree": {
            "value_for_every_method_and_prompt": 1.0,
            "informative": False,
            "used_for_motion_claim": False,
            "used_for_method_ranking": False,
            "provenance": dynamic_provenance,
        },
        "old_aggregate_scores": _aggregate_from_summary(summary, dynamic_value=None),
        "corrected_aggregate_scores": _aggregate_from_summary(
            summary, dynamic_value=1.0
        ),
        "comparisons": comparisons,
        "candidate_status": candidate_status,
        "passing_candidates": passing,
        "recommendation": recommendation,
        "manual_review_required": False,
        "paper_claim_ready": False,
        "claim_boundary": (
            "This report repairs an exploratory v183 metric artifact. The old "
            "Dynamic Degree and any Quality Score derived from it are invalid. "
            "A new method still requires a frozen paired comparison with canonical "
            "SF and continuous motion diagnostics."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v202 Corrected v183 Evidence",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        "- Corrected Dynamic Degree: `1.0` for every method and prompt",
        "- Dynamic Degree used for ranking: `False`",
        "",
        "| Method | Old Quality | Corrected Quality | Delta vs corrected SF | Paper efficacy signal |",
        "|---|---:|---:|---:|---:|",
    ]
    corrected = report["corrected_aggregate_scores"]
    sf_quality = corrected["sf_native"]["official_quality_score"]
    for method in METHODS:
        old = report["old_aggregate_scores"][method]["official_quality_score"]
        current = corrected[method]["official_quality_score"]
        signal = (
            report["candidate_status"]
            .get(method, {})
            .get("paper_efficacy_signal", False)
        )
        lines.append(
            f"| {method} | {old:.4f} | {current:.4f} | "
            f"{current - sf_quality:+.4f} | {signal} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.comparison_root / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    dynamic, provenance = load_corrected_dynamic(args.parts_root)
    rows = corrected_rows(args.parts_root, summary, dynamic)
    report = analyze(
        manifest,
        summary,
        rows,
        dynamic_provenance=provenance,
    )
    report["source"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
        "stale_summary": str(args.summary.resolve()),
        "stale_summary_sha256": sha256(args.summary),
        "parts_root": str(args.parts_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v202-v183-correction] "
        f"recommendation={report['recommendation']} "
        f"passing={report['passing_candidates']}"
    )


if __name__ == "__main__":
    main()
