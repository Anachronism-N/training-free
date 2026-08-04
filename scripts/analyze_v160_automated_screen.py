#!/usr/bin/env python3
"""Build a diagnostic screen and adaptive review plan for v160.

The screen is deliberately not a promotion metric. It detects likely failures,
localizes metric disagreement, and selects a small informative review subset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v160_automated_diagnostic_screen"
SOURCE_EXPERIMENT = "v160_fresh_motion_moviebench16"
REPORT_TITLE = "v160 Automated Diagnostic Screen"
LOG_PREFIX = "v160-screen"
PROMPT_COUNT = 16
PRIMARY = "ours_middle10_reservoir2_freshmotionpair1"
CURRENT = "ours_middle10_reservoir2_motionpair1_reference"
RESERVOIR = "ours_middle10_reservoir4_reference"
METHODS = (
    "sf_native",
    PRIMARY,
    CURRENT,
    RESERVOIR,
    "ours_all_recent8_reference",
)
REVIEW_METHODS = (PRIMARY, CURRENT, RESERVOIR)
REFERENCES = (CURRENT, RESERVOIR)

# Direction is +1 when larger is preferable and -1 when smaller is preferable.
# These signals are used only for paired triage and prompt selection.
TEMPORAL_FEATURES = {
    "motion_coverage_fraction": 1,
    "late_motion_ratio": 1,
    "longest_low_motion_run_fraction": -1,
    "temporal_jump": -1,
    "appearance_outlier_fraction": -1,
    "flow_accel_outlier_fraction": -1,
    "dark_frame_fraction": -1,
    "bright_frame_fraction": -1,
    "low_contrast_frame_fraction": -1,
    "edge_density_outlier_fraction": -1,
}
COMPREHENSIVE_FEATURES = {
    "m1_dino_consistency": 1,
    "m1_min_stability": 1,
    "m1_first_last_gap": -1,
    "m2_drift_slope": 1,
    "m3_motion_smoothness": -1,
    "m4_arcface_id_sim": 1,
    "m5_temporal_flickering": -1,
    "m5_max_flicker": -1,
    "m6_clip_text_alignment": 1,
    "m6_clip_text_min": 1,
    "m7_background_consistency": 1,
    "m7_background_drift": -1,
    "m8_loop_score": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-csv", required=True, type=Path)
    parser.add_argument("--comprehensive-json", type=Path)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=ROOT / "prompts" / "moviegen_128_qwen_v154_diverse16.json",
    )
    parser.add_argument("--published-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_coverage(rows: dict[tuple[str, int], dict], *, label: str) -> None:
    expected = {
        (method, prompt_index)
        for method in METHODS
        for prompt_index in range(PROMPT_COUNT)
    }
    actual = set(rows)
    if actual != expected:
        raise ValueError(
            f"{label} coverage mismatch: missing={sorted(expected-actual)[:20]} "
            f"extra={sorted(actual-expected)[:20]}"
        )


def load_temporal(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (str(raw["method"]), int(raw["prompt_index"]))
            if key in rows:
                raise ValueError(f"duplicate temporal row: {key}")
            rows[key] = {
                feature: value
                for feature in TEMPORAL_FEATURES
                if (value := finite_float(raw.get(feature))) is not None
            }
    validate_coverage(rows, label="temporal")
    missing = {
        key: sorted(set(TEMPORAL_FEATURES) - set(values))
        for key, values in rows.items()
        if set(values) != set(TEMPORAL_FEATURES)
    }
    if missing:
        raise ValueError(f"temporal features missing: {list(missing.items())[:5]}")
    return rows


def load_comprehensive(
    path: Path | None,
) -> dict[tuple[str, int], dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_video = payload.get("per_video")
    if not isinstance(per_video, dict):
        raise ValueError("comprehensive result has no per_video mapping")
    rows = {}
    for raw in per_video.values():
        key = (str(raw["method"]), int(raw["prompt_index"]))
        if key in rows:
            raise ValueError(f"duplicate comprehensive row: {key}")
        metrics = raw.get("metrics", {})
        rows[key] = {
            feature: value
            for feature in COMPREHENSIVE_FEATURES
            if (value := finite_float(metrics.get(feature))) is not None
        }
    validate_coverage(rows, label="comprehensive")
    return rows


def robust_scale(values: list[float]) -> float:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    if mad > 1e-9:
        return 1.4826 * mad
    spread = max(values) - min(values)
    return max(spread / 4.0, 1e-6)


def feature_scales(
    rows: dict[tuple[str, int], dict[str, float]],
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for metrics in rows.values():
        for feature, value in metrics.items():
            values.setdefault(feature, []).append(value)
    return {
        feature: robust_scale(feature_values)
        for feature, feature_values in values.items()
        if len(feature_values) >= PROMPT_COUNT
    }


def paired_delta(
    rows: dict[tuple[str, int], dict[str, float]],
    *,
    prompt_index: int,
    reference: str,
    directions: dict[str, int],
    scales: dict[str, float],
) -> dict[str, float]:
    primary = rows[(PRIMARY, prompt_index)]
    control = rows[(reference, prompt_index)]
    result = {}
    for feature, direction in directions.items():
        if feature not in primary or feature not in control or feature not in scales:
            continue
        result[feature] = (
            direction * (primary[feature] - control[feature]) / scales[feature]
        )
    return result


def automatic_flags(
    temporal: dict[tuple[str, int], dict[str, float]],
    comprehensive: dict[tuple[str, int], dict[str, float]],
    prompt_index: int,
) -> list[str]:
    primary = temporal[(PRIMARY, prompt_index)]
    refs = [temporal[(method, prompt_index)] for method in REFERENCES]
    flags = []
    if (
        primary["longest_low_motion_run_fraction"] > 0.20
        and primary["longest_low_motion_run_fraction"]
        > max(row["longest_low_motion_run_fraction"] for row in refs) + 0.10
    ):
        flags.append("long_low_motion_run")
    if (
        primary["late_motion_ratio"] < 0.55
        and primary["late_motion_ratio"]
        < min(row["late_motion_ratio"] for row in refs) - 0.20
    ):
        flags.append("late_motion_collapse")
    if (
        primary["temporal_jump"]
        > 1.35 * max(row["temporal_jump"] for row in refs)
        and primary["appearance_outlier_fraction"]
        > max(row["appearance_outlier_fraction"] for row in refs) + 0.02
    ):
        flags.append("temporal_discontinuity")
    if (
        primary["dark_frame_fraction"] > 0.02
        or primary["bright_frame_fraction"] > 0.02
        or primary["low_contrast_frame_fraction"] > 0.05
    ):
        flags.append("luminance_or_contrast_failure")
    if comprehensive:
        primary_c = comprehensive[(PRIMARY, prompt_index)]
        refs_c = [comprehensive[(method, prompt_index)] for method in REFERENCES]
        if (
            "m1_dino_consistency" in primary_c
            and all("m1_dino_consistency" in row for row in refs_c)
            and primary_c["m1_dino_consistency"]
            < min(row["m1_dino_consistency"] for row in refs_c) - 0.03
        ):
            flags.append("subject_consistency_drop")
        if (
            "m7_background_drift" in primary_c
            and all("m7_background_drift" in row for row in refs_c)
            and primary_c["m7_background_drift"]
            > max(row["m7_background_drift"] for row in refs_c) + 0.05
        ):
            flags.append("background_drift")
    return flags


def score_prompts(
    temporal: dict[tuple[str, int], dict[str, float]],
    comprehensive: dict[tuple[str, int], dict[str, float]],
) -> list[dict]:
    temporal_scales = feature_scales(temporal)
    comprehensive_scales = feature_scales(comprehensive) if comprehensive else {}
    rows = []
    for prompt_index in range(PROMPT_COUNT):
        deltas = {}
        flattened = []
        for reference in REFERENCES:
            values = paired_delta(
                temporal,
                prompt_index=prompt_index,
                reference=reference,
                directions=TEMPORAL_FEATURES,
                scales=temporal_scales,
            )
            if comprehensive:
                values.update(
                    paired_delta(
                        comprehensive,
                        prompt_index=prompt_index,
                        reference=reference,
                        directions=COMPREHENSIVE_FEATURES,
                        scales=comprehensive_scales,
                    )
                )
            deltas[reference] = values
            flattened.extend(values.values())
        negative = [-value for value in flattened if value < 0]
        positive = [value for value in flattened if value > 0]
        flags = automatic_flags(temporal, comprehensive, prompt_index)
        rows.append(
            {
                "prompt_index": prompt_index,
                "risk_score": (
                    (sum(negative) / max(1, len(negative))) + 2.0 * len(flags)
                ),
                "gain_score": sum(positive) / max(1, len(positive)),
                "disagreement_score": (
                    statistics.pstdev(flattened) if len(flattened) > 1 else 0.0
                ),
                "automatic_flags": flags,
                "paired_oriented_deltas": deltas,
            }
        )
    risk_median = statistics.median(row["risk_score"] for row in rows)
    gain_median = statistics.median(row["gain_score"] for row in rows)
    for row in rows:
        row["typical_distance"] = abs(row["risk_score"] - risk_median) + abs(
            row["gain_score"] - gain_median
        )
    return rows


def choose_review_prompts(rows: list[dict], prompt_items: list[dict]) -> dict:
    by_index = {row["prompt_index"]: row for row in rows}
    selected: set[int] = set()
    used_tags: set[str] = set()

    rankings = {
        "highest_automatic_risk": sorted(
            rows, key=lambda row: (-row["risk_score"], row["prompt_index"])
        ),
        "largest_predicted_gain": sorted(
            rows, key=lambda row: (-row["gain_score"], row["prompt_index"])
        ),
        "largest_metric_disagreement": sorted(
            rows,
            key=lambda row: (-row["disagreement_score"], row["prompt_index"]),
        ),
        "typical_case": sorted(
            rows, key=lambda row: (row["typical_distance"], row["prompt_index"])
        ),
    }

    def pick(reason: str) -> dict:
        candidates = [
            row for row in rankings[reason] if row["prompt_index"] not in selected
        ]
        if not candidates:
            raise ValueError("adaptive review selection exhausted prompts")
        diverse = [
            row
            for row in candidates
            if set(prompt_items[row["prompt_index"]].get("tags", [])) - used_tags
        ]
        row = (diverse or candidates)[0]
        prompt_index = row["prompt_index"]
        selected.add(prompt_index)
        tags = list(prompt_items[prompt_index].get("tags", []))
        used_tags.update(tags)
        return {
            "prompt_index": prompt_index,
            "reason": reason,
            "tags": tags,
            "automatic_flags": row["automatic_flags"],
            "risk_score": row["risk_score"],
            "gain_score": row["gain_score"],
            "disagreement_score": row["disagreement_score"],
        }

    wave1 = [pick(reason) for reason in rankings]
    wave2 = [pick(reason) for reason in rankings]
    return {
        "version": 1,
        "methods": list(REVIEW_METHODS),
        "videos_per_wave": len(REVIEW_METHODS) * 4,
        "maximum_review_videos": len(REVIEW_METHODS) * 8,
        "wave1": wave1,
        "wave2": wave2,
        "selection_is_diagnostic_only": True,
        "wave2_condition": "run only when wave1 human decision is inconclusive",
        "rows": [by_index[index] for index in sorted(by_index)],
    }


def method_summary(
    rows: dict[tuple[str, int], dict[str, float]],
) -> dict[str, dict[str, float]]:
    summary = {}
    for method in METHODS:
        features = sorted(
            {
                feature
                for prompt_index in range(PROMPT_COUNT)
                for feature in rows[(method, prompt_index)]
            }
        )
        summary[method] = {
            feature: statistics.median(
                rows[(method, prompt_index)][feature]
                for prompt_index in range(PROMPT_COUNT)
                if feature in rows[(method, prompt_index)]
            )
            for feature in features
        }
    return summary


def markdown(report: dict) -> str:
    lines = [
        f"# {REPORT_TITLE}",
        "",
        "This report is for failure triage and adaptive review selection. "
        "It is not a paper metric or a promotion gate.",
        "",
        "Automatic safety screen: "
        f"**{'PASS' if report['automatic_safety_screen'] else 'FLAGGED'}**",
        "",
        "## Flagged prompts",
        "",
    ]
    flagged = [
        row
        for row in report["review_plan"]["rows"]
        if row["automatic_flags"]
    ]
    if flagged:
        for row in flagged:
            lines.append(
                f"- Prompt {row['prompt_index']}: "
                + ", ".join(row["automatic_flags"])
            )
    else:
        lines.append("- None")
    for wave in ("wave1", "wave2"):
        lines.extend(["", f"## {wave.title()}", ""])
        for item in report["review_plan"][wave]:
            lines.append(
                f"- Prompt {item['prompt_index']}: {item['reason']} "
                f"(tags={','.join(item['tags'])})"
            )
    lines.extend(
        [
            "",
            "Wave 1 contains 12 videos (4 prompts x 3 methods). Prepare Wave 2 "
            "only if the prespecified human decision is inconclusive.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    published = json.loads(args.published_manifest.read_text(encoding="utf-8"))
    prompt_manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8"))
    if (
        not published.get("ok")
        or published.get("experiment") != SOURCE_EXPERIMENT
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or tuple(row["key"] for row in published.get("methods", [])) != METHODS
        or prompt_manifest.get("suite") != "v154_qwen_moviebench_diverse16"
        or int(prompt_manifest.get("prompt_count", -1)) != PROMPT_COUNT
    ):
        raise ValueError(
            f"{SOURCE_EXPERIMENT} screen inputs violate the frozen manifest contract"
        )
    temporal = load_temporal(args.temporal_csv)
    comprehensive = load_comprehensive(args.comprehensive_json)
    prompt_rows = score_prompts(temporal, comprehensive)
    plan = choose_review_prompts(prompt_rows, prompt_manifest["items"])
    flagged = [row for row in prompt_rows if row["automatic_flags"]]
    automatic_safety = (
        len(flagged) <= 2
        and not any(len(row["automatic_flags"]) >= 2 for row in flagged)
    )
    report = {
        "version": 1,
        "experiment": EXPERIMENT,
        "inputs": {
            "temporal_csv": str(args.temporal_csv.resolve()),
            "temporal_csv_sha256": sha256(args.temporal_csv),
            "comprehensive_json": (
                str(args.comprehensive_json.resolve())
                if args.comprehensive_json
                else None
            ),
            "comprehensive_json_sha256": (
                sha256(args.comprehensive_json) if args.comprehensive_json else None
            ),
            "published_manifest_sha256": sha256(args.published_manifest),
            "prompt_manifest_sha256": sha256(args.prompt_manifest),
        },
        "automatic_safety_screen": automatic_safety,
        "automatic_safety_is_not_promotion": True,
        "flagged_prompt_count": len(flagged),
        "method_medians": {
            "temporal": method_summary(temporal),
            "comprehensive": (
                method_summary(comprehensive) if comprehensive else None
            ),
        },
        "review_plan": plan,
        "claim_boundary": (
            "The adaptive sample is selected after observing diagnostics and "
            "therefore cannot support an unbiased paper comparison. It only "
            "reduces exploratory review load; later held-out evaluation remains required."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "automated_screen.json"
    plan_path = args.output_dir / "review_plan.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "automated_screen.md").write_text(
        markdown(report),
        encoding="utf-8",
    )
    print(
        f"[{LOG_PREFIX}] safety={automatic_safety} flagged={len(flagged)} "
        f"wave1={[row['prompt_index'] for row in plan['wave1']]} "
        f"wave2={[row['prompt_index'] for row in plan['wave2']]}"
    )


if __name__ == "__main__":
    main()
