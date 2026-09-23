#!/usr/bin/env python3
"""Fixed-seed, zero-generation motion/imaging diagnostics for paper closure."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import export_lphc_horizon_evidence as horizon

evidence = horizon.evidence
MOTION_TOLERANCE = 1e-9
METRICS = ("imaging_quality", "dynamic_degree", "official_quality_score", "subject_consistency")
SHORT_PAIRS = (("ours_correct", "sf_fifo21"), ("ours_correct", "ours_random"),
               ("ours_correct", "pooled_correct"), ("ours_random", "sf_fifo21"),
               ("pooled_correct", "sf_fifo21"), ("pooled_correct", "ours_random"))


def vectors(report, candidate, control, window):
    sources = report["source_indices"]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("empty or duplicated prompt sources")
    rows, arrays = {}, {}
    for metric in METRICS:
        matches = [r for r in report["comparisons"] if
                   (r["candidate"], r["control"], r["window"], r["metric"]) ==
                   (candidate, control, window, metric)]
        if len(matches) != 1:
            raise ValueError(f"missing or duplicated contrast: {candidate}/{control}/{window}/{metric}")
        row = matches[0]
        values = np.asarray(row["per_prompt_delta"], dtype=float)
        if (values.shape != (len(sources),) or not np.isfinite(values).all()
                or not math.isclose(float(values.mean()), row["mean_delta"], abs_tol=1e-9)):
            raise ValueError("invalid per-prompt contrast or inconsistent mean")
        arrays[metric], rows[metric] = values, row
    return rows, arrays


def motion_masks(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("motion deltas must be a finite vector")
    return {"dd_down": values < -MOTION_TOLERANCE,
            "dd_tied": np.abs(values) <= MOTION_TOLERANCE,
            "dd_up": values > MOTION_TOLERANCE}


def influence(values, sources):
    """Sensitivity summaries never remove prompts from the reported estimate."""
    x = np.asarray(values, dtype=float)
    if x.shape != (len(sources),) or not len(x) or not np.isfinite(x).all():
        raise ValueError("influence needs one finite difference per prompt")
    order = np.argsort(x, kind="stable")
    trim = len(x) // 10
    trimmed = np.sort(x)[trim:len(x)-trim] if trim else x
    positive = sorted((i for i in range(len(x)) if x[i] > 0), key=lambda i: (-x[i], sources[i]))
    total_positive = float(np.maximum(x, 0).sum())
    concentrated = []
    for count in (1, 3):
        chosen = positive[:count]
        mass = float(x[chosen].sum())
        concentrated.append({"requested_top_positive_count": count, "actual_count": len(chosen),
                             "source_indices": [sources[i] for i in chosen],
                             "share_of_positive_delta_mass": mass/total_positive if total_positive else None,
                             "mean_without_these_prompts": float((x.sum()-mass)/(len(x)-len(chosen)))
                             if len(chosen) < len(x) else None,
                             "diagnostic_only_not_an_alternative_cohort": True})
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "trim_each_tail_count": trim, "trimmed10_mean": float(trimmed.mean()),
            "positive_count": int((x > 0).sum()), "negative_count": int((x < 0).sum()),
            "exact_tie_count": int((x == 0).sum()),
            "leave_one_out_min_mean": float((x.sum()-x.max())/(len(x)-1)) if len(x) > 1 else None,
            "leave_one_out_max_mean": float((x.sum()-x.min())/(len(x)-1)) if len(x) > 1 else None,
            "smallest_delta_source": sources[int(order[0])], "largest_delta_source": sources[int(order[-1])],
            "positive_concentration": concentrated}


def diagnose(report, candidate, control, window, seed):
    rows, arrays = vectors(report, candidate, control, window)
    imaging, motion = arrays["imaging_quality"], arrays["dynamic_degree"]
    sources, n = report["source_indices"], len(imaging)
    masks = motion_masks(motion)
    groups = []
    for index, (name, mask) in enumerate(masks.items()):
        values = imaging[mask]
        # The contributions partition the FULL mean; no subgroup replaces it.
        contribution = np.where(mask, imaging, 0.)
        groups.append({"motion_group": name, "prompt_count": int(mask.sum()),
                       "source_indices": [s for s, selected in zip(sources, mask) if selected],
                       "imaging_mean": float(values.mean()) if len(values) else None,
                       "imaging_ci95": evidence.paired.bootstrap_ci(values.tolist(), seed=seed+index)
                       if len(values) >= 2 else None,
                       "contribution_to_full_imaging_mean": float(contribution.mean()),
                       "dynamic_mean": float(motion[mask].mean()) if len(values) else None,
                       "quality_mean": float(arrays["official_quality_score"][mask].mean()) if len(values) else None,
                       "subject_mean": float(arrays["subject_consistency"][mask].mean()) if len(values) else None})
    if not math.isclose(sum(g["contribution_to_full_imaging_mean"] for g in groups), float(imaging.mean()), abs_tol=1e-12):
        raise ValueError("motion groups do not reconstruct the full estimate")
    correlation = None
    if n >= 2 and float(np.std(imaging)) > 1e-12 and float(np.std(motion)) > 1e-12:
        correlation = float(np.corrcoef(imaging, motion)[0, 1])
    per_prompt = [{"source_index": source, "motion_group": next(k for k, v in masks.items() if v[i]),
                   **{f"delta_{metric}": float(values[i]) for metric, values in arrays.items()}}
                  for i, source in enumerate(sources)]
    return {"candidate": candidate, "control": control, "window": window, "prompt_count": n,
            "original_contrasts": {m: {"mean_delta": rows[m]["mean_delta"],
                                       "ci95": rows[m]["bootstrap_ci95"]} for m in METRICS},
            "posthoc_descriptive": True, "causal_motion_independence_established": False,
            "groups": groups, "pearson_imaging_vs_dynamic_delta": correlation,
            "imaging_up_and_dd_not_down_count": int(((imaging > 0) & ~masks["dd_down"]).sum()),
            "imaging_up_and_dd_down_count": int(((imaging > 0) & masks["dd_down"]).sum()),
            "imaging_influence": influence(imaging, sources), "per_prompt": per_prompt}


def make_packet(short, long=None):
    if long is not None:
        horizon.bind_horizons(short, long)
    bundles = [short] + ([long] if long is not None else [])
    analyses = []
    for bundle in bundles:
        report = bundle["report"]
        pairs = SHORT_PAIRS if bundle["label"] == "v219" else SHORT_PAIRS[:1]
        for candidate, control in pairs:
            for window in evidence.old.WINDOWS:
                row = diagnose(report, candidate, control, window, seed=222000+10*len(analyses))
                row.update(cohort=bundle["label"], base_seed=bundle["inputs"]["base_seed"])
                analyses.append(row)
    return {"version": 1, "experiment": "v222_fixed_seed_motion_imaging_diagnostic",
            "status": "v220_pending" if long is None else "30s_60s_diagnostics_available",
            "new_generation_count": 0, "new_seed_count": 0, "requested_review_pairs": [],
            "motion_tolerance": MOTION_TOLERANCE, "analyses": analyses,
            "source_report_sha256": {b["label"]: b["report_sha256"] for b in bundles},
            "automatic_risks": horizon.make_packet(short, long)["automatic_risks"],
            "boundaries": [
                "Posthoc diagnostics; the frozen full-video imaging endpoint is unchanged.",
                "Motion groups condition on observed treatment outcomes, not randomized subgroups or causal controls.",
                "Dynamic-degree ties do not establish equal motion: the metric is a coarse clip-level threshold statistic.",
                "Subgroup intervals are descriptive, unadjusted for multiple comparisons, and not new primary tests.",
                "Trimmed/leave-out means diagnose concentration only; every prompt remains in the main table.",
                "Correlations and a positive tied-motion subgroup do not prove that motion is preserved.",
                "Keep SF, random and pooled comparisons, including negative effects. No seed/method winner is selected.",
                "No new video, metric-model inference, or human review is requested; no automatic acceptance verdict."]}


def render(packet):
    lines = ["# Fixed-seed motion/imaging diagnostics", "", f"Status: {packet['status']}", "",
             "All differences below use a 0-100 scale. Descriptive diagnostics, not causal evidence.", "",
             "| Cohort | Candidate - control | Imaging | DD | Median imaging | Trimmed imaging | Imaging wins without DD drop |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in packet["analyses"]:
        if row["window"] != "full":
            continue
        original, inf = row["original_contrasts"], row["imaging_influence"]
        lines.append(f"| {row['cohort']} | {row['candidate']} - {row['control']} | "
                     f"{100*original['imaging_quality']['mean_delta']:+.4f} | "
                     f"{100*original['dynamic_degree']['mean_delta']:+.4f} | {100*inf['median']:+.4f} | "
                     f"{100*inf['trimmed10_mean']:+.4f} | {row['imaging_up_and_dd_not_down_count']}/{row['prompt_count']} |")
    for row in packet["analyses"]:
        if row["window"] != "full" or (row["candidate"], row["control"]) != SHORT_PAIRS[0]:
            continue
        lines += ["", f"## {row['cohort']} Ours - SF", "",
                  "| Observed DD change | Prompts | Mean imaging delta | Contribution to full imaging delta |",
                  "|---|---:|---:|---:|"]
        for group in row["groups"]:
            mean = "n/a" if group["imaging_mean"] is None else f"{100*group['imaging_mean']:+.4f}"
            lines.append(f"| {group['motion_group']} | {group['prompt_count']} | {mean} | "
                         f"{100*group['contribution_to_full_imaging_mean']:+.4f} |")
        for top in row["imaging_influence"]["positive_concentration"]:
            share = top["share_of_positive_delta_mass"]
            share_text = "n/a" if share is None else f"{100*share:.1f}%"
            lines.append(f"- Top {top['actual_count']} positive prompts: sources {top['source_indices']}, "
                         f"{share_text} of positive-delta mass. They are NOT removed from the main estimate.")
    lines += ["", "## Interpretation boundaries", "", *[f"- {s}" for s in packet["boundaries"]], ""]
    return "\n".join(lines)


def export(packet, root):
    evidence.dev.frozen_json(root/"motion_imaging.json", packet)
    evidence.dev.write_frozen(root/"motion_imaging.md", render(packet).encode())
    groups, prompts = [], []
    for row in packet["analyses"]:
        keys = {k: row[k] for k in ("cohort", "base_seed", "candidate", "control", "window")}
        for group in row["groups"]:
            ci = group["imaging_ci95"] or [None, None]
            groups.append({**keys, **{k: v for k, v in group.items() if k not in ("source_indices", "imaging_ci95")},
                           "imaging_ci_low": ci[0], "imaging_ci_high": ci[1]})
        for item in row["per_prompt"]:
            prompts.append({**keys, "effective_seed": row["base_seed"]+item["source_index"], **item})
    evidence.write_csv(root/"motion_groups.csv", groups)
    evidence.write_csv(root/"paired_prompt_deltas.csv", prompts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v219-root", type=Path, required=True)
    parser.add_argument("--v220-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    packet = make_packet(horizon.load(args.v219_root, "v219"),
                         horizon.load(args.v220_root, "v220") if args.v220_root else None)
    export(packet, args.output_root)
    print(f"[lphc-motion] status={packet['status']} contrasts={len(packet['analyses'])} new_videos=0 new_seeds=0")
    print(render(packet))


if __name__ == "__main__":
    main()
