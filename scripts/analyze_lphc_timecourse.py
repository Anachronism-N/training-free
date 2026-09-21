#!/usr/bin/env python3
"""GPU-free paired time curves from completed VBench parts, without new generation."""
import argparse
import json
from pathlib import Path

import numpy as np

import analyze_v165_final_decision as detail
from analyze_v212_lphc import load_validated_inputs
from export_lphc_paper_evidence import RAW_METRICS, write_csv
from v212_lphc_protocol import load_protocol, frozen_json, write_frozen, sha256


class ReadOnlyProtocol:
    """A new analysis can read old runs; generation code never gets this adapter."""
    def __init__(self, protocol):
        self.protocol = protocol

    def __getattr__(self, name):
        return getattr(self.protocol, name)

    def verify(self, repo, out, *, runtime=True):
        return self.protocol.verify(repo, out, runtime=False)


def paired_curve(candidate, control, *, seed, samples=5000):
    a, b = np.asarray(candidate, dtype=float), np.asarray(control, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 2 or a.shape[1] != 15:
        raise ValueError("expected matched prompt-by-15-clip arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or samples < 100:
        raise ValueError("finite values and at least 100 bootstrap samples required")
    delta = a-b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[indices].mean(axis=1)
    center = delta.mean(axis=0)
    lo, hi = np.quantile(means, [.025, .975], axis=0)
    radius = float(np.quantile(np.abs(means-center).max(axis=1), .95))
    return [{"clip_index": j+1, "candidate_mean": float(a[:, j].mean()), "control_mean": float(b[:, j].mean()),
             "mean_delta": float(center[j]), "pointwise_ci_low": float(lo[j]), "pointwise_ci_high": float(hi[j]),
             "within_metric_band_low": float(center[j]-radius), "within_metric_band_high": float(center[j]+radius)}
            for j in range(15)]


def analyze(run_root, campaign):
    p = ReadOnlyProtocol(load_protocol(campaign, run_root))
    _, _, source, manifest = load_validated_inputs(p.out, protocol=p)
    root = p.out/"evaluation"
    summary = json.loads((root/"metrics/vbench_core9_summary.json").read_text())
    contrasts = [("ours_correct", c) for c in ("sf_fifo21", "ours_random", "pooled_correct") if c in p.METHODS]
    curves, detail_hashes = [], {}
    for i, metric in enumerate(RAW_METRICS):
        values = {}
        for method in p.METHODS:
            path = root/f"metrics/vbench_long_parts/{method}/{metric}/results.json"
            # load_validated_inputs has authenticated all result and job-contract hashes.
            detail_hashes[f"{method}/{metric}"] = sha256(path)
            clips = detail.load_dimension(path, metric, prompt_count=len(p.SOURCE_INDICES), clips_per_video=15)
            array = np.asarray([clips[j] for j in range(len(p.SOURCE_INDICES))], dtype=float)
            factor = detail.scale_factor(float(array.mean()), summary["methods"][method][metric], name=f"{method}/{metric}")
            values[method] = factor*array
        for j, (candidate, control) in enumerate(contrasts):
            for row in paired_curve(values[candidate], values[control], seed=2191000+100*i+j):
                curves.append({"candidate": candidate, "control": control, "metric": metric, **row})
    return {"campaign": campaign, "source_root": str(p.out), "source": source,
            "source_indices": list(p.SOURCE_INDICES), "prompt_count": len(p.SOURCE_INDICES),
            "method_specs": manifest["methods"], "detail_sha256": detail_hashes, "curves": curves,
            "posthoc_descriptive": True, "generation_changed": False, "bootstrap_samples": 5000,
            "boundary": "Prompt-paired bootstrap resamples each prompt's entire 15-clip trajectory. Pointwise CIs and centered max-deviation bands across 15 clips within each metric are descriptive, not new primary tests. No correction across metrics/contrasts, no selection of favorable clips. Clip index is not exact wall-clock time or person re-identification."}


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    controls = list(dict.fromkeys(row["control"] for row in report["curves"]))
    for control in controls:
        fig, axes = plt.subplots(3, 3, figsize=(12, 8), squeeze=False)
        for metric, ax in zip(RAW_METRICS, axes.flat):
            rows = [r for r in report["curves"] if r["metric"] == metric and r["control"] == control]
            x = [r["clip_index"] for r in rows]
            ax.plot(x, [r["mean_delta"] for r in rows], color="#2166ac", linewidth=1.4)
            ax.fill_between(x, [r["within_metric_band_low"] for r in rows],
                            [r["within_metric_band_high"] for r in rows], color="#2166ac", alpha=.16)
            ax.axhline(0, color="#444444", linewidth=.7)
            ax.set(title=metric.replace("_", " "), xlabel="Ordered evaluation clip index", ylabel="Ours minus control")
            ax.set_xticks([1, 4, 8, 12, 15])
        fig.suptitle(f"{report['campaign']}: Ours - {control}, {report['prompt_count']} paired prompts (descriptive)")
        fig.tight_layout()
        fig.savefig(out/f"timecourse_vs_{control}.png", dpi=200)
        fig.savefig(out/f"timecourse_vs_{control}.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("v216", "v217", "v219"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="requires matplotlib; writes all nine dimensions, not a chosen subset")
    args = parser.parse_args()
    report = analyze(args.run_root, args.campaign)
    frozen_json(args.output_root/"timecourse.json", report)
    write_csv(args.output_root/"timecourse.csv", report["curves"])
    write_frozen(args.output_root/"README.md", ("# Paired timecourse\n\n"+report["boundary"]+"\n").encode())
    if args.plot:
        plot(report, args.output_root)
    print(f"[timecourse] campaign={args.campaign} prompts={report['prompt_count']} curves={len(report['curves'])} new_videos=0")


if __name__ == "__main__":
    main()
