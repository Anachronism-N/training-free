#!/usr/bin/env python3
"""Paired untouched-holdout analysis for strict RCCP membership."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_v174_paired_metrics as base


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_runtime_fingerprint(
    parts_root: Path,
    methods: tuple[str, ...],
    dimensions: tuple[str, ...],
) -> dict:
    """Require one metric implementation contract across every paired job."""
    normalized_contracts = []
    for method in methods:
        for dimension in dimensions:
            path = parts_root / method / dimension / "job_contract.json"
            if not path.is_file():
                raise ValueError(f"missing VBench job contract: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("method") != method
                or payload.get("dimension") != dimension
            ):
                raise ValueError(f"mixed VBench job contract: {path}")
            dependencies = payload.get("dependencies") or {}
            dependency_hashes = {}
            for name, artifact in sorted(dependencies.items()):
                digest = artifact.get("sha256") if isinstance(artifact, dict) else None
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError(f"invalid VBench dependency hash: {path}:{name}")
                dependency_hashes[name] = digest
            if not dependency_hashes:
                raise ValueError(f"VBench dependency contract is absent: {path}")
            model_loading = payload.get("model_loading") or {}
            normalized_contracts.append(
                {
                    "contract_version": int(payload.get("version", -1)),
                    "vbench_commit": str(payload.get("vbench_commit", "")),
                    "dependency_sha256": dependency_hashes,
                    "prompt_mapping": payload.get("prompt_mapping"),
                    "mode": payload.get("mode"),
                    "dev_flag": payload.get("dev_flag"),
                    "num_of_samples_per_prompt": payload.get(
                        "num_of_samples_per_prompt"
                    ),
                    "local_models": bool(model_loading.get("local_models", False)),
                }
            )
    encoded = {
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in normalized_contracts
    }
    if len(encoded) != 1:
        raise ValueError("VBench runtime contract differs across paired jobs")
    contract = normalized_contracts[0]
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "version": 1,
        "sha256": digest,
        "job_contract_count": len(normalized_contracts),
        "contract": contract,
        "path_fields_ignored": True,
    }


def analyze(manifest: dict, summary: dict, parts_root: Path) -> dict:
    if (
        manifest.get("experiment") != "v178_rccp_holdout_vbench"
        or manifest.get("profile_contract") != "v177"
        or manifest.get("generation_prompts_used_for_membership") is not False
    ):
        raise ValueError("invalid or leaked v178 comparison manifest")
    methods = tuple(row["key"] for row in manifest["methods"])
    prompt_count = int(manifest["prompt_count"])
    if tuple(summary.get("methods") or {}) != methods or summary.get("missing"):
        raise ValueError("v178 paired analysis received an incomplete summary")
    if prompt_count != 32 or "matched" not in methods or "all_recent" not in methods:
        raise ValueError("v178 requires matched/all-recent on 32 prompts")
    negatives = tuple(method for method in methods if method.startswith("hard_negative_"))
    if len(negatives) != 4:
        raise ValueError("v178 requires four layer/count-matched hard negatives")
    raw = base.load_prompt_rows(parts_root, summary, methods, prompt_count)
    rows = base.derived_rows(raw, methods, prompt_count)
    controls = ("all_recent", *negatives)
    comparisons = []
    for control_index, control in enumerate(controls):
        for metric_index, metric in enumerate(base.METRICS):
            deltas = [
                rows[("matched", prompt)][metric] - rows[(control, prompt)][metric]
                for prompt in range(prompt_count)
            ]
            comparisons.append(
                {
                    "comparison": f"matched_minus_{control}",
                    "control": control,
                    "metric": metric,
                    "mean_delta": float(np.mean(deltas)),
                    "median_delta": float(np.median(deltas)),
                    "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
                    "bootstrap_ci95": base.bootstrap_ci(
                        deltas, seed=1782026 + control_index * 101 + metric_index
                    ),
                    "p_value": base.sign_p(deltas),
                    "per_prompt_delta": deltas,
                }
            )
    for metric_index, metric in enumerate(base.METRICS):
        deltas = []
        for prompt in range(prompt_count):
            negative_mean = float(
                np.mean([rows[(method, prompt)][metric] for method in negatives])
            )
            deltas.append(rows[("matched", prompt)][metric] - negative_mean)
        comparisons.append(
            {
                "comparison": "matched_minus_hard_negative_ensemble",
                "control": "hard_negative_ensemble",
                "metric": metric,
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "win_fraction": float(np.mean(np.asarray(deltas) > 0.0)),
                "bootstrap_ci95": base.bootstrap_ci(
                    deltas, seed=1789026 + metric_index
                ),
                "p_value": base.sign_p(deltas),
                "per_prompt_delta": deltas,
            }
        )
    base.bh(comparisons)

    primary_metrics = {"official_quality_score", "identity_background"}
    primary = [
        row
        for row in comparisons
        if row["control"] == "hard_negative_ensemble"
        and row["metric"] in primary_metrics
    ]
    operator_primary = [
        row
        for row in comparisons
        if row["control"] == "all_recent" and row["metric"] in primary_metrics
    ]
    dynamic_rows = [
        row
        for row in comparisons
        if row["control"] in {"hard_negative_ensemble", "all_recent"}
        and row["metric"] == "dynamic_degree"
    ]
    gate_checks = {
        "ensemble_primary_mean_positive": all(
            row["mean_delta"] > 0.0 for row in primary
        ),
        "ensemble_primary_ci95_above_zero": all(
            row["bootstrap_ci95"][0] > 0.0 for row in primary
        ),
        "ensemble_primary_bh_q_le_0p10": all(
            row["q_value"] <= 0.10 for row in primary
        ),
        "ensemble_primary_win_fraction_ge_0p55": all(
            row["win_fraction"] >= 0.55 for row in primary
        ),
        "all_recent_primary_mean_positive": all(
            row["mean_delta"] > 0.0 for row in operator_primary
        ),
        "dynamic_mean_delta_ge_minus_0p02": all(
            row["mean_delta"] >= -0.02 for row in dynamic_rows
        ),
    }
    dynamic_nonregression = gate_checks["dynamic_mean_delta_ge_minus_0p02"]
    hypothesis_gate = all(gate_checks.values())
    return {
        "version": 1,
        "experiment": manifest["experiment"],
        "profile_contract": "v177",
        "prompt_count": prompt_count,
        "methods": list(methods),
        "hard_negative_controls": list(negatives),
        "per_prompt_metrics": {
            method: [
                {
                    "prompt_index": prompt,
                    **{
                        metric: rows[(method, prompt)][metric]
                        for metric in base.METRICS
                    },
                }
                for prompt in range(prompt_count)
            ]
            for method in methods
        },
        "comparisons": comparisons,
        "gate_checks": gate_checks,
        "failed_gate_checks": [
            name for name, passed in gate_checks.items() if not passed
        ],
        "membership_hypothesis_gate": bool(hypothesis_gate),
        "dynamic_nonregression_observed": bool(dynamic_nonregression),
        "decision": (
            "advance_rccp_membership_to_broader_generation"
            if hypothesis_gate
            else "reject_static_rccp_membership_for_generation"
        ),
        "claim_boundary": (
            "Only matched superiority over the layer/count-matched hard-negative "
            "ensemble on untouched prompts supports RCCP membership. All-recent "
            "isolates operator utility but cannot validate head selection."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v178 Paired RCCP Holdout Analysis",
        "",
        f"Prompts: {report['prompt_count']}",
        f"Membership gate: {report['membership_hypothesis_gate']}",
        f"Decision: {report['decision']}",
        f"Failed checks: {report['failed_gate_checks'] or 'none'}",
        "",
        "| Comparison | Metric | Mean delta | CI95 | Win | q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        if row["control"] != "hard_negative_ensemble":
            continue
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_delta']:.6f} | "
            f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}] | "
            f"{row['win_fraction']:.3f} | {row['q_value']:.4g} |"
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
    report = analyze(manifest, summary, args.parts_root)
    report["metric_runtime_fingerprint"] = metric_runtime_fingerprint(
        args.parts_root,
        tuple(row["key"] for row in manifest["methods"]),
        tuple(summary["dimensions"]),
    )
    report["input_provenance"] = {
        "comparison_manifest": str(manifest_path.resolve()),
        "comparison_manifest_sha256": sha256(manifest_path),
        "metric_summary": str(args.summary.resolve()),
        "metric_summary_sha256": sha256(args.summary),
        "parts_root": str(args.parts_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[v178-paired] "
        f"gate={report['membership_hypothesis_gate']} "
        f"decision={report['decision']} output={args.output}"
    )


if __name__ == "__main__":
    main()
