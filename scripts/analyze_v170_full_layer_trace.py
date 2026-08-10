#!/usr/bin/env python3
"""Audit every active v170 layer from head-filtered policy traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_v166_multiscale_motion_trace as v166
import analyze_v169_soft_cross_scale_trace as v169
import v169_soft_cross_scale_contract as v169_contract
import v170_matched_attribution_contract as contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_active_layer_rows(path: Path) -> dict[int, list[dict]]:
    by_layer: dict[int, dict[int, dict]] = {}
    observed_heads: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if (
                row.get("event") != "middle_selection"
                or row.get("branch") != "cond"
                or int(row.get("label", -1)) != 10
            ):
                continue
            strategy = next(
                (
                    item
                    for item in row.get("strategies", [])
                    if item.get("name") == "CoherentMotionStrategy"
                ),
                None,
            )
            if strategy is None:
                continue
            layer = int(row["layer"])
            head = int(row["head"])
            observed_heads.add(head)
            if head not in contract.TRACE_HEADS:
                raise ValueError(f"unexpected traced head {head} in {path}")
            sync_t = int(row["sync_t"])
            layer_rows = by_layer.setdefault(layer, {})
            if sync_t in layer_rows:
                raise ValueError(
                    f"duplicate layer={layer} head={head} t={sync_t} in {path}"
                )
            layer_rows[sync_t] = {
                "line_number": line_number,
                "layer": layer,
                "head": head,
                "sync_t": sync_t,
                "strategy": strategy,
            }
    expected_layers = set(contract.ACTIVE_LAYERS)
    if set(by_layer) != expected_layers:
        raise ValueError(
            f"active-layer coverage mismatch in {path}: "
            f"observed={sorted(by_layer)} expected={sorted(expected_layers)}"
        )
    if observed_heads != set(contract.TRACE_HEADS):
        raise ValueError(
            f"trace-head coverage mismatch in {path}: {sorted(observed_heads)}"
        )
    return {
        layer: [rows[key] for key in sorted(rows)]
        for layer, rows in sorted(by_layer.items())
    }


def analyzer_spec(method: str):
    kind = contract.ANALYZER_KIND[method]
    if kind == "v166_multiscale_motion":
        return v166.analyze_prompt, v166.aggregate_method, v166.MULTISCALE_MOTION
    if kind == "v169_query_weighted":
        return (
            v169.analyze_prompt,
            v169.aggregate_method,
            v169_contract.QUERY_WEIGHTED,
        )
    raise ValueError(f"unsupported analyzer kind: {kind}")


def analyze_method(trace_dir: Path, *, method: str) -> dict:
    paths = sorted(trace_dir.glob(f"{method}__p*.policy.jsonl"))
    if len(paths) != contract.PROMPT_COUNT:
        raise ValueError(
            f"expected {contract.PROMPT_COUNT} traces for {method}, found {len(paths)}"
        )
    analyze_prompt, aggregate_method, analyzer_method = analyzer_spec(method)
    layer_prompts = {layer: [] for layer in contract.ACTIVE_LAYERS}
    prompt_reports = []
    for path in paths:
        by_layer = load_active_layer_rows(path)
        prompt_layers = {}
        for layer, rows in by_layer.items():
            result = analyze_prompt(path, method=analyzer_method, rows=rows)
            if result["representative_layer"] != layer:
                raise ValueError(f"analyzer layer mismatch in {path}")
            if result["representative_head"] not in contract.TRACE_HEADS:
                raise ValueError(f"analyzer head mismatch in {path}")
            result["layer"] = layer
            prompt_layers[str(layer)] = result
            layer_prompts[layer].append(result)
        prompt_reports.append(
            {
                "prompt_index": next(iter(prompt_layers.values()))["prompt_index"],
                "trace": str(path),
                "layers": prompt_layers,
            }
        )
    if [row["prompt_index"] for row in prompt_reports] != list(
        range(contract.PROMPT_COUNT)
    ):
        raise ValueError(f"prompt coverage mismatch for {method}")

    per_layer = {
        str(layer): aggregate_method(rows, method=analyzer_method)
        for layer, rows in layer_prompts.items()
    }
    flattened = [
        row for layer in contract.ACTIVE_LAYERS for row in layer_prompts[layer]
    ]
    aggregate = aggregate_method(flattened, method=analyzer_method)
    aggregate.update(
        {
            "method": method,
            "analyzer_method": analyzer_method,
            "prompt_count": contract.PROMPT_COUNT,
            "layer_count": len(contract.ACTIVE_LAYERS),
            "layer_prompt_count": len(flattened),
        }
    )
    for row in per_layer.values():
        row["method"] = method
        row["analyzer_method"] = analyzer_method
    return {
        "kind": contract.ANALYZER_KIND[method],
        "aggregate": aggregate,
        "layers": per_layer,
        "prompts": prompt_reports,
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# v170 Full Active-layer Trace Audit",
        "",
        f"Overall mechanism gate: **{report['mechanism_gate']}**",
        "",
        (
            "| Method | Gate | Reads | Retrievals | Multi-candidate | "
            "Changed selector | Old recalls | Age p95 | Budget errors | Failures |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in contract.METHODS:
        row = report["methods"][method]["aggregate"]
        changed = row.get(
            "changed_from_v166_count",
            row.get("changed_from_legacy_count", 0),
        )
        lines.append(
            f"| {method} | {row['mechanism_gate']} | {row['read_count']} | "
            f"{row['retrieval_count']} | {row['multi_candidate_count']} | "
            f"{changed} | {row.get('old_recall_count', 'n/a')} | "
            f"{row['selected_age']['p95']} | "
            f"{row['read_budget_violation_count']} | "
            f"{row['contract_failure_count']} |"
        )
    for method in contract.METHODS:
        lines.extend(
            [
                "",
                f"## {method}",
                "",
                (
                    "| Layer | Gate | Retrievals | Changed | Old recalls | "
                    "Age median | Failures |"
                ),
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for layer in contract.ACTIVE_LAYERS:
            row = report["methods"][method]["layers"][str(layer)]
            changed = row.get(
                "changed_from_v166_count",
                row.get("changed_from_legacy_count", 0),
            )
            lines.append(
                f"| {layer} | {row['mechanism_gate']} | "
                f"{row['retrieval_count']} | {changed} | "
                f"{row.get('old_recall_count', 'n/a')} | "
                f"{row['selected_age']['median']} | "
                f"{row['contract_failure_count']} |"
            )
    lines.extend(
        [
            "",
            (
                "Per-layer gates are diagnostics only; the frozen mechanism gate "
                "requires complete coverage and a valid aggregate execution for "
                "each replica. Selector changes need not occur in every layer."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = {
        method: analyze_method(args.trace_dir, method=method)
        for method in contract.METHODS
    }
    mechanism_gate = all(row["aggregate"]["mechanism_gate"] for row in methods.values())
    report = {
        "version": 1,
        "experiment": "v170_full_active_layer_trace",
        "mechanism_gate": mechanism_gate,
        "trace_contract": {
            "layers": list(contract.ACTIVE_LAYERS),
            "heads": list(contract.TRACE_HEADS),
            "prompt_count": contract.PROMPT_COUNT,
            "method_count": len(contract.METHODS),
        },
        "methods": methods,
        "gate_definition": (
            "all four replicas have complete prompt/layer coverage, exact "
            "selector recomputation, exercised retrieval, atomic reads, and "
            "zero cache-contract or read-budget violations"
        ),
        "claim_boundary": (
            "the audit validates execution across active layers; it does not "
            "establish video-quality superiority or a head taxonomy"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    if not mechanism_gate:
        raise SystemExit("v170 full-layer mechanism gate failed")
    print(
        json.dumps(
            {method: row["aggregate"] for method, row in methods.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
