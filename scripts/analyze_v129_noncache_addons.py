#!/usr/bin/env python3
"""Audit and summarize v129 non-cache Value-calibration traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def method_from_task(task: str) -> str:
    marker = "__p"
    if marker not in task:
        raise ValueError(f"invalid task name: {task}")
    return task.split(marker, 1)[0]


def finite_values(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("trace contains non-finite numeric samples")
    return result


def mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    contract_path = run_root / "contracts" / "experiment.json"
    if not contract_path.is_file():
        raise SystemExit(f"missing experiment contract: {contract_path}")
    contract = load_json(contract_path)
    methods = [
        str(row["key"])
        for row in contract.get("methods", [])
        if isinstance(row, dict)
    ]
    prompt_count = int(contract.get("prompt_count", 0))
    expected = {
        f"{method}__p{index:03d}"
        for method in methods
        for index in range(prompt_count)
    }

    traces: dict[str, dict[str, Any]] = {}
    for task_dir in sorted((run_root / "videos").glob("*")):
        if not task_dir.is_dir():
            continue
        trace_path = task_dir / "value_alignment_trace.json"
        if trace_path.is_file():
            traces[task_dir.name] = load_json(trace_path)

    failures: list[str] = []
    missing = sorted(expected - set(traces))
    extra = sorted(set(traces) - expected)
    if missing:
        failures.append(f"missing traces: {len(missing)}")
    if extra:
        failures.append(f"unexpected traces: {len(extra)}")

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task, trace in traces.items():
        method = method_from_task(task)
        changed = int(trace.get("changed_calls", 0))
        nonfinite = int(trace.get("sampled_nonfinite", 0))
        if nonfinite:
            failures.append(f"{task}: sampled_nonfinite={nonfinite}")
        if method.endswith("value_control") and changed != 0:
            failures.append(f"{task}: control unexpectedly changed Value tensors")
        if not method.endswith("value_control") and changed <= 0:
            failures.append(f"{task}: add-on never changed a Value tensor")
        by_method[method].append(trace)

    rows: list[dict[str, Any]] = []
    for method in methods:
        items = by_method.get(method, [])
        relative = [
            value
            for item in items
            for value in finite_values(item.get("relative_delta_samples"))
        ]
        maximum = [
            value
            for item in items
            for value in finite_values(item.get("max_abs_delta_samples"))
        ]
        rows.append(
            {
                "method": method,
                "tasks": len(items),
                "calls": sum(int(item.get("calls", 0)) for item in items),
                "changed_calls": sum(
                    int(item.get("changed_calls", 0)) for item in items
                ),
                "sampled_calls": sum(
                    int(item.get("sampled_calls", 0)) for item in items
                ),
                "mean_relative_delta": mean(relative),
                "max_sampled_abs_delta": max(maximum) if maximum else None,
            }
        )

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else run_root / "analysis" / "value_alignment"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "run_root": str(run_root),
        "contract": str(contract_path),
        "prompt_count": prompt_count,
        "expected_tasks": len(expected),
        "observed_traces": len(traces),
        "rows": rows,
        "failures": failures,
        "ok": not failures,
    }
    (output_root / "value_alignment_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# v129 Non-cache Value Alignment Trace",
        "",
        f"- Prompt count: {prompt_count}",
        f"- Expected tasks: {len(expected)}",
        f"- Observed traces: {len(traces)}",
        f"- Audit: {'PASS' if not failures else 'FAIL'}",
        "",
        "| Method | Tasks | Calls | Changed | Mean relative delta | Max abs delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        relative = row["mean_relative_delta"]
        maximum = row["max_sampled_abs_delta"]
        lines.append(
            f"| {row['method']} | {row['tasks']} | {row['calls']} | "
            f"{row['changed_calls']} | "
            f"{'n/a' if relative is None else f'{relative:.6g}'} | "
            f"{'n/a' if maximum is None else f'{maximum:.6g}'} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    (output_root / "value_alignment_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        "[v129-addon-analysis] "
        f"ok={not failures} traces={len(traces)}/{len(expected)} "
        f"output={output_root}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
