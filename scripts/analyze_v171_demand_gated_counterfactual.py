#!/usr/bin/env python3
"""Replay v171 selectors on frozen v170 full-layer traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import analyze_v168_cross_scale_consensus_trace as v168
import v171_demand_gated_contract as contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = (
    ROOT
    / "runs"
    / "v170_matched_attribution_moviebench16"
    / "full8"
    / "traces"
)
DEFAULT_OUTPUT = (
    ROOT
    / "runs"
    / "v171_demand_gated_motion_moviebench16"
    / "offline"
    / "v171_counterfactual.json"
)
SOURCE_METHOD = "ours_v170_queryweighted_a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def strategy_state(row: dict) -> dict | None:
    strategy = next(
        (
            item
            for item in row.get("strategies", [])
            if item.get("name") == "CoherentMotionStrategy"
        ),
        None,
    )
    if strategy is None:
        return None
    return strategy.get("state", {})


def first_pair(value: object) -> list[int] | None:
    pair = v168.first_pair(value if isinstance(value, list) else [])
    return None if pair is None else [int(pair[0]), int(pair[1])]


def changed_amplitude_delta(
    candidates: list[dict],
    *,
    baseline: list[int] | None,
    selected: list[int] | None,
) -> tuple[float, float] | None:
    if baseline is None or selected is None or baseline == selected:
        return None
    by_pair = {tuple(item["pair"]): item for item in candidates}
    old = by_pair.get(tuple(baseline))
    new = by_pair.get(tuple(selected))
    if old is None or new is None:
        return None
    return (
        float(new["candidate_local_magnitude"])
        - float(old["candidate_local_magnitude"]),
        float(new["candidate_context_magnitude_per_step"])
        - float(old["candidate_context_magnitude_per_step"]),
    )


def analyze(trace_dir: Path) -> dict:
    paths = sorted(trace_dir.glob(f"{SOURCE_METHOD}__p*.policy.jsonl"))
    failures: list[str] = []
    method_counts = {method: Counter() for method in contract.CANDIDATES}
    prompt_counts = {
        method: defaultdict(Counter) for method in contract.CANDIDATES
    }
    layer_counts = {
        method: defaultdict(Counter) for method in contract.CANDIDATES
    }
    age_deltas = {method: [] for method in contract.CANDIDATES}
    amplitude_deltas = {"local": [], "context": []}
    observed_layers: set[int] = set()
    observed_heads: set[int] = set()
    trace_hashes = {}
    read_events = 0
    retrieval_events = 0
    gate_ready_count = 0
    gate_triggered_count = 0
    full_query_changes = 0

    if len(paths) != contract.PROMPT_COUNT:
        failures.append(
            f"expected {contract.PROMPT_COUNT} source traces, found {len(paths)}"
        )
    for path in paths:
        prompt_index = int(path.name.split("__p", 1)[1].split(".", 1)[0])
        trace_hashes[path.name] = sha256(path)
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
                state = strategy_state(row)
                if state is None:
                    continue
                layer = int(row["layer"])
                head = int(row["head"])
                observed_layers.add(layer)
                observed_heads.add(head)
                read_events += 1
                retrieval = state.get("last_retrieval", {})
                if not retrieval:
                    continue
                retrieval_events += 1
                if retrieval.get("selection_mode") != (
                    "query_weighted_multiscale_magnitude"
                ):
                    failures.append(
                        f"{path.name}:{line_number}: unexpected source mode"
                    )
                    continue
                deficit = dict(retrieval.get("motion_deficit") or {})
                ready = bool(deficit.get("ready", False))
                triggered = bool(deficit.get("triggered", False))
                gate_ready_count += int(ready)
                gate_triggered_count += int(triggered)
                if triggered and not ready:
                    failures.append(
                        f"{path.name}:{line_number}: trigger before warmup"
                    )
                candidates = list(retrieval.get("candidates", []))
                traced_baseline = first_pair(
                    retrieval.get("motion_signature_selected", [])
                )
                traced_query = first_pair(
                    retrieval.get("query_weighted_selected", [])
                )
                if traced_baseline != traced_query:
                    full_query_changes += 1

                for method in contract.CANDIDATES:
                    expected = contract.expected_selection(
                        candidates,
                        method=method,
                        motion_deficit=deficit,
                    )
                    counts = method_counts[method]
                    counts["retrievals"] += 1
                    counts["ready"] += int(ready)
                    counts["triggered"] += int(triggered)
                    if expected["baseline"] != traced_baseline:
                        failures.append(
                            f"{path.name}:{line_number}: v166 replay mismatch"
                        )
                    if expected["query_weighted"] != traced_query:
                        failures.append(
                            f"{path.name}:{line_number}: query replay mismatch"
                        )
                    changed = bool(
                        expected["selected"] is not None
                        and expected["baseline"] is not None
                        and expected["selected"] != expected["baseline"]
                    )
                    counts["changed"] += int(changed)
                    counts["healthy_changed"] += int(changed and not triggered)
                    counts["deficit_changed"] += int(changed and triggered)
                    prompt_counts[method][prompt_index]["changed"] += int(changed)
                    prompt_counts[method][prompt_index]["triggered"] += int(
                        triggered
                    )
                    layer_counts[method][layer]["changed"] += int(changed)
                    layer_counts[method][layer]["triggered"] += int(triggered)
                    if changed and expected["selected"] and expected["baseline"]:
                        age_deltas[method].append(
                            int(expected["baseline"][1])
                            - int(expected["selected"][1])
                        )
                    if method == contract.DEFICIT_BASELINE and changed:
                        delta = changed_amplitude_delta(
                            candidates,
                            baseline=expected["baseline"],
                            selected=expected["selected"],
                        )
                        if delta is not None:
                            amplitude_deltas["local"].append(delta[0])
                            amplitude_deltas["context"].append(delta[1])

    if observed_layers != set(contract.ACTIVE_LAYERS):
        failures.append(
            f"active layer mismatch: observed={sorted(observed_layers)}"
        )
    if observed_heads != set(contract.TRACE_HEADS):
        failures.append(
            f"trace head mismatch: observed={sorted(observed_heads)}"
        )

    methods = {}
    for method in contract.CANDIDATES:
        counts = method_counts[method]
        ages = age_deltas[method]
        methods[method] = {
            **dict(counts),
            "change_rate_over_retrievals": (
                counts["changed"] / counts["retrievals"]
                if counts["retrievals"]
                else 0.0
            ),
            "age_delta": {
                "count": len(ages),
                "mean": statistics.fmean(ages) if ages else None,
                "median": statistics.median(ages) if ages else None,
                "min": min(ages) if ages else None,
                "max": max(ages) if ages else None,
                "meaning": "positive means v171 selected an older pair than v166",
            },
            "per_prompt": {
                str(index): dict(prompt_counts[method][index])
                for index in range(contract.PROMPT_COUNT)
            },
            "per_layer": {
                str(layer): dict(layer_counts[method][layer])
                for layer in contract.ACTIVE_LAYERS
            },
        }

    local = amplitude_deltas["local"]
    context = amplitude_deltas["context"]
    methods[contract.DEFICIT_BASELINE]["changed_candidate_amplitude_delta"] = {
        "local_mean": statistics.fmean(local) if local else None,
        "local_median": statistics.median(local) if local else None,
        "context_mean": statistics.fmean(context) if context else None,
        "context_median": statistics.median(context) if context else None,
        "meaning": "candidate magnitude selected by v171 minus v166",
    }
    offline_gate = bool(
        not failures
        and len(paths) == contract.PROMPT_COUNT
        and methods[contract.DEFICIT_QUERY]["changed"] > 0
        and methods[contract.DEFICIT_QUERY]["changed"] < full_query_changes
        and methods[contract.DEFICIT_BASELINE]["changed"] > 0
        and methods[contract.DEFICIT_QUERY]["healthy_changed"] == 0
        and methods[contract.DEFICIT_BASELINE]["healthy_changed"] == 0
        and local
        and context
        and statistics.fmean(local) > 0.0
        and statistics.fmean(context) > 0.0
    )
    return {
        "version": 1,
        "experiment": "v171_demand_gated_offline_counterfactual",
        "offline_gate": offline_gate,
        "source": {
            "method": SOURCE_METHOD,
            "trace_dir": portable_path(trace_dir),
            "trace_count": len(paths),
            "trace_sha256": trace_hashes,
        },
        "coverage": {
            "read_events": read_events,
            "retrieval_events": retrieval_events,
            "layers": sorted(observed_layers),
            "heads": sorted(observed_heads),
            "gate_ready": gate_ready_count,
            "gate_triggered": gate_triggered_count,
            "full_query_weighted_changes": full_query_changes,
        },
        "methods": methods,
        "failures": failures,
        "claim_boundary": (
            "Counterfactual replay proves branch activity and selector sparsity, "
            "not video quality; autoregressive generation must be rerun."
        ),
    }


def write_markdown(path: Path, report: dict) -> None:
    coverage = report["coverage"]
    lines = [
        "# v171 Offline Demand-gated Counterfactual",
        "",
        f"Offline gate: **{report['offline_gate']}**",
        "",
        (
            f"The frozen v170 trace contains {coverage['retrieval_events']} "
            f"retrieval decisions. Motion deficit triggered "
            f"{coverage['gate_triggered']} times. Full Query weighting changed "
            f"{coverage['full_query_weighted_changes']} v166 choices."
        ),
        "",
        "| Candidate | Changed | Deficit changed | Healthy changed | Change rate | Age delta mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in contract.CANDIDATES:
        row = report["methods"][method]
        lines.append(
            f"| {method} | {row['changed']} | {row['deficit_changed']} | "
            f"{row['healthy_changed']} | "
            f"{row['change_rate_over_retrievals']:.4%} | "
            f"{row['age_delta']['mean']:.3f} |"
        )
    amplitude = report["methods"][contract.DEFICIT_BASELINE][
        "changed_candidate_amplitude_delta"
    ]
    lines.extend(
        [
            "",
            (
                "For changed baseline-calibrated decisions, selected historical "
                f"motion magnitude increased by {amplitude['local_mean']:.6f} "
                f"(local) and {amplitude['context_mean']:.6f} (context) on average."
            ),
            "",
            report["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = analyze(args.trace_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output.with_suffix(".md"), report)
    print(json.dumps({"offline_gate": report["offline_gate"], **report["coverage"]}))
    if not report["offline_gate"]:
        raise SystemExit("v171 offline counterfactual gate failed")


if __name__ == "__main__":
    main()
