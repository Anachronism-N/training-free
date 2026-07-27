#!/usr/bin/env python3
"""Summarize v115/v116 role-memory policy and feature traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def summarize_policy(path: Path) -> dict[str, Any]:
    records = 0
    strategy_counts: Counter[str] = Counter()
    decision_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    union_equivalents: list[float] = []
    retrieval_selected: list[int] = []
    retrieval_similarity: list[float] = []
    retrieval_archive: list[int] = []
    prototype_bank: list[int] = []
    prototype_counts: list[int] = []
    snapshot_bank: list[int] = []
    snapshot_tokens: list[int] = []
    motion_pair_bank: list[int] = []
    contract_failures = 0
    overlap_failures = 0
    max_counters: dict[str, dict[str, int]] = defaultdict(dict)

    for event in _load_jsonl(path):
        if event.get("event") != "middle_selection":
            continue
        records += 1
        if not bool(event.get("cache_contract_pass", False)):
            contract_failures += 1
        if event.get("middle_sink_overlap") or event.get("middle_recent_overlap"):
            overlap_failures += 1
        frame_seqlen = int(event.get("frame_seqlen", 0))
        if frame_seqlen > 0:
            union_equivalents.append(
                float(event.get("union_token_count", 0)) / frame_seqlen
            )
        for item in event.get("strategies", []):
            name = str(item.get("name", "unknown"))
            strategy_counts[name] += 1
            state = item.get("state")
            if not isinstance(state, dict):
                continue
            decision = state.get("last_decision", {})
            if isinstance(decision, dict):
                reason = decision.get("reason")
                if reason:
                    decision_reasons[name][str(reason)] += 1
            for counter_name in (
                "accepted_count",
                "rejected_count",
                "evicted_count",
                "compressed_count",
                "created_count",
            ):
                if counter_name in state:
                    key = (
                        f"{event.get('layer')}:{event.get('head')}:"
                        f"{event.get('seq')}:{event.get('branch')}"
                    )
                    max_counters[name][f"{key}:{counter_name}"] = max(
                        int(state[counter_name]),
                        max_counters[name].get(f"{key}:{counter_name}", 0),
                    )
            if name == "SemanticRetrievalStrategy":
                retrieval_archive.append(
                    len(state.get("archive_frame_ids", []))
                )
                retrieval = state.get("last_retrieval", {})
                selected = retrieval.get("selected", [])
                retrieval_selected.append(len(selected))
                retrieval_similarity.extend(
                    float(row["similarity"])
                    for row in selected
                    if isinstance(row, dict) and "similarity" in row
                )
            elif name == "TemporalPrototypeStrategy":
                prototype_bank.append(len(state.get("prototype_spans", [])))
                prototype_counts.extend(
                    int(value) for value in state.get("prototype_counts", [])
                )
            elif name in {
                "UniqueSnapshotStrategy",
                "SparseSnapshotStrategy",
            }:
                snapshot_bank.append(
                    len(state.get("snapshot_frame_ids", []))
                )
                snapshot_tokens.extend(
                    int(value)
                    for value in state.get("snapshot_token_counts", [])
                )
            elif name == "CoherentMotionStrategy":
                motion_pair_bank.append(
                    len(state.get("pair_frame_ids", []))
                )

    counter_totals: dict[str, dict[str, int]] = {}
    for strategy, keyed in max_counters.items():
        totals: Counter[str] = Counter()
        for key, value in keyed.items():
            totals[key.rsplit(":", maxsplit=1)[1]] += int(value)
        counter_totals[strategy] = dict(sorted(totals.items()))
    return {
        "records": records,
        "strategy_records": dict(sorted(strategy_counts.items())),
        "decision_reasons": {
            name: dict(sorted(values.items()))
            for name, values in sorted(decision_reasons.items())
        },
        "counter_totals": counter_totals,
        "middle_frame_equivalents_mean": _round(_mean(union_equivalents)),
        "middle_frame_equivalents_max": (
            _round(max(union_equivalents)) if union_equivalents else None
        ),
        "retrieval": {
            "archive_size_mean": _round(_mean(retrieval_archive)),
            "archive_size_max": max(retrieval_archive, default=0),
            "selected_mean": _round(_mean(retrieval_selected)),
            "selected_max": max(retrieval_selected, default=0),
            "similarity_mean": _round(_mean(retrieval_similarity)),
            "similarity_min": (
                _round(min(retrieval_similarity))
                if retrieval_similarity
                else None
            ),
        },
        "prototype": {
            "bank_size_mean": _round(_mean(prototype_bank)),
            "bank_size_max": max(prototype_bank, default=0),
            "represented_frames_mean": _round(_mean(prototype_counts)),
            "represented_frames_max": max(prototype_counts, default=0),
        },
        "snapshot": {
            "bank_size_mean": _round(_mean(snapshot_bank)),
            "bank_size_max": max(snapshot_bank, default=0),
            "tokens_mean": _round(_mean(snapshot_tokens)),
            "tokens_min": min(snapshot_tokens, default=0),
            "tokens_max": max(snapshot_tokens, default=0),
        },
        "motion_pair": {
            "bank_size_mean": _round(_mean(motion_pair_bank)),
            "bank_size_max": max(motion_pair_bank, default=0),
        },
        "contract_failures": contract_failures,
        "overlap_failures": overlap_failures,
    }


def summarize_features(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "records": 0,
            "contexts": {},
            "motion_mean": None,
            "semantic_mean": None,
            "sparse_token_score_mean": None,
        }
    records = 0
    contexts: Counter[str] = Counter()
    motion: list[float] = []
    semantic: list[float] = []
    token_means: list[float] = []
    for event in _load_jsonl(path):
        if event.get("event") != "role_event_features":
            continue
        records += 1
        contexts[str(event.get("context_key", "unknown"))] += 1
        motion.extend(float(value) for value in event.get("motion_scores", []))
        semantic.extend(
            float(value)
            for value in event.get("adjacent_semantic_similarity", [])
        )
        token_summary = event.get("token_score_summary")
        if isinstance(token_summary, dict) and "mean" in token_summary:
            token_means.append(float(token_summary["mean"]))
    return {
        "records": records,
        "contexts": dict(sorted(contexts.items())),
        "motion_mean": _round(_mean(motion)),
        "semantic_mean": _round(_mean(semantic)),
        "sparse_token_score_mean": _round(_mean(token_means)),
    }


def cell_name_from_policy(path: Path) -> str:
    suffix = ".policy.jsonl"
    if not path.name.endswith(suffix):
        raise ValueError(f"not a policy trace: {path}")
    return path.name[: -len(suffix)]


def write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    output_dir = run_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "role_memory_trace_summary.json"
    csv_path = output_dir / "role_memory_trace_summary.csv"
    md_path = output_dir / "role_memory_trace_summary.md"
    json_path.write_text(
        json.dumps({"cells": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    flat_rows = []
    for row in rows:
        policy = row["policy"]
        features = row["features"]
        flat_rows.append(
            {
                "cell": row["cell"],
                "policy_records": policy["records"],
                "strategies": "+".join(policy["strategy_records"]),
                "middle_eq_mean": policy["middle_frame_equivalents_mean"],
                "middle_eq_max": policy["middle_frame_equivalents_max"],
                "retrieval_selected_mean": policy["retrieval"]["selected_mean"],
                "retrieval_similarity_mean": policy["retrieval"][
                    "similarity_mean"
                ],
                "prototype_represented_mean": policy["prototype"][
                    "represented_frames_mean"
                ],
                "snapshot_tokens_mean": policy["snapshot"]["tokens_mean"],
                "motion_pair_bank_mean": policy["motion_pair"][
                    "bank_size_mean"
                ],
                "feature_motion_mean": features["motion_mean"],
                "feature_semantic_mean": features["semantic_mean"],
                "contract_failures": policy["contract_failures"],
                "overlap_failures": policy["overlap_failures"],
            }
        )
    fields = list(flat_rows[0]) if flat_rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)

    lines = [
        "# Role-memory trace summary",
        "",
        "| cell | strategies | middle eq mean/max | retrieval k/sim | "
        "prototype span | snapshot tokens | motion bank | failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat_rows:
        lines.append(
            f"| {row['cell']} | {row['strategies']} | "
            f"{row['middle_eq_mean']}/{row['middle_eq_max']} | "
            f"{row['retrieval_selected_mean']}/"
            f"{row['retrieval_similarity_mean']} | "
            f"{row['prototype_represented_mean']} | "
            f"{row['snapshot_tokens_mean']} | "
            f"{row['motion_pair_bank_mean']} | "
            f"{row['contract_failures'] + row['overlap_failures']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    trace_root = run_root / "traces"
    policy_paths = sorted(trace_root.glob("*.policy.jsonl"))
    if not policy_paths:
        raise SystemExit(f"no policy traces found under {trace_root}")
    rows = []
    for policy_path in policy_paths:
        cell = cell_name_from_policy(policy_path)
        role_path = trace_root / f"{cell}.role_event.jsonl"
        rows.append(
            {
                "cell": cell,
                "policy_path": str(policy_path),
                "feature_path": str(role_path) if role_path.is_file() else None,
                "policy": summarize_policy(policy_path),
                "features": summarize_features(
                    role_path if role_path.is_file() else None
                ),
            }
        )
    write_outputs(run_root, rows)


if __name__ == "__main__":
    main()
