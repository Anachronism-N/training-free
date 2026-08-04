#!/usr/bin/env python3
"""Audit v159 coherent-motion traces and freeze the v160 diagnosis basis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v159_motion_pair_trace_diagnosis"
PROMPT_COUNT = 16
METHOD_LAYERS = {
    "ours_interleaved10_reservoir2_motionpair1": 7,
    "ours_interleaved10_motionpair2": 7,
    "ours_middle10_reservoir2_motionpair1": 15,
}
TRACE_PATTERN = re.compile(r"^traces/(.+)__p(\d{3})\.policy\.jsonl$")


def parse_args() -> argparse.Namespace:
    default_tar = (
        ROOT
        / "docs"
        / "results"
        / "v159_motion_coherent_reservoir_moviebench16"
        / "v159_diagnostics.tar.gz"
    )
    default_output = default_tar.parent / "v159_motion_pair_trace_diagnosis.json"
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--diagnostics-tar", type=Path, default=default_tar)
    source.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-json", type=Path, default=default_output)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def iter_tar_traces(path: Path) -> Iterable[tuple[str, int, list[str]]]:
    with tarfile.open(path, "r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and TRACE_PATTERN.fullmatch(member.name)
            ),
            key=lambda member: member.name,
        )
        for member in members:
            match = TRACE_PATTERN.fullmatch(member.name)
            assert match is not None
            method, prompt_raw = match.groups()
            if method not in METHOD_LAYERS:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read tar member: {member.name}")
            yield method, int(prompt_raw), handle.read().decode("utf-8").splitlines()


def iter_directory_traces(path: Path) -> Iterable[tuple[str, int, list[str]]]:
    trace_dir = path / "traces" if (path / "traces").is_dir() else path
    for trace_path in sorted(trace_dir.glob("*.policy.jsonl")):
        match = TRACE_PATTERN.fullmatch(f"traces/{trace_path.name}")
        if match is None:
            continue
        method, prompt_raw = match.groups()
        if method not in METHOD_LAYERS:
            continue
        yield method, int(prompt_raw), trace_path.read_text(
            encoding="utf-8"
        ).splitlines()


def trace_rows(lines: list[str], *, method: str, prompt_index: int) -> list[dict]:
    target_layer = METHOD_LAYERS[method]
    selected = []
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{method} prompt={prompt_index} line={line_number}: {error}"
            ) from error
        if (
            event.get("event") == "middle_selection"
            and int(event.get("layer", -1)) == target_layer
            and int(event.get("head", -1)) == 0
            and int(event.get("label", -1)) == 10
        ):
            selected.append(event)
    if len(selected) != 40:
        raise ValueError(
            f"{method} prompt={prompt_index}: expected 40 representative "
            f"updates at L{target_layer}/H0, got {len(selected)}"
        )
    return selected


def analyze_trace(events: list[dict]) -> dict:
    reasons: Counter[str] = Counter()
    pair_ages: list[float] = []
    union_counts: list[float] = []
    final_state = None
    for event in events:
        union_counts.append(float(event.get("union_frame_count", 0)))
        motion_items = [
            item
            for item in event.get("strategies", [])
            if item.get("name") == "CoherentMotionStrategy"
        ]
        if len(motion_items) != 1:
            raise ValueError("representative event lacks one coherent-motion state")
        state = motion_items[0].get("state")
        if not isinstance(state, dict):
            raise ValueError("coherent-motion state is not a mapping")
        final_state = state
        decision = state.get("last_decision", {})
        if decision.get("reason"):
            reasons[str(decision["reason"])] += 1
        sync_t = int(event.get("sync_t", 0))
        for pair in state.get("pair_frame_ids", []):
            pair_ages.append(float(sync_t - int(pair[1])))
    assert final_state is not None
    return {
        "updates": len(events),
        "accepted_count": int(final_state.get("accepted_count", -1)),
        "rejected_count": int(final_state.get("rejected_count", -1)),
        "evicted_count": int(final_state.get("evicted_count", -1)),
        "reason_counts": dict(sorted(reasons.items())),
        "pair_age": summarize(pair_ages),
        "union_frame_count": summarize(union_counts),
        "union_full4_fraction": (
            sum(value == 4.0 for value in union_counts) / len(union_counts)
        ),
        "union_zero_fraction": (
            sum(value == 0.0 for value in union_counts) / len(union_counts)
        ),
    }


def aggregate(rows: list[dict]) -> dict:
    reasons: Counter[str] = Counter()
    for row in rows:
        reasons.update(row["reason_counts"])
    return {
        "prompt_count": len(rows),
        "updates_per_prompt": sorted({row["updates"] for row in rows}),
        "accepted_per_prompt": summarize(
            [float(row["accepted_count"]) for row in rows]
        ),
        "rejected_per_prompt": summarize(
            [float(row["rejected_count"]) for row in rows]
        ),
        "evicted_per_prompt": summarize(
            [float(row["evicted_count"]) for row in rows]
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "pair_age_p95_per_prompt": summarize(
            [float(row["pair_age"]["p95"]) for row in rows]
        ),
        "pair_age_max": max(float(row["pair_age"]["max"]) for row in rows),
        "union_mean_per_prompt": summarize(
            [float(row["union_frame_count"]["mean"]) for row in rows]
        ),
        "union_full4_fraction": (
            sum(float(row["union_full4_fraction"]) for row in rows) / len(rows)
        ),
        "union_zero_fraction": (
            sum(float(row["union_zero_fraction"]) for row in rows) / len(rows)
        ),
    }


def markdown(report: dict) -> str:
    lines = [
        "# v159 Motion-Pair Trace Diagnosis",
        "",
        "This is a mechanism audit, not a quality comparison.",
        "",
        "| Method | Accept/prompt | Reject/prompt | Pair-age p95 | Max age | Union mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, payload in report["methods"].items():
        lines.append(
            f"| {method} | {payload['accepted_per_prompt']['mean']:.3f} | "
            f"{payload['rejected_per_prompt']['mean']:.3f} | "
            f"{payload['pair_age_p95_per_prompt']['mean']:.3f} | "
            f"{payload['pair_age_max']:.0f} | "
            f"{payload['union_mean_per_prompt']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen conclusion",
            "",
            "The motion route executed and respected its read budget, but the "
            "motion-quantile gate was the dominant rejection path. The existing "
            "`max_pair_age=24` only relaxed replacement; it did not bypass the "
            "quantile gate, so retained pairs could remain substantially older "
            "than 24 frames.",
            "",
            "v160 therefore changes only the selected Middle10 route: stale "
            "pairs use a 12-frame refresh horizon and may bypass the motion "
            "quantile gate. Positive-motion and semantic-coherence eligibility, "
            "pair adjacency, spacing, sink/recent allocation, and the 9-FFE read "
            "budget remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.trace_root is not None:
        source_path = args.trace_root.resolve()
        iterator = iter_directory_traces(source_path)
        source = {"type": "trace_root", "path": str(source_path)}
    else:
        source_path = args.diagnostics_tar.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        iterator = iter_tar_traces(source_path)
        source = {
            "type": "diagnostics_tar",
            "path": str(source_path),
            "sha256": sha256(source_path),
        }
    per_method: dict[str, list[dict]] = {method: [] for method in METHOD_LAYERS}
    observed: dict[str, set[int]] = {method: set() for method in METHOD_LAYERS}
    for method, prompt_index, lines in iterator:
        if prompt_index in observed[method]:
            raise ValueError(f"duplicate trace: {method} prompt={prompt_index}")
        observed[method].add(prompt_index)
        per_method[method].append(
            {
                "prompt_index": prompt_index,
                **analyze_trace(
                    trace_rows(lines, method=method, prompt_index=prompt_index)
                ),
            }
        )
    expected = set(range(PROMPT_COUNT))
    for method, indices in observed.items():
        if indices != expected:
            raise ValueError(
                f"incomplete traces for {method}: missing={sorted(expected-indices)} "
                f"extra={sorted(indices-expected)}"
            )
    method_reports = {
        method: aggregate(sorted(rows, key=lambda row: row["prompt_index"]))
        for method, rows in per_method.items()
    }
    for method, payload in method_reports.items():
        reasons = payload["reason_counts"]
        dominant = max(reasons, key=reasons.get)
        if dominant != "motion_quantile_gate" or payload["pair_age_max"] <= 24:
            raise ValueError(
                f"{method} no longer supports the frozen diagnosis: "
                f"dominant={dominant} max_age={payload['pair_age_max']}"
            )
    report = {
        "version": 1,
        "experiment": EXPERIMENT,
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "source": source,
        "representative_trace": {
            method: {"layer": layer, "head": 0, "label": 10}
            for method, layer in METHOD_LAYERS.items()
        },
        "methods": method_reports,
        "diagnosis": {
            "mechanism_executed": True,
            "dominant_rejection": "motion_quantile_gate",
            "max_pair_age_is_not_a_hard_refresh_bound": True,
            "next_isolated_change": (
                "Middle10 reservoir2+motionpair1; max_pair_age=12; stale "
                "eligible pair bypasses motion quantile"
            ),
        },
        "claim_boundary": (
            "Trace evidence localizes mechanism behavior only and cannot "
            "establish video-quality improvement."
        ),
    }
    output_md = args.output_md or args.output_json.with_suffix(".md")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(markdown(report), encoding="utf-8")
    print(f"[v159-trace] wrote {args.output_json}")
    print(f"[v159-trace] wrote {output_md}")


if __name__ == "__main__":
    main()
