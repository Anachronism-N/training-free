#!/usr/bin/env python3
"""Audit whether the v160 stale-only motion refresh actually executed."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from analyze_v160_automated_screen import METHODS, PRIMARY, PROMPT_COUNT


ROOT = Path(__file__).resolve().parents[1]
TARGET_LAYER = 15
TARGET_HEAD = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs" / "v160_fresh_motion_moviebench16" / "full8",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def selected_events(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if (
                event.get("event") == "middle_selection"
                and int(event.get("layer", -1)) == TARGET_LAYER
                and int(event.get("head", -1)) == TARGET_HEAD
                and int(event.get("label", -1)) == 10
            ):
                events.append(event)
    if len(events) != 40:
        raise ValueError(f"{path}: expected 40 L15/H0 events, got {len(events)}")
    return events


def analyze_prompt(path: Path, prompt_index: int) -> dict:
    reasons: Counter[str] = Counter()
    pair_ages: list[float] = []
    bypasses = []
    early_freshness_refreshes = []
    final_state = None
    for event in selected_events(path):
        if (
            not event.get("cache_contract_pass")
            or not event.get("explicit_composition_owns_dynamic")
            or int(event.get("sink_frames", -1)) != 1
            or int(event.get("recent_frames", -1)) != 4
            or int(event.get("union_frame_count", 0)) > 4
        ):
            raise ValueError(
                f"prompt {prompt_index}: selected cache contract failed"
            )
        motion_items = [
            item
            for item in event.get("strategies", [])
            if item.get("name") == "CoherentMotionStrategy"
        ]
        if len(motion_items) != 1:
            raise ValueError(f"prompt {prompt_index}: missing coherent-motion state")
        state = motion_items[0].get("state")
        if (
            not isinstance(state, dict)
            or int(state.get("max_pair_age", -1)) != 12
            or state.get("stale_refresh_bypass_quantile") is not True
            or int(state.get("pair_capacity", -1)) != 1
        ):
            raise ValueError(
                f"prompt {prompt_index}: fresh-motion state is not configured"
            )
        final_state = state
        decision = state.get("last_decision", {})
        reason = decision.get("reason")
        if reason:
            reasons[str(reason)] += 1
        if decision.get("stale_quantile_bypass"):
            required = {
                "accepted": True,
                "victim_stale": True,
                "motion_quantile_pass": False,
                "motion_ok": True,
                "reason": "stale_quantile_refresh",
            }
            mismatched = {
                key: {"expected": value, "actual": decision.get(key)}
                for key, value in required.items()
                if decision.get(key) != value
            }
            if mismatched:
                raise ValueError(
                    f"prompt {prompt_index}: invalid stale bypass {mismatched}"
                )
            bypasses.append(
                {
                    "candidate_pair": decision.get("candidate_pair"),
                    "motion": decision.get("motion"),
                    "motion_threshold": decision.get("motion_threshold"),
                    "victim_end_t": decision.get("victim_end_t"),
                }
            )
        candidate_pair = decision.get("candidate_pair")
        victim_end_t = decision.get("victim_end_t")
        victim_age = decision.get("victim_age")
        if (
            victim_age is None
            and isinstance(candidate_pair, list)
            and len(candidate_pair) == 2
            and victim_end_t is not None
        ):
            victim_age = int(candidate_pair[1]) - int(victim_end_t)
        if (
            decision.get("accepted") is True
            and decision.get("victim_stale") is True
            and victim_age is not None
            and 12 <= int(victim_age) < 24
            and decision.get("improves_victim") is False
        ):
            early_freshness_refreshes.append(
                {
                    "candidate_pair": candidate_pair,
                    "victim_end_t": victim_end_t,
                    "victim_age": int(victim_age),
                    "reason": decision.get("reason"),
                }
            )
        sync_t = int(event.get("sync_t", 0))
        for pair in state.get("pair_frame_ids", []):
            pair_ages.append(float(sync_t - int(pair[1])))
    assert final_state is not None
    return {
        "prompt_index": prompt_index,
        "accepted_count": int(final_state.get("accepted_count", -1)),
        "rejected_count": int(final_state.get("rejected_count", -1)),
        "evicted_count": int(final_state.get("evicted_count", -1)),
        "stale_quantile_bypass_count": len(bypasses),
        "stale_quantile_bypasses": bypasses,
        "early_freshness_refresh_count": len(early_freshness_refreshes),
        "early_freshness_refreshes": early_freshness_refreshes,
        "reason_counts": dict(sorted(reasons.items())),
        "pair_age": summary(pair_ages),
    }


def markdown(report: dict) -> str:
    aggregate = report["aggregate"]
    return "\n".join(
        [
            "# v160 Fresh-Motion Trace Audit",
            "",
            f"Mechanism gate: **{report['mechanism_gate']}**",
            "",
            "- Actual below-quantile stale bypasses: "
            f"{aggregate['stale_quantile_bypass_count']}",
            "- Accepted 12-23-frame freshness refreshes: "
            f"{aggregate['early_freshness_refresh_count']}",
            "- Accepted updates per prompt: "
            f"{aggregate['accepted_per_prompt']['mean']:.3f}",
            "- Pair-age p95 per prompt: "
            f"{aggregate['pair_age_p95_per_prompt']['mean']:.3f}",
            f"- Maximum observed pair age: {aggregate['pair_age_max']:.0f}",
            "",
            "This gate proves only that the intended mechanism executed under "
            "the audited cache contract. Video quality is evaluated separately.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    published = json.loads(
        (run_root / "published_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not published.get("ok")
        or published.get("experiment") != "v160_fresh_motion_moviebench16"
        or tuple(row["key"] for row in published.get("methods", [])) != METHODS
    ):
        raise ValueError("v160 published manifest violates the trace contract")
    prompts = []
    for prompt_index in range(PROMPT_COUNT):
        trace = run_root / "traces" / f"{PRIMARY}__p{prompt_index:03d}.policy.jsonl"
        if not trace.is_file():
            raise FileNotFoundError(trace)
        prompts.append(analyze_prompt(trace, prompt_index))
    reason_counts: Counter[str] = Counter()
    for row in prompts:
        reason_counts.update(row["reason_counts"])
    aggregate = {
        "prompt_count": len(prompts),
        "accepted_per_prompt": summary(
            [float(row["accepted_count"]) for row in prompts]
        ),
        "rejected_per_prompt": summary(
            [float(row["rejected_count"]) for row in prompts]
        ),
        "stale_quantile_bypass_count": sum(
            row["stale_quantile_bypass_count"] for row in prompts
        ),
        "early_freshness_refresh_count": sum(
            row["early_freshness_refresh_count"] for row in prompts
        ),
        "prompts_with_stale_quantile_bypass": sum(
            row["stale_quantile_bypass_count"] > 0 for row in prompts
        ),
        "pair_age_p95_per_prompt": summary(
            [float(row["pair_age"]["p95"]) for row in prompts]
        ),
        "pair_age_max": max(float(row["pair_age"]["max"]) for row in prompts),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    mechanism_gate = (
        aggregate["stale_quantile_bypass_count"] > 0
        or aggregate["early_freshness_refresh_count"] > 0
    )
    report = {
        "version": 1,
        "experiment": "v160_fresh_motion_trace_audit",
        "method": PRIMARY,
        "representative_trace": {"layer": TARGET_LAYER, "head": TARGET_HEAD},
        "mechanism_gate": mechanism_gate,
        "aggregate": aggregate,
        "prompts": prompts,
        "claim_boundary": (
            "The mechanism gate validates execution and cache ownership only; "
            "it is not evidence of video-quality improvement."
        ),
    }
    output = args.output or run_root / "automated_screen" / "fresh_motion_trace.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    if not mechanism_gate:
        raise SystemExit(
            "v160 freshness path never changed an accepted update; quality "
            "results cannot test the proposed mechanism"
        )
    print(
        f"[v160-trace] PASS bypasses={aggregate['stale_quantile_bypass_count']} "
        f"early_refresh={aggregate['early_freshness_refresh_count']} "
        f"prompts={aggregate['prompts_with_stale_quantile_bypass']} output={output}"
    )


if __name__ == "__main__":
    main()
