#!/usr/bin/env python3
"""Audit continuous-flow results without comparing incompatible prompt grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np

EXPERIMENT = "v204_continuous_flow_protocol_audit"
CLIPS_PER_PROMPT = 15
CLIP_PATTERN = re.compile(r"(?:^|[/\\])(\d{6})-0_(\d{3})\.mp4$")
WINDOWS = {
    "full": tuple(range(CLIPS_PER_PROMPT)),
    "early": tuple(range(7)),
    "late": tuple(range(8, CLIPS_PER_PROMPT)),
}
PROTOCOL_FIELDS = (
    "prompt_text_sha256",
    "source_index_sha256",
    "prompt_count",
    "seed",
    "reseed_per_prompt",
    "num_output_frames",
    "decoded_video_contract",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: expected a numeric value") from error
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite value")
    return result


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sign_p(values: np.ndarray) -> float:
    nonzero = values[values != 0.0]
    if not nonzero.size:
        return 1.0
    wins = int(np.sum(nonzero > 0.0))
    n = int(nonzero.size)
    tail = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
    return min(1.0, 2.0 * min(tail, 1.0 - tail + math.comb(n, wins) / 2**n))


def _bootstrap_ci(values: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(10000, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _bh(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_value"])
    running = 1.0
    total = len(order)
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = total - reverse_rank + 1
        running = min(running, rows[index]["p_value"] * total / rank)
        rows[index]["q_value"] = min(1.0, running)


def _method_contract(manifest: dict) -> dict:
    prompt_count = int(manifest.get("prompt_count", -1))
    prompt_items = manifest.get("prompt_items") or ()
    if prompt_count <= 0 or len(prompt_items) != prompt_count:
        raise ValueError("manifest has an incomplete prompt grid")
    texts = [str(row.get("text", "")) for row in prompt_items]
    if any(not text for text in texts):
        raise ValueError("manifest contains an empty prompt")
    source_indices = [row.get("source_index") for row in prompt_items]
    return {
        "prompt_text_sha256": hashlib.sha256(
            "\n".join(texts).encode("utf-8")
        ).hexdigest(),
        "source_index_sha256": _json_sha256(source_indices),
        "source_index_min": min(source_indices)
        if all(isinstance(value, int) for value in source_indices)
        else None,
        "source_index_max": max(source_indices)
        if all(isinstance(value, int) for value in source_indices)
        else None,
        "prompt_count": prompt_count,
        "seed": manifest.get("seed"),
        "reseed_per_prompt": manifest.get("reseed_per_prompt"),
        "num_output_frames": manifest.get("num_output_frames"),
        "decoded_video_contract": manifest.get("decoded_video_contract"),
        "prompt_suite_label": manifest.get("prompt_suite"),
    }


def load_method(name: str, manifest_path: Path, result_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = [str(row.get("key", "")) for row in manifest.get("methods") or ()]
    if methods.count(name) != 1:
        raise ValueError(f"{name}: method is not unique in {manifest_path}")
    contract = _method_contract(manifest)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    value = payload.get("dynamic_degree")
    if not isinstance(value, list) or len(value) not in {2, 3}:
        raise ValueError(f"{name}: malformed Dynamic Degree result")
    details = next(
        (
            rows
            for rows in reversed(value[1:])
            if isinstance(rows, list)
            and rows
            and isinstance(rows[0], dict)
            and "video_mean_score" in rows[0]
        ),
        None,
    )
    if details is None:
        raise ValueError(f"{name}: continuous per-clip scores are absent")
    clips: dict[tuple[int, int], float] = {}
    for row in details:
        match = CLIP_PATTERN.search(str(row.get("video_path", "")))
        if match is None:
            raise ValueError(
                f"{name}: cannot parse clip path {row.get('video_path')!r}"
            )
        key = (int(match.group(1)), int(match.group(2)))
        if key in clips:
            raise ValueError(f"{name}: duplicate clip {key}")
        score = _finite(row.get("video_mean_score"), label=f"{name}:{key}")
        if score < 0.0:
            raise ValueError(f"{name}: negative continuous flow score")
        clips[key] = score
    expected = {
        (prompt, clip)
        for prompt in range(contract["prompt_count"])
        for clip in range(CLIPS_PER_PROMPT)
    }
    if set(clips) != expected:
        missing = sorted(expected - set(clips))[:12]
        extra = sorted(set(clips) - expected)[:12]
        raise ValueError(
            f"{name}: incomplete clip grid missing={missing} extra={extra}"
        )
    per_prompt = {
        window: np.asarray(
            [
                np.mean([clips[(prompt, clip)] for clip in clip_ids])
                for prompt in range(contract["prompt_count"])
            ],
            dtype=np.float64,
        )
        for window, clip_ids in WINDOWS.items()
    }
    return {
        "name": name,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "result": str(result_path.resolve()),
        "result_sha256": sha256(result_path),
        "contract": contract,
        "clip_count": len(clips),
        "aggregate_continuous_flow": float(per_prompt["full"].mean()),
        "per_prompt": per_prompt,
    }


def compare(candidate: dict, control: dict, *, seed: int) -> dict:
    mismatches = [
        field
        for field in PROTOCOL_FIELDS
        if candidate["contract"].get(field) != control["contract"].get(field)
    ]
    base = {
        "candidate": candidate["name"],
        "control": control["name"],
        "protocol_match": not mismatches,
        "protocol_mismatches": mismatches,
    }
    if mismatches:
        return {
            **base,
            "status": "invalid_cross_protocol_comparison",
            "statistics": [],
        }
    statistics = []
    for window_index, window in enumerate(WINDOWS):
        candidate_values = candidate["per_prompt"][window]
        control_values = control["per_prompt"][window]
        delta = candidate_values - control_values
        statistics.append(
            {
                "window": window,
                "candidate_mean": float(candidate_values.mean()),
                "control_mean": float(control_values.mean()),
                "mean_delta": float(delta.mean()),
                "relative_delta": float(
                    delta.mean() / max(control_values.mean(), 1e-12)
                ),
                "median_delta": float(np.median(delta)),
                "win_fraction": float(np.mean(delta > 0.0)),
                "bootstrap_ci95": _bootstrap_ci(delta, seed=seed + 101 * window_index),
                "p_value": _sign_p(delta),
            }
        )
    _bh(statistics)
    return {**base, "status": "valid_paired_diagnostic", "statistics": statistics}


def analyze(
    methods: list[tuple[str, Path, Path]], comparisons: list[tuple[str, str]]
) -> dict:
    loaded = {}
    for name, manifest, result in methods:
        if name in loaded:
            raise ValueError(f"duplicate method input: {name}")
        loaded[name] = load_method(name, manifest, result)
    rows = []
    for index, (candidate, control) in enumerate(comparisons):
        if candidate not in loaded or control not in loaded or candidate == control:
            raise ValueError(f"invalid comparison: {candidate} vs {control}")
        rows.append(compare(loaded[candidate], loaded[control], seed=204000 + index))
    valid = [row for row in rows if row["protocol_match"]]
    valid_sf = [
        row
        for row in valid
        if row["candidate"] == "sf_native" or row["control"] == "sf_native"
    ]
    serializable_methods = {}
    for name, row in loaded.items():
        serializable_methods[name] = {
            key: value for key, value in row.items() if key != "per_prompt"
        }
    return {
        "version": 1,
        "experiment": EXPERIMENT,
        "metric": "uncompensated_raft_flow_magnitude",
        "diagnostic_only": True,
        "implementation_provenance_available": False,
        "paper_metric_ready": False,
        "methods": serializable_methods,
        "comparisons": rows,
        "valid_comparison_count": len(valid),
        "valid_sf_comparison_count": len(valid_sf),
        "recommendation": (
            "raw_flow_has_same_protocol_sf_context"
            if valid_sf
            else "no_same_protocol_sf_comparison"
        ),
        "manual_review_required": False,
        "claim_boundary": (
            "Raw RAFT magnitude is camera-sensitive and the producing implementation "
            "is not committed in this repository. Cross-protocol aggregate differences "
            "are invalid. Use the repository-owned camera-compensated v193 diagnostic "
            "and paired quality evidence before making a motion claim."
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# v204 Continuous-Flow Protocol Audit",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Valid paired comparisons: {report['valid_comparison_count']}",
        f"- Valid comparisons involving SF: {report['valid_sf_comparison_count']}",
        "- Paper metric ready: `False`",
        "",
        "| Candidate | Control | Protocol | Full delta | Late delta |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["comparisons"]:
        if not row["protocol_match"]:
            mismatch = ", ".join(row["protocol_mismatches"])
            lines.append(
                f"| {row['candidate']} | {row['control']} | invalid: {mismatch} | - | - |"
            )
            continue
        stats = {item["window"]: item for item in row["statistics"]}
        lines.append(
            f"| {row['candidate']} | {row['control']} | valid | "
            f"{stats['full']['mean_delta']:+.4f} | {stats['late']['mean_delta']:+.4f} |"
        )
    lines.extend(["", report["claim_boundary"], ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        nargs=3,
        action="append",
        metavar=("NAME", "MANIFEST", "RESULT"),
        required=True,
    )
    parser.add_argument(
        "--comparison",
        nargs=2,
        action="append",
        metavar=("CANDIDATE", "CONTROL"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        [
            (name, Path(manifest), Path(result))
            for name, manifest, result in args.method
        ],
        [(candidate, control) for candidate, control in args.comparison],
    )
    _atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(args.output.with_suffix(".md"), render(report))
    print(
        "[v204-flow-audit] "
        f"recommendation={report['recommendation']} "
        f"valid={report['valid_comparison_count']} "
        f"valid_sf={report['valid_sf_comparison_count']}"
    )


if __name__ == "__main__":
    main()
