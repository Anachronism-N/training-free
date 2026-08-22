#!/usr/bin/env python3
"""Post-hoc audit for the legacy v185 PF-native 128-prompt baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from audit_indexed_videos import audit_interval


PROMPT_COUNT = 128
SHARDS = 16
COMPLETION = re.compile(r"\[(\d+)/128\]")
FAILURES = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_logs(log_root: Path) -> dict:
    paths = sorted(log_root.glob("shard*.log"))
    expected_names = [f"shard{rank:02d}.log" for rank in range(SHARDS)]
    if [path.name for path in paths] != expected_names:
        raise ValueError("legacy v185 requires exactly shard00..shard15 logs")
    rows = []
    errors = []
    observed_completions = []
    for rank, path in enumerate(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        completions = [int(match.group(1)) for match in COMPLETION.finditer(text)]
        expected = list(range(rank + 1, PROMPT_COUNT + 1, SHARDS))
        required = {
            "prompt_count": "Number of prompts: 128" in text,
            "native_head_map": (
                "Loading PyramidKV config from "
                "configs/head_configs/best_labels.csv"
            ) in text,
            "completion_order": completions == expected,
            "no_cache_compatibility_override": (
                "CacheCompatibility" not in text
                and "coverage_policy=" not in text
                and "HistoryPolarityPolicy" not in text
            ),
        }
        failures = [marker for marker in FAILURES if marker in text]
        if not all(required.values()) or failures:
            errors.append(
                {
                    "log": path.name,
                    "required": required,
                    "failures": failures,
                    "observed_completions": completions,
                    "expected_completions": expected,
                }
            )
        observed_completions.extend(completions)
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "completion_count": len(completions),
                "first_completion": completions[0] if completions else None,
                "last_completion": completions[-1] if completions else None,
            }
        )
    if sorted(observed_completions) != list(range(1, PROMPT_COUNT + 1)):
        errors.append("completion markers do not cover prompts 1..128 exactly once")
    return {"ok": not errors, "errors": errors, "logs": rows}


def audit_inputs(prompt_file: Path, config: Path, head_map: Path) -> dict:
    prompts = prompt_file.read_text(encoding="utf-8").splitlines()
    if len(prompts) != PROMPT_COUNT or any(not value.strip() for value in prompts):
        raise ValueError("legacy v185 prompt file must contain 128 non-empty lines")
    config_text = config.read_text(encoding="utf-8")
    required_config = (
        "use_pyramidkv: true",
        "use_adaptive_pyramidkv: true",
        "pyramidkv_config_path: configs/head_configs/best_labels.csv",
        "pyramidkv_policy_csv_path: configs/head_configs/best_labels.csv",
        "pyramidkv_label_phase_bucket_map:",
        "pyramidkv_label_stride_enabled_map:",
        "pyramidkv_label_merge_enabled_map:",
    )
    missing = [marker for marker in required_config if marker not in config_text]
    if missing:
        raise ValueError(f"legacy v185 PF config markers are missing: {missing}")
    with head_map.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 30 or any(len(row) != 12 for row in rows):
        raise ValueError("PF best_labels.csv must be a complete 30x12 matrix")
    labels = Counter(int(value) for row in rows for value in row)
    if set(labels) != {-1, 1, 2}:
        raise ValueError(f"unexpected PF label set: {dict(labels)}")
    return {
        "prompt_file": str(prompt_file.resolve()),
        "prompt_file_sha256": sha256(prompt_file),
        "config": str(config.resolve()),
        "config_sha256": sha256(config),
        "head_map": str(head_map.resolve()),
        "head_map_sha256": sha256(head_map),
        "head_label_counts": {str(key): labels[key] for key in sorted(labels)},
        "generation_time_hash_binding_available": False,
    }


def audit(
    run_root: Path,
    prompt_file: Path,
    config: Path,
    head_map: Path,
    *,
    require_media: bool,
    decode: bool,
) -> dict:
    logs = audit_logs(run_root / "logs")
    inputs = audit_inputs(prompt_file, config, head_map)
    raw = run_root / "raw"
    media = None
    if raw.is_dir() and any(raw.glob("*.mp4")):
        media = audit_interval(
            raw,
            start_idx=0,
            end_idx=PROMPT_COUNT,
            sample_idx=0,
            expected_frames=477,
            expected_fps=16.0,
            expected_width=832,
            expected_height=480,
            fps_tolerance=0.05,
            allow_outside_interval=False,
            decode=decode,
        )
    media_ok = bool(media and media["ok"])
    if require_media and not media_ok:
        raise ValueError("legacy v185 media is required but missing or invalid")
    ok = bool(logs["ok"] and (media_ok or not require_media))
    decision = (
        "reuse_for_development_metrics"
        if media_ok
        else "generation_logs_complete_media_not_uploaded"
    )
    return {
        "version": 1,
        "experiment": "legacy_v185_pf_native_baseline_audit",
        "ok": ok,
        "legacy_number_collision": "v185_recovered_v181_long60",
        "method": "pf_native_best_labels",
        "prompt_count": PROMPT_COUNT,
        "num_output_frames": 120,
        "seed": 0,
        "logs": logs,
        "inputs": inputs,
        "media": media,
        "media_available": media is not None,
        "media_valid": media_ok,
        "provenance_grade": "posthoc_log_bound",
        "paper_ready": False,
        "decision": decision,
        "claim_boundary": (
            "The logs establish complete PF-native execution and identify the "
            "loaded head-map path. Current file hashes were not emitted at "
            "generation time, so this artifact is a development baseline until "
            "a frozen generation contract or equivalent server snapshot is bound."
        ),
    }


def render(report: dict) -> str:
    return "\n".join(
        [
            "# Legacy v185 PF-native Baseline Audit",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Logs valid: `{report['logs']['ok']}`",
            f"- Media available: `{report['media_available']}`",
            f"- Media valid: `{report['media_valid']}`",
            f"- Provenance grade: `{report['provenance_grade']}`",
            f"- Paper ready: `{report['paper_ready']}`",
            "",
            report["claim_boundary"],
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-media", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.run_root,
        args.prompt_file,
        args.config,
        args.head_map,
        require_media=args.require_media,
        decode=not args.skip_decode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(
        "[legacy-v185-pf-audit] "
        f"decision={report['decision']} media_valid={report['media_valid']}"
    )


if __name__ == "__main__":
    main()
