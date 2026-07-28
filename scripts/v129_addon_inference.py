#!/usr/bin/env python3
"""PF inference wrapper with sampled Value-calibration diagnostics."""

from __future__ import annotations

import atexit
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any


PF_ROOT = Path.cwd().resolve()
if str(PF_ROOT) not in sys.path:
    sys.path.insert(0, str(PF_ROOT))


def _argument_value(flag: str) -> str | None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return sys.argv[index + 1]


OUTPUT_FOLDER = _argument_value("--output_folder")
TRACE_ENABLED = os.environ.get("V129_ADDON_VALUE_TRACE", "0") == "1"
SAMPLE_STRIDE = max(
    1, int(os.environ.get("V129_ADDON_VALUE_TRACE_SAMPLE_STRIDE", "128"))
)
STATS: dict[str, Any] = {
    "version": 1,
    "trace_enabled": TRACE_ENABLED,
    "sample_stride": SAMPLE_STRIDE,
    "calls": 0,
    "changed_calls": 0,
    "unchanged_calls": 0,
    "sampled_calls": 0,
    "sampled_nonfinite": 0,
    "enabled_sequences": 0,
    "total_sequences": 0,
    "moment_modes": Counter(),
    "strengths": Counter(),
    "relative_delta_samples": [],
    "max_abs_delta_samples": [],
}


def _serializable_stats() -> dict[str, Any]:
    payload = dict(STATS)
    payload["moment_modes"] = dict(STATS["moment_modes"])
    payload["strengths"] = dict(STATS["strengths"])
    return payload


def _write_trace() -> None:
    if not TRACE_ENABLED or not OUTPUT_FOLDER:
        return
    output = Path(OUTPUT_FOLDER)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "value_alignment_trace.json"
    temporary = target.with_suffix(".json.tmp")
    payload = _serializable_stats()
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    print(
        "[V129ValueTrace] "
        f"calls={payload['calls']} changed={payload['changed_calls']} "
        f"sampled={payload['sampled_calls']} "
        f"nonfinite={payload['sampled_nonfinite']} path={target}",
        flush=True,
    )


atexit.register(_write_trace)


if TRACE_ENABLED:
    import torch
    from wan.modules.attention import core as attention_core

    _original = attention_core.renormalize_stale_history_values

    def traced_renormalize(*args: Any, **kwargs: Any):
        values = kwargs.get("values")
        if values is None and args:
            values = args[0]
        result = _original(*args, **kwargs)

        STATS["calls"] += 1
        mode = str(kwargs.get("moment_mode", "full"))
        strength = float(kwargs.get("strength", 0.0))
        STATS["moment_modes"][mode] += 1
        STATS["strengths"][f"{strength:.6g}"] += 1

        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is None and len(args) > 1:
            cu_seqlens = args[1]
        num_sequences = (
            max(0, int(cu_seqlens.numel()) - 1)
            if cu_seqlens is not None
            else 0
        )
        sequence_enabled = kwargs.get("sequence_enabled")
        enabled_sequences = (
            num_sequences
            if sequence_enabled is None
            else sum(bool(value) for value in sequence_enabled)
        )
        STATS["total_sequences"] += num_sequences
        STATS["enabled_sequences"] += enabled_sequences

        changed = result is not values
        if changed:
            STATS["changed_calls"] += 1
        else:
            STATS["unchanged_calls"] += 1

        if (
            changed
            and values is not None
            and STATS["changed_calls"] % SAMPLE_STRIDE == 1
        ):
            with torch.no_grad():
                delta = result.float() - values.float()
                finite = bool(torch.isfinite(delta).all().item())
                STATS["sampled_calls"] += 1
                if not finite:
                    STATS["sampled_nonfinite"] += 1
                else:
                    denominator = values.float().norm().clamp_min(1e-12)
                    relative = float((delta.norm() / denominator).item())
                    maximum = float(delta.abs().max().item())
                    STATS["relative_delta_samples"].append(relative)
                    STATS["max_abs_delta_samples"].append(maximum)
        return result

    attention_core.renormalize_stale_history_values = traced_renormalize


inference_path = PF_ROOT / "inference.py"
if not inference_path.is_file():
    raise SystemExit(f"expected PF inference.py in cwd, found {PF_ROOT}")
runpy.run_path(str(inference_path), run_name="__main__")
