#!/usr/bin/env python3
"""Run isolated non-cache add-ons on the frozen v129 base method.

This entrypoint intentionally adds no new cache policy. It reuses the
Prototype4 + Retrieval1(age<=24) base and changes only read-time historical
Value calibration. Keeping this in a separate entrypoint prevents an
in-progress v129 contract from changing.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import run_v100_fast_selection_1video as fast_runner
import run_v120_moviebench32_main as runner
from run_v100_fast_selection_1video import Cell


PROMPT_COUNT = int(os.environ.get("V129_ADDON_PROMPT_COUNT", "1"))
if PROMPT_COUNT not in {1, 16, 128}:
    raise SystemExit("V129_ADDON_PROMPT_COUNT must be 1, 16, or 128")

DURATION_SECONDS = int(os.environ.get("V129_ADDON_DURATION_SECONDS", "30"))
if DURATION_SECONDS != 30:
    raise SystemExit("v129 non-cache add-ons currently require 30 seconds")

NUM_OUTPUT_FRAMES = 120
TRACE_WRAPPER = Path(__file__).with_name("v129_addon_inference.py").resolve()


VALUE_VARIANTS: dict[str, dict[str, str]] = {
    "value_var_s025": {
        "strength": "0.25",
        "labels": "10",
        "layer_start": "0",
        "layer_end": "-1",
        "transition_lambda": "0.0",
    },
    "value_var_s050": {
        "strength": "0.5",
        "labels": "10",
        "layer_start": "0",
        "layer_end": "-1",
        "transition_lambda": "0.0",
    },
    "value_var_s050_mid": {
        "strength": "0.5",
        "labels": "10",
        "layer_start": "10",
        "layer_end": "20",
        "transition_lambda": "0.0",
    },
    "value_var_s050_mid_t3": {
        "strength": "0.5",
        "labels": "10",
        "layer_start": "10",
        "layer_end": "20",
        "transition_lambda": "3.0",
    },
}


def _set_cli_option(command: list[str], flag: str, value: str) -> None:
    count = command.count(flag)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {flag} in inference command, found {count}"
        )
    index = command.index(flag)
    if index + 1 >= len(command):
        raise RuntimeError(f"missing value after {flag}")
    command[index + 1] = str(value)


def variant_from_task_name(task_name: str) -> str | None:
    for key in VALUE_VARIANTS:
        if task_name.startswith(f"ours_{key}__p"):
            return key
    return None


_base_inference_command = fast_runner.inference_command


def addon_inference_command(
    args: Any,
    *,
    cell: Cell,
    output: Path,
    transition_trace: Path,
    scene_trace: Path,
) -> tuple[list[str], Path, Path, int]:
    command, cwd, head_map, prompt_index = _base_inference_command(
        args,
        cell=cell,
        output=output,
        transition_trace=transition_trace,
        scene_trace=scene_trace,
    )
    if not TRACE_WRAPPER.is_file():
        raise RuntimeError(f"missing v129 add-on inference wrapper: {TRACE_WRAPPER}")
    if len(command) < 2 or Path(command[1]).name != "inference.py":
        raise RuntimeError(f"unexpected PF inference command prefix: {command[:2]}")
    command[1] = str(TRACE_WRAPPER)

    variant = variant_from_task_name(cell.name)
    if variant is None:
        if cell.variance_refresh:
            raise RuntimeError(
                f"unregistered Value calibration task: {cell.name}"
            )
        return command, cwd, head_map, prompt_index

    if not cell.variance_refresh:
        raise RuntimeError(f"Value variant did not enable refresh: {cell.name}")
    values = VALUE_VARIANTS[variant]
    _set_cli_option(
        command,
        "--pyramidkv_history_value_renorm_strength",
        values["strength"],
    )
    _set_cli_option(
        command,
        "--pyramidkv_history_value_recent_frames",
        "4",
    )
    _set_cli_option(
        command,
        "--pyramidkv_history_value_gate_lambda",
        "3.0",
    )
    _set_cli_option(
        command,
        "--pyramidkv_history_value_labels",
        values["labels"],
    )
    _set_cli_option(
        command,
        "--pyramidkv_history_value_layer_start",
        values["layer_start"],
    )
    _set_cli_option(
        command,
        "--pyramidkv_history_value_layer_end",
        values["layer_end"],
    )
    _set_cli_option(
        command,
        "--pyramidkv_history_value_moment_mode",
        "variance_only",
    )
    _set_cli_option(
        command,
        "--pyramidkv_history_value_target_frames",
        "4",
    )
    _set_cli_option(
        command,
        "--pyramidkv_history_value_max_std_ratio",
        "1.5",
    )
    command.extend(
        [
            "--pyramidkv_history_value_transition_lambda",
            values["transition_lambda"],
        ]
    )
    return command, cwd, head_map, prompt_index


_base_experiment_contract = runner.experiment_contract


def addon_experiment_contract(*args: Any, **kwargs: Any) -> dict[str, object]:
    payload = _base_experiment_contract(*args, **kwargs)
    payload["addon_contract"] = {
        "kind": "read_time_history_value_calibration",
        "cache_policy_changed": False,
        "base": {
            "supportive": "sink1 + TemporalPrototype4 + recent4",
            "suppressive": "sink1 + Retrieval1(age<=24) + recent7",
            "max_full_frame_equivalents": 9,
        },
        "value_variants": VALUE_VARIANTS,
        "fixed_parameters": {
            "moment_mode": "variance_only",
            "recent_frames": 4,
            "target_frames": 4,
            "compatibility_gate_lambda": 3.0,
            "max_std_ratio": 1.5,
        },
        "trace_wrapper": str(TRACE_WRAPPER),
        "trace_wrapper_sha256": hashlib.sha256(
            TRACE_WRAPPER.read_bytes()
        ).hexdigest(),
    }
    return payload


CONTROL_CELL = Cell(
    "prototype4_retrieval1_age24_value_control",
    "v129_noncache_control",
    "single",
    support_policy="prototype",
    suppress_policy="retrieval1_age24",
)

ADDON_CELLS = tuple(
    Cell(
        f"prototype4_retrieval1_age24_{key}",
        "v129_noncache_candidate",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        variance_refresh=True,
    )
    for key in VALUE_VARIANTS
)

ADDON_DEFAULT_CANDIDATES = (
    "value_control",
    "value_var_s025",
    "value_var_s050",
    "value_var_s050_mid",
    "value_var_s050_mid_t3",
)
ADDON_CELLS_BY_NAME = {
    cell.name: cell for cell in (CONTROL_CELL, *ADDON_CELLS)
}
ADDON_CANDIDATE_SPECS = {
    "value_control": (
        CONTROL_CELL.name,
        "exact_v125_base_control",
    ),
    **{
        key: (
            f"prototype4_retrieval1_age24_{key}",
            "noncache_history_value_calibration",
        )
        for key in VALUE_VARIANTS
    },
}


def configure_runner() -> None:
    runner.EXPERIMENT = (
        f"v129_noncache_addons_{PROMPT_COUNT}prompt_{DURATION_SECONDS}s"
    )
    runner.PROMPT_COUNT = PROMPT_COUNT
    runner.TASK_STAGE = f"v129_noncache_{PROMPT_COUNT}prompt"
    runner.PUBLISHED_TAG = f"v129_addon_{DURATION_SECONDS}s"
    runner.RUN_LABEL = f"v129-addon-{PROMPT_COUNT}p"
    runner.NUM_OUTPUT_FRAMES = NUM_OUTPUT_FRAMES
    runner.DEFAULT_PROMPT_PATH = os.environ.get("V129_ADDON_PROMPTS")
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = True
    runner.MAX_CANDIDATES = len(ADDON_DEFAULT_CANDIDATES)
    runner.DEFAULT_CANDIDATES = ADDON_DEFAULT_CANDIDATES
    runner._CELLS_BY_NAME = dict(ADDON_CELLS_BY_NAME)
    runner._CANDIDATE_SPECS = dict(ADDON_CANDIDATE_SPECS)

    # run_cell is imported by run_v120, but resolves inference_command through
    # the defining module's globals. Patch only this add-on process.
    fast_runner.inference_command = addon_inference_command
    runner.experiment_contract = addon_experiment_contract
    os.environ["V129_ADDON_VALUE_TRACE"] = "1"


if __name__ == "__main__":
    configure_runner()
    if runner.DEFAULT_PROMPT_PATH is None:
        raise SystemExit("set V129_ADDON_PROMPTS or pass --prompts")
    runner.main()
