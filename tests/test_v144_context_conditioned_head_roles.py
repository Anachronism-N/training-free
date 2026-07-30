import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v144_context_conditioned_head_roles.py"
)
SPEC = importlib.util.spec_from_file_location("v144_context_roles_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_context_csv(path: Path) -> None:
    rows = []
    head_pattern = np.tile(np.linspace(-1.0, 1.0, 12), 30)
    layer_pattern = np.repeat(np.linspace(-10.0, 10.0, 30), 12)
    for context_index, frame in enumerate((54, 60)):
        for split in MODULE.SPLITS:
            split_noise = 0.0 if split == "all" else (
                0.01 if split == "discovery" else -0.01
            )
            for flat_head in range(MODULE.TOTAL_HEADS):
                stable = (
                    layer_pattern[flat_head]
                    + head_pattern[flat_head]
                    + split_noise
                )
                state_flipped = (
                    layer_pattern[flat_head]
                    + ((-1.0) ** context_index) * head_pattern[flat_head]
                    + split_noise
                )
                row = {
                    "prompt_split": split,
                    "switch_type": "identity_scene",
                    "mode": "clean",
                    "current_frame": frame,
                    "nominal_timestep": 0,
                    "episode": "A" if frame < 57 else "B",
                    "stale_a_visible": int(frame >= 57),
                    "boundary_offset": frame - 57,
                    "layer": flat_head // 12,
                    "head": flat_head % 12,
                    "prompt_history_excess": stable,
                    "policy_prompt_score": state_flipped,
                    "stale_a_mass": stable,
                    "persistent_content": stable,
                    "persistent_positioned": stable,
                    "persistent_output": stable,
                }
                rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_context_audit_separates_static_and_state_conditioned_axes(tmp_path):
    source = tmp_path / "contexts.csv"
    output = tmp_path / "analysis"
    _write_context_csv(source)
    report = MODULE.analyze(
        source,
        output,
        minimum_split_rho=0.5,
        minimum_context_rho=0.3,
    )
    stable = report["axis_summary"]["prompt_history_excess"]
    dynamic = report["axis_summary"]["policy_prompt_score"]
    assert stable["static_head_axis_gate"] == 1
    assert dynamic["static_head_axis_gate"] == 0
    assert stable["median_layer_eta_squared"] > 0.9
    assert stable["median_layer_residual_cross_context_spearman"] > 0.99
    assert dynamic["median_layer_residual_cross_context_spearman"] < -0.99
    saved = json.loads(
        (output / "context_role_report.json").read_text(encoding="utf-8")
    )
    assert saved["static_head_axis_count"] >= 1
