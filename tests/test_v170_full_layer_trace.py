from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v170_full_layer_trace as audit
import v170_matched_attribution_contract as contract


def trace_row(layer: int, *, head: int = 0, sync_t: int = 9) -> dict:
    return {
        "event": "middle_selection",
        "branch": "cond",
        "label": 10,
        "layer": layer,
        "head": head,
        "sync_t": sync_t,
        "strategies": [
            {
                "name": "CoherentMotionStrategy",
                "frame_ids": [],
                "state": {},
            }
        ],
    }


def write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_active_layer_rows_requires_all_ten_layers(tmp_path: Path) -> None:
    path = tmp_path / "ours_v170_v166_a__p000.policy.jsonl"
    write_trace(path, [trace_row(layer) for layer in contract.ACTIVE_LAYERS])
    rows = audit.load_active_layer_rows(path)
    assert tuple(rows) == contract.ACTIVE_LAYERS
    assert all(len(value) == 1 for value in rows.values())


def test_load_active_layer_rows_rejects_missing_layer(tmp_path: Path) -> None:
    path = tmp_path / "ours_v170_v166_a__p000.policy.jsonl"
    write_trace(path, [trace_row(layer) for layer in contract.ACTIVE_LAYERS[:-1]])
    with pytest.raises(ValueError, match="active-layer coverage mismatch"):
        audit.load_active_layer_rows(path)


def test_load_active_layer_rows_rejects_unrequested_head(tmp_path: Path) -> None:
    path = tmp_path / "ours_v170_v166_a__p000.policy.jsonl"
    rows = [trace_row(layer) for layer in contract.ACTIVE_LAYERS]
    rows.append(trace_row(contract.ACTIVE_LAYERS[0], head=1, sync_t=12))
    write_trace(path, rows)
    with pytest.raises(ValueError, match="unexpected traced head"):
        audit.load_active_layer_rows(path)
