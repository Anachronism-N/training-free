from __future__ import annotations

import json

import pytest

from pyramidkv.prompt_warmup import (
    PromptWarmupShield,
    PromptWarmupShieldConfig,
)


def _shield(**overrides) -> PromptWarmupShield:
    values = {
        "enabled": True,
        "blocks": 4,
        "release_span": 6,
        "mode": "history",
        "shield_labels": (-1,),
        "layer_start": 0,
        "layer_end": -1,
    }
    values.update(overrides)
    return PromptWarmupShield(
        PromptWarmupShieldConfig(**values),
        layer_idx=0,
        head_labels=(-1, 1, -1, -1),
    )


def test_enabled_shield_requires_positive_warmup():
    with pytest.raises(ValueError, match="blocks > 0"):
        _shield(blocks=0)


def test_prompt_roles_and_staggered_release_are_deterministic():
    shield = _shield()

    assert shield.active_mask(0) == (True, False, True, True)
    assert shield.active_mask(4) == (False, False, True, True)
    assert shield.active_mask(6) == (False, False, True, False)
    assert shield.active_mask(10) == (False, False, False, False)


def test_middle_mode_preserves_sink_visibility():
    shield = _shield(mode="middle", release_span=0)

    assert shield.shields_middle(0, 0)
    assert not shield.shields_sink(0, 0)
    assert not shield.shields_middle(0, 4)


def test_layer_range_disables_out_of_range_controller():
    config = PromptWarmupShieldConfig(
        enabled=True,
        blocks=4,
        mode="history",
        layer_start=1,
        layer_end=3,
    )
    shield = PromptWarmupShield(
        config,
        layer_idx=0,
        head_labels=(-1, -1),
    )

    assert not shield.enabled
    assert shield.active_mask(0) == (False, False)


def test_trace_records_role_counts_and_deduplicates(tmp_path):
    trace = tmp_path / "warmup.jsonl"
    shield = _shield(trace_path=str(trace), release_span=0)

    first = shield.active_mask(0)
    shield.record(block_id=0, branch="cond", active_mask=first)
    shield.record(block_id=0, branch="cond", active_mask=first)
    shield.record(block_id=4, branch="cond")

    rows = [
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["event"] == "prompt_warmup_shield"
    assert rows[0]["eligible_heads"] == 3
    assert rows[0]["active_heads"] == 3
    assert rows[1]["active_heads"] == 0
