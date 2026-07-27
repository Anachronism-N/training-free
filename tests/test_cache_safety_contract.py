from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SAFETY = (
    ROOT
    / "third_party"
    / "Pyramid-Forcing"
    / "pyramidkv"
    / "safety.py"
)
ADAPTIVE_CACHE = SAFETY.with_name("adaptive_cache.py")


def _load_safety():
    spec = importlib.util.spec_from_file_location(
        "pyramidkv_safety_no_torch",
        SAFETY,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exclusive_opening_partition_preserves_recent_frames():
    validate = _load_safety().validate_exclusive_opening_partition

    assert validate(
        sink_frames=1,
        recent_frames=7,
        opening_frames=3,
    ) == 2
    assert validate(
        sink_frames=3,
        recent_frames=4,
        opening_frames=4,
    ) == 1


def test_exclusive_opening_partition_rejects_all_sink_layout():
    validate = _load_safety().validate_exclusive_opening_partition

    with pytest.raises(ValueError, match="leaving zero recent frames"):
        validate(
            sink_frames=3,
            recent_frames=4,
            opening_frames=3,
        )


def test_adaptive_cache_wires_guard_and_debug_fields():
    text = ADAPTIVE_CACHE.read_text(encoding="utf-8")

    assert "validate_exclusive_opening_partition(" in text
    assert "self.composition_owns_dynamic" in text
    assert '"opening_recent_starved"' in text
    assert '"sink_rope_time"' in text
    assert '"sink_collapses_multiple_physical_times"' in text
