from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_head_profile_threshold_free_audit import (  # noqa: E402
    bh_adjust,
    spearman,
)


def test_spearman_is_rank_based() -> None:
    left = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert spearman(left, left**3) == 1.0
    assert spearman(left, left[::-1]) == -1.0


def test_bh_adjust_is_monotone_in_sorted_p_values() -> None:
    values = [0.04, 0.001, 0.02, 0.5]
    adjusted = bh_adjust(values)
    ordered = sorted(zip(values, adjusted))
    assert all(
        left[1] <= right[1] for left, right in zip(ordered, ordered[1:])
    )
    assert all(value >= raw for raw, value in zip(values, adjusted))
