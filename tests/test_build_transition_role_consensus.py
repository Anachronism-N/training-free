from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_transition_role_consensus.py"
)
SPEC = importlib.util.spec_from_file_location("build_transition_role_consensus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_consensus_keeps_agreement_and_neutralizes_disagreement():
    consensus, report = MODULE.build_consensus(
        [[1, -1, 1], [-1, 1, -1]],
        [[1, 1, 1], [-1, -1, -1]],
    )

    assert consensus == [[1, 0, 1], [-1, 0, -1]]
    assert report["agreement"] == pytest.approx(4 / 6)
    assert report["label_counts"] == {
        "persistent": 2,
        "reactive": 2,
        "neutral": 2,
    }


def test_consensus_rejects_different_shapes():
    with pytest.raises(ValueError, match="different shapes"):
        MODULE.build_consensus([[1, -1]], [[1], [-1]])
