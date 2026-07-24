from pathlib import Path
import importlib.util

import pytest


torch = pytest.importorskip("torch")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "Echo-Forcing"
    / "utils"
    / "uniqueness_snapshot.py"
)
SPEC = importlib.util.spec_from_file_location("echo_uniqueness_snapshot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_relevance_selects_a_complete_source_frame():
    candidate_k = torch.zeros(3, 2, 2, 2)
    candidate_v = torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2)
    query = torch.ones(2, 2)
    candidate_k[1] = 1.0

    selected, diagnostics = MODULE.select_coherent_unique_snapshot(
        candidate_k,
        candidate_v,
        query,
        uniqueness_weight=0.0,
        endpoint_bonus=0.0,
    )

    assert selected == 1
    assert diagnostics["selected"] == 1
    assert torch.equal(candidate_v[selected], candidate_v[1])


def test_uniqueness_can_select_nonredundant_frame():
    candidate_k = torch.ones(3, 2, 1, 2)
    candidate_v = torch.zeros(3, 2, 1, 2)
    candidate_v[0, :, 0, 0] = 1
    candidate_v[1, :, 0, 0] = 1
    candidate_v[2, :, 0, 1] = 1
    query = torch.ones(1, 2)

    selected, diagnostics = MODULE.select_coherent_unique_snapshot(
        candidate_k,
        candidate_v,
        query,
        uniqueness_weight=1.0,
        endpoint_bonus=0.0,
    )

    assert selected == 2
    assert diagnostics["uniqueness"][2] > diagnostics["uniqueness"][0]


def test_invalid_weight_fails_closed():
    candidate = torch.ones(1, 1, 1, 1)
    with pytest.raises(ValueError, match="uniqueness_weight"):
        MODULE.select_coherent_unique_snapshot(
            candidate,
            candidate,
            torch.ones(1, 1),
            uniqueness_weight=1.1,
        )
