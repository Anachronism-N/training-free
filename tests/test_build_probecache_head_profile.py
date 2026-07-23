import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_probecache_head_profile.py"
SPEC = importlib.util.spec_from_file_location("build_probecache_head_profile", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _profile(kind, side, records):
    return {
        "path": f"{kind}_{side}.pt",
        "kind": kind,
        "side": side,
        "pair_id": "pair",
        "seed": 0,
        "records": records,
    }


def _records(head_deltas):
    records = {}
    base = torch.ones((2, 4), dtype=torch.float32)
    for layer in range(2):
        for current_start in (0, 2, 4):
            key = (0, layer, current_start, "noisy", 0)
            sketch = base.clone()
            for head, delta in enumerate(head_deltas):
                sketch[head] += delta
            records[key] = {"sketch": sketch}
    return records


def test_counterfactual_profile_separates_remote_and_prompt_heads():
    base = _records([0.0, 0.0])
    profiles = [
        _profile("prompt", "a", base),
        _profile("prompt", "b", _records([0.05, 0.8])),
        _profile("history", "full", base),
        _profile("history", "recent", _records([0.8, 0.05])),
    ]
    matrix, report = MODULE.build_profile(
        profiles,
        num_layers=2,
        num_heads=2,
        cache_update_mode="noisy",
        call_indices={0},
        bootstrap_rounds=20,
        bootstrap_seed=7,
    )
    assert matrix == [[1, -1], [1, -1]]
    assert report["label_counts"] == {"reactive": 2, "persistent": 2}
    assert report["acceptance_gates"]["accepted"]
    assert all(
        entry["bootstrap_agreement"] == 1.0 for entry in report["entries"]
    )


def test_incomplete_counterfactual_pair_is_rejected():
    profiles = [_profile("prompt", "a", _records([0.0, 0.0]))]
    try:
        MODULE._pair_profiles(profiles)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("expected incomplete pair failure")
