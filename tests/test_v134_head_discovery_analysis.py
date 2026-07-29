import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v134_head_discovery.py"
)
SPEC = importlib.util.spec_from_file_location("v134_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(branch, layer, signature):
    return {
        "branch": branch,
        "mode": "noisy",
        "current_frame": 21,
        "nominal_timestep": 1000,
        "actual_timestep": 999.0,
        "call_index": 0,
        "layer": layer,
        "history_frames": 3,
        "recent_frames": 1,
        "residual_signature": signature.clone(),
        "full_signature": signature.clone(),
        "recent_signature": signature.clone(),
        "native_signature": signature.clone(),
        "query_signature": signature.clone(),
        "current_key_signature": signature.clone(),
        "temporal_logits": torch.ones((12, 3)),
        "temporal_probs": torch.full((12, 3), 1 / 3),
        "history_frame_ids": torch.tensor([18, 19, 20]),
    }


def _profiles():
    base = torch.ones((12, 4))
    semantic = base.clone()
    null = base.clone()
    semantic[:6] += 1.0
    null[:6] += 0.1
    semantic[6:] += 0.05
    null[6:] += 0.5
    counter_records = []
    observational_records = []
    for layer in range(30):
        counter_records.extend(
            (
                _record("base", layer, base),
                _record("semantic", layer, semantic),
                _record("null", layer, null),
            )
        )
        observational_records.append(_record("base", layer, base))
    counter = {
        "_path": "counter.pt",
        "version": 2,
        "job": {
            "dataset_index": 0,
            "job_id": "counter",
            "family_id": "family",
            "kind": "counterfactual",
            "factor": "scene",
        },
        "records": counter_records,
    }
    observational = {
        "_path": "observational.pt",
        "version": 2,
        "job": {
            "dataset_index": 0,
            "job_id": "observational",
            "kind": "observational",
            "factor": "natural",
        },
        "records": observational_records,
    }
    return observational, counter


def test_analysis_recovers_prompt_conditional_and_invariant_heads(tmp_path):
    observational, counter = _profiles()
    report = MODULE.analyze(
        [observational],
        [counter],
        output_dir=tmp_path,
        bootstrap_rounds=20,
        bootstrap_seed=4,
        expected_count=1,
        legacy_specs=[],
    )
    assert report["label_counts"] == {
        "prompt_conditional": 180,
        "prompt_invariant": 180,
    }
    matrix = [
        line.split(",")
        for line in (tmp_path / "head_map.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(matrix) == 30
    assert all(row == ["1"] * 6 + ["0"] * 6 for row in matrix)
    assert (tmp_path / "classification_report.json").is_file()
    assert (tmp_path / "analysis_summary.md").is_file()
