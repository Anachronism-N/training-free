import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v138_history_interventions.py"
)
SPEC = importlib.util.spec_from_file_location("v138_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _signature(value):
    return torch.full((12, 8), float(value))


def _descriptors(job_index):
    query = torch.zeros((12, 2, 16))
    history = torch.zeros((12, 6, 2, 16))
    query[:6, :, job_index] = 1.0
    history[:6, :, :, job_index] = 1.0
    query[6:, :, 2] = 1.0
    history[6:, :, :, 2] = 1.0
    return query, history


def _record(job_index, layer, current_frame):
    query, history = _descriptors(job_index)
    full = _signature(1.0)
    reverse = full.clone()
    phase = full.clone()
    frozen = full.clone()
    mismatch = full.clone()
    reverse[:6] += 1.0
    reverse[6:] += 0.1
    phase[:6] += 0.5
    phase[6:] += 0.05
    frozen[:6] += 0.8
    frozen[6:] += 0.08
    mismatch[:6] += 0.7
    mismatch[6:] += 0.07
    return {
        "branch": "base",
        "mode": "noisy",
        "current_frame": current_frame,
        "nominal_timestep": 1000,
        "layer": layer,
        "history_frames": 6,
        "recent_frames": 4,
        "full_history_signature": full,
        "recent_history_signature": _signature(0.5),
        "history_reverse_signature": reverse,
        "history_phase_shift_signature": phase,
        "history_freeze_latest_signature": frozen,
        "history_value_mismatch_signature": mismatch,
        "query_projection": query,
        "history_key_projection": history,
        "history_frame_ids": torch.arange(
            current_frame - 6, current_frame
        ),
        "history_intervention_pre_rope_sidecar": 1.0,
        "history_intervention_rope_reconstruction_relative_max": 0.0,
        "history_intervention_rope_reconstruction_relative_rms": 0.0,
        "history_intervention_recent_value_preservation_max": 0.0,
    }


def _profile(job_index):
    records = []
    for current_frame in (21, 63):
        for layer in range(30):
            records.append(_record(job_index, layer, current_frame))
    return {
        "version": 4,
        "_path": f"profile_{job_index}.pt",
        "job": {
            "dataset_index": job_index,
            "job_id": f"job_{job_index}",
            "kind": "history_intervention",
            "base_prompt": (
                "a dancer in a bright studio"
                if job_index == 0
                else "a vehicle on a rainy mountain road"
            ),
            "seed": 0,
        },
        "metadata": {
            "history_interventions": True,
            "projection_seed": 20260729,
            "projection_dim": 16,
            "run_commit": "test-commit",
        },
        "records": records,
    }


def test_v138_analysis_recovers_specific_and_order_responsive_heads(tmp_path):
    report = MODULE.analyze(
        [_profile(0), _profile(1)],
        output_dir=tmp_path,
        expected_count=2,
        expected_states=2,
        recent_frames=4,
        bootstrap_rounds=20,
        bootstrap_seed=3,
        v136_head_axes=None,
    )
    assert report["head_count"] == 360
    assert report["profile_contract_passed"]
    assert report["specificity_counts"] == {
        "self_history_specific": 180,
        "no_self_history_preference": 180,
    }
    assert (tmp_path / "head_axes.csv").is_file()
    assert (tmp_path / "donor_audit.csv").is_file()
    assert (tmp_path / "analysis_report.json").is_file()
    assert (tmp_path / "analysis_summary.md").is_file()
