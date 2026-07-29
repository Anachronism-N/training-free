import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v136_multi_axis_head_discovery.py"
)
SPEC = importlib.util.spec_from_file_location("v136_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _signature(value):
    return torch.full((12, 8), float(value), dtype=torch.float32)


def _temporal_tensors(history_frames, branch):
    if history_frames <= 4:
        logits = torch.zeros((12, history_frames))
        probs = torch.full((12, history_frames), 1.0 / history_frames)
        return logits, probs

    logits = torch.empty((12, history_frames))
    logits[:6, :4] = 1.0
    logits[:6, 4:] = -1.0
    logits[6:, :4] = -1.0
    logits[6:, 4:] = 1.0
    base = logits.softmax(dim=-1)
    if branch == "base":
        return logits, base

    shifted = base.clone()
    if branch == "semantic":
        shifted[:6, :4] *= 2.0
        shifted[:6, 4:] *= 0.5
        shifted[6:, :4] *= 1.02
        shifted[6:, 4:] *= 0.98
    else:
        shifted[:6, :4] *= 1.02
        shifted[:6, 4:] *= 0.98
        shifted[6:, :4] *= 2.0
        shifted[6:, 4:] *= 0.5
    shifted /= shifted.sum(dim=-1, keepdim=True)
    return logits, shifted


def _record(branch, layer, current_frame):
    history_frames = 3 if current_frame == 3 else 8
    logits, probs = _temporal_tensors(history_frames, branch)
    residual = _signature(0.0 if current_frame == 3 else 1.0)
    if current_frame > 3:
        if branch == "semantic":
            residual[:6] += 1.0
            residual[6:] += 0.05
        elif branch == "null":
            residual[:6] += 0.1
            residual[6:] += 0.5
    native = _signature(2.0)
    query = _signature(1.0)
    current_key = _signature(1.5)
    if branch == "semantic":
        native += 0.2
        query += 0.3
        current_key += 0.25
    elif branch == "null":
        native += 0.1
        query += 0.1
        current_key += 0.1
    return {
        "branch": branch,
        "mode": "noisy",
        "current_frame": current_frame,
        "nominal_timestep": 1000,
        "actual_timestep": 999.0,
        "call_index": 0,
        "layer": layer,
        "history_frames": history_frames,
        "recent_frames": min(4, history_frames),
        "residual_signature": residual,
        "native_signature": native,
        "query_signature": query,
        "current_key_signature": current_key,
        "temporal_logits": logits,
        "temporal_probs": probs,
        "history_frame_ids": torch.arange(
            current_frame - history_frames, current_frame
        ),
    }


def _profile(index, kind):
    records = []
    branches = ("base",) if kind == "observational" else (
        "base",
        "semantic",
        "null",
    )
    for current_frame in (3, 21):
        for layer in range(30):
            for branch in branches:
                records.append(_record(branch, layer, current_frame))
    return {
        "_path": f"{kind}_{index}.pt",
        "version": 2,
        "job": {
            "dataset_index": index,
            "job_id": f"{kind}_{index}",
            "family_id": f"family_{index}",
            "kind": kind,
            "factor": "scene" if kind == "counterfactual" else "natural",
        },
        "records": records,
    }


def test_multi_axis_analysis_excludes_no_old_history_states(tmp_path):
    observational = [_profile(index, "observational") for index in range(2)]
    counterfactual = [_profile(index, "counterfactual") for index in range(2)]
    report = MODULE.analyze(
        observational,
        counterfactual,
        output_dir=tmp_path,
        recent_frames=4,
        expected_count=2,
        expected_states=2,
        bootstrap_rounds=20,
        bootstrap_seed=7,
    )

    assert report["head_count"] == 360
    assert report["negative_control"]["state_count"] == 60
    assert report["negative_control"]["median_cphi_semantic"] == 0.0
    assert report["label_counts"]["prompt_label"] == {
        "prompt_conditional": 180,
        "prompt_invariant": 180,
    }
    assert report["label_counts"]["history_polarity_label"] == {
        "history_supportive": 180,
        "recent_preferred": 180,
    }
    assert (tmp_path / "head_axes.csv").is_file()
    assert (tmp_path / "axis_diagnostics.csv").is_file()
    assert (tmp_path / "state_eligibility_audit.csv").is_file()
    assert (tmp_path / "multi_axis_report.json").is_file()
    assert (tmp_path / "multi_axis_summary.md").is_file()


def test_temporal_metrics_have_shift_invariant_middle_margin():
    record = _record("base", layer=0, current_frame=21)
    original = MODULE._temporal_metrics(record, recent_frames=4)
    shifted = dict(record)
    shifted["temporal_logits"] = record["temporal_logits"] + 13.0
    shifted_metrics = MODULE._temporal_metrics(shifted, recent_frames=4)
    torch.testing.assert_close(
        original["middle_recent_margin"],
        shifted_metrics["middle_recent_margin"],
    )


def test_temporal_js_is_zero_for_identical_distributions():
    probs = torch.tensor([[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]])
    result = MODULE._js_per_head(probs, probs)
    torch.testing.assert_close(result, torch.zeros(2))
