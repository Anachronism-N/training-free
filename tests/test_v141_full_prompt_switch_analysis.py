import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v141_full_prompt_switch_profiles.py"
)
SPEC = importlib.util.spec_from_file_location("v141_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _signature(values):
    return torch.tensor(values, dtype=torch.float32).view(12, 1).repeat(1, 8)


def _record(branch, layer, frame):
    base = torch.ones(12)
    high = torch.cat((torch.full((6,), 1.0), torch.full((6,), 0.02)))
    paraphrase = torch.full((12,), 0.08)
    query_switch = torch.full((12,), 0.08)
    query_paraphrase = torch.full((12,), 0.04)
    episode_index, episode_label = MODULE._episode(frame)
    matched = "exact_b" if episode_label == "B" else "exact_a"
    opposite = "exact_a" if episode_label == "B" else "exact_b"
    if branch == matched or branch == "base":
        residual = base
        query = base
    elif branch == opposite:
        residual = base + high
        query = base + query_switch
    else:
        residual = base + paraphrase
        query = base + query_paraphrase
    probabilities = torch.full((12, 6), 1.0 / 6.0)
    return {
        "branch": branch,
        "mode": "noisy",
        "current_frame": frame,
        "nominal_timestep": 1000,
        "layer": layer,
        "episode_index": episode_index,
        "episode_label": episode_label,
        "residual_signature": _signature(residual),
        "native_signature": _signature(residual),
        "query_signature": _signature(query),
        "current_key_signature": _signature(query),
        "temporal_probs": probabilities,
        "history_frame_ids": torch.arange(frame - 6, frame),
    }


def _profile(index):
    frames = (39, 78)
    branches = ("base", "exact_a", "exact_b", "paraphrase_a", "paraphrase_b")
    records = [
        _record(branch, layer, frame)
        for frame in frames
        for branch in branches
        for layer in range(30)
    ]
    calls = [
        {
            "branch": branch,
            "mode": "noisy",
            "current_frame": frame,
            "nominal_timestep": 1000,
        }
        for frame in frames
        for branch in branches
    ]
    return {
        "version": 5,
        "job": {
            "dataset_index": index,
            "job_id": f"aba_scene_action_{index:02d}",
            "kind": "full_prompt_switch",
            "family_index": index,
            "switch_type": "scene_action",
            "seed": 0,
            "switch_frames": [39, 78],
            "shadow_prompts": {
                "exact_a": "a",
                "exact_b": "b",
                "paraphrase_a": "aa",
                "paraphrase_b": "bb",
            },
        },
        "metadata": {
            "allow_prompt_schedule": True,
            "switch_frames": [39, 78],
            "captured_calls": len(calls),
            "record_count": len(records),
            "run_commit": "test-commit",
        },
        "calls": calls,
        "records": records,
    }


def test_v141_analysis_recovers_held_out_switch_axis(tmp_path):
    report = MODULE.analyze(
        [_profile(0), _profile(1)],
        output_dir=tmp_path,
        expected_count=2,
        expected_states=2,
    )
    assert report["profile_contract_passed"]
    assert report["gates"]["exact_prompt_shadow_parity"]
    assert report["gates"]["full_switch_exceeds_local_paraphrase"]
    assert report["held_out"]["zero_label_agreement"] == 1.0
    assert report["held_out"]["validation_positive"] == 174
    assert (tmp_path / "head_axes.csv").is_file()
    assert (tmp_path / "head_episode_axes.csv").is_file()
    assert (tmp_path / "analysis_summary.md").is_file()
