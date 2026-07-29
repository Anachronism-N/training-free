import json
from pathlib import Path

import torch

from lifecycle_kv.head_profile import HeadProfileConfig, HeadProfileSession


def test_generic_shadows_and_schedule_episode_metadata(tmp_path):
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset_index": 0,
                "job_id": "aba_scene_action_00",
                "kind": "full_prompt_switch",
                "base_prompt": "prompt A || prompt B || prompt A",
                "schedule_prompts": ["prompt A", "prompt B", "prompt A"],
                "switch_frames": [39, 78],
                "segment_labels": ["A1", "B", "A2"],
                "shadow_prompts": {
                    "exact_a": "prompt A",
                    "exact_b": "prompt B",
                    "paraphrase_a": "rewritten prompt A",
                    "paraphrase_b": "rewritten prompt B",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    session = HeadProfileSession(
        HeadProfileConfig(
            enabled=True,
            output_dir=tmp_path / "profiles",
            manifest_path=manifest,
            noisy_frames=(),
            noisy_timesteps=(),
            clean_frames=(),
            recent_frames=4,
            spatial_samples=4,
            strict=True,
            allow_prompt_schedule=True,
        )
    )
    session.begin_video(
        dataset_index=0,
        text_prompts=["prompt A || prompt B || prompt A"],
        num_frames=120,
        frame_seq_length=4,
        num_frame_per_block=3,
        local_attn_size=21,
    )
    assert session.alternate_prompts() == [
        ("exact_a", "prompt A"),
        ("exact_b", "prompt B"),
        ("paraphrase_a", "rewritten prompt A"),
        ("paraphrase_b", "rewritten prompt B"),
    ]
    session.register_prompt_schedule(
        prompts=["prompt A", "prompt B", "prompt A"],
        switch_frames=[39, 78],
    )
    assert not session.set_call_context(
        branch="base",
        mode="noisy",
        current_frame=39,
        nominal_timestep=1000,
        actual_timestep=999.0,
    )
    assert session.context["episode_index"] == 1
    assert session.context["episode_label"] == "B"
    output = session.end_video(expected_layers=30)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["version"] == 5
    assert payload["metadata"]["switch_frames"] == [39, 78]
    assert payload["metadata"]["schedule_prompts"] == [
        "prompt A",
        "prompt B",
        "prompt A",
    ]


def test_pipeline_routes_generic_shadows_and_registers_runtime_schedule():
    root = Path(__file__).parents[1]
    source = (
        root
        / "third_party"
        / "Self-Forcing"
        / "pipeline"
        / "causal_inference.py"
    ).read_text(encoding="utf-8")
    assert "for branch, conditioning in alternate_conditionings.items()" in source
    assert "HEAD_PROFILE_ALLOW_PROMPT_SCHEDULE=1" in source
    assert "profile_session.register_prompt_schedule(" in source
    assert '"self_cache={schedule_cache_mode} "' in source
