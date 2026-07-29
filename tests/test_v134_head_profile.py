import json

import torch

from lifecycle_kv.head_profile import HeadProfileConfig, HeadProfileSession


def _attention(query, key, value):
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
    probs = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, value)


def test_head_profile_records_counterfactual_signatures(tmp_path):
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset_index": 0,
                "job_id": "pair",
                "kind": "counterfactual",
                "base_prompt": "base",
                "semantic_prompt": "semantic",
                "null_prompt": "paraphrase",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = HeadProfileConfig(
        enabled=True,
        output_dir=tmp_path / "profiles",
        manifest_path=manifest,
        noisy_frames=(3,),
        noisy_timesteps=(1000,),
        clean_frames=(),
        recent_frames=1,
        spatial_samples=2,
        strict=True,
    )
    session = HeadProfileSession(config)
    session.begin_video(
        dataset_index=0,
        text_prompts=["base"],
        num_frames=120,
        frame_seq_length=4,
        num_frame_per_block=3,
        local_attn_size=21,
    )
    assert session.alternate_prompts() == [
        ("semantic", "semantic"),
        ("null", "paraphrase"),
    ]
    assert session.set_call_context(
        branch="base",
        mode="noisy",
        current_frame=3,
        nominal_timestep=1000,
        actual_timestep=999.0,
    )
    generator = torch.Generator().manual_seed(4)
    query = torch.randn((1, 8, 2, 4), generator=generator)
    current_key = torch.randn((1, 8, 2, 4), generator=generator)
    history_key = torch.randn((1, 12, 2, 4), generator=generator)
    history_value = torch.randn((1, 12, 2, 4), generator=generator)
    native_output = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, history_value[:, :8]), dim=1),
    )
    session.record_attention(
        layer=0,
        query=query,
        current_key=current_key,
        history_key=history_key,
        history_value=history_value,
        native_output=native_output,
        frame_seq_length=4,
        attention_fn=_attention,
    )
    output = session.end_video(expected_layers=1)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["version"] == 2
    assert payload["metadata"]["record_count"] == 1
    record = payload["records"][0]
    assert record["history_frames"] == 3
    assert record["recent_frames"] == 1
    assert record["residual_signature"].shape == (2, 8)
    assert record["temporal_probs"].shape == (2, 3)
    torch.testing.assert_close(
        record["temporal_probs"].float().sum(dim=-1),
        torch.ones(2),
        atol=1e-3,
        rtol=1e-3,
    )


def test_head_profile_uses_semantic_threshold_selection(tmp_path):
    config = HeadProfileConfig(
        enabled=True,
        output_dir=tmp_path,
        manifest_path=None,
        noisy_frames=(21,),
        noisy_timesteps=(750,),
        clean_frames=(63,),
        recent_frames=4,
        spatial_samples=4,
        strict=True,
    )
    session = HeadProfileSession(config)
    session.begin_video(
        dataset_index=0,
        text_prompts=["prompt"],
        num_frames=120,
        frame_seq_length=4,
        num_frame_per_block=3,
        local_attn_size=21,
    )
    assert session.should_capture(
        mode="noisy", current_frame=21, nominal_timestep=750
    )
    assert not session.should_capture(
        mode="noisy", current_frame=21, nominal_timestep=500
    )
    assert session.should_capture(
        mode="clean", current_frame=63, nominal_timestep=0
    )
