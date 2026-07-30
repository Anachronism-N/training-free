import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from lifecycle_kv.head_profile import HeadProfileConfig, HeadProfileSession


def _attention(query, key, value):
    logits = torch.einsum(
        "bthd,bshd->bhts", query, key
    ) / math.sqrt(query.shape[-1])
    probabilities = logits.softmax(dim=-1)
    return torch.einsum("bhts,bshd->bthd", probabilities, value)


def _session(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset_index": 0,
                "job_id": "v142_test",
                "kind": "full_prompt_switch",
                "base_prompt": "A || B || A",
                "switch_frames": [39, 78],
                "segment_labels": ["A1", "B", "A2"],
                "shadow_prompts": {"exact_a": "A", "exact_b": "B"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return HeadProfileSession(
        HeadProfileConfig(
            enabled=True,
            output_dir=tmp_path / "profiles",
            manifest_path=manifest,
            noisy_frames=(),
            noisy_timesteps=(),
            clean_frames=(39,),
            recent_frames=2,
            spatial_samples=2,
            strict=True,
            allow_prompt_schedule=True,
            causal_policy_metrics=True,
            policy_budget_frames=4,
            persistent_probe=True,
            persistent_capture_frames=(0,),
            persistent_probe_frames=(39,),
            persistent_spatial_samples=2,
        )
    )


def test_v142_records_policy_errors_and_persistent_a_probe(tmp_path):
    torch.manual_seed(7)
    session = _session(tmp_path)
    session.begin_video(
        dataset_index=0,
        text_prompts=["A || B || A"],
        num_frames=120,
        frame_seq_length=4,
        num_frame_per_block=3,
        local_attn_size=21,
    )

    session.set_call_context(
        branch="base",
        mode="clean",
        current_frame=0,
        nominal_timestep=0,
        actual_timestep=0.0,
    )
    raw_archive_key = torch.randn(1, 12, 2, 4)
    archive_key = raw_archive_key + 0.01
    archive_value = torch.randn_like(archive_key)
    session.capture_persistent_tokens(
        layer=0,
        raw_current_key=raw_archive_key,
        current_key=archive_key,
        current_value=archive_value,
        frame_seq_length=4,
    )

    assert session.set_call_context(
        branch="base",
        mode="clean",
        current_frame=39,
        nominal_timestep=0,
        actual_timestep=0.0,
    )
    raw_query = torch.randn(1, 12, 2, 4)
    query = raw_query + 0.02
    raw_current_key = torch.randn_like(query)
    current_key = raw_current_key + 0.03
    current_value = torch.randn_like(query)
    history_key = torch.randn(1, 32, 2, 4)
    history_value = torch.randn_like(history_key)
    native_output = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, current_value), dim=1),
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
        raw_query=raw_query,
        raw_current_key=raw_current_key,
        current_value=current_value,
        output_projection_weight=torch.eye(8),
    )

    output = session.end_video(expected_layers=1)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["version"] == 6
    assert payload["metadata"]["persistent_capture_count"] == 1
    record = payload["records"][0]
    policy = record["causal_policy_metrics"]
    assert set(policy) == {
        "current_only",
        "recent4",
        "recent_budget",
        "boundary_recent",
        "uniform_recent",
    }
    assert policy["recent_budget"]["projected_relative_error"].shape == (2,)
    metadata = record["causal_policy_metadata"]
    assert metadata["eligible_budget_comparison"] is True
    assert metadata["native_reconstruction_relative_rms"] < 1e-6
    assert metadata["frame_indices"]["recent_budget"].tolist() == [4, 5, 6, 7]
    assert metadata["frame_indices"]["boundary_recent"].tolist() == [0, 1, 2, 7]
    assert metadata["frame_indices"]["uniform_recent"].tolist() == [0, 5, 6, 7]

    persistent = record["persistent_probe_metrics"]
    assert persistent["content_top1_cosine"].shape == (2,)
    assert persistent["positioned_normalized_entropy"].shape == (2,)
    assert persistent["output_projected_relative_error"].shape == (2,)
    assert record["persistent_probe_metadata"]["archive_tokens"] == 6


def test_persistent_probe_excludes_same_frame_clean_capture(tmp_path):
    session = _session(tmp_path)
    session.config = replace(
        session.config,
        persistent_capture_frames=(0, 39),
        persistent_probe_frames=(39,),
    )
    session.begin_video(
        dataset_index=0,
        text_prompts=["A || B || A"],
        num_frames=120,
        frame_seq_length=4,
        num_frame_per_block=3,
        local_attn_size=21,
    )

    torch.manual_seed(39)
    for frame in (0, 39):
        session.set_call_context(
            branch="base",
            mode="clean",
            current_frame=frame,
            nominal_timestep=0,
            actual_timestep=0.0,
        )
        raw_key = torch.randn(1, 12, 2, 4)
        key = raw_key + 0.01
        value = torch.randn_like(key)
        session.capture_persistent_tokens(
            layer=0,
            raw_current_key=raw_key,
            current_key=key,
            current_value=value,
            frame_seq_length=4,
        )

    raw_query = torch.randn(1, 12, 2, 4)
    query = raw_query + 0.02
    current_key = torch.randn_like(query)
    current_value = torch.randn_like(query)
    history_key = torch.randn(1, 32, 2, 4)
    history_value = torch.randn_like(history_key)
    native_output = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, current_value), dim=1),
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
        raw_query=raw_query,
        raw_current_key=current_key,
        current_value=current_value,
        output_projection_weight=torch.eye(8),
    )

    metadata = session.records[0]["persistent_probe_metadata"]
    assert metadata["capture_frames"] == [0]
    assert metadata["configured_capture_frames"] == [0, 39]
    assert metadata["strictly_older_than_frame"] == 39


def test_v142_model_passes_raw_and_projection_inputs_to_profiler():
    root = Path(__file__).parents[1]
    source = (
        root
        / "third_party"
        / "Self-Forcing"
        / "wan"
        / "modules"
        / "causal_model.py"
    ).read_text(encoding="utf-8")
    assert "profile_session.capture_persistent_tokens(" in source
    assert "raw_query=q" in source
    assert "raw_current_key=k" in source
    assert "current_value=v" in source
    assert "output_projection_weight=self.o.weight" in source
