import json

import pytest
import torch

from lifecycle_kv.downstream_probe import (
    apply_history_policy,
    load_probe_plan,
    output_delta_metrics,
    qk_value_motion_correspondence,
)
from lifecycle_kv.head_profile import HeadProfileConfig, HeadProfileSession


def _attention(query, key, value):
    logits = torch.einsum("bqhd,bkhd->bhqk", query, key)
    logits = logits / query.shape[-1] ** 0.5
    weights = logits.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", weights, value)


def test_probe_plan_rejects_invalid_or_empty_head_maps(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "layers": 2,
                "heads": 4,
                "probes": [
                    {
                        "name": "top_recent4",
                        "policy": "recent4",
                        "group": "top",
                        "head_map": {"0": [1, 3]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = load_probe_plan(path)
    assert plan is not None
    assert plan["probes"][0]["head_map"] == {0: [1, 3]}
    assert plan["probes"][0]["selected_head_count"] == 2
    assert len(plan["sha256"]) == 64

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["probes"][0]["head_map"] = {"0": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selects no heads"):
        load_probe_plan(path)


def test_recent_policy_changes_only_selected_heads():
    torch.manual_seed(147)
    batch, tokens, heads, dim = 1, 2, 4, 5
    query = torch.randn(batch, tokens, heads, dim)
    current_key = torch.randn_like(query)
    current_value = torch.randn_like(query)
    history_key = torch.randn(batch, 12, heads, dim)
    history_value = torch.randn_like(history_key)
    native = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, current_value), dim=1),
    )
    output, metadata = apply_history_policy(
        policy="recent4",
        selected_heads=[1, 3],
        query=query,
        current_key=current_key,
        current_value=current_value,
        history_key=history_key,
        history_value=history_value,
        native_output=native,
        frame_seq_length=tokens,
        attention_fn=_attention,
    )
    assert torch.equal(output[:, :, 0], native[:, :, 0])
    assert torch.equal(output[:, :, 2], native[:, :, 2])
    assert not torch.equal(output[:, :, 1], native[:, :, 1])
    assert not torch.equal(output[:, :, 3], native[:, :, 3])
    assert metadata["frame_indices"].tolist() == [2, 3, 4, 5]
    assert metadata["replacement_relative_rms"] > 0


def test_retrieval_and_value_shift_are_frame_aligned():
    torch.manual_seed(148)
    query = torch.randn(2, 4, 3, 4)
    current_key = torch.randn_like(query)
    current_value = torch.randn_like(query)
    history_key = torch.randn(2, 40, 3, 4)
    history_value = torch.randn_like(history_key)
    native = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, current_value), dim=1),
    )
    _, retrieval = apply_history_policy(
        policy="q_retrieval8",
        selected_heads=[0, 2],
        query=query,
        current_key=current_key,
        current_value=current_value,
        history_key=history_key,
        history_value=history_value,
        native_output=native,
        frame_seq_length=4,
        attention_fn=_attention,
    )
    assert retrieval["frame_indices"].shape == (2, 2, 8)
    assert torch.all(retrieval["frame_indices"][..., -4:] >= 6)

    shifted_output, shifted = apply_history_policy(
        policy="value_shift",
        selected_heads=[1],
        query=query,
        current_key=current_key,
        current_value=current_value,
        history_key=history_key,
        history_value=history_value,
        native_output=native,
        frame_seq_length=4,
        attention_fn=_attention,
    )
    assert shifted["recent_frames_preserved"] == 4
    assert shifted["shifted_old_frames"] == 6
    assert torch.equal(shifted_output[:, :, 0], native[:, :, 0])
    assert not torch.equal(shifted_output[:, :, 1], native[:, :, 1])


def test_downstream_output_metrics_have_exact_zero_replay():
    tensor = torch.randn(1, 3, 2, 4, 5)
    metrics = output_delta_metrics(tensor, tensor.clone(), sketch_size=9)
    assert metrics["relative_rms"] == 0
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["per_frame_relative_rms"].shape == (3,)
    assert metrics["delta_sketch"].numel() == 9


def test_qkv_correspondence_recovers_perfect_spatial_identity():
    tokens, heads = 4, 2
    basis = torch.eye(tokens).view(1, tokens, 1, tokens)
    basis = basis.expand(1, tokens, heads, tokens).contiguous()
    metrics = qk_value_motion_correspondence(
        query=basis,
        current_value=basis,
        history_key=basis,
        history_value=basis,
        frame_seq_length=tokens,
        spatial_grid_shape=(2, 2),
        spatial_samples=tokens,
        topk=2,
    )
    assert torch.allclose(
        metrics["raw_value_coordinate_error"], torch.zeros(heads)
    )
    assert torch.allclose(
        metrics["refined_value_coordinate_error"], torch.zeros(heads)
    )
    assert torch.allclose(
        metrics["raw_value_top1_match"], torch.ones(heads)
    )
    assert torch.allclose(
        metrics["refined_value_top1_match"], torch.ones(heads)
    )


def test_profile_session_records_native_replay_and_intervention(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "version": 1,
                "layers": 2,
                "heads": 4,
                "probes": [
                    {
                        "name": "top_recent4",
                        "policy": "recent4",
                        "group": "top",
                        "head_map": {"0": [1]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset_index": 0,
                "job_id": "session_test",
                "kind": "causal_transport_profile",
                "seed": 7,
                "base_prompt": "test prompt",
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
            noisy_frames=(117,),
            noisy_timesteps=(500,),
            clean_frames=(),
            recent_frames=4,
            spatial_samples=2,
            strict=True,
            downstream_probe_plan_path=plan_path,
            downstream_probe_frames=(117,),
            downstream_probe_timesteps=(500,),
            downstream_probe_clean=False,
        )
    )
    session.begin_video(
        dataset_index=0,
        text_prompts=["test prompt"],
        num_frames=120,
        frame_seq_length=2,
        num_frame_per_block=2,
        local_attn_size=8,
    )
    reference = torch.randn(1, 2, 3, 4, 4)
    session.begin_downstream_probe_context(
        mode="noisy",
        current_frame=117,
        nominal_timestep=500,
        actual_timestep=500,
        base_flow=reference,
        base_x0=reference,
    )
    probes = session.downstream_probes()
    session.activate_downstream_probe(probes[0])
    session.record_downstream_probe_output(flow=reference, x0=reference)

    torch.manual_seed(149)
    query = torch.randn(1, 2, 4, 3)
    current_key = torch.randn_like(query)
    current_value = torch.randn_like(query)
    history_key = torch.randn(1, 12, 4, 3)
    history_value = torch.randn_like(history_key)
    native = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, current_value), dim=1),
    )
    session.activate_downstream_probe(probes[1])
    replaced = session.apply_downstream_probe(
        layer=0,
        query=query,
        current_key=current_key,
        current_value=current_value,
        history_key=history_key,
        history_value=history_value,
        native_output=native,
        frame_seq_length=2,
        attention_fn=_attention,
    )
    assert torch.equal(replaced[:, :, 0], native[:, :, 0])
    assert not torch.equal(replaced[:, :, 1], native[:, :, 1])
    session.record_downstream_probe_output(
        flow=reference + 0.1,
        x0=reference + 0.2,
    )
    session.end_downstream_probe_context()
    output = session.end_video(expected_layers=2)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["version"] == 8
    assert payload["downstream_probe_expected_count"] == 2
    assert len(payload["downstream_probe_records"]) == 2
