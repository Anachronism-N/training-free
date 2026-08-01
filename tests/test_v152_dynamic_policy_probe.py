import json

import pytest

torch = pytest.importorskip("torch")

from lifecycle_kv.downstream_probe import (
    compute_dynamic_head_scores,
    load_probe_plan,
    select_dynamic_heads,
)
from lifecycle_kv.head_profile import HeadProfileConfig, HeadProfileSession


def _attention(query, key, value):
    logits = torch.einsum("bqhd,bkhd->bhqk", query, key) / query.shape[-1] ** 0.5
    probabilities = logits.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value)


def _tensors():
    generator = torch.Generator().manual_seed(152)
    query = torch.randn(1, 2, 4, 2, generator=generator)
    current_key = torch.randn(1, 2, 4, 2, generator=generator)
    current_value = torch.randn(1, 2, 4, 2, generator=generator)
    history_key = torch.randn(1, 20, 4, 2, generator=generator)
    history_value = torch.randn(1, 20, 4, 2, generator=generator)
    native = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, current_value), dim=1),
    )
    return query, current_key, current_value, history_key, history_value, native


def _selector(selector_type, direction="high"):
    return {
        "type": selector_type,
        "direction": direction,
        "heads_per_layer": 2,
        "budget_frames": 8,
        "recent_frames": 4,
        "spatial_samples": 2,
    }


def test_dynamic_plan_schema_is_versioned_and_counted(tmp_path):
    payload = {
        "version": 2,
        "suite": "test",
        "layers": 2,
        "heads": 4,
        "probes": [
            {
                "name": "dynamic_uniform",
                "group": "dynamic",
                "policy": "uniform8",
                "head_selector": _selector("qk_policy_margin"),
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    plan = load_probe_plan(path)
    assert plan["probes"][0]["head_map"] == {}
    assert plan["probes"][0]["head_selector"]["type"] == "qk_policy_margin"
    assert plan["probes"][0]["selected_head_count"] == 4

    payload["version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires downstream plan version"):
        load_probe_plan(path)


@pytest.mark.parametrize(
    "selector_type",
    ["policy_error_margin", "qk_policy_margin", "old_history_mass"],
)
def test_dynamic_scores_are_finite_and_high_low_are_disjoint(selector_type):
    (
        query,
        current_key,
        current_value,
        history_key,
        history_value,
        native,
    ) = _tensors()
    high_spec = _selector(selector_type, "high")
    bundle = compute_dynamic_head_scores(
        selector=high_spec,
        query=query,
        current_key=current_key,
        current_value=current_value,
        history_key=history_key,
        history_value=history_value,
        native_output=native,
        frame_seq_length=2,
        attention_fn=_attention,
        output_projection_weight=torch.eye(8),
    )
    assert bundle["scores"].shape == (4,)
    assert torch.isfinite(bundle["scores"]).all()
    high = select_dynamic_heads(bundle, high_spec)
    low = select_dynamic_heads(bundle, _selector(selector_type, "low"))
    assert len(high["selected_heads"]) == len(low["selected_heads"]) == 2
    assert set(high["selected_heads"]).isdisjoint(low["selected_heads"])


def test_dynamic_tail_tie_breaks_use_disjoint_ends():
    bundle = {"scores": torch.zeros(4), "type": "qk_policy_margin"}
    high = select_dynamic_heads(
        bundle, _selector("qk_policy_margin", "high")
    )
    low = select_dynamic_heads(bundle, _selector("qk_policy_margin", "low"))
    assert high["selected_heads"] == [0, 1]
    assert low["selected_heads"] == [2, 3]


def test_session_freezes_native_selector_across_policy_replays(tmp_path):
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset_index": 0,
                "job_id": "v152_test",
                "kind": "v152_online_policy_core",
                "prompt_slot": 0,
                "source_prompt_index": 0,
                "seed_replicate": 0,
                "seed": 152000,
                "base_prompt": "test prompt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selector = _selector("qk_policy_margin")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "version": 2,
                "suite": "v152_online_policy_core",
                "layers": 2,
                "heads": 4,
                "probes": [
                    {
                        "name": f"qk_{policy}",
                        "group": "qk",
                        "policy": policy,
                        "head_selector": selector,
                    }
                    for policy in ("uniform8", "recent8")
                ],
            }
        ),
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
        num_frame_per_block=1,
        local_attn_size=21,
    )
    session.begin_downstream_probe_context(
        mode="noisy",
        current_frame=117,
        nominal_timestep=500,
        actual_timestep=499.0,
        base_flow=torch.ones(1, 1, 1, 1, 1),
        base_x0=torch.ones(1, 1, 1, 1, 1),
    )
    tensors = _tensors()
    selections = []
    for probe_index, probe in enumerate(session.downstream_probes()):
        session.activate_downstream_probe(probe)
        for layer in range(2):
            query = tensors[0] + (100.0 if probe_index else 0.0)
            session.apply_downstream_probe(
                layer=layer,
                query=query,
                current_key=tensors[1],
                current_value=tensors[2],
                history_key=tensors[3],
                history_value=tensors[4],
                native_output=tensors[5],
                frame_seq_length=2,
                attention_fn=_attention,
                output_projection_weight=torch.eye(8),
            )
        if probe["name"] != "native_replay":
            selections.append(
                [
                    session._downstream_layer_metadata[layer]["head_selector"][
                        "selected_heads"
                    ]
                    for layer in range(2)
                ]
            )
        session.record_downstream_probe_output(
            flow=torch.ones(1, 1, 1, 1, 1),
            x0=torch.ones(1, 1, 1, 1, 1),
        )
    session.end_downstream_probe_context()
    assert selections[0] == selections[1]
    assert session._downstream_selector_cache == {}
