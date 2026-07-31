import json

import pytest
import torch

from lifecycle_kv.downstream_probe import (
    apply_history_policy,
    load_probe_plan,
)


def _attention(query, key, value):
    logits = torch.einsum("bqhd,bkhd->bhqk", query, key)
    logits = logits / query.shape[-1] ** 0.5
    weights = logits.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", weights, value)


def _inputs():
    torch.manual_seed(149)
    query = torch.randn(1, 4, 4, 3)
    current_key = torch.randn_like(query)
    current_value = torch.randn_like(query)
    history_key = torch.randn(1, 40, 4, 3)
    history_value = torch.randn_like(history_key)
    native = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, current_value), dim=1),
    )
    projection = torch.randn(12, 12)
    bias = torch.randn(12)
    return {
        "query": query,
        "current_key": current_key,
        "current_value": current_value,
        "history_key": history_key,
        "history_value": history_value,
        "native_output": native,
        "frame_seq_length": 4,
        "attention_fn": _attention,
        "output_projection_weight": projection,
        "output_projection_bias": bias,
    }


def test_probe_plan_normalizes_calibration_and_policy_contrast(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "layers": 2,
                "heads": 4,
                "probes": [
                    {
                        "name": "policy_top",
                        "policy": "policy_contrast",
                        "policy_args": {
                            "left": "uniform8",
                            "right": "recent8",
                        },
                        "calibration": {
                            "mode": "projected_relative_rms",
                            "target": 0.05,
                            "min_scale": 0.001,
                            "max_scale": 1000,
                        },
                        "head_map": {"0": [1, 3]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = load_probe_plan(path)
    probe = plan["probes"][0]
    assert probe["policy_args"] == {
        "left": "uniform8",
        "right": "recent8",
    }
    assert probe["calibration"]["target"] == pytest.approx(0.05)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["probes"][0]["policy_args"]["right"] = "uniform8"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid policy contrast"):
        load_probe_plan(path)


@pytest.mark.parametrize(
    ("policy", "policy_args"),
    [
        ("key_shift", None),
        ("value_shift", None),
        (
            "policy_contrast",
            {"left": "uniform8", "right": "recent8"},
        ),
    ],
)
def test_projected_calibration_hits_target_without_mutating_history(
    policy, policy_args
):
    inputs = _inputs()
    key_before = inputs["history_key"].clone()
    value_before = inputs["history_value"].clone()
    target = 0.05
    output, metadata = apply_history_policy(
        policy=policy,
        selected_heads=[1, 3],
        policy_args=policy_args,
        calibration={
            "mode": "projected_relative_rms",
            "target": target,
            "min_scale": 0.001,
            "max_scale": 1000,
        },
        **inputs,
    )

    assert torch.equal(inputs["history_key"], key_before)
    assert torch.equal(inputs["history_value"], value_before)
    assert torch.equal(output[:, :, [0, 2]], inputs["native_output"][:, :, [0, 2]])
    assert not bool(metadata["calibration_clipped"])
    assert not bool(metadata["calibration_degenerate"])
    assert float(metadata["projected_replacement_relative_rms"]) == pytest.approx(
        target, rel=2e-5, abs=2e-6
    )

    projected_native = torch.nn.functional.linear(
        inputs["native_output"].flatten(2),
        inputs["output_projection_weight"],
        inputs["output_projection_bias"],
    )
    projected_output = torch.nn.functional.linear(
        output.flatten(2),
        inputs["output_projection_weight"],
        inputs["output_projection_bias"],
    )
    achieved = (
        (projected_output - projected_native).square().mean().sqrt()
        / projected_native.square().mean().sqrt()
    )
    assert float(achieved) == pytest.approx(target, rel=2e-5, abs=2e-6)
    assert float(metadata["replacement_relative_rms"]) > 0
    assert float(metadata["raw_replacement_relative_rms"]) > 0
    if policy == "policy_contrast":
        assert metadata["frame_indices"]["uniform8"].numel() == 8
        assert metadata["frame_indices"]["recent8"].numel() == 8
        assert not torch.equal(
            metadata["frame_indices"]["uniform8"],
            metadata["frame_indices"]["recent8"],
        )


def test_projected_calibration_requires_projection_weight():
    inputs = _inputs()
    inputs.pop("output_projection_weight")
    inputs.pop("output_projection_bias")
    with pytest.raises(ValueError, match="requires output projection weight"):
        apply_history_policy(
            policy="key_shift",
            selected_heads=[0],
            calibration={
                "mode": "projected_relative_rms",
                "target": 0.05,
            },
            **inputs,
        )
