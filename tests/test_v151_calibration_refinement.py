import json

import pytest

torch = pytest.importorskip("torch")

from lifecycle_kv.downstream_probe import (
    apply_history_policy,
    load_probe_plan,
)


def _attention(query, key, value):
    logits = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float())
    weights = (logits / query.shape[-1] ** 0.5).softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", weights, value.float())


def _inputs():
    torch.manual_seed(151)
    query = torch.randn(1, 4, 4, 3)
    current_key = torch.randn_like(query)
    current_value = torch.randn_like(query)
    history_key = torch.randn(1, 40, 4, 3)
    history_value = torch.randn_like(history_key)
    native = _attention(
        query,
        torch.cat((history_key, current_key), dim=1),
        torch.cat((history_value, current_value), dim=1),
    ).to(torch.bfloat16)
    projection = torch.randn(12, 12).to(torch.bfloat16)
    bias = torch.randn(12).to(torch.bfloat16)
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


def test_plan_normalizes_and_bounds_refinement_steps(tmp_path):
    payload = {
        "version": 1,
        "layers": 1,
        "heads": 4,
        "probes": [
            {
                "name": "refined",
                "policy": "key_shift",
                "head_map": {"0": [0, 1]},
                "calibration": {
                    "mode": "projected_relative_rms",
                    "target": 0.02,
                    "refinement_steps": 4,
                },
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    plan = load_probe_plan(path)
    assert plan["probes"][0]["calibration"]["refinement_steps"] == 4

    payload["probes"][0]["calibration"]["refinement_steps"] = 9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="between 0 and 8"):
        load_probe_plan(path)


def test_refinement_uses_exact_cast_and_projection_path():
    inputs = _inputs()
    common = {
        "mode": "projected_relative_rms",
        "target": 0.01,
        "min_scale": 0.001,
        "max_scale": 50.0,
    }
    _, baseline = apply_history_policy(
        policy="key_shift",
        selected_heads=[1, 3],
        calibration={**common, "refinement_steps": 0},
        **inputs,
    )
    _, refined = apply_history_policy(
        policy="key_shift",
        selected_heads=[1, 3],
        calibration={**common, "refinement_steps": 4},
        **inputs,
    )
    assert int(refined["calibration_refinement_steps"]) == 4
    assert not bool(refined["calibration_refinement_bound_hit"])
    assert float(refined["calibration_relative_error"]) <= float(
        baseline["calibration_relative_error"]
    )
    assert float(refined["calibration_relative_error"]) <= 0.02
