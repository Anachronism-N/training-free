import importlib.util
import json
import sys
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v147_causal_transport_profiles.py"
)
SPEC = importlib.util.spec_from_file_location("v147_analysis_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _probe_specs(layers):
    maps = {
        "top": {str(layer): [1] for layer in range(layers)},
        "bottom": {str(layer): [0] for layer in range(layers)},
        "random": {str(layer): [0] for layer in range(layers)},
        "all": {str(layer): [0, 1] for layer in range(layers)},
    }
    specs = []
    for group in ("top", "bottom", "random"):
        specs.append((group, "recent4"))
    specs.append(("all", "recent4"))
    for group in ("top", "random"):
        specs.append((group, "uniform8"))
    for group in ("top", "bottom", "random"):
        specs.append((group, "q_retrieval8"))
    for group in ("top", "bottom", "random"):
        specs.append((group, "value_shift"))
    return [
        {
            "name": f"{group}_{policy}",
            "group": group,
            "policy": policy,
            "head_map": maps[group],
        }
        for group, policy in specs
    ]


def _profile(prompt, replicate, plan_sha, layers):
    contexts = (("noisy", 1000), ("noisy", 500), ("clean", 0))
    records = []
    for mode, timestep in contexts:
        for layer in range(layers):
            metrics = {
                name: torch.tensor(
                    [0.1 + prompt * 0.01, 0.3 + prompt * 0.02]
                )
                for name in MODULE.MOTION_METRICS
            }
            metrics.update({"sample_count": 4, "topk": 2})
            records.append(
                {
                    "mode": mode,
                    "current_frame": 117,
                    "nominal_timestep": timestep,
                    "layer": layer,
                    "motion_correspondence_metrics": metrics,
                }
            )
    downstream = []
    for mode, timestep in contexts:
        base_top = 0.25 + 0.02 * prompt + 0.002 * replicate
        values = {
            "native_replay": 0.0,
            "top_recent4": base_top,
            "bottom_recent4": 0.05 + 0.002 * prompt,
            "random_recent4": 0.07 + 0.003 * prompt,
            "all_recent4": base_top,
            "top_uniform8": base_top * (0.65 + 0.01 * prompt),
            "random_uniform8": 0.06 + 0.003 * prompt,
            "top_q_retrieval8": base_top * (0.40 + 0.02 * prompt),
            "bottom_q_retrieval8": 0.03 + 0.001 * prompt,
            "random_q_retrieval8": 0.04 + 0.002 * prompt,
            "top_value_shift": base_top * 1.4,
            "bottom_value_shift": 0.06 + 0.002 * prompt,
            "random_value_shift": 0.08 + 0.002 * prompt,
        }
        for name, value in values.items():
            policy = (
                "native"
                if name == "native_replay"
                else name.split("_", 1)[1]
            )
            group = "native" if name == "native_replay" else name.split("_")[0]
            layer_metadata = {}
            if name != "native_replay":
                layer_metadata = {
                    layer: {
                        "replacement_relative_rms": value * 0.5,
                        **(
                            {"shifted_old_frames": 6}
                            if policy == "value_shift"
                            else {}
                        ),
                    }
                    for layer in range(layers)
                }
            metric = {
                "relative_rms": value,
                "cosine": 1.0 - value * 0.01,
                "max_abs_delta": value * 2,
            }
            downstream.append(
                {
                    "mode": mode,
                    "current_frame": 117,
                    "nominal_timestep": timestep,
                    "probe_name": name,
                    "policy": policy,
                    "group": group,
                    "selected_head_count": 0 if name == "native_replay" else layers,
                    "flow_metrics": metric,
                    "x0_metrics": metric,
                    "layer_metadata": layer_metadata,
                }
            )
    seed = 1000 + prompt + replicate * 100
    return {
        "version": 8,
        "job": {
            "dataset_index": prompt * 2 + replicate,
            "kind": "causal_transport_profile",
            "prompt_slot": prompt,
            "source_prompt_index": prompt * 3,
            "seed_replicate": replicate,
            "seed": seed,
            "reference_seed": seed,
        },
        "metadata": {
            "seed": seed,
            "captured_calls": 3,
            "record_count": 3 * layers,
            "incomplete_calls": [],
            "downstream_probe_plan": {"sha256": plan_sha},
        },
        "records": records,
        "downstream_probe_records": downstream,
        "downstream_probe_expected_count": 39,
    }


def test_analysis_validates_replay_and_detects_ranked_causal_effect(
    tmp_path, monkeypatch
):
    prompts, layers, heads = 4, 2, 2
    monkeypatch.setattr(MODULE, "PROMPTS", prompts)
    monkeypatch.setattr(MODULE, "LAYERS", layers)
    monkeypatch.setattr(MODULE, "HEADS", heads)
    plan_path = tmp_path / "plan.json"
    plan = {
        "version": 1,
        "layers": layers,
        "heads": heads,
        "source": {
            "selected_axis": {
                "variant": "identity",
                "axis": "k_shift",
            }
        },
        "probes": _probe_specs(layers),
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    _, digest = MODULE._load_plan(plan_path)
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    for prompt in range(prompts):
        for replicate in (0, 1):
            torch.save(
                _profile(prompt, replicate, digest, layers),
                profile_dir / f"{prompt}_{replicate}.pt",
            )
    report = MODULE.analyze(
        profile_dir=profile_dir,
        probe_plan_path=plan_path,
        output_dir=tmp_path / "analysis",
        expected_count=prompts * 2,
    )
    assert report["gates"]["g0_native_replay_parity"]
    assert report["gates"]["g1_ranked_heads_have_reproducible_downstream_effect"]
    assert report["gates"]["g2_q_retrieval_rescues_ranked_heads"]
    assert report["gates"]["g3_value_shift_is_non_degenerate"]
    assert report["gates"]["g4_q_retrieval_is_head_selective"]
    assert (tmp_path / "analysis" / "report.md").exists()
    assert (
        tmp_path / "analysis" / "downstream_comparisons.csv"
    ).exists()
