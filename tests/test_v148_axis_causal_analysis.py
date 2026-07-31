import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v148_axis_causal_profiles.py"
)
SPEC = importlib.util.spec_from_file_location("v148_analysis_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


AXES = ("k", "v", "policy")
POLICIES = ("key_shift", "value_shift", "recent4")
MATCHED = {"k": "key_shift", "v": "value_shift", "policy": "recent4"}


def _core_plan(layers):
    head_map = {str(layer): [0] for layer in range(layers)}
    probes = []
    hypotheses = []
    for axis in AXES:
        for policy in POLICIES:
            top = f"{axis}_top_{policy}"
            bottom = f"{axis}_bottom_{policy}"
            probes.extend(
                [
                    {
                        "name": top,
                        "group": f"{axis}_top",
                        "policy": policy,
                        "head_map": head_map,
                    },
                    {
                        "name": bottom,
                        "group": f"{axis}_bottom",
                        "policy": policy,
                        "head_map": head_map,
                    },
                ]
            )
            random_probes = []
            if policy == MATCHED[axis]:
                for index in range(2):
                    name = f"{axis}_random{index}_{policy}"
                    probes.append(
                        {
                            "name": name,
                            "group": f"{axis}_random{index}",
                            "policy": policy,
                            "head_map": head_map,
                        }
                    )
                    random_probes.append(name)
            hypotheses.append(
                {
                    "axis": axis,
                    "policy": policy,
                    "matched": policy == MATCHED[axis],
                    "top_probe": top,
                    "bottom_probe": bottom,
                    "random_probes": random_probes,
                }
            )
    pf_hypotheses = []
    for axis in AXES:
        policy = MATCHED[axis]
        top = f"{axis}_pfmatched_top_{policy}"
        bottom = f"{axis}_pfmatched_bottom_{policy}"
        probes.extend(
            [
                {
                    "name": top,
                    "group": f"{axis}_pfmatched_top",
                    "policy": policy,
                    "head_map": head_map,
                },
                {
                    "name": bottom,
                    "group": f"{axis}_pfmatched_bottom",
                    "policy": policy,
                    "head_map": head_map,
                },
            ]
        )
        pf_hypotheses.append(
            {
                "axis": axis,
                "policy": policy,
                "top_probe": top,
                "bottom_probe": bottom,
                "metadata": {},
            }
        )
    assert len(probes) == 30
    return {
        "version": 1,
        "layers": layers,
        "heads": 2,
        "suite": "v148_axis_core",
        "source": {"axes": {}},
        "probes": probes,
        "hypotheses": hypotheses,
        "pf_matched_hypotheses": pf_hypotheses,
    }


def _dose_plan(layers):
    head_map = {str(layer): [0] for layer in range(layers)}
    probes = []
    hypotheses = []
    for axis in AXES:
        policy = MATCHED[axis]
        pairs = []
        for dose in range(1, 5):
            top = f"{axis}_top{dose}_{policy}"
            bottom = f"{axis}_bottom{dose}_{policy}"
            probes.extend(
                [
                    {
                        "name": top,
                        "group": f"{axis}_top{dose}",
                        "policy": policy,
                        "head_map": head_map,
                    },
                    {
                        "name": bottom,
                        "group": f"{axis}_bottom{dose}",
                        "policy": policy,
                        "head_map": head_map,
                    },
                ]
            )
            pairs.append(
                {
                    "dose": dose,
                    "top_probe": top,
                    "bottom_probe": bottom,
                }
            )
        hypotheses.append({"axis": axis, "policy": policy, "pairs": pairs})
    assert len(probes) == 24
    return {
        "version": 1,
        "layers": layers,
        "heads": 2,
        "suite": "v148_axis_dose",
        "source": {"axes": {}},
        "probes": probes,
        "dose_hypotheses": hypotheses,
    }


def _profile(prompt, replicate, plan, digest, layers):
    contexts = (("noisy", 1000), ("noisy", 500))
    records = [
        {
            "mode": mode,
            "current_frame": 117,
            "nominal_timestep": timestep,
            "layer": layer,
        }
        for mode, timestep in contexts
        for layer in range(layers)
    ]
    downstream = []
    for mode, timestep in contexts:
        native_metric = {
            "relative_rms": 0.0,
            "cosine": 1.0,
            "max_abs_delta": 0.0,
        }
        downstream.append(
            {
                "mode": mode,
                "current_frame": 117,
                "nominal_timestep": timestep,
                "probe_name": "native_replay",
                "policy": "native",
                "group": "native",
                "selected_head_count": 0,
                "flow_metrics": native_metric,
                "x0_metrics": native_metric,
                "layer_metadata": {},
            }
        )
        for probe in plan["probes"]:
            name = probe["name"]
            axis = next(axis for axis in AXES if name.startswith(f"{axis}_"))
            matched = probe["policy"] == MATCHED[axis]
            if "pfmatched_top" in name:
                value = 0.28 + 0.02 * prompt + 0.002 * replicate
            elif "pfmatched_bottom" in name:
                value = 0.08 + 0.004 * prompt + 0.001 * replicate
            elif "_top_" in name:
                value = (
                    0.42 + 0.025 * prompt + 0.003 * replicate
                    if matched
                    else 0.14 + 0.006 * prompt + 0.001 * replicate
                )
            elif "_bottom_" in name:
                value = 0.07 + 0.003 * prompt + 0.0005 * replicate
            else:
                value = 0.15 + 0.006 * prompt + 0.001 * replicate
            metric = {
                "relative_rms": value,
                "cosine": 1.0 - value * 0.01,
                "max_abs_delta": value * 2,
            }
            metadata = {
                layer: {
                    "replacement_relative_rms": value * 0.5,
                    **(
                        {"shifted_old_frames": 8}
                        if probe["policy"] in {"key_shift", "value_shift"}
                        else {}
                    ),
                }
                for layer in range(layers)
            }
            downstream.append(
                {
                    "mode": mode,
                    "current_frame": 117,
                    "nominal_timestep": timestep,
                    "probe_name": name,
                    "policy": probe["policy"],
                    "group": probe["group"],
                    "selected_head_count": layers,
                    "flow_metrics": metric,
                    "x0_metrics": metric,
                    "layer_metadata": metadata,
                }
            )
    seed = 5000 + prompt + replicate * 100
    return {
        "version": 8,
        "job": {
            "dataset_index": prompt * 2 + replicate,
            "kind": "v148_axis_core",
            "prompt_slot": prompt,
            "source_prompt_index": prompt * 3,
            "seed_replicate": replicate,
            "seed": seed,
            "reference_seed": seed,
        },
        "metadata": {
            "seed": seed,
            "captured_calls": 2,
            "record_count": 2 * layers,
            "incomplete_calls": [],
            "downstream_probe_plan": {"sha256": digest},
        },
        "records": records,
        "downstream_probe_records": downstream,
        "downstream_probe_expected_count": 2 * (len(plan["probes"]) + 1),
    }


def _dose_profile(prompt, replicate, plan, digest, layers):
    contexts = (("noisy", 1000), ("noisy", 500))
    records = [
        {
            "mode": mode,
            "current_frame": 117,
            "nominal_timestep": timestep,
            "layer": layer,
        }
        for mode, timestep in contexts
        for layer in range(layers)
    ]
    downstream = []
    for mode, timestep in contexts:
        native_metric = {
            "relative_rms": 0.0,
            "cosine": 1.0,
            "max_abs_delta": 0.0,
        }
        downstream.append(
            {
                "mode": mode,
                "current_frame": 117,
                "nominal_timestep": timestep,
                "probe_name": "native_replay",
                "policy": "native",
                "group": "native",
                "selected_head_count": 0,
                "flow_metrics": native_metric,
                "x0_metrics": native_metric,
                "layer_metadata": {},
            }
        )
        for probe in plan["probes"]:
            marker = "_top" if "_top" in probe["group"] else "_bottom"
            dose = int(probe["group"].split(marker, 1)[1])
            bottom = 0.04 * dose + 0.001 * prompt
            ratio = (1.2 + 0.2 * dose) * (
                1.0 + 0.01 * prompt + 0.001 * replicate
            )
            value = bottom * ratio if "_top" in probe["group"] else bottom
            metric = {
                "relative_rms": value,
                "cosine": 1.0 - value * 0.01,
                "max_abs_delta": value * 2,
            }
            metadata = {
                layer: {
                    "replacement_relative_rms": value * 0.5,
                    **(
                        {"shifted_old_frames": 8}
                        if probe["policy"] in {"key_shift", "value_shift"}
                        else {}
                    ),
                }
                for layer in range(layers)
            }
            downstream.append(
                {
                    "mode": mode,
                    "current_frame": 117,
                    "nominal_timestep": timestep,
                    "probe_name": probe["name"],
                    "policy": probe["policy"],
                    "group": probe["group"],
                    "selected_head_count": layers * dose,
                    "flow_metrics": metric,
                    "x0_metrics": metric,
                    "layer_metadata": metadata,
                }
            )
    seed = 6000 + prompt + replicate * 100
    return {
        "version": 8,
        "job": {
            "dataset_index": prompt * 2 + replicate,
            "kind": "v148_axis_dose",
            "prompt_slot": prompt,
            "source_prompt_index": prompt * 3,
            "seed_replicate": replicate,
            "seed": seed,
            "reference_seed": seed,
        },
        "metadata": {
            "seed": seed,
            "captured_calls": 2,
            "record_count": 2 * layers,
            "incomplete_calls": [],
            "downstream_probe_plan": {"sha256": digest},
        },
        "records": records,
        "downstream_probe_records": downstream,
        "downstream_probe_expected_count": 2 * (len(plan["probes"]) + 1),
    }


def test_core_analysis_requires_axis_random_and_pf_controls(
    tmp_path, monkeypatch
):
    prompts, layers = 5, 2
    monkeypatch.setattr(MODULE, "LAYERS", layers)
    monkeypatch.setattr(MODULE, "HEADS", 2)
    monkeypatch.setattr(MODULE, "_expected_prompt_count", lambda plan: prompts)
    plan = _core_plan(layers)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    for prompt in range(prompts):
        for replicate in (0, 1):
            torch.save(
                _profile(prompt, replicate, plan, digest, layers),
                profile_dir / f"{prompt}_{replicate}.pt",
            )
    report = MODULE.analyze(
        profile_dir=profile_dir,
        probe_plan_path=plan_path,
        output_dir=tmp_path / "analysis",
        expected_count=prompts * 2,
    )
    assert report["native_replay_pass"]
    assert report["shift_interventions_non_degenerate"]
    assert all(
        report["gates"]["g1_axis_matched_causal_effect"].values()
    )
    assert all(report["gates"]["g2_pf_independent_effect"].values())
    assert all(report["gates"]["g3_intervention_specificity"].values())
    assert (tmp_path / "analysis" / "axis_comparisons.csv").exists()
    assert (tmp_path / "analysis" / "axis_specificity.csv").exists()


def test_dose_analysis_uses_equal_count_rank_separation(tmp_path, monkeypatch):
    prompts, layers = 5, 2
    monkeypatch.setattr(MODULE, "LAYERS", layers)
    monkeypatch.setattr(MODULE, "HEADS", 2)
    monkeypatch.setattr(MODULE, "_expected_prompt_count", lambda plan: prompts)
    plan = _dose_plan(layers)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    for prompt in range(prompts):
        for replicate in (0, 1):
            torch.save(
                _dose_profile(prompt, replicate, plan, digest, layers),
                profile_dir / f"{prompt}_{replicate}.pt",
            )
    report = MODULE.analyze(
        profile_dir=profile_dir,
        probe_plan_path=plan_path,
        output_dir=tmp_path / "analysis",
        expected_count=prompts * 2,
    )
    assert all(
        report["gates"][
            "g1_positive_rank_separation_at_multiple_doses"
        ].values()
    )
    assert all(report["gates"]["g2_dose4_exceeds_dose1"].values())
