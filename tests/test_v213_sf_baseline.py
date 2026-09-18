from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v212_lphc_protocol as previous
import v213_lphc_protocol as protocol
import v213_sf_baseline_contract as contract
import run_v213_sf_baseline as runner
import run_v212_lphc as generation


def test_production_config_matches_sampling_references(tmp_path):
    base = ROOT / "third_party/Self-Forcing/configs"
    config = previous.merged(yaml.safe_load((base / "default_config.yaml").read_text(encoding="utf-8")),
                             yaml.safe_load((base / "self_forcing_dmd.yaml").read_text(encoding="utf-8")))
    config.update(use_pyramidkv=False, use_teacache=False, compile_ffn=False,
                  vae_decode_mode="batch", independent_first_frame=False, few_step_cfg_enabled=False)
    config["model_kwargs"].update(local_attn_size=21, sink_size=0)
    frozen = tmp_path / "sf.yaml"
    frozen.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = contract.config_alignment(ROOT, ROOT / "third_party/Self-Forcing", frozen)
    assert report["pass"] and set(report["references"]) == {"official_sf", "vendored_pf_plain_sf"}
    for key, value in (("denoising_step_list", [1000, 500]), ("few_step_cfg_enabled", True),
                       ("context_noise", 1), ("vae_decode_mode", "stream")):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="unexpected"):
            contract.validate_config(changed)
    config["model_kwargs"]["sink_size"] = 1
    with pytest.raises(ValueError, match="model_kwargs"):
        contract.validate_config(config)


def test_reference_is_pinned_pristine_and_complete(tmp_path, monkeypatch):
    files = ("pipeline/causal_inference.py", "wan/modules/causal_model.py")
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    replies = {("rev-parse", "HEAD"): contract.UPSTREAM_COMMIT.encode(),
               ("status", "--porcelain", "--untracked-files=no"): b"",
               ("ls-files", "-z"): ("\0".join(files) + "\0").encode()}
    monkeypatch.setattr(contract.subprocess, "check_output", lambda cmd, **kw: replies[tuple(cmd[1:])])
    assert set(contract.reference_inventory(tmp_path)["files"]) == set(files)
    extra = tmp_path / "pipeline/untracked.py"
    extra.write_text("# injected\n")
    with pytest.raises(ValueError, match="untracked Python"):
        contract.reference_inventory(tmp_path)
    replies[("rev-parse", "HEAD")] = b"wrong"
    with pytest.raises(ValueError, match="pinned"):
        contract.reference_inventory(tmp_path)


def make_trace(job, *, modify=None, gpu="uuid1", seed=21303):
    job.mkdir(parents=True)
    rows = []
    for event, index in sorted(contract.expected_records()):
        arrays = {"latent": np.array([1., 2.], dtype=np.float32)}
        if event == "generator_input":
            arrays["rng"] = np.array([1., 2.], dtype=np.float32)
            arrays["timestep"] = np.array([750.], dtype=np.float32)
        if event == "generator_output":
            arrays["k_0"] = np.array([1., 2.], dtype=np.float32)
        if modify:
            modify(event, index, arrays)
        path = job / f"{event}_{index:03d}.npz"
        np.savez(path, **arrays)
        rows.append({"event": event, "index": index, "file": path.name,
                     "sha256": previous.sha256(path), "metadata": {"index": index},
                     "tensors": {k: {"shape": list(v.shape), "dtype": "torch.float32", "full": True}
                                 for k, v in arrays.items()}})
    previous.frozen_json(job / "events.json", {"events": rows})
    previous.frozen_json(job / "done.json", {
        "counts": contract.COUNTS, "contract_sha256": "contract", "grad_enabled": False,
        "events_sha256": previous.sha256(job / "events.json"), "gpu": {"uuid": gpu},
        "torch": "fixture", "cuda": "fixture", "source": 3, "effective_seed": seed})
    return job


def test_all_events_pair_and_decode_is_diagnostic(tmp_path):
    left = make_trace(tmp_path / "a")
    def modify(event, index, arrays):
        if event == "decoded_sample":
            arrays["latent"] *= 2
    right = make_trace(tmp_path / "b", modify=modify)
    report = runner.compare(left, right, "contract")
    assert report["pass"] and report["compared_fields"] > 100
    assert report["decoded_diagnostic_only"][0]["pass"] is False
    assert report["relative_tolerance"] == 1e-5


@pytest.mark.parametrize("event,field", [("generator_input", "rng"), ("generator_input", "timestep"),
                                        ("generator_output", "k_0"), ("final_latents", "latent")])
def test_numerical_divergence_is_not_hidden(tmp_path, event, field):
    left = make_trace(tmp_path / "a")
    def modify(name, index, arrays):
        if name == event and index == 0:
            arrays[field] += .1
    right = make_trace(tmp_path / "b", modify=modify)
    report = runner.compare(left, right, "contract")
    assert not report["pass"]
    assert report["first_failures"][0]["tensor"] == field


def test_corrupt_coverage_and_hardware_are_rejected(tmp_path):
    left = make_trace(tmp_path / "a")
    right = make_trace(tmp_path / "b", gpu="uuid2")
    with pytest.raises(ValueError, match="pairing"):
        runner.compare(left, right, "contract")
    (left / "input_noise_000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.load_trace(left, "contract")
    with pytest.raises(ValueError, match="nonfinite"):
        runner.array_metrics(np.array([np.nan]), np.array([1.]))
    with pytest.raises(ValueError, match="shape"):
        runner.array_metrics(np.ones(2), np.ones(3))


def test_exact_rng_not_relaxed_by_float_tolerance():
    a, b = np.array([1.]), np.array([1. + 1e-6])
    assert runner.array_metrics(a, b)["pass"]
    assert not runner.array_metrics(a, b, exact=True)["pass"]


def test_certificate_requires_complete_receipts(tmp_path):
    out = tmp_path / "v213_test"
    previous.frozen_json(out / "inputs/manifest.json", {"test": True})
    previous.frozen_json(out / "baseline/contract.json", {"test": True})
    with pytest.raises(ValueError, match="run v213 baseline"):
        contract.require_baseline(out)
    report = {"pass": True, "upstream_commit": contract.UPSTREAM_COMMIT,
              "input_manifest_sha256": previous.sha256(out / "inputs/manifest.json"),
              "contract_sha256": previous.sha256(out / "baseline/contract.json"),
              "jobs": [], "comparisons": []}
    path = out / "decisions/sf_upstream_gate.json"
    previous.frozen_json(path, report)
    with pytest.raises(ValueError, match="coverage"):
        contract.require_baseline(out)
    for source in contract.SOURCES:
        for mode in contract.MODES:
            job = out / "baseline/jobs" / f"source_{source:03d}" / mode / "done.json"
            previous.frozen_json(job, {"source": source, "mode": mode})
            report["jobs"].append({"source": source, "mode": mode, "path": str(job), "sha256": previous.sha256(job)})
        for kind in ("official_repeat", "local_vs_official"):
            report["comparisons"].append({"source": source, "kind": kind, "pass": True})
    path.write_text(json.dumps(report))
    assert contract.require_baseline(out)["pass"]
    job.write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        contract.require_baseline(out)


def test_v213_cannot_start_gate0_without_reference(tmp_path):
    assert not hasattr(previous, "require_baseline")
    with pytest.raises(ValueError, match="run v213 baseline"):
        generation.gate0(ROOT, tmp_path, {}, "0", protocol=protocol)


def test_coverage_cannot_be_shortened(tmp_path):
    job = make_trace(tmp_path / "a")
    rows = json.loads((job / "events.json").read_text())
    rows["events"].pop()
    (job / "events.json").write_text(json.dumps(rows))
    done = json.loads((job / "done.json").read_text())
    done["events_sha256"] = previous.sha256(job / "events.json")
    (job / "done.json").write_text(json.dumps(done))
    with pytest.raises(ValueError, match="coverage"):
        runner.load_trace(job, "contract")
