from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_v209_sf_protocol as prepare
import analyze_v209_canary as canary
import run_v209_sf_protocol as runner

spec = importlib.util.spec_from_file_location("dense_cache_trace", ROOT / "src/lifecycle_kv/dense_cache_trace.py")
dense = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dense)


@pytest.mark.parametrize("capacity,sink", [(21, 0), (21, 1), (13, 1), (9, 1), (9, 3)])
def test_dense_sidecar_tracks_nondivisible_budget_and_clean_rewrites(capacity, sink):
    cache = {}
    previous_length = 0
    for start in range(0, 120, 3):
        end = min(capacity, previous_length + 3)
        for _ in range(5):
            observed = dense.observe_write(cache, current_start=start, write_start=end-3,
                                           write_end=end, capacity=capacity, sink=sink)
            assert observed == canary.expected_ids(start + 3, capacity, sink)
        previous_length = end
    assert len(observed) == capacity


def test_sidecar_rejects_late_attachment_and_write_gap():
    with pytest.raises(ValueError, match="start at frame zero"):
        dense.observe_write({}, current_start=3, write_start=0, write_end=3, capacity=21, sink=1)
    with pytest.raises(ValueError, match="write gap"):
        dense.observe_write({}, current_start=0, write_start=1, write_end=3, capacity=21, sink=1)


def test_frozen_configs_change_only_explicit_sf_protocol(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("\n".join(f"MovieGen prompt {i}" for i in range(128)) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    out = tmp_path / "inputs"
    payload = prepare.prepare(ROOT, source, checkpoint, out)
    assert prepare.verify(out / "manifest.json", ROOT) == payload
    assert payload["prompt_items"][16]["source_index"] == 65
    configs = {}
    for method, (budget, sink) in prepare.SPECS.items():
        config = yaml.safe_load(Path(payload["configs"][method]["path"]).read_text(encoding="utf-8"))
        assert config["model_kwargs"].pop("local_attn_size") == budget
        assert config["model_kwargs"].pop("sink_size") == sink
        assert config["compile_ffn"] is False
        assert config["use_pyramidkv"] is False
        assert config["vae_decode_mode"] == "batch"
        configs[method] = config
    assert all(config == configs["sf_fifo21"] for config in configs.values())
    path = Path(payload["configs"]["sf_fifo21"]["path"])
    path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config drift"):
        prepare.verify(out / "manifest.json", ROOT)


def test_environment_scrubs_prior_runtime_and_reference_mode():
    result = runner.scrub_env({"PYRAMIDKV_ROPE_REFERENCE": "1", "SF_PARITY_REFERENCE_ATTENTION": "1", "STRUCTURED_MEMORY_GATE": "0.1", "PATH": "normal"})
    assert result == {"PATH": "normal"}


def test_incomplete_canary_cannot_authorize_screen(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "manifest.json").write_text("{}", encoding="utf-8")
    report = canary.analyze(tmp_path, "production")
    assert report["native_budget_screen_ready"] is False
    assert report["pf_adaptive_runtime_ready"] is False


def test_reference_backend_cannot_authorize_production(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "manifest.json").write_text("{}", encoding="utf-8")
    audit = {"available": True, "pass": True, "worker": {"hostname": "same", "gpu_uuid": "gpu"}}
    monkeypatch.setattr(canary, "load_index", lambda *a, **k: ({}, audit))
    monkeypatch.setattr(canary, "compare", lambda *a, **k: ({"pass": True}, {}))
    report = canary.analyze(tmp_path, "reference")
    assert report["native_budget_screen_ready"] is False
    assert report["pf_adaptive_runtime_ready"] is False


def test_native_screen_can_advance_while_adaptive_is_incomplete(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "manifest.json").write_text("{}", encoding="utf-8")
    def load(job, *args, **kwargs):
        available = job.parent.name.startswith("sf_")
        return {}, {"available": available, "pass": available, "worker": {"hostname": "same", "gpu_uuid": "gpu"}}
    monkeypatch.setattr(canary, "load_index", load)
    monkeypatch.setattr(canary, "compare", lambda *a, **k: ({"pass": True}, {}))
    report = canary.analyze(tmp_path, "production")
    assert report["native_budget_screen_ready"] is True
    assert report["pf_adaptive_runtime_ready"] is False


def test_stream_comparison_ignores_run_label_but_detects_tensor_change(tmp_path):
    torch = pytest.importorskip("torch")
    jobs, indices = [], []
    for name, value in (("left", 1.0), ("right", 1.0)):
        job = tmp_path / name
        (job / "trace").mkdir(parents=True)
        tensor = torch.tensor([value])
        payload = {"tensors": {"latent": {"full": tensor, "sample": tensor}}}
        torch.save(payload, job / "trace/a.pt")
        row = {"event": "final_latents", "counter": 0, "context": {"runtime": name}, "metadata": {}, "file": "a.pt"}
        jobs.append(job)
        indices.append({canary.identity(row): row})
    result, _ = canary.compare(*jobs, *indices, None)
    assert result["pass"] is True
    payload["tensors"]["latent"]["full"] = torch.tensor([2.0])
    torch.save(payload, jobs[1] / "trace/a.pt")
    result, _ = canary.compare(*jobs, *indices, None)
    assert result["pass"] is False
    assert result["first_divergence"]["event"] == "final_latents"


def test_budget_analysis_retains_both_sf_controls():
    import analyze_v209_budget as budget

    rows = {(method, p): {metric: 1.0 for metric in budget.base.ANALYSIS_METRICS}
            for method in prepare.METHODS for p in range(32)}
    temporal = {(method, p): {feature: 0.0 for feature in budget.temporal.TEMPORAL_FEATURES}
                for method in prepare.METHODS for p in range(32)}
    report = budget.analyze({window: rows for window in budget.base.WINDOWS}, temporal)
    assert report["future_addon_controls"] == ["sf_fifo21", "sf_sink1_21"]
    assert set(report["baseline_axis_directions"].values()) == {"inconclusive"}
    assert report["new_method_efficacy_established"] is False
