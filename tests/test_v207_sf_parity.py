from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import prepare_v207_sf_parity as prepare  # noqa: E402

try:  # The lightweight local audit environment does not bundle PyTorch.
    import torch

    import compare_v207_sf_parity as compare  # noqa: E402
    from lifecycle_kv import parity_trace  # noqa: E402

    HAS_TORCH = True
except ImportError:
    torch = None
    compare = None
    parity_trace = None
    HAS_TORCH = False


def _source(path: Path) -> None:
    path.write_text(
        "\n".join(f"Qwen rewrite prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )


def test_parity_manifest_binds_prompt_checkpoint_and_runtime(tmp_path: Path) -> None:
    source = tmp_path / "moviegen128.txt"
    checkpoint = tmp_path / "checkpoint.pt"
    sf_config = tmp_path / "sf.yaml"
    plain_config = tmp_path / "plain.yaml"
    adaptive_config = tmp_path / "adaptive.yaml"
    _source(source)
    checkpoint.write_bytes(b"checkpoint")
    sf_config.write_text("kind: sf\n", encoding="utf-8")
    plain_config.write_text("kind: plain\n", encoding="utf-8")
    adaptive_config.write_text("kind: adaptive\n", encoding="utf-8")
    output = tmp_path / "inputs"

    payload = prepare.prepare(
        repo_root=ROOT,
        source_prompts=source,
        checkpoint=checkpoint,
        sf_config=sf_config,
        pf_plain_config=plain_config,
        pf_adaptive_config=adaptive_config,
        output_root=output,
    )
    verified = prepare.verify(output / "manifest.json", ROOT)

    assert verified == payload
    assert payload["run_order"] == list(prepare.RUN_ORDER)
    assert payload["num_output_frames"] == 9
    assert payload["num_ar_blocks"] == 3
    assert payload["prompt_text"] == "Qwen rewrite prompt 3"
    assert payload["adaptive_recent21_contract"]["read_frame_equivalents"] == 21


@pytest.mark.skipif(not HAS_TORCH, reason="parity tensor trace requires torch")
def test_opt_in_trace_is_inert_then_writes_full_pipeline_tensor(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SF_PARITY_TRACE_DIR", raising=False)
    parity_trace.reset_for_tests()
    assert parity_trace.enabled() is False

    directory = tmp_path / "trace"
    monkeypatch.setenv("SF_PARITY_TRACE_DIR", str(directory))
    monkeypatch.setenv("SF_PARITY_RUN_KIND", "unit")
    parity_trace.reset_for_tests()
    parity_trace.set_context(ar_block=0, call_kind="noisy", call_index=0)
    value = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    parity_trace.record_event("input_noise", {"noise": value})

    rows = [json.loads(row) for row in (directory / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    payload = torch.load(directory / rows[0]["file"], weights_only=False)
    assert torch.equal(payload["tensors"]["noise"]["full"], value)
    assert payload["context"]["call_index"] == 0
    parity_trace.reset_for_tests()


def _write_single_event_trace(directory: Path, value: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "trace_meta.json").write_text("{}\n", encoding="utf-8")
    tensor = torch.tensor([value], dtype=torch.float32)
    payload = {
        "event": "input_noise",
        "context": {
            "video_index": 0,
            "runtime": directory.name,
            "ar_block": None,
            "current_start_frame": None,
            "call_kind": None,
            "call_index": None,
            "call_count": None,
            "timestep": None,
        },
        "metadata": {},
        "tensors": {
            "noise": {
                "shape": (1,),
                "sample": tensor,
                "full": tensor,
            }
        },
    }
    torch.save(payload, directory / "000000_input_noise.pt")
    row = {
        "event": "input_noise",
        "context": payload["context"],
        "metadata": {},
        "file": "000000_input_noise.pt",
    }
    (directory / "events.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="parity tensor trace requires torch")
def test_comparison_separates_plain_runtime_from_adaptive_failure(
    tmp_path: Path, monkeypatch
) -> None:
    trace_root = tmp_path / "traces"
    values = {
        "sf_native_a": 1.0,
        "sf_native_b": 1.0,
        "pf_plain_sf21": 1.0,
        "adaptive_recent21": 2.0,
    }
    for run, value in values.items():
        _write_single_event_trace(trace_root / run, value)
    monkeypatch.setattr(compare, "EXPECTED_COUNTS", {"input_noise": 1})
    monkeypatch.setattr(compare, "GATED_EVENTS", {"input_noise"})

    report = compare.analyze(
        trace_root=trace_root,
        output_dir=tmp_path / "report",
        relative_tolerance=1e-5,
        absolute_tolerance=5e-4,
        floor_multiplier=5.0,
    )

    assert report["comparisons"]["vendored_runtime_parity"]["pass"] is True
    assert report["comparisons"]["adaptive_cache_parity"]["pass"] is False
    assert report["decision"] == "stop_fix_adaptive_recent21_before_large_scale"
    first = report["comparisons"]["adaptive_cache_parity"]["first_divergence"]
    assert first["event"] == "input_noise"


def test_runtime_hooks_and_large_screen_gate_are_present() -> None:
    runner = (SCRIPTS / "run_v207_sf_parity.sh").read_text(encoding="utf-8")
    screen = (SCRIPTS / "run_v207_context_budget_phase_screen_32gpu.sh").read_text(
        encoding="utf-8"
    )
    sf_pipeline = (
        ROOT / "third_party" / "Self-Forcing" / "pipeline" / "causal_inference.py"
    ).read_text(encoding="utf-8")
    pf_core = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "wan"
        / "modules"
        / "attention"
        / "core.py"
    ).read_text(encoding="utf-8")

    assert "sf_native_a" in runner and "sf_native_b" in runner
    assert "pf_plain_sf21" in runner and "adaptive_recent21" in runner
    assert "--num_output_frames \"$FRAMES\"" in runner
    assert "require_sf_parity" in screen
    assert '"flow_prediction"' in sf_pipeline
    assert "record_varlen_cache_readout" in pf_core
