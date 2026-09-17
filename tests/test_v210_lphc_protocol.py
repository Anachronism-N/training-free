from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_v210_lphc_trace as auditor
import prepare_v210_lphc as prepare
import run_v210_lphc as controller
import run_v210_worker as worker


@pytest.fixture(autouse=True)
def allow_test_checkout(monkeypatch):
    monkeypatch.setattr(prepare, "require_clean_checkout", lambda root: None)


def prepared(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "MovieGen_128_qwen.txt"
    source.write_text("\n".join(f"MovieGen prompt {index}" for index in range(128)) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "self_forcing_dmd.pt"
    checkpoint.write_bytes(b"v210 test checkpoint")
    output_root = tmp_path / "v210"
    wan_model = tmp_path / "Wan2.1-T2V-1.3B"
    wan_model.mkdir()
    for relative in prepare.WAN_REQUIRED_FILES:
        path = wan_model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{relative}".encode())
    tokenizer = wan_model / prepare.WAN_REQUIRED_DIRECTORIES[0] / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_text("{}", encoding="utf-8")
    (wan_model / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    payload = prepare.prepare(
        ROOT,
        source,
        checkpoint,
        output_root / "inputs",
        prepare.AUTHORIZED_NODES,
        wan_model,
        require_clean=False,
    )
    return output_root, payload


def test_source_indices_and_effective_seeds_are_frozen():
    assert prepare.SOURCE_INDICES == (1, 17, 33, 49, 65, 81, 97, 113)
    assert [prepare.effective_seed(index) for index in prepare.SOURCE_INDICES] == [
        21001, 21017, 21033, 21049, 21065, 21081, 21097, 21113
    ]
    with pytest.raises(ValueError, match="outside"):
        prepare.effective_seed(0)


def test_default_root_binds_short_commit_and_rejects_immutable_v209():
    assert prepare.short_git_commit(ROOT) in str(prepare.default_output_root(ROOT))
    with pytest.raises(ValueError, match="immutable"):
        prepare.validate_output_root(Path("runs/v209_sf_protocol_budget_86f10607_sixnode"))
    with pytest.raises(ValueError, match="immutable"):
        prepare.validate_output_root(Path("/tmp/copy-v209_sf_protocol_budget_86f10607_sixnode-retry"))


def test_prepare_freezes_prompt_config_checkpoint_runtime_and_nodes(tmp_path):
    output_root, payload = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    assert prepare.verify(manifest_path, ROOT) == payload
    assert payload["source_commit"] == prepare.git_commit(ROOT)
    assert len(payload["authorized_nodes"]) == 6
    assert tuple(payload["authorized_nodes"]) == prepare.AUTHORIZED_NODES
    assert payload["checkpoint"]["sha256"] == prepare.sha256(Path(payload["checkpoint"]["path"]))
    assert payload["prompt_source"]["count"] == 128
    assert len(payload["prompt_items"]) == 8
    assert all(Path(item["path"]).read_text().count("\n") == 1 for item in payload["prompt_items"])
    assert payload["wan_model"]["runtime_present"] is True
    assert payload["wan_model"]["weights_present"] is True
    assert payload["wan_model"]["inventory"]
    assert all({"relative_path", "bytes", "mtime_ns", "sha256"} <= set(row) for row in payload["wan_model"]["inventory"])
    assert "scripts/run_v210_worker.py" in payload["runtime_paths"]
    assert "src/lifecycle_kv/lphc.py" in payload["runtime_paths"]
    for method in prepare.METHODS:
        config = yaml.safe_load(Path(payload["configs"][method]["path"]).read_text(encoding="utf-8"))
        spec = prepare.METHOD_SPECS[method]
        assert config["model_kwargs"]["local_attn_size"] == spec["local_attn_size"]
        assert config["model_kwargs"]["sink_size"] == spec["sink_size"]
        assert config["lphc"]["archive_frames"] == 12
        assert config["lphc"]["history_frames"] == 4


def test_method_matrix_and_gate0_contract_are_exact():
    assert prepare.METHODS == (
        "sf_fifo21", "sf_sink1_21", "sf_fifo25", "lphc_e1_a002_correct",
        "lphc_e1_a005_correct", "lphc_e1_a010_correct",
        "lphc_e2_a010_correct", "lphc_full_a010_correct",
    )
    assert prepare.GATE0_SOURCE_INDICES == (1, 65)
    assert set(prepare.GATE0_METHODS) == {"native", "lphc_alpha0"}
    assert prepare.GATE0_METHODS["lphc_alpha0"]["alpha"] == 0.0
    assert prepare.SCREEN_FRAMES == 120
    assert prepare.GATE_FRAMES == 30


def test_verify_detects_config_prompt_and_source_commit_drift(monkeypatch, tmp_path):
    output_root, payload = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    config = Path(payload["configs"]["sf_fifo21"]["path"])
    config.write_text(config.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config drift"):
        prepare.verify(manifest_path, ROOT)
    config.write_text(config.read_text(encoding="utf-8").removesuffix("# drift\n"), encoding="utf-8")
    model_file = Path(payload["wan_model"]["weights_path"]) / payload["wan_model"]["inventory"][0]["relative_path"]
    original = model_file.read_bytes()
    model_file.write_bytes(original + b"drift")
    with pytest.raises(ValueError, match="Wan model stamp drift"):
        prepare.verify(manifest_path, ROOT)
    model_file.write_bytes(original)
    monkeypatch.setattr(prepare, "git_commit", lambda root: "0" * 40)
    with pytest.raises(ValueError, match="source commit drift"):
        prepare.verify(manifest_path, ROOT)


def test_environment_scrubbing_and_lphc_binding():
    dirty = {
        "PATH": "/bin",
        "PYRAMIDKV_MODE": "x",
        "LIFECACHE_ENABLE": "1",
        "STRUCTURED_MEMORY_GATE": "1",
        "SF_PARITY_REFERENCE_ATTENTION": "1",
        "LPHC_ALPHA": "9",
    }
    assert worker.scrub_env(dirty) == {"PATH": "/bin"}
    env = worker.lphc_environment(
        {"lphc": True, "alpha": 0.1, "phase": "full", "retrieval_mode": "correct"},
        Path("trace.jsonl"),
    )
    assert env["LPHC_ALPHA"] == "0.1"
    assert env["LPHC_ARCHIVE_FRAMES"] == "12"
    assert env["LPHC_HISTORY_FRAMES"] == "4"
    assert worker.lphc_environment({"lphc": False}, Path("unused")) == {}


def test_exact_node_allowlist_and_six_node_schedule(monkeypatch):
    with pytest.raises(ValueError, match="frozen six-IP"):
        prepare._normalize_nodes(["only-one"])
    manifest = {"authorized_nodes": list(prepare.AUTHORIZED_NODES)}
    monkeypatch.delenv("V210_NODE_ADDRESS", raising=False)
    with pytest.raises(PermissionError, match="required"):
        worker.assert_authorized_node(manifest, hostname="TENCENT64.site")
    address = prepare.AUTHORIZED_NODES[0]
    assert worker.assert_authorized_node(
        manifest, address, interface_addresses=frozenset({address})
    ) == address
    with pytest.raises(PermissionError, match="local network interface"):
        worker.assert_authorized_node(
            manifest, address, interface_addresses=frozenset({"127.0.0.1"})
        )
    for forbidden in prepare.FORBIDDEN_NODES:
        with pytest.raises(PermissionError, match="allowlist"):
            worker.assert_authorized_node(
                manifest, forbidden, interface_addresses=frozenset({forbidden})
            )
    gpus = tuple(str(index) for index in range(8))
    assignments = [controller.screen_assignments(rank, gpus) for rank in range(6)]
    jobs = [job for assignment in assignments for lane in assignment.values() for job in lane]
    assert len(jobs) == len(prepare.METHODS) * len(prepare.SOURCE_INDICES)
    assert len(set(jobs)) == len(jobs)
    assert controller.screen_assignments(2, gpus) == assignments[2]


def test_ssh_launcher_targets_only_frozen_nodes_and_sets_identity(tmp_path):
    _, manifest = prepared(tmp_path)
    commands = controller.screen_ssh_commands(manifest, tmp_path / "out", "0,1,2,3,4,5,6,7")
    assert len(commands) == 6
    targets = [command[-2] for command in commands]
    assert targets == [f"root@{node}" for node in prepare.AUTHORIZED_NODES]
    assert not any(forbidden in " ".join(command) for forbidden in prepare.FORBIDDEN_NODES for command in commands)
    for rank, (node, command) in enumerate(zip(prepare.AUTHORIZED_NODES, commands)):
        rendered = " ".join(command)
        assert command[2] == "36000"
        assert f"V210_NODE_ADDRESS={node}" in rendered
        assert f"NODE_RANK={rank}" in rendered
        assert controller.CONDA_ACTIVATION in rendered


def test_job_stamp_and_completion_marker_bind_identity(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "source_commit": "abc",
        "method_specs": prepare.METHOD_SPECS,
    }
    stamp = worker.make_stamp(manifest, manifest_path, "screen8", "sf_fifo21", 17)
    assert stamp == {
        "stage": "screen8", "method": "sf_fifo21", "source_index": 17,
        "effective_seed": 21017, "input_manifest_sha256": prepare.sha256(manifest_path),
        "source_commit": "abc",
        "requires_lphc_trace": False,
    }
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    done = {
        "stamp": stamp,
        "contract_sha256": stamp["input_manifest_sha256"],
        "media": {"path": str(media), "sha256": prepare.sha256(media), "validation": {"valid": True}},
    }
    assert worker.done_matches(done, stamp)
    assert not worker.done_matches(done, {**stamp, "effective_seed": 999})
    media.write_bytes(b"changed")
    assert not worker.done_matches(done, stamp)


def write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_trace_auditor_accepts_bounded_protocol(tmp_path):
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [{
        "event": "attention_call", "call_kind": "noisy", "phase_index": 0, "layer_idx": 0,
        "history_source": "noisy",
        "clean_history_reads": 0, "archive_frames": 12,
        "local_frame_indices": [20, 21, 22],
        "eligible_frame_indices": [1, 5, 9, 13],
        "selected_history_frames": [1, 5, 9, 13],
        "correction_ratio": 0.099, "lookup_count": 1, "random_count": 0,
        "second_attention_count": 1,
    }])
    report = auditor.audit_trace(trace, 0.1, expect_second_attention=True)
    assert report["pass"] is True
    assert report["maximums"]["selected"] == 4
    assert report["totals"] == {"lookup": 1, "random": 0, "second_attention": 1}


def test_trace_auditor_requires_enabled_lookup_selection_and_second_attention(tmp_path):
    trace = tmp_path / "inactive.jsonl"
    write_trace(trace, [{
        "event": "attention_call", "call_kind": "noisy", "phase_index": 0,
        "local_frame_indices": [20], "eligible_frame_indices": [], "archive_frames": 4,
        "selected_history_frames": [], "selected_count": 0, "clean_history_reads": 0,
        "correction_ratio": 0.0, "lookup_count": 0, "random_count": 0,
        "second_attention_count": 0,
    }])
    report = auditor.audit_trace(trace, 0.1, expect_second_attention=True, phase="e1")
    assert report["pass"] is False
    joined = "\n".join(report["errors"])
    assert "lookup" in joined and "eligible and selected" in joined and "second attention" in joined


def test_trace_auditor_rejects_activity_on_clean_or_disabled_phase(tmp_path):
    trace = tmp_path / "wrong-phase.jsonl"
    write_trace(trace, [{
        "event": "attention_call", "call_kind": "clean", "phase_index": 4,
        "local_frame_indices": [20], "eligible_frame_indices": [1], "archive_frames": 4,
        "selected_history_frames": [1], "selected_count": 1, "clean_history_reads": 0,
        "correction_ratio": 0.01, "lookup_count": 1, "random_count": 0,
        "second_attention_count": 1,
    }])
    report = auditor.audit_trace(trace, 0.1, expect_second_attention=True, phase="e1")
    assert report["pass"] is False
    assert any("outside expected phase" in error for error in report["errors"])


def test_trace_auditor_rejects_each_invariant_and_alpha0_activity(tmp_path):
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [{
        "event": "attention_call", "call_kind": "noisy", "history_source": "clean", "archive_size": 13,
        "local_frames": [7, 8], "selected_frames": [1, 2, 3, 7, 9],
        "correction_ratio": float("inf"), "lookup_count": 1,
        "random_count": 1, "second_attention_count": 1,
    }])
    report = auditor.audit_trace(trace, 0.0, expect_second_attention=False)
    assert report["pass"] is False
    joined = "\n".join(report["errors"])
    assert "clean history" in joined
    assert "overlap" in joined
    assert "archive size" in joined
    assert "selected" in joined
    assert "not finite" in joined
    assert "unexpected second" in joined
    assert "alpha0" in joined


def test_gate0_compares_full_tensors_not_only_media(tmp_path):
    torch = pytest.importorskip("torch")

    def make(name: str, final_value: float) -> dict:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "trace_meta.json").write_text(json.dumps({
            "contract_sha256": "contract", "reference_attention": False,
            "trace_layers": [0],
        }), encoding="utf-8")
        rows = []
        counter = 0
        for event, count in controller.GATE_EVENT_COUNTS.items():
            for occurrence in range(count):
                value = final_value if event == "final_latents" else 1.0
                file = f"{counter}.pt"
                torch.save({"tensors": {"latent": {"full": torch.tensor([value])}}}, directory / file)
                rows.append({
                    "event": event,
                    "counter": counter,
                    "context": {"video_index": 0, "call_index": occurrence},
                    "metadata": {},
                    "file": file,
                })
                counter += 1
        (directory / "events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return {"contract_sha256": "contract", "tensor_trace": {"path": str(directory), "present": True}}

    native = make("native", 2.0)
    assert controller.compare_gate_tensors(native, make("same", 2.0))["pass"] is True
    assert controller.compare_gate_tensors(native, make("different", 3.0))["pass"] is False


def test_smoke_is_one_full_length_lphc_job(tmp_path, monkeypatch):
    output_root, manifest = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    trace = tmp_path / "smoke.jsonl"
    write_trace(trace, [{
        "event": "attention_call", "call_kind": "noisy", "phase_index": 0, "layer_idx": 0,
        "clean_history_reads": 0, "local_frame_indices": [20, 21], "archive_frames": 4,
        "eligible_frame_indices": [1], "selected_history_frames": [1], "correction_ratio": 0.05,
        "lookup_count": 1, "random_count": 0, "second_attention_count": 1,
    }])
    launched = []
    monkeypatch.setattr(
        controller,
        "assert_authorized_node",
        lambda manifest, address=None: prepare.AUTHORIZED_NODES[0],
    )
    monkeypatch.setattr(controller, "require_decision", lambda *args: {"pass": True})
    monkeypatch.setattr(controller, "launch_worker", lambda *args: launched.append(args[3:]))
    monkeypatch.setattr(controller, "load_done", lambda *args: {"trace": {"path": str(trace), "present": True}})
    report = controller.run_smoke(ROOT, output_root, manifest_path, manifest, "0")
    assert launched == [(controller.SMOKE_METHOD, 1, "0")]
    assert report["latent_frames"] == 120
    assert report["pass"] is True


def test_stage_decisions_bind_manifest_and_block_screen(tmp_path, monkeypatch):
    output_root, manifest = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    report = {
        "pass": True,
        "input_manifest_sha256": prepare.sha256(manifest_path),
        "source_commit": manifest["source_commit"],
    }
    gate = controller.decision_path(output_root, "gate0")
    controller.write_decision(gate, report)
    assert controller.valid_decision(gate, manifest_path, manifest)
    stale = dict(report, input_manifest_sha256="bad")
    smoke = controller.decision_path(output_root, "smoke")
    smoke.parent.mkdir(parents=True, exist_ok=True)
    smoke.write_text(json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(
        controller,
        "assert_authorized_node",
        lambda manifest, address=None: prepare.AUTHORIZED_NODES[0],
    )
    with pytest.raises(RuntimeError, match="smoke decision"):
        controller.run_screen8(
            ROOT,
            output_root,
            manifest_path,
            manifest,
            tuple(str(index) for index in range(8)),
            0,
        )


def test_recover_only_quarantines_incomplete_jobs_inside_v210_root(tmp_path):
    output_root, manifest = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    incomplete = controller.job_path(output_root, "screen8", "sf_fifo21", 1)
    incomplete.mkdir(parents=True)
    (incomplete / "stdout.log").write_text("partial", encoding="utf-8")
    quarantined = controller.recover(output_root, manifest_path, manifest)
    assert len(quarantined) == 1
    target = Path(quarantined[0])
    assert target.is_relative_to(output_root)
    assert (target / "quarantine.json").is_file()
    assert not incomplete.exists()
