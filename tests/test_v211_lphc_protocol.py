from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_v211_lphc_trace as auditor
import prepare_v211_lphc as prepare
import run_v211_lphc as controller
import run_v211_worker as worker


@pytest.fixture(autouse=True)
def allow_test_checkout(monkeypatch):
    monkeypatch.setattr(prepare, "require_clean_checkout", lambda root: None)


def prepared(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "MovieGen_128_qwen.txt"
    source.write_text(
        "\n".join(f"MovieGen prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "self_forcing_dmd.pt"
    checkpoint.write_bytes(b"v211 test checkpoint")
    output_root = tmp_path / "v211"
    wan_model = tmp_path / "Wan2.1-T2V-1.3B"
    wan_model.mkdir(exist_ok=True)
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


def test_frozen_source_suites_and_effective_seeds():
    assert prepare.EXPERIMENT == "v211_lphc_sink_followup"
    assert prepare.DEV32_SOURCE_INDICES == tuple(range(3, 128, 4))
    assert prepare.SCREEN8_SOURCE_INDICES == (3, 19, 35, 51, 67, 83, 99, 115)
    assert prepare.SOURCE_INDICES == prepare.SCREEN8_SOURCE_INDICES
    assert prepare.GATE0_SOURCE_INDICES == (3, 67)
    assert prepare.BASE_SEED == 21100
    assert [prepare.effective_seed(index) for index in prepare.SCREEN8_SOURCE_INDICES] == [
        21103, 21119, 21135, 21151, 21167, 21183, 21199, 21215
    ]
    assert prepare.effective_seed(7) == 21107
    with pytest.raises(ValueError, match="outside"):
        prepare.effective_seed(4)


def test_default_root_binds_full_commit_and_rejects_legacy_roots():
    assert prepare.DEFAULT_WAN_MODEL == Path(
        "/apdcephfs_gy2/share_302533218/cedricnie/model_cache/Wan2.1-T2V-1.3B"
    )
    commit = prepare.git_commit(ROOT)
    default = prepare.default_output_root(ROOT)
    assert commit in default.name
    assert prepare.short_git_commit(ROOT) != commit
    for path in (
        Path("runs/v209_sf_protocol_budget_86f10607_sixnode"),
        Path("runs/v210_lphc_deadbeef_sixnode"),
        Path("/tmp/copy-v210-retry"),
    ):
        with pytest.raises(ValueError, match="v209/v210"):
            prepare.validate_output_root(path)


def test_method_matrix_gate0_and_frame_contract_are_exact():
    assert prepare.METHODS == (
        "sf_fifo21",
        "sf_sink1_21",
        "lphc_sink1_e1_a002_r4",
        "lphc_sink1_e1_a010_r4",
    )
    assert prepare.METHOD_SPECS["sf_fifo21"] == {
        "local_attn_size": 21, "sink_size": 0, "lphc": False
    }
    assert prepare.METHOD_SPECS["sf_sink1_21"] == {
        "local_attn_size": 21, "sink_size": 1, "lphc": False
    }
    a002 = prepare.METHOD_SPECS["lphc_sink1_e1_a002_r4"]
    a010 = prepare.METHOD_SPECS["lphc_sink1_e1_a010_r4"]
    assert (a002["sink_size"], a002["phase"], a002["alpha"], a002["history_frames"]) == (1, "e1", 0.02, 4)
    assert (a010["sink_size"], a010["phase"], a010["alpha"], a010["history_frames"]) == (1, "e1", 0.10, 4)
    for spec in (a002, a010):
        assert spec["local_attn_size"] == 21
        assert spec["archive_frames"] == 12
        assert spec["retrieval_mode"] == "correct"
        assert spec["protocol"] == "v211"
        assert spec["local_policy"] == "sink1_recent20"
    assert tuple(prepare.GATE0_METHODS) == ("sf_sink1_21_native", "sink1_lphc_alpha0")
    assert prepare.GATE0_METHODS["sf_sink1_21_native"]["config"] == "sf_sink1_21"
    assert prepare.GATE0_METHODS["sink1_lphc_alpha0"]["alpha"] == 0.0
    assert prepare.SCREEN_FRAMES == 120
    assert prepare.GATE_FRAMES == 30


def test_prepare_freezes_manifest_source_runtime_configs_and_model(tmp_path):
    output_root, payload = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    assert prepare.verify(manifest_path, ROOT) == payload
    assert payload["source_commit"] == prepare.git_commit(ROOT)
    assert payload["dev32_source_indices"] == list(prepare.DEV32_SOURCE_INDICES)
    assert payload["screen8_source_indices"] == list(prepare.SCREEN8_SOURCE_INDICES)
    assert payload["source_indices"] == list(prepare.SCREEN8_SOURCE_INDICES)
    assert len(payload["prompt_items"]) == 8
    assert tuple(payload["authorized_nodes"]) == prepare.AUTHORIZED_NODES
    assert payload["execution"]["gpu_slots"] == [str(index) for index in range(8)]
    assert payload["checkpoint"]["sha256"] == prepare.sha256(Path(payload["checkpoint"]["path"]))
    assert payload["prompt_source"]["count"] == 128
    assert payload["wan_model"]["inventory"]
    for name in prepare.V211_RUNTIME_FILES:
        assert name in payload["runtime_paths"]
    for name in (
        "third_party/Self-Forcing/inference.py",
        "third_party/Self-Forcing/pipeline/causal_inference.py",
        "third_party/Self-Forcing/wan/modules/causal_model.py",
    ):
        assert name in payload["runtime_paths"]
    for method in prepare.METHODS:
        config = yaml.safe_load(Path(payload["configs"][method]["path"]).read_text(encoding="utf-8"))
        spec = prepare.METHOD_SPECS[method]
        assert config["model_kwargs"]["local_attn_size"] == spec["local_attn_size"]
        assert config["model_kwargs"]["sink_size"] == spec["sink_size"]
        assert config["lphc"]["archive_frames"] == 12
        assert config["lphc"]["protocol"] == "v211"
        assert config["lphc"]["local_policy"] == "sink1_recent20"
        if spec["lphc"]:
            assert config["lphc"]["history_frames"] == spec["history_frames"]


def test_frozen_inputs_cannot_be_rewritten(tmp_path):
    output_root, payload = prepared(tmp_path)
    prompt = Path(payload["prompt_items"][0]["path"])
    prompt.write_text("different\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen v211 input differs"):
        prepared(tmp_path)


def test_environment_scrubbing_and_explicit_v211_binding():
    dirty = {
        "PATH": "/bin",
        "PYRAMIDKV_MODE": "x",
        "LPHC_PROTOCOL": "v210",
        "V209_SOURCE_INDEX": "1",
        "V210_NODE_ADDRESS": "old",
        "V211_EFFECTIVE_SEED": "stale",
    }
    assert worker.scrub_env(dirty) == {"PATH": "/bin"}
    env = worker.lphc_environment(
        prepare.METHOD_SPECS["lphc_sink1_e1_a010_r4"],
        Path("trace.jsonl"),
        3,
    )
    assert env["LPHC_PROTOCOL"] == "v211"
    assert env["LPHC_LOCAL_POLICY"] == "sink1_recent20"
    assert env["LPHC_ARCHIVE_FRAMES"] == "12"
    assert env["LPHC_HISTORY_FRAMES"] == "4"
    assert env["LPHC_CONTROL_SEED"] == "21103"
    assert worker.lphc_environment({"lphc": False}, Path("unused")) == {}


def test_v211_node_identity_and_screen_schedule(monkeypatch):
    manifest = {"authorized_nodes": list(prepare.AUTHORIZED_NODES)}
    monkeypatch.delenv("V211_NODE_ADDRESS", raising=False)
    monkeypatch.setenv("V210_NODE_ADDRESS", prepare.AUTHORIZED_NODES[0])
    with pytest.raises(PermissionError, match="V211_NODE_ADDRESS"):
        worker.assert_authorized_node(manifest, interface_addresses=frozenset(prepare.AUTHORIZED_NODES))
    address = prepare.AUTHORIZED_NODES[0]
    assert worker.assert_authorized_node(
        manifest, address, interface_addresses=frozenset({address})
    ) == address
    for forbidden in prepare.FORBIDDEN_NODES:
        with pytest.raises(PermissionError, match="allowlist"):
            worker.assert_authorized_node(manifest, forbidden, interface_addresses=frozenset({forbidden}))
    gpus = tuple(str(index) for index in range(8))
    assignments = [controller.screen_assignments(rank, gpus) for rank in range(6)]
    jobs = [job for assignment in assignments for lane in assignment.values() for job in lane]
    assert len(jobs) == 32
    assert len(set(jobs)) == 32
    assert all(sum(len(lane) for lane in assignment.values()) > 0 for assignment in assignments)
    assert sum(prepare.METHOD_SPECS[method]["lphc"] for method, _ in jobs) == 16


def test_ssh_launch_is_source_bound_and_v211_only(tmp_path):
    _, manifest = prepared(tmp_path)
    commands = controller.screen_ssh_commands(manifest, tmp_path / "out", "0,1,2,3,4,5,6,7")
    assert len(commands) == 6
    assert [command[-2] for command in commands] == [
        f"root@{node}" for node in prepare.AUTHORIZED_NODES
    ]
    for rank, (node, command) in enumerate(zip(prepare.AUTHORIZED_NODES, commands)):
        rendered = " ".join(command)
        assert manifest["source_commit"] in rendered
        assert f"V211_NODE_ADDRESS={node}" in rendered
        assert "V210_NODE_ADDRESS=" not in rendered
        assert f"NODE_RANK={rank}" in rendered
        assert "scripts/run_v211_lphc.py" in rendered
    assert not any(
        forbidden in " ".join(command)
        for forbidden in prepare.FORBIDDEN_NODES
        for command in commands
    )


def write_trace(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def lphc_trace_block(
    history_budget: int, *, block_id: int = 0, local: list[int] | None = None
) -> list[dict]:
    end = (block_id + 1) * 3
    expected_local = list(range(end)) if end <= 21 else [0, *range(end - 20, end)]
    rows = [{
        "event": "video_start",
        "protocol": "v211",
        "local_policy": "sink1_recent20",
        "history_frames": history_budget,
    }]
    for phase_index in range(5):
        active = phase_index == 0
        selected = list(range(100, 100 + history_budget)) if active else []
        rows.append({
            "event": "attention_call",
            "block_id": block_id,
            "call_kind": "noisy" if phase_index < 4 else "clean",
            "phase_index": phase_index,
            "layer_idx": 0,
            "history_source": "noisy",
            "clean_history_reads": 0,
            "archive_frames": 12,
            "local_frame_indices": expected_local if local is None else local,
            "eligible_frame_indices": list(range(100, 112)) if active else [],
            "selected_history_frames": selected,
            "selected_count": len(selected),
            "correction_ratio": 0.019 if active else 0.0,
            "last_scale_min": 0.5,
            "last_scale_max": 1.0,
            "lookup_count": int(active),
            "random_count": 0,
            "second_attention_count": int(active),
        })
    return rows


@pytest.mark.parametrize("history_budget", [1, 4])
def test_trace_auditor_accepts_supported_history_budgets_and_sink_topology(tmp_path, history_budget):
    trace = tmp_path / f"history-{history_budget}.jsonl"
    write_trace(trace, lphc_trace_block(history_budget, block_id=8))
    report = auditor.audit_trace(
        trace,
        0.02,
        expect_second_attention=True,
        phase="e1",
        expected_history_budget=history_budget,
        expected_layers=1,
    )
    assert report["pass"] is True
    assert report["maximums"]["selected"] == history_budget
    assert report["totals"] == {"lookup": 1, "random": 0, "second_attention": 1}


def test_trace_auditor_rejects_budget_topology_overlap_random_and_ratio(tmp_path):
    rows = lphc_trace_block(4, local=list(range(20, 41)))
    rows[1].update({
        "eligible_frame_indices": [1, 2, 3, 4],
        "selected_history_frames": [1, 2, 3, 4, 21],
        "selected_count": 5,
        "random_count": 1,
        "correction_ratio": 0.021,
    })
    trace = tmp_path / "bad.jsonl"
    write_trace(trace, rows)
    report = auditor.audit_trace(
        trace, 0.02, expect_second_attention=True, phase="e1",
        expected_history_budget=4, expected_blocks=1, expected_layers=1,
    )
    assert report["pass"] is False
    joined = "\n".join(report["errors"])
    assert "local topology" in joined
    assert "overlap" in joined
    assert "requested budget" in joined
    assert "not a subset" in joined
    assert "random retrieval" in joined
    assert "exceeds alpha" in joined


def test_trace_auditor_requires_exact_requested_selection_when_eligible(tmp_path):
    rows = lphc_trace_block(4)
    rows[1]["selected_history_frames"] = [100, 101, 102]
    rows[1]["selected_count"] = 3
    trace = tmp_path / "short.jsonl"
    write_trace(trace, rows)
    report = auditor.audit_trace(
        trace, 0.02, expect_second_attention=True,
        expected_history_budget=4, expected_blocks=1, expected_layers=1,
    )
    assert report["pass"] is False
    assert any("!= requested budget 4" in error for error in report["errors"])


def test_trace_auditor_accepts_exact_warmup_sink_topology(tmp_path):
    rows = lphc_trace_block(4, block_id=0)
    rows[1].update({
        "eligible_frame_indices": [],
        "selected_history_frames": [],
        "selected_count": 0,
        "correction_ratio": 0.0,
        "lookup_count": 0,
        "second_attention_count": 0,
    })
    trace = tmp_path / "warmup.jsonl"
    write_trace(trace, rows)
    report = auditor.audit_trace(
        trace, 0.0, expect_second_attention=False, phase="e1",
        expected_history_budget=4, expected_blocks=1, expected_layers=1,
    )
    assert report["pass"] is True


def test_trace_auditor_requires_header_and_exact_post_roll_topology(tmp_path):
    good = tmp_path / "post-roll.jsonl"
    write_trace(good, lphc_trace_block(4, block_id=7))
    report = auditor.audit_trace(
        good, 0.02, expect_second_attention=True, phase="e1",
        expected_history_budget=4, expected_layers=1,
    )
    assert report["pass"] is True

    rows = lphc_trace_block(4, block_id=7)
    rows[0]["protocol"] = "v210"
    rows[0]["local_policy"] = "fifo21"
    rows[0]["history_frames"] = 1
    rows[1]["local_frame_indices"] = [0, *range(3, 23)]
    bad = tmp_path / "bad-header-topology.jsonl"
    write_trace(bad, rows)
    report = auditor.audit_trace(
        bad, 0.02, expect_second_attention=True, phase="e1",
        expected_history_budget=4, expected_layers=1,
    )
    assert report["pass"] is False
    joined = "\n".join(report["errors"])
    assert "protocol must be v211" in joined
    assert "local_policy must be sink1_recent20" in joined
    assert "history_frames does not match" in joined
    assert "local topology" in joined


def test_trace_auditor_requires_five_calls_and_e1_only(tmp_path):
    trace = tmp_path / "incomplete.jsonl"
    write_trace(trace, lphc_trace_block(1)[:1])
    report = auditor.audit_trace(
        trace, 0.1, expect_second_attention=True, phase="e1",
        expected_history_budget=1, expected_blocks=1, expected_layers=1,
    )
    assert report["pass"] is False
    assert any("call trajectory" in error for error in report["errors"])
    with pytest.raises(ValueError, match="phase must be e1"):
        auditor.audit_trace(trace, 0.1, phase="full", expected_history_budget=1)
    with pytest.raises(ValueError, match="must be 1 or 4"):
        auditor.audit_trace(trace, 0.1, expected_history_budget=2)


def test_gate0_fails_when_media_differs_even_if_tensor_parity_passes(tmp_path, monkeypatch):
    output_root, manifest = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    trace = tmp_path / "alpha0.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(controller, "assert_authorized_node", lambda *args: prepare.AUTHORIZED_NODES[0])
    monkeypatch.setattr(controller, "launch_worker", lambda *args: None)

    def fake_done(_output, _manifest_path, _manifest, _stage, method, _source):
        return {
            "hostname": "host",
            "gpu_uuid": "uuid",
            "cuda_visible_devices": "0",
            "media": {"sha256": method},
            "trace": {"path": str(trace), "present": True},
        }

    monkeypatch.setattr(controller, "load_done", fake_done)
    monkeypatch.setattr(controller, "compare_gate_tensors", lambda *args: {"pass": True, "errors": []})
    monkeypatch.setattr(controller, "audit_trace", lambda *args, **kwargs: {"pass": True, "errors": []})
    report = controller.run_gate0(ROOT, output_root, manifest_path, manifest, "0")
    assert report["pass"] is False
    assert len([error for error in report["errors"] if "decoded media differs" in error]) == 2


def test_smoke_runs_both_r4_candidates_on_source_three(tmp_path, monkeypatch):
    output_root, manifest = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    launched = []
    budgets = []
    monkeypatch.setattr(
        controller, "assert_authorized_node",
        lambda manifest, address=None: prepare.AUTHORIZED_NODES[0],
    )
    monkeypatch.setattr(controller, "require_decision", lambda *args: {"pass": True})
    monkeypatch.setattr(controller, "launch_worker", lambda *args: launched.append(args[3:]))
    monkeypatch.setattr(
        controller, "load_done",
        lambda output_root, manifest_path, manifest, stage, method, source_index: {
            "trace": {"path": str(tmp_path / f"{method}.jsonl"), "present": True}
        },
    )
    for method in controller.SMOKE_METHODS:
        (tmp_path / f"{method}.jsonl").write_text("{}\n", encoding="utf-8")

    def fake_audit(*args, **kwargs):
        budgets.append(kwargs["expected_history_budget"])
        return {"pass": True, "errors": []}

    monkeypatch.setattr(controller, "audit_trace", fake_audit)
    report = controller.run_smoke(ROOT, output_root, manifest_path, manifest, "0")
    assert launched == [
        ("lphc_sink1_e1_a002_r4", 3, "0"),
        ("lphc_sink1_e1_a010_r4", 3, "0"),
    ]
    assert budgets == [4, 4]
    assert tuple(report["audits"]) == controller.SMOKE_METHODS
    assert tuple(report["completions"]) == controller.SMOKE_METHODS
    assert report["latent_frames"] == 120
    assert report["pass"] is True
    smoke_decision = controller.decision_path(output_root, "smoke")
    assert json.loads(smoke_decision.read_text(encoding="utf-8")) == report
    assert list(smoke_decision.parent.glob("smoke*.json")) == [smoke_decision]


def test_gate_and_smoke_decisions_are_both_required_for_screen(tmp_path, monkeypatch):
    output_root, manifest = prepared(tmp_path)
    manifest_path = output_root / "inputs" / "manifest.json"
    valid = {
        "pass": True,
        "input_manifest_sha256": prepare.sha256(manifest_path),
        "source_commit": manifest["source_commit"],
        "errors": [],
    }
    gate = {
        **valid,
        "attention": "production",
        "same_gpu_sequential": True,
        "sequence": [
            {"source_index": source_index, "mode": mode, "gpu": "0"}
            for source_index in prepare.GATE0_SOURCE_INDICES
            for mode in prepare.GATE0_METHODS
        ],
        "pairs": [
            {
                "source_index": source_index,
                "media_equal": True,
                "tensor_comparison": {"pass": True},
                "alpha0_audit": {"pass": True},
            }
            for source_index in prepare.GATE0_SOURCE_INDICES
        ],
    }
    controller.write_decision(controller.decision_path(output_root, "gate0"), gate)
    stale = dict(valid, input_manifest_sha256="bad")
    smoke = controller.decision_path(output_root, "smoke")
    smoke.parent.mkdir(parents=True, exist_ok=True)
    smoke.write_text(json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(
        controller, "assert_authorized_node",
        lambda manifest, address=None: prepare.AUTHORIZED_NODES[0],
    )
    with pytest.raises(RuntimeError, match="smoke decision"):
        controller.run_screen8(
            ROOT, output_root, manifest_path, manifest,
            tuple(str(index) for index in range(8)), 0,
        )
