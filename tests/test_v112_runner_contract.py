from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

RUNNER_PATH = SCRIPTS / "run_v112_role_event_cache_32prompt.py"
spec = importlib.util.spec_from_file_location(
    "v112_role_event_runner_no_torch",
    RUNNER_PATH,
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_full_suite_is_four_methods_times_32_prompts():
    candidate = "support_landmark_suppress_motion"
    methods = runner.methods_for(candidate, "full")
    tasks = runner.all_tasks(candidate, "full")

    assert [method.key for method in methods] == [
        "candidate_support_landmark_suppress_motion",
        "control_all_recent8",
        "control_all_landmark4",
        "control_all_motion_pair2",
    ]
    assert len(tasks) == 128
    assert len({cell.name for _, _, cell in tasks}) == 128
    assert all(
        cell.support_policy not in {"stride", "cyclic", "hybrid"}
        and cell.suppress_policy not in {"merge", "cyclic"}
        for _, _, cell in tasks
    )


def test_minimal_suite_keeps_only_candidate_and_recent_control():
    methods = runner.methods_for(
        "support_hybrid_suppress_recent",
        "minimal",
    )

    assert [method.key for method in methods] == [
        "candidate_support_hybrid_suppress_recent",
        "control_all_recent8",
    ]


def test_four_node_partition_is_complete_and_nonoverlapping():
    candidate = "support_hybrid_suppress_motion"
    shards = [
        runner.selected_tasks(
            candidate,
            "full",
            node_rank=rank,
            num_nodes=4,
        )
        for rank in range(4)
    ]
    names = [cell.name for shard in shards for _, _, cell in shard]

    assert [len(shard) for shard in shards] == [32, 32, 32, 32]
    assert len(names) == len(set(names)) == 128


def test_published_audit_checks_all_32_markers_per_method(tmp_path):
    candidate = "support_landmark_suppress_recent"
    contract_sha = "a" * 64
    args = SimpleNamespace(
        candidate=candidate,
        suite="minimal",
        out_root=tmp_path,
    )
    for method in runner.methods_for(candidate, "minimal"):
        for prompt_index in range(32):
            source = (
                tmp_path
                / "sources"
                / method.key
                / f"{prompt_index:06d}.mp4"
            )
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"video")
            target = (
                tmp_path
                / "published"
                / method.key
                / f"{prompt_index:06d}.mp4"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
            marker = (
                tmp_path
                / "status"
                / "published"
                / f"{method.key}.p{prompt_index:03d}.json"
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "experiment_contract_sha256": contract_sha,
                        "method": method.key,
                        "prompt_index": prompt_index,
                        "task_cell": "test",
                        "source": str(source),
                        "target": str(target),
                        "size": source.stat().st_size,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    payload = runner.audit_published(
        args,
        contract_sha256=contract_sha,
    )

    assert payload["ok"] is True
    assert [row["video_count"] for row in payload["methods"]] == [32, 32]
