import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_v98_policy_traces as policy_audit


SCRIPT = ROOT / "scripts" / "build_v98_history_polarity_maps.py"
RUNNER = ROOT / "scripts" / "run_v98_history_polarity_4node_32gpu.sh"
POSTPROCESS = ROOT / "scripts" / "postprocess_v98_history_polarity.sh"
INFERENCE = ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array(script: str, name: str) -> list[str]:
    match = re.search(rf"{name}=\(\n(.*?)\n\)", script, flags=re.DOTALL)
    assert match is not None
    return [
        line.strip().strip('"')
        for line in match.group(1).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_v98_map_builder_uses_neutral_labels_and_natural_zero(tmp_path):
    scores = tmp_path / "scores.csv"
    fields = (
        "layer",
        "head",
        "consensus_score",
        "positive_rate",
        "mean_logit",
        "mean_abs_logit",
        "signed_logit_mass",
        "sign_switch_rate",
    )
    rows = [
        (0, 0, -0.3, 0.1, -1.0, 1.2, -0.8, 0.1),
        (0, 1, -0.1, 0.4, -0.1, 1.0, -0.05, 0.4),
        (0, 2, 0.1, 0.6, 0.1, 1.0, 0.05, 0.4),
        (0, 3, 0.3, 0.9, 1.0, 1.2, 0.8, 0.1),
    ]
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)

    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "files": {"score_csv_sha256": _sha256(scores)},
                "head_count": 4,
            }
        ),
        encoding="utf-8",
    )
    pf_labels = tmp_path / "pf.csv"
    pf_labels.write_text("1,-1,2,2\n", encoding="utf-8")
    output = tmp_path / "maps"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--scores",
            str(scores),
            "--score-artifact",
            str(artifact),
            "--pf-labels",
            str(pf_labels),
            "--output-dir",
            str(output),
            "--num-layers",
            "1",
            "--num-heads",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    primary = [
        [int(value) for value in row]
        for row in csv.reader(
            (output / "history_polarity_zero.csv").open(
                encoding="utf-8"
            )
        )
    ]
    assert primary == [[11, 11, 10, 10]]
    manifest = json.loads(
        (output / "history_polarity_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["support_label"] == 10
    assert manifest["suppress_label"] == 11
    assert manifest["claims"]["primary_classifier"] == (
        "history_polarity_zero"
    )
    assert not manifest["claims"]["pf_labels_used_for_primary_classifier"]
    source = manifest["maps"]["history_polarity_zero"]
    assert source["threshold"] == 0
    assert source["threshold_provenance"] == "natural_zero_no_pf_labels"


def test_v98_32gpu_matrix_covers_eight_methods_and_four_shards():
    runner = RUNNER.read_text(encoding="utf-8")
    postprocess = POSTPROCESS.read_text(encoding="utf-8")
    methods = _array(runner, "METHODS")
    cells = [line.split("|", 1)[0] for line in _array(runner, "CELLS")]

    assert len(methods) == 8
    assert len(set(methods)) == 8
    assert methods == cells
    assert methods == _array(postprocess, "METHODS")
    assert "global_rank=$((NODE_RANK * 8 + local_slot))" in runner
    assert "method_index=$((global_rank / 4))" in runner
    assert "shard=$((global_rank % 4))" in runner
    assert "NODE_RANK must be one of 0,1,2,3" in runner
    assert "FRAMES=\"${FRAMES:-120}\"" in runner
    assert "MovieGenVideoBench_num32.txt" in runner
    assert "MovieGenVideoBench_num128.txt" in runner


def test_v98_runner_has_parity_and_neutral_policy_gates():
    runner = RUNNER.read_text(encoding="utf-8")
    postprocess = POSTPROCESS.read_text(encoding="utf-8")
    inference = INFERENCE.read_text(encoding="utf-8")

    assert "pf_explicit_parity" in runner
    assert "--pyramidkv_binary_responsive_policy cyclic" in runner
    assert "--pyramidkv_history_polarity" in runner
    assert "legacy_pf_labels=false" in runner
    assert "refusing mixed resume" in runner
    assert "history_polarity_hybrid_merge_v78" in runner
    assert "audit_v98_policy_traces.py" in postprocess
    assert "--strict" in postprocess
    assert "history-polarity head map must be a complete 30x12 matrix" in (
        inference
    )
    assert "neutral labels 10/11" in inference


def test_v98_trace_audit_validates_neutral_hybrid_and_merge(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("10,11\n", encoding="utf-8")
    config = tmp_path / "method.shard0.env"
    config.write_text(
        "\n".join(
            [
                "name=method",
                "shard=0",
                f"labels={labels}",
                f"label_sha256={_sha256(labels)}",
                "route=history_hybrid_merge",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace = tmp_path / "method.shard0.policy.jsonl"
    events = []
    for branch in ("cond", "uncond"):
        events.extend(
            [
                {
                    "event": "middle_selection",
                    "layer": 0,
                    "head": 0,
                    "label": 10,
                    "branch": branch,
                    "sink_frames": 3,
                    "recent_frames": 4,
                    "strategies": [
                        {"name": "CyclicStrategy"},
                        {"name": "StrideStrategy"},
                    ],
                    "union_frame_ids": [1, 6],
                    "union_frame_count": 2,
                    "union_token_count": 32,
                },
                {
                    "event": "middle_selection",
                    "layer": 0,
                    "head": 1,
                    "label": 11,
                    "branch": branch,
                    "sink_frames": 3,
                    "recent_frames": 4,
                    "strategies": [{"name": "MergeStrategy"}],
                    "union_frame_ids": [3],
                    "union_frame_count": 1,
                    "union_token_count": 8,
                },
            ]
        )
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    result = policy_audit.audit_trace(
        method="method",
        shard=0,
        config_path=config,
        trace_path=trace,
        expected_layers={0},
        num_layers=1,
        num_heads=2,
    )

    assert result["status"] == "nominal"
    assert result["label_events"] == {10: 2, 11: 2}
