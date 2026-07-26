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
        "middle_relative_logit_margin",
        "uniform_stride_margin",
        "uniform_merge_margin",
        "topology_sign_agreement",
        "profile_observation_count",
        "record_observation_count",
        "profile_positive_fraction",
        "bootstrap_sign_agreement",
    )
    rows = [
        (0, 0, -0.8, -0.8, -0.8, 1, 64, 800, 0.0, 1.0),
        (0, 1, -0.05, -0.05, -0.05, 1, 64, 800, 0.0, 1.0),
        (0, 2, 0.05, 0.05, 0.05, 1, 64, 800, 1.0, 1.0),
        (0, 3, 0.8, 0.8, 0.8, 1, 64, 800, 1.0, 1.0),
    ]
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)

    profiles = []
    for policy in ("uniform_stride", "uniform_merge"):
        for pair_index in range(8):
            for side in ("a", "b"):
                for seed in (0, 1):
                    stem = f"{policy}_{pair_index}_{side}_{seed}"
                    profile = tmp_path / f"{stem}.pt"
                    profile.write_bytes(stem.encode())
                    prompt = tmp_path / f"{stem}.txt"
                    prompt.write_text(f"prompt-{pair_index}-{side}\n")
                    profiles.append(
                        {
                            "path": str(profile),
                            "sha256": _sha256(profile),
                            "probe_policy": policy,
                            "pair_id": f"pair_{pair_index}",
                            "side": side,
                            "seed": seed,
                            "prompt_path": str(prompt),
                            "prompt_sha256": _sha256(prompt),
                        }
                    )
    score_values = [-0.8, -0.05, 0.05, 0.8]
    observation_heads = {}
    for head, value in enumerate(score_values):
        observation_heads[f"L0H{head}"] = {
            policy: {
                f"pair_{pair_index}|{side}|{seed}": value
                for pair_index in range(8)
                for side in ("a", "b")
                for seed in (0, 1)
            }
            for policy in ("uniform_stride", "uniform_merge")
        }
    observations = tmp_path / "observations.json"
    observations.write_text(
        json.dumps(
            {
                "version": 3,
                "method": "v98_middle_relative_qk_head_scores",
                "primary_field": "middle_relative_logit_margin",
                "per_head_policy_profile_margins": observation_heads,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dependencies = {}
    for name in (
        "config.yaml",
        "checkpoint.pt",
        "pairs.json",
        "stride.csv",
        "merge.csv",
    ):
        path = tmp_path / name
        path.write_text(name + "\n", encoding="utf-8")
        dependencies[name] = path
    dependencies["pairs.json"].write_text(
        json.dumps(
            {
                "prompt_pairs": [
                    {
                        "id": f"pair_{pair_index}",
                        "a": f"prompt-{pair_index}-a",
                        "b": f"prompt-{pair_index}-b",
                    }
                    for pair_index in range(8)
                ]
            }
        ),
        encoding="utf-8",
    )
    dependencies["stride.csv"].write_text(
        "1,1,1,1\n", encoding="utf-8"
    )
    dependencies["merge.csv"].write_text(
        "2,2,2,2\n", encoding="utf-8"
    )
    profile_protocol = {
        "EXPERIMENT": "v98_middle_relative_scores",
        "RUN_COMMIT": "a" * 40,
        "TRACKED_WORKTREE_DIRTY": "0",
        "CONFIG": str(dependencies["config.yaml"]),
        "CONFIG_SHA256": _sha256(dependencies["config.yaml"]),
        "CHECKPOINT": str(dependencies["checkpoint.pt"]),
        "CHECKPOINT_SHA256": _sha256(dependencies["checkpoint.pt"]),
        "PAIR_JSON": str(dependencies["pairs.json"]),
        "PAIR_SHA256": _sha256(dependencies["pairs.json"]),
        "PROBE_MAP_STRIDE": str(dependencies["stride.csv"]),
        "PROBE_MAP_STRIDE_SHA256": _sha256(dependencies["stride.csv"]),
        "PROBE_MAP_MERGE": str(dependencies["merge.csv"]),
        "PROBE_MAP_MERGE_SHA256": _sha256(dependencies["merge.csv"]),
        "PROFILE_FRAMES": "120",
        "PROFILE_BRANCHES": "cond",
        "PROFILE_UPDATE_MODES": "noisy",
        "FEW_STEP_CFG_ENABLED": "0",
        "PROBE_POLICIES": "uniform_stride,uniform_merge",
        "PAIR_COUNT": "8",
        "PROFILE_COUNT_PER_POLICY": "32",
        "PROFILE_COUNT": "64",
        "SEEDS": "0 1",
        "SINK_FRAMES": "3",
        "RECENT_FRAMES": "4",
    }
    run_manifest = tmp_path / "run_manifest.env"
    run_manifest.write_text(
        "".join(f"{key}={value}\n" for key, value in profile_protocol.items()),
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "version": 2,
                "method": "v98_middle_relative_qk_head_scores",
                "accepted": True,
                "num_layers": 1,
                "num_heads": 4,
                "head_count": 4,
                "profile_audit": profiles,
                "profile_protocol": profile_protocol,
                "score_definition": {
                    "primary_field": "middle_relative_logit_margin",
                    "branch": "cond",
                    "update_mode": "noisy",
                    "sink_frames_excluded": 3,
                    "recent_distinct_key_frames": 4,
                    "common_logit_shift_invariant": True,
                    "pf_labels_used": False,
                    "probe_policy_balanced": True,
                    "probe_policies": [
                        "uniform_stride",
                        "uniform_merge",
                    ],
                    "bootstrap_unit": "counterfactual_prompt_pair",
                },
                "bootstrap_protocol": {
                    "rounds": 500,
                    "seed": 20260726,
                    "zero_effect_is_stable": False,
                },
                "acceptance_protocol": {
                    "min_profiles_per_policy_head": 32,
                    "min_stable_head_fraction": 0.80,
                    "min_head_bootstrap_agreement": 0.75,
                    "min_topology_sign_agreement_fraction": 0.80,
                    "min_minority_fraction": 0.05,
                },
                "acceptance_gates": {
                    "complete_head_grid": {
                        "observed": 4,
                        "required": 4,
                        "passed": True,
                    },
                    "bootstrap_stable_head_fraction": {
                        "observed": 1.0,
                        "required": 0.80,
                        "per_head_threshold": 0.75,
                        "passed": True,
                    },
                    "topology_sign_agreement_fraction": {
                        "observed": 1.0,
                        "required": 0.80,
                        "passed": True,
                    },
                    "minority_role_fraction": {
                        "observed": 0.5,
                        "required": 0.05,
                        "passed": True,
                    },
                },
                "label_counts_at_zero": {
                    "10": 2,
                    "11": 2,
                },
                "files": {
                    "score_csv_sha256": _sha256(scores),
                    "observations": observations.name,
                    "observations_sha256": _sha256(observations),
                    "run_manifest": run_manifest.name,
                    "run_manifest_sha256": _sha256(run_manifest),
                },
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
    assert source["threshold_provenance"] == (
        "shift_invariant_equal_preference_zero_no_pf_labels"
    )
    assert source["path"] == "history_polarity_zero.csv"
    assert manifest["claims"]["common_logit_shift_invariant"]

    tampered = json.loads(artifact.read_text(encoding="utf-8"))
    tampered["acceptance_gates"][
        "bootstrap_stable_head_fraction"
    ]["observed"] = 0.9
    artifact.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = subprocess.run(
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
            str(tmp_path / "tampered_maps"),
            "--num-layers",
            "1",
            "--num-heads",
            "4",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "does not match the score CSV" in rejected.stderr


def test_v98_32gpu_matrix_covers_eight_methods_and_four_shards():
    runner = RUNNER.read_text(encoding="utf-8")
    postprocess = POSTPROCESS.read_text(encoding="utf-8")
    methods = _array(runner, "METHODS")
    cells = [line.split("|", 1)[0] for line in _array(runner, "CELLS")]

    assert len(methods) == 8
    assert len(set(methods)) == 8
    assert methods == cells
    assert methods == _array(postprocess, "METHODS")
    assert "method_index=$local_slot" in runner
    assert "shard=$NODE_RANK" in runner
    assert "global_rank / 4" not in runner
    assert "history_polarity_zero_random_hybrid_merge" in methods
    assert "history_polarity_hybrid_merge_v78" not in methods
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
    assert "refusing mixed experiment_contract.json" in runner
    assert "refusing mixed cross-node experiment contract" in runner
    assert "refusing mixed cell config" in runner
    assert "followup_history_polarity_hybrid_merge_v78" in runner
    assert "FOLLOWUP_V78" in runner
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
