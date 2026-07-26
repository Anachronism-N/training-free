from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_v98_history_polarity as analysis
import audit_v98_policy_traces as audit
import prepare_blind_review as blind_review
from compute_temporal_jump_diagnostic import _indexed_video_paths
from merge_comprehensive_results import merge


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_matrix(path: Path, values: list[int]) -> None:
    path.write_text(
        ",".join(str(value) for value in values) + "\n",
        encoding="utf-8",
    )


def _serialized_policies(
    routes: dict[int, dict[str, object]],
) -> dict[str, object]:
    return {
        str(label): {
            "sink_frames": policy["sink"],
            "recent_frames": policy["recent"],
            "policy_type": policy["policy_type"],
            "strategies": [
                {
                    "name": strategy["name"],
                    "params": {
                        key: value
                        for key, value in strategy.items()
                        if key != "name"
                    },
                }
                for strategy in policy["strategies"]
            ],
            "max_union_frames": 4,
        }
        for label, policy in routes.items()
    }


def _contract_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run_root = tmp_path / "run"
    map_dir = run_root / "maps"
    map_dir.mkdir(parents=True)
    score_csv = tmp_path / "scores.csv"
    score_csv.write_text("layer,head,middle_relative_logit_margin\n", encoding="utf-8")
    score_artifact = tmp_path / "score_artifact.json"
    score_artifact.write_text(
        json.dumps(
            {
                "version": 2,
                "method": "v98_middle_relative_qk_head_scores",
                "accepted": True,
                "score_definition": {
                    "primary_field": "middle_relative_logit_margin",
                    "bootstrap_unit": "counterfactual_prompt_pair",
                },
            }
        ),
        encoding="utf-8",
    )
    pf = map_dir / "pf.csv"
    natural = map_dir / "history_polarity_zero.csv"
    random_map = map_dir / "history_polarity_zero_random.csv"
    positive = map_dir / "positive_rate_half.csv"
    pf_aw = map_dir / "pf_aw_binary_control.csv"
    _write_matrix(pf, [-1, 1, 2])
    for path in (natural, random_map, positive, pf_aw):
        _write_matrix(path, [10, 11, 10])

    def map_entry(path: Path, **extra: object) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "label_counts": {"10": 2, "11": 1},
            **extra,
        }

    manifest_path = map_dir / "history_polarity_manifest.json"
    manifest = {
        "version": 2,
        "support_label": 10,
        "suppress_label": 11,
        "reserved_pf_labels": [-1, 1, 2],
        "score_csv_sha256": _sha256(score_csv),
        "score_artifact": str(score_artifact.resolve()),
        "score_artifact_sha256": _sha256(score_artifact),
        "pf_labels": str(pf.resolve()),
        "pf_labels_sha256": _sha256(pf),
        "maps": {
            "history_polarity_zero": map_entry(
                natural,
                score_column="middle_relative_logit_margin",
                support_rule="middle_relative_logit_margin >= 0",
            ),
            "history_polarity_zero_random": map_entry(
                random_map,
                reference="history_polarity_zero",
            ),
            "positive_rate_half": map_entry(positive),
            "pf_aw_binary_control": map_entry(pf_aw),
        },
        "claims": {
            "primary_classifier": "history_polarity_zero",
            "pf_labels_used_for_primary_classifier": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "".join(f"prompt {index}\n" for index in range(32)),
        encoding="utf-8",
    )

    map_bindings = {
        "pf_labels": pf,
        "history_polarity_zero": natural,
        "history_polarity_zero_random": random_map,
        "positive_rate_half": positive,
        "pf_aw_binary_control": pf_aw,
    }
    methods = []
    for index, name in enumerate(audit.METHODS):
        expected = audit.METHOD_CONTRACTS[name]
        if expected.map_role is None:
            map_key = None
            map_path = None
            map_hash = None
            expected_labels = None
            policies = None
        else:
            map_key = (
                "history_polarity_zero"
                if expected.map_role == "primary"
                else expected.map_role
            )
            bound_path = map_bindings[map_key]
            map_path = str(bound_path.resolve())
            map_hash = _sha256(bound_path)
            routes = audit.expected_routes(expected.route)
            expected_labels = sorted(routes)
            policies = _serialized_policies(routes)
        methods.append(
            {
                "method_index": index,
                "name": name,
                "engine": expected.engine,
                "route": expected.route,
                "route_parameters": {},
                "map_key": map_key,
                "map_path": map_path,
                "map_sha256": map_hash,
                "transition": {
                    "enabled": False,
                    "branches": [],
                    "parameters": None,
                },
                "expected_labels": expected_labels,
                "policies": policies,
                "few_step_cfg_enabled": False,
                "policy_trace_branches": (
                    ["cond"] if expected.engine == "pf" else []
                ),
            }
        )
    contract = {
        "version": 2,
        "experiment": "v98_history_polarity",
        "phase": "primary",
        "mode": "screen32",
        "run_commit": "abc123",
        "tracked_worktree_dirty": False,
        "prompt": {
            "path": str(prompts.resolve()),
            "sha256": _sha256(prompts),
            "count": 32,
            "seed": 0,
            "reseed_per_prompt": True,
        },
        "frames": 120,
        "seed": 0,
        "shards": 4,
        "few_step_cfg_enabled": False,
        "runtime": {
            "few_step_cfg_enabled": False,
            "policy_trace": {
                "layers": [0],
                "stride": 3,
                "max_records": 60000,
            },
        },
        "video": {
            "latent_frames": 120,
            "decoded_frames": 477,
            "sample_index": 0,
        },
        "sharding": {"shards": 4, "shard_size": 8},
        "score": {
            "artifact_path": str(score_artifact.resolve()),
            "artifact_sha256": _sha256(score_artifact),
            "csv_path": str(score_csv.resolve()),
            "csv_sha256": _sha256(score_csv),
            "map_manifest_path": str(manifest_path.resolve()),
            "map_manifest_sha256": _sha256(manifest_path),
            "artifact_version": 2,
            "artifact_method": "v98_middle_relative_qk_head_scores",
            "artifact_accepted": True,
            "primary_field": "middle_relative_logit_margin",
            "bootstrap_unit": "counterfactual_prompt_pair",
        },
        "method_contract_sha256": "a" * 64,
        "methods": methods,
    }
    contract["run_fingerprint"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    contract_path = run_root / "experiment_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path, manifest


def _metric_row(value: float) -> dict[str, float]:
    return {
        metric: value
        for metric in analysis.COMPREHENSIVE_METRICS
    }


def _comprehensive_part(
    path: Path,
    *,
    method: str,
    prompts: list[str],
    value: float,
) -> None:
    per_video = {}
    for index in reversed(range(len(prompts))):
        metrics = _metric_row(value + index / 1000)
        per_video[f"{method}/{index}-0_ema"] = {
            "method": method,
            "prompt_index": index,
            "sample_index": 0,
            "video_name": f"{index}-0_ema.mp4",
            "video_path": str(path.parent / method / f"{index}-0_ema.mp4"),
            "prompt": prompts[index],
            "metrics": metrics,
        }
    per_method = {
        metric: statistics.fmean(
            row["metrics"][metric] for row in per_video.values()
        )
        for metric in analysis.COMPREHENSIVE_METRICS
    }
    per_method.update(
        {
            "num_videos": len(prompts),
            "prompt_indices": list(range(len(prompts))),
        }
    )
    path.write_text(
        json.dumps(
            {
                "per_video": per_video,
                "per_method": {method: per_method},
            }
        ),
        encoding="utf-8",
    )


def test_merge_preserves_numeric_prompt_observations(tmp_path: Path) -> None:
    prompts = [f"prompt {index}" for index in range(12)]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _comprehensive_part(first, method="first", prompts=prompts, value=0.4)
    _comprehensive_part(second, method="second", prompts=prompts, value=0.5)

    payload = merge(
        [first, second],
        expected_methods=["first", "second"],
        expected_videos=12,
    )

    assert len(payload["per_video"]) == 24
    assert payload["merge"]["prompt_indices"] == list(range(12))
    assert payload["per_video"]["first/10-0_ema"]["prompt_index"] == 10
    assert payload["per_video"]["first/10-0_ema"]["prompt"] == "prompt 10"


def test_merge_rejects_cross_method_prompt_text_mismatch(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _comprehensive_part(
        first, method="first", prompts=["same"], value=0.4
    )
    _comprehensive_part(
        second, method="second", prompts=["different"], value=0.5
    )

    with pytest.raises(ValueError, match="prompt text mismatch"):
        merge([first, second], expected_videos=1)


def test_temporal_inputs_are_numeric_and_cartesian(tmp_path: Path) -> None:
    for method in ("first", "second"):
        directory = tmp_path / method
        directory.mkdir()
        for index in (10, 2, 1, 0, 11, 9, 8, 7, 6, 5, 4, 3):
            (directory / f"{index}-0_ema.mp4").write_bytes(b"video")

    rows = _indexed_video_paths(
        [tmp_path / "second", tmp_path / "first"],
        expected_videos=12,
    )

    assert [(method, index) for method, index, _ in rows[:12]] == [
        ("first", index) for index in range(12)
    ]
    (tmp_path / "second" / "7-0_ema.mp4").unlink()
    with pytest.raises(ValueError, match="coverage mismatch"):
        _indexed_video_paths(
            [tmp_path / "first", tmp_path / "second"],
            expected_videos=12,
        )


def test_audit_contract_binds_primary_statistic_and_random_map(
    tmp_path: Path,
) -> None:
    contract_path, _ = _contract_fixture(tmp_path)
    manifest = audit._load_manifest_contract(
        contract_path.parent,
        num_layers=1,
        num_heads=3,
    )

    contract = audit._load_experiment_contract(
        contract_path,
        manifest=manifest,
    )

    assert contract["pass"]
    assert manifest["primary_score"]["score_column"] == (
        "middle_relative_logit_margin"
    )
    random_cell = contract["methods"][
        "history_polarity_zero_random_hybrid_merge"
    ]
    assert random_cell["map_key"] == "history_polarity_zero_random"

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["methods"][4]["map_key"] = "history_polarity_zero_random"
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("run_fingerprint")
    payload["run_fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    contract = audit._load_experiment_contract(
        contract_path,
        manifest=manifest,
    )
    assert not contract["pass"]
    assert any(
        "map key/path/hash contract mismatch" in failure
        for failure in contract["failures"]
    )


def test_full_policy_audit_uses_frozen_contract_and_observed_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, _ = _contract_fixture(tmp_path)
    run_root = contract_path.parent
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_hash = _sha256(contract_path)
    configs = run_root / "configs"
    traces = run_root / "traces"
    configs.mkdir()
    traces.mkdir()
    score = contract_payload["score"]
    prompt = contract_payload["prompt"]

    for method in contract_payload["methods"]:
        name = method["name"]
        label_matrix = None
        if method["map_path"]:
            with Path(method["map_path"]).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                label_matrix = [
                    [int(value) for value in row]
                    for row in csv.reader(handle)
                    if row
                ]
        for shard in range(4):
            start = shard * 8
            end = start + 8
            config = {
                "contract_version": "3",
                "name": name,
                "phase": "primary",
                "mode": "screen32",
                "node_rank": str(shard),
                "shard": str(shard),
                "start_idx": str(start),
                "end_idx": str(end),
                "gpu": "0",
                "engine": method["engine"],
                "labels": method["map_path"] or "",
                "label_sha256": method["map_sha256"] or "",
                "route": method["route"],
                "transition": "0",
                "score_sha256": score["csv_sha256"],
                "score_artifact_sha256": score["artifact_sha256"],
                "map_manifest_sha256": score["map_manifest_sha256"],
                "method_contract_sha256": contract_payload[
                    "method_contract_sha256"
                ],
                "experiment_contract_sha256": contract_hash,
                "run_commit": contract_payload["run_commit"],
                "prompt_sha256": prompt["sha256"],
                "prompt_count": "32",
                "frames": "120",
                "expected_video_frames": "477",
                "seed": "0",
                "reseed_per_prompt": "1",
                "few_step_cfg_enabled": "0",
                "policy_trace_layers": "0",
                "policy_trace_stride": "3",
                "policy_trace_max_records": "60000",
            }
            (configs / f"{name}.shard{shard}.env").write_text(
                "".join(f"{key}={value}\n" for key, value in config.items()),
                encoding="utf-8",
            )
            if method["engine"] == "sf":
                continue
            events = []
            policies = method["policies"]
            assert label_matrix is not None
            for prompt_id in range(1, 9):
                for head, label in enumerate(label_matrix[0]):
                    policy = policies[str(label)]
                    for sync_t in range(0, 120, 3):
                        strategy_rows = []
                        union_costs: dict[int, int] = {}
                        all_ids = []
                        for strategy in policy["strategies"]:
                            strategy_name = strategy["name"]
                            params = strategy["params"]
                            tail_min = (
                                sync_t - policy["recent_frames"] + 1
                            )
                            if strategy_name == "StrideStrategy":
                                candidates = [
                                    frame
                                    for frame in range(sync_t)
                                    if frame >= policy["sink_frames"]
                                    and frame < tail_min
                                    and frame % params["interval"] == 0
                                ]
                                frame_ids = candidates[-params["capacity"] :]
                                token_cost = 8
                            elif strategy_name == "CyclicStrategy":
                                candidates = [
                                    frame
                                    for frame in range(sync_t)
                                    if frame >= policy["sink_frames"]
                                    and frame < tail_min
                                    and frame % params["period"]
                                    == sync_t % params["period"]
                                ]
                                frame_ids = candidates[
                                    -params["bucket_cap"] :
                                ]
                                token_cost = 8
                            else:
                                block_frames = params["block_frames"]
                                candidates = [
                                    (start + start + block_frames - 1) // 2
                                    for start in range(
                                        0, sync_t, block_frames
                                    )
                                    if start >= policy["sink_frames"]
                                    and start + block_frames - 1 < tail_min
                                ]
                                frame_ids = candidates[-params["capacity"] :]
                                token_cost = 2
                            token_count = token_cost * len(frame_ids)
                            strategy_rows.append(
                                {
                                    "name": strategy_name,
                                    **params,
                                    "frame_ids": frame_ids,
                                    "token_count": token_count,
                                }
                            )
                            all_ids.extend(frame_ids)
                            for frame_id in frame_ids:
                                union_costs.setdefault(frame_id, token_cost)
                        union_ids = sorted(set(all_ids))
                        sink_ids = list(range(policy["sink_frames"]))
                        recent_ids = list(
                            range(
                                max(
                                    policy["sink_frames"],
                                    sync_t
                                    + 3
                                    - policy["recent_frames"],
                                ),
                                sync_t + 3,
                            )
                        )
                        events.append(
                            {
                                "event": "middle_selection",
                                "prompt_id": prompt_id,
                                "layer": 0,
                                "head": head,
                                "seq": head,
                                "branch": "cond",
                                "sync_t": sync_t,
                                "label": label,
                                "sink_frames": policy["sink_frames"],
                                "recent_frames": policy["recent_frames"],
                                "policy_type": policy["policy_type"],
                                "strategies": strategy_rows,
                                "union_frame_ids": union_ids,
                                "union_frame_count": len(union_ids),
                                "union_token_count": sum(
                                    union_costs.values()
                                ),
                                "sink_frame_ids": sink_ids,
                                "sink_frame_count": len(sink_ids),
                                "sink_token_count": 8 * len(sink_ids),
                                "recent_frame_ids": recent_ids,
                                "recent_frame_count": len(recent_ids),
                                "recent_token_count": 8 * len(recent_ids),
                                "middle_sink_overlap": [],
                                "middle_recent_overlap": [],
                                "composition_present": True,
                                "dynamic_policy_owner": "composition_recent",
                                "explicit_composition_owns_dynamic": True,
                                "cache_contract_violations": [],
                                "cache_contract_pass": True,
                            }
                        )
            (traces / f"{name}.shard{shard}.policy.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--run-root",
            str(run_root),
            "--experiment-contract",
            str(contract_path),
            "--expected-layers",
            "0",
            "--num-layers",
            "1",
            "--num-heads",
            "3",
            "--shards",
            "4",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ],
    )

    audit.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["strict_pass"]
    assert payload["pf_parity_observed_contract"]["pass"]
    assert payload["pf_parity_observed_contract"]["compared_events"] == 3840
    assert len(payload["shards"]) == 32


def test_followup_contract_is_two_cell_and_transition_is_isolated(
    tmp_path: Path,
) -> None:
    primary_contract_path, _ = _contract_fixture(tmp_path)
    primary = json.loads(
        primary_contract_path.read_text(encoding="utf-8")
    )
    natural = next(
        item
        for item in primary["methods"]
        if item["name"] == "history_polarity_hybrid_merge"
    )
    transition_parameters = {
        "mode": "full",
        "min_reliability": 0.55,
        "min_novelty": 0.01,
        "max_commit_fraction": 0.75,
        "stagger_period": 1,
        "max_age_blocks": 6,
        "branches": "cond",
        "denoise_weight": 2.0,
    }
    followup_methods = []
    for index, name in enumerate(audit.FOLLOWUP_METHODS):
        enabled = index == 1
        followup_methods.append(
            {
                **natural,
                "method_index": index,
                "name": name,
                "transition": {
                    "enabled": enabled,
                    "branches": ["cond"] if enabled else [],
                    "parameters": (
                        transition_parameters if enabled else None
                    ),
                },
            }
        )
    evidence_dir = primary_contract_path.parent / "primary_evidence"
    evidence_dir.mkdir()
    primary_evidence = {}
    for name in (
        "primary_manifest",
        "primary_experiment_contract",
        "primary_analysis",
        "primary_blind_frozen",
        "primary_blind_verification",
        "primary_blind_completion",
        "primary_blind_scorecard",
        "primary_blind_key",
    ):
        path = evidence_dir / f"{name}.json"
        path.write_text(f"{name}\n", encoding="utf-8")
        primary_evidence[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    followup = {
        **primary,
        "phase": "followup_v78",
        "methods": followup_methods,
        "method_contract_sha256": "c" * 64,
        "primary_manifest_sha256": primary_evidence[
            "primary_manifest"
        ]["sha256"],
        "primary_gate_evidence": primary_evidence,
        "inputs": {
            **primary.get("inputs", {}),
            **primary_evidence,
        },
    }
    followup.pop("run_fingerprint")
    followup["run_fingerprint"] = hashlib.sha256(
        json.dumps(
            followup,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    followup_root = primary_contract_path.parent / "followup_v78"
    followup_root.mkdir()
    followup_path = followup_root / "experiment_contract.json"
    followup_path.write_text(json.dumps(followup), encoding="utf-8")
    manifest = audit._load_manifest_contract(
        followup_root,
        num_layers=1,
        num_heads=3,
        manifest_path=Path(followup["score"]["map_manifest_path"]),
    )

    loaded = audit._load_experiment_contract(
        followup_path,
        manifest=manifest,
    )

    assert loaded["pass"]
    assert loaded["phase"] == "followup_v78"
    assert tuple(loaded["methods"]) == audit.FOLLOWUP_METHODS
    assert loaded["methods"][audit.FOLLOWUP_METHODS[0]][
        "transition"
    ]["enabled"] is False
    assert loaded["methods"][audit.FOLLOWUP_METHODS[1]][
        "transition"
    ]["enabled"] is True


def _trace_event(
    *,
    head: int,
    label: int,
    strategy_items: list[dict[str, object]],
    union_ids: list[int],
    union_tokens: int,
) -> dict[str, object]:
    return {
        "event": "middle_selection",
        "prompt_id": 5,
        "layer": 0,
        "head": head,
        "seq": head,
        "branch": "cond",
        "sync_t": 13,
        "label": label,
        "sink_frames": 3,
        "recent_frames": 4,
        "policy_type": "stride" if label == 10 else "merge",
        "strategies": strategy_items,
        "union_frame_ids": union_ids,
        "union_frame_count": len(union_ids),
        "union_token_count": union_tokens,
        "sink_frame_ids": [0, 1, 2],
        "sink_frame_count": 3,
        "sink_token_count": 24,
        "recent_frame_ids": [9, 10, 11, 12],
        "recent_frame_count": 4,
        "recent_token_count": 32,
        "middle_sink_overlap": [],
        "middle_recent_overlap": [],
        "composition_present": True,
        "dynamic_policy_owner": "composition_recent",
        "explicit_composition_owns_dynamic": True,
        "cache_contract_violations": [],
        "cache_contract_pass": True,
    }


def _strict_trace_events() -> list[dict[str, object]]:
    return [
        _trace_event(
            head=0,
            label=10,
            strategy_items=[
                {
                    "name": "CyclicStrategy",
                    "period": 6,
                    "bucket_cap": 2,
                    "dynamic_rope": True,
                    "frame_ids": [7],
                    "token_count": 8,
                },
                {
                    "name": "StrideStrategy",
                    "interval": 6,
                    "capacity": 2,
                    "dynamic_rope": True,
                    "frame_ids": [6],
                    "token_count": 8,
                },
            ],
            union_ids=[6, 7],
            union_tokens=16,
        ),
        _trace_event(
            head=1,
            label=11,
            strategy_items=[
                {
                    "name": "MergeStrategy",
                    "patch_size": 2,
                    "block_frames": 4,
                    "capacity": 4,
                    "dynamic_rope": True,
                    "frame_ids": [5],
                    "token_count": 2,
                }
            ],
            union_ids=[5],
            union_tokens=2,
        ),
    ]


def test_trace_audit_requires_strategy_parameters_and_cartesian_cells(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in _strict_trace_events()),
        encoding="utf-8",
    )

    result = audit._audit_events(
        method="primary",
        shard=0,
        trace_path=trace,
        labels=[[10, 11]],
        routes=audit.HISTORY_HYBRID,
        expected_branches={"cond"},
        expected_layers={0},
        num_heads=2,
        expected_prompt_count=1,
        strict_schema=True,
    )

    assert result["status"] == "nominal"
    assert result["observed_cartesian_cells"] == 2

    broken = _strict_trace_events()
    broken[0]["strategies"][0].pop("period")
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in broken),
        encoding="utf-8",
    )
    result = audit._audit_events(
        method="primary",
        shard=0,
        trace_path=trace,
        labels=[[10, 11]],
        routes=audit.HISTORY_HYBRID,
        expected_branches={"cond"},
        expected_layers={0},
        num_heads=2,
        expected_prompt_count=1,
        strict_schema=True,
    )
    assert result["status"] == "failed"
    assert any("missing parameter period" in row for row in result["failures"])


def test_trace_audit_rejects_future_or_recent_middle_frames(
    tmp_path: Path,
) -> None:
    events = _strict_trace_events()
    events[0]["strategies"][1]["frame_ids"] = [11]
    events[0]["union_frame_ids"] = [7, 11]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in events),
        encoding="utf-8",
    )

    result = audit._audit_events(
        method="primary",
        shard=0,
        trace_path=trace,
        labels=[[10, 11]],
        routes=audit.HISTORY_HYBRID,
        expected_branches={"cond"},
        expected_layers={0},
        num_heads=2,
        expected_prompt_count=1,
        strict_schema=True,
    )

    assert result["status"] == "failed"
    assert any("sink/recent/future" in row for row in result["failures"])


def test_trace_audit_rejects_legacy_dynamic_history_leak(
    tmp_path: Path,
) -> None:
    events = _strict_trace_events()
    events[0]["recent_frame_ids"] = [8, 9, 10, 11, 12]
    events[0]["recent_frame_count"] = 5
    events[0]["recent_token_count"] = 40
    events[0]["cache_contract_violations"] = [
        "recent_frame_budget_exceeded"
    ]
    events[0]["cache_contract_pass"] = False
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in events),
        encoding="utf-8",
    )

    result = audit._audit_events(
        method="primary",
        shard=0,
        trace_path=trace,
        labels=[[10, 11]],
        routes=audit.HISTORY_HYBRID,
        expected_branches={"cond"},
        expected_layers={0},
        num_heads=2,
        expected_prompt_count=1,
        strict_schema=True,
    )

    assert result["status"] == "failed"
    assert any(
        "dynamic cache leaked non-recent frames" in row
        for row in result["failures"]
    )


def _paired_inputs(
    difference: float,
) -> tuple[
    dict[str, dict[int, dict[str, float]]],
    dict[str, dict[int, float]],
    dict[str, dict[str, float]],
]:
    comprehensive = {"pf_native": {}, "pf_explicit_parity": {}}
    temporal = {"pf_native": {}, "pf_explicit_parity": {}}
    for index in range(4):
        comprehensive["pf_native"][index] = _metric_row(0.5)
        comprehensive["pf_explicit_parity"][index] = _metric_row(
            0.5 + (difference if index == 0 else 0.0)
        )
        temporal["pf_native"][index] = 1.0
        temporal["pf_explicit_parity"][index] = 1.0
    vbench = {
        method: {
            f"vbench_{metric}": 0.5
            for metric in analysis.VBENCH_METRICS
        }
        for method in comprehensive
    }
    return comprehensive, temporal, vbench


def test_parity_gate_uses_per_metric_max_paired_delta() -> None:
    comprehensive, temporal, vbench = _paired_inputs(0.0)
    comparison = analysis.build_comparison(
        name="implementation_parity",
        baseline="pf_native",
        candidate="pf_explicit_parity",
        purpose="test",
        comprehensive=comprehensive,
        temporal=temporal,
        vbench=vbench,
        prompt_count=4,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert analysis.parity_metric_gate(comparison)["pass"]

    comprehensive, temporal, vbench = _paired_inputs(0.01)
    comparison = analysis.build_comparison(
        name="implementation_parity",
        baseline="pf_native",
        candidate="pf_explicit_parity",
        purpose="test",
        comprehensive=comprehensive,
        temporal=temporal,
        vbench=vbench,
        prompt_count=4,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    gate = analysis.parity_metric_gate(comparison)
    assert not gate["pass"]
    assert not gate["metrics"]["m1_dino_consistency"]["pass"]
    assert gate["metrics"]["m1_dino_consistency"][
        "max_abs_delta"
    ] == pytest.approx(0.01)


def test_full_analysis_preserves_pairing_and_passes_clean_controls(
    tmp_path: Path,
) -> None:
    contract_path, _ = _contract_fixture(tmp_path)
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    comprehensive_rows = {}
    comprehensive_methods = {}
    temporal_path = tmp_path / "temporal.csv"
    temporal_rows = []
    vbench_methods = {}
    vbench_sources = {}
    for method_index, method in enumerate(analysis.PRIMARY_METHODS):
        base = 0.5 if method_index < 3 else 0.5 + method_index / 100
        metric_values = {metric: [] for metric in analysis.COMPREHENSIVE_METRICS}
        for prompt_index in range(32):
            value = base + prompt_index / 10000
            # Native/explicit parity must be exactly matched prompt by prompt.
            if method == "pf_explicit_parity":
                value = 0.5 + prompt_index / 10000
            metrics = _metric_row(value)
            for metric, metric_value in metrics.items():
                metric_values[metric].append(metric_value)
            comprehensive_rows[f"{method}/{prompt_index}-0_ema"] = {
                "method": method,
                "prompt_index": prompt_index,
                "sample_index": 0,
                "video_name": f"{prompt_index}-0_ema.mp4",
                "video_path": str(
                    tmp_path / method / f"{prompt_index}-0_ema.mp4"
                ),
                "prompt": f"prompt {prompt_index}",
                "metrics": metrics,
            }
            temporal_rows.append(
                {
                    "method": method,
                    "prompt_index": prompt_index,
                    "sample_index": 0,
                    "video": str(
                        tmp_path / method / f"{prompt_index}-0_ema.mp4"
                    ),
                    "temporal_jump": 1.0 + (
                        0 if method_index < 3 else method_index / 100
                    ),
                }
            )
        comprehensive_methods[method] = {
            **{
                metric: statistics.fmean(values)
                for metric, values in metric_values.items()
            },
            "num_videos": 32,
            "prompt_indices": list(range(32)),
        }
        vbench_methods[method] = {
            metric: 0.6 + (0 if method_index < 3 else method_index / 100)
            for metric in analysis.VBENCH_METRICS
        }
        vbench_sources[method] = str(tmp_path / method / "results.json")

    comprehensive_path = tmp_path / "comprehensive.json"
    comprehensive_path.write_text(
        json.dumps(
            {
                "per_video": comprehensive_rows,
                "per_method": comprehensive_methods,
            }
        ),
        encoding="utf-8",
    )
    vbench_path = tmp_path / "vbench.json"
    vbench_path.write_text(
        json.dumps(
            {
                "methods": vbench_methods,
                "dimensions": list(analysis.VBENCH_METRICS),
                "sources": vbench_sources,
                "missing": [],
            }
        ),
        encoding="utf-8",
    )
    with temporal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(temporal_rows[0])
        )
        writer.writeheader()
        writer.writerows(temporal_rows)
    policy_audit_path = tmp_path / "policy_audit.json"
    policy_audit_path.write_text(
        json.dumps(
                {
                    "strict_pass": True,
                    "experiment_contract_sha256": _sha256(contract_path),
                    "expected_methods": list(analysis.PRIMARY_METHODS),
                    "pf_parity_observed_contract": {"pass": True},
                    "shards": [],
                }
            ),
            encoding="utf-8",
        )
    metric_manifest_path = tmp_path / "metric_manifest.json"
    metric_manifest = {
        "version": 2,
        "generation": {
            "global_manifest_sha256": "a" * 64,
            "experiment_contract_sha256": _sha256(contract_path),
            "run_fingerprint": contract_payload["run_fingerprint"],
        },
        "blind": {},
        "video_inputs": [
            {
                "method": method,
                "count": 32,
                "input_fingerprint": hashlib.sha256(
                    method.encode("utf-8")
                ).hexdigest(),
            }
            for method in analysis.PRIMARY_METHODS
        ],
        "stages": {
            "vbench": True,
            "comprehensive": True,
            "temporal": True,
            "analysis": True,
        },
        "parameters": analysis.FROZEN_METRIC_PARAMETERS,
        "vbench": {
            "commit": "a" * 40,
            "dirty": False,
            "diff_sha256": "b" * 64,
            "status_sha256": "c" * 64,
            "evaluator_sha256": "d" * 64,
            "info_sha256": "e" * 64,
        },
        "evaluators": {
            str(Path(analysis.__file__).resolve()): _sha256(
                Path(analysis.__file__)
            )
        },
    }
    metric_manifest["metric_input_fingerprint"] = hashlib.sha256(
        json.dumps(
            metric_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metric_manifest_path.write_text(
        json.dumps(metric_manifest), encoding="utf-8"
    )

    payload = analysis.analyze(
        comprehensive_path=comprehensive_path,
        vbench_path=vbench_path,
        temporal_path=temporal_path,
        map_manifest_path=contract_path.parent
        / "maps"
        / "history_polarity_manifest.json",
        policy_audit_path=policy_audit_path,
        metric_manifest_path=metric_manifest_path,
        experiment_contract_path=contract_path,
        transition_summary_path=None,
        blind_scorecard_path=None,
        blind_key_path=None,
        blind_verification_path=None,
        bootstrap_samples=100,
        bootstrap_seed=5,
    )

    assert not payload["hard_gate_pass"]
    assert not payload["gates"]["blind_scorecard"]["pass"]
    assert payload["gates"]["blind_scorecard"]["status"] == (
        "missing_required_frozen_human_review"
    )
    assert payload["gates"]["pf_parity_metrics"]["pass"]
    assert payload["sample_contract"]["prompt_indices"] == list(range(32))
    assert payload["input_artifacts"]["comprehensive"]["sha256"] == _sha256(
        comprehensive_path
    )
    assert payload["input_artifacts"]["policy_audit"]["sha256"] == _sha256(
        policy_audit_path
    )
    random_comparison = next(
        row
        for row in payload["comparisons"]
        if row["name"] == "polarity_vs_count_matched_random"
    )
    assert random_comparison["paired_deltas"]["composite"]["n"] == 32
    assert payload["map_contract"]["primary_score"]["score_column"] == (
        "middle_relative_logit_margin"
    )


def test_optional_transition_summary_requires_accept_and_reject(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transition.json"
    path.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "status": "nominal",
                        "failures": [],
                        "events": 2,
                        "total": 4,
                        "accepted": 2,
                        "branches": {"cond": 1, "uncond": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert analysis.load_transition_summary(path)["pass"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summaries"][0]["accepted"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not analysis.load_transition_summary(path)["pass"]


def test_blind_scorecard_is_accepted_only_when_frozen_and_complete(
    tmp_path: Path,
) -> None:
    methods = ("first", "second")
    run_root = tmp_path / "run"
    for method in methods:
        method_dir = run_root / method
        method_dir.mkdir(parents=True)
        (method_dir / "0-0_ema.mp4").write_bytes(method.encode("utf-8"))
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("test prompt\n", encoding="utf-8")
    public = tmp_path / "blind_public"
    private = tmp_path / "blind_private"
    blind_review.create_package(
        run_root=run_root,
        methods=list(methods),
        prompts=prompts,
        output=public,
        private_output=private,
        prompt_count=1,
        seed=7,
        force=False,
    )
    key = private / "key_private.json"
    scorecard = public / "scorecard.csv"
    key_payload = json.loads(key.read_text(encoding="utf-8"))
    method_by_label = {
        candidate["label"]: candidate["method"]
        for candidate in key_payload["items"][0]["candidates"]
    }
    fieldnames = list(analysis.BLIND_SCORE_FIELDS)
    with scorecard.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        method = method_by_label[row["label"]]
        row.update({field: "5" for field in analysis.BLIND_RATING_FIELDS})
        row.update({field: "0" for field in analysis.BLIND_FLAG_FIELDS})
        row["polygon_noise_0_or_1"] = "1" if method == "second" else "0"
        row["overall_rank"] = "1" if row["label"] == "A" else "2"
    with scorecard.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    package_verification = blind_review.verify_package(
        run_root=run_root,
        methods=list(methods),
        prompts=prompts,
        output=public,
        private_output=private,
        prompt_count=1,
        seed=7,
    )
    blind_review.freeze_package(
        output=public,
        private_output=private,
        prompt_count=1,
        method_count=2,
        verification=package_verification,
        force=False,
    )
    verification_path = tmp_path / "blind_verification.json"
    verification_path.write_text(
        json.dumps(
            blind_review.verify_frozen_package(
                output=public,
                private_output=private,
                prompt_count=1,
                method_count=2,
                verification=package_verification,
            )
        ),
        encoding="utf-8",
    )

    result = analysis.load_blind_review(
        scorecard,
        key,
        run_root=run_root,
        prompts_path=prompts,
        verification_path=verification_path,
        methods=methods,
        prompt_count=1,
    )

    assert result["pass"]
    assert result["frozen_verified"]
    assert result["methods"]["first"]["usable"]
    assert not result["methods"]["second"]["usable"]

    for row in rows:
        method = method_by_label[row["label"]]
        row["identity_1_to_5"] = "1" if method == "first" else "5"
        row["polygon_noise_0_or_1"] = "0"
    with scorecard.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    blind_review.freeze_package(
        output=public,
        private_output=private,
        prompt_count=1,
        method_count=2,
        verification=package_verification,
        force=True,
    )
    verification_path.write_text(
        json.dumps(
            blind_review.verify_frozen_package(
                output=public,
                private_output=private,
                prompt_count=1,
                method_count=2,
                verification=package_verification,
            )
        ),
        encoding="utf-8",
    )
    identity_failure = analysis.load_blind_review(
        scorecard,
        key,
        run_root=run_root,
        prompts_path=prompts,
        verification_path=verification_path,
        methods=methods,
        prompt_count=1,
    )
    assert identity_failure["pass"]
    assert not identity_failure["methods"]["first"]["usable"]
    assert identity_failure["methods"]["second"]["usable"]

    scorecard.write_text(
        scorecard.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    stale = analysis.load_blind_review(
        scorecard,
        key,
        run_root=run_root,
        prompts_path=prompts,
        verification_path=verification_path,
        methods=methods,
        prompt_count=1,
    )
    assert not stale["pass"]
    assert not stale["frozen_verified"]
