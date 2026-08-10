#!/usr/bin/env python3
"""Run the v170 matched, order-balanced attribution experiment."""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_v100_fast_selection_1video as fast
import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
import run_v157_layer_gated_moviebench16 as v157
import v170_matched_attribution_contract as contract
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v170_matched_attribution_moviebench16"
PROMPT_COUNT = contract.PROMPT_COUNT
_PARENT_RUN_TASK = runner.run_task

V170_CELLS = (
    Cell(
        "v170_v166_a",
        "v170_matched_reference",
        "single",
        support_policy="reservoir2_multiscalemotion1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v170_queryweighted_a",
        "v170_matched_candidate",
        "single",
        support_policy="reservoir2_multiscalequeryweighted1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v170_v166_b",
        "v170_matched_reference",
        "single",
        support_policy="reservoir2_multiscalemotion1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v170_queryweighted_b",
        "v170_matched_candidate",
        "single",
        support_policy="reservoir2_multiscalequeryweighted1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
)

CANDIDATE_KEYS = (
    "v170_v166_a",
    "v170_queryweighted_a",
    "v170_v166_b",
    "v170_queryweighted_b",
)


def configure_parent_runner() -> None:
    runner.EXPERIMENT = EXPERIMENT
    runner.PROMPT_COUNT = PROMPT_COUNT
    runner.TASK_STAGE = "matched_moviebench16"
    runner.PUBLISHED_TAG = "v170"
    runner.RUN_LABEL = "v170"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = True
    runner.MAX_CANDIDATES = len(CANDIDATE_KEYS)
    runner.DEFAULT_CANDIDATES = CANDIDATE_KEYS
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V170_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "v170_v166_a": ("v170_v166_a", "v166_replica_lane_a"),
            "v170_queryweighted_a": (
                "v170_queryweighted_a",
                "queryweighted_replica_lane_a",
            ),
            "v170_v166_b": ("v170_v166_b", "v166_replica_lane_b"),
            "v170_queryweighted_b": (
                "v170_queryweighted_b",
                "queryweighted_replica_lane_b",
            ),
        }
    )
    runner.run_task = _PARENT_RUN_TASK
    fast.TRACE_LAYERS = contract.ACTIVE_LAYERS
    configured_heads = os.environ.get("PYRAMIDKV_POLICY_TRACE_HEADS", "0")
    if configured_heads.strip() != "0":
        raise ValueError("v170 requires PYRAMIDKV_POLICY_TRACE_HEADS=0")
    os.environ["PYRAMIDKV_POLICY_TRACE_HEADS"] = "0"


def v169_evidence() -> dict[str, object]:
    root = ROOT / "runs" / "v169_soft_cross_scale_moviebench16" / "full8"
    paths = {
        "metrics": root / "analysis" / "v169_corrected_metrics.json",
        "review": root / "minimal_review" / "reviewer" / "review_sheet.csv",
        "blind_key": root / "minimal_review" / "private" / "blind_key.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"missing frozen v169 evidence: {path}")
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    blind = json.loads(paths["blind_key"].read_text(encoding="utf-8"))
    with paths["review"].open("r", encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    noted = [row for row in review_rows if str(row.get("notes", "")).strip()]
    if (
        metrics.get("experiment") != "v169_corrected_metric_analysis"
        or blind.get("experiment") != "v169_minimal_blind_review"
        or len(review_rows) != 4
        or len(noted) != 2
    ):
        raise ValueError("v169 metrics/review evidence is incomplete")
    return {
        "root": str(root.resolve()),
        "files": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "reviewed_video_count": len(review_rows),
        "noted_pair_count": len(noted),
        "resolved_conclusion": (
            "both selected prompt pairs preferred v166 after unblinding; "
            "prompt 3 contradicted the positive VBench delta"
        ),
    }


def build_contract(
    args,
    *,
    methods,
    prompts: list[str],
    prompt_manifest: dict,
    layer_manifest: dict,
    map_audits: dict,
) -> dict[str, object]:
    payload = runner.experiment_contract(
        args,
        methods=methods,
        prompts=prompts,
        map_audit=map_audits["middle10"],
    )
    implementation = ROOT / "scripts" / "v170_matched_attribution_contract.py"
    payload.update(
        {
            "version": 2,
            "prompt_suite": prompt_manifest,
            "v169_evidence": v169_evidence(),
            "layer_map_manifest": layer_manifest,
            "layer_membership": map_audits["middle10"],
            "cache_contract": {
                "active_layers": list(contract.ACTIVE_LAYERS),
                "active": "sink1 + reservoir2 + recalled atomic pair2 + recent4",
                "other": "sink1 + recent8",
                "archive_pairs": 4,
                "read_pairs": 1,
                "max_read_age": 24,
                "max_full_frame_equivalents": 9,
            },
            "matched_design": {
                "methods": list(contract.METHODS),
                "replicas_per_policy": 2,
                "same_gpu_within_lane": True,
                "order_balanced_by_lane_and_prompt_parity": True,
                "num_nodes": contract.NUM_NODES,
                "gpus_per_node": contract.GPUS_PER_NODE,
                "total_new_videos": PROMPT_COUNT * len(contract.METHODS),
                "schedule": list(contract.full_schedule()),
            },
            "trace_contract": {
                "layers": list(contract.ACTIVE_LAYERS),
                "heads": list(contract.TRACE_HEADS),
                "stride": 3,
                "reason": (
                    "audit every active layer while avoiding twelvefold "
                    "head-replicated trace volume"
                ),
            },
        }
    )
    payload["implementation_hashes"][str(implementation.relative_to(ROOT))] = sha256(
        implementation
    )
    return payload


def task_for(methods_by_key: dict, method_key: str, prompt_index: int):
    method = methods_by_key[method_key]
    return method, prompt_index, runner.task_cell(method, prompt_index)


def worker_plan(methods, *, node_rank: int, num_nodes: int, smoke: bool):
    methods_by_key = {method.key: method for method in methods}
    if set(methods_by_key) != set(contract.METHODS):
        raise ValueError("v170 method set mismatch")
    if smoke:
        prompt_index = int(os.environ.get("V170_SMOKE_PROMPT_INDEX", "3"))
        if not 0 <= prompt_index < PROMPT_COUNT:
            raise ValueError("V170_SMOKE_PROMPT_INDEX must be within [0, 15]")
        rows = []
        for lane_offset, lane in enumerate(("a", "b")):
            tasks = [
                task_for(methods_by_key, key, prompt_index)
                for key in contract.lane_methods(prompt_index, lane)
            ]
            rows.append((lane_offset, tasks))
        return rows
    return [
        (
            gpu_slot,
            [
                task_for(methods_by_key, str(row["method"]), int(row["prompt_index"]))
                for row in contract.node_schedule(node_rank, num_nodes)
                if int(row["gpu_slot"]) == gpu_slot
            ],
        )
        for gpu_slot in range(contract.GPUS_PER_NODE)
    ]


def main() -> None:
    configure_parent_runner()
    args = runner.parse_args()
    prompts, prompt_manifest = v155.load_prompt_suite(args)
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != contract.METHODS:
        raise SystemExit(f"v170 requires method order {contract.METHODS}")
    layer_manifest, layer_paths, map_audits = v157.load_layer_maps(args)
    args.head_maps = layer_paths
    args.head_map_audits = map_audits
    for name in (
        "logs",
        "traces",
        "configs",
        "status",
        "status/published",
        "diagnostics",
        "contracts",
        "videos",
        "published",
        "published_indexed",
    ):
        (args.out_root / name).mkdir(parents=True, exist_ok=True)
    frozen = build_contract(
        args,
        methods=methods,
        prompts=prompts,
        prompt_manifest=prompt_manifest,
        layer_manifest=layer_manifest,
        map_audits=map_audits,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    contract_sha = (
        write_frozen(contract_path, frozen)
        if args.node_rank == 0
        else wait_for_frozen(
            contract_path,
            frozen,
            args.contract_wait_seconds,
        )
    )
    print(
        "[V170Contract] "
        + canonical_json(
            {
                "contract_sha256": contract_sha,
                "methods": list(contract.METHODS),
                "new_videos": PROMPT_COUNT * len(contract.METHODS),
                "trace_layers": list(contract.ACTIVE_LAYERS),
                "trace_heads": list(contract.TRACE_HEADS),
            }
        )
        .decode("utf-8")
        .strip(),
        flush=True,
    )
    smoke = os.environ.get("V170_SMOKE", "0") == "1"
    gpus = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
    required_gpus = 2 if smoke else contract.GPUS_PER_NODE
    if len(gpus) != required_gpus or len(gpus) != len(set(gpus)):
        raise SystemExit(f"v170 requires exactly {required_gpus} unique GPUs")
    if not smoke and args.num_nodes != contract.NUM_NODES:
        raise SystemExit("v170 full run requires NUM_NODES=4")
    plans = worker_plan(
        methods,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
        smoke=smoke,
    )
    if args.mode == "preflight":
        print(
            "[v170-preflight] PASS "
            + json.dumps(
                {
                    "node_rank": args.node_rank,
                    "num_nodes": args.num_nodes,
                    "workers": [
                        {
                            "gpu": gpus[slot],
                            "tasks": [
                                runner.task_name(method, prompt)
                                for method, prompt, _ in tasks
                            ],
                        }
                        for slot, tasks in plans
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.mode == "audit":
        if args.node_rank != 0:
            raise SystemExit("v170 audit requires NODE_RANK=0")
        payload = runner.audit_published(
            args,
            methods=methods,
            contract_sha256=contract_sha,
        )
        print(
            f"[v170-audit] PASS methods={len(payload['methods'])} "
            f"manifest={args.out_root / 'published_manifest.json'}",
            flush=True,
        )
        return

    results: list[dict] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(plans)) as executor:
        futures = {
            executor.submit(
                runner.run_worker,
                args,
                gpu=gpus[slot],
                tasks=tasks,
                contract_sha256=contract_sha,
            ): gpus[slot]
            for slot, tasks in plans
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                results.extend(future.result())
            except Exception as error:  # noqa: BLE001 - preserve all worker failures
                failures.append(f"gpu={gpu}: {error}")
    failures.extend(
        f"{row.get('name')}: {row.get('error')}"
        for row in results
        if row.get("status") == "failed"
    )
    expected = sum(len(tasks) for _, tasks in plans)
    summary = {
        "version": 1,
        "experiment": EXPERIMENT,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "task_count": expected,
        "result_count": len(results),
        "results": sorted(
            results,
            key=lambda row: (
                str(row.get("method", row.get("name", ""))),
                int(row.get("prompt_index", -1)),
            ),
        ),
        "failures": failures,
        "ok": not failures and len(results) == expected,
    }
    suffix = "smoke" if smoke else f"node{args.node_rank}"
    summary_path = args.out_root / "status" / f"{suffix}.summary.json"
    write_runtime_json(summary_path, summary)
    if not summary["ok"]:
        raise SystemExit("\n".join(failures or ["v170 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
