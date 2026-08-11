#!/usr/bin/env python3
"""Run the v171 demand-gated motion-recall experiment."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_v100_fast_selection_1video as fast
import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
import run_v157_layer_gated_moviebench16 as v157
import run_v159_motion_coherent_reservoir_moviebench16 as v159
import v171_demand_gated_contract as contract
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v171_demand_gated_motion_moviebench16"
PROMPT_COUNT = contract.PROMPT_COUNT
REUSE_METHODS = {contract.V166: "ours_v170_v166_a"}
NEW_METHODS = set(contract.CANDIDATES)
_PARENT_RUN_TASK = runner.run_task


V171_CELLS = (
    Cell(
        "v171_v166_reference",
        "v171_reused_reference",
        "single",
        support_policy="reservoir2_multiscalemotion1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v171_deficit_query",
        "v171_demand_gated_control",
        "single",
        support_policy="reservoir2_deficitquery1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v171_deficit_baseline",
        "v171_demand_gated_primary",
        "single",
        support_policy="reservoir2_deficitbaseline1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
)


def text_sha256_lf(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def configure_parent_runner() -> None:
    runner.EXPERIMENT = EXPERIMENT
    runner.PROMPT_COUNT = PROMPT_COUNT
    runner.TASK_STAGE = "demand_gated_moviebench16"
    runner.PUBLISHED_TAG = "v171"
    runner.RUN_LABEL = "v171"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = True
    runner.MAX_CANDIDATES = len(contract.METHODS)
    runner.DEFAULT_CANDIDATES = (
        "middle10_reservoir2_multiscalemotion1",
        "middle10_reservoir2_deficitquery1",
        "middle10_reservoir2_deficitbaseline1",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V171_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "middle10_reservoir2_multiscalemotion1": (
                "v171_v166_reference",
                "v166_reused_reference",
            ),
            "middle10_reservoir2_deficitquery1": (
                "v171_deficit_query",
                "deficit_gated_query_control",
            ),
            "middle10_reservoir2_deficitbaseline1": (
                "v171_deficit_baseline",
                "baseline_calibrated_motion_recall",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse
    fast.TRACE_LAYERS = contract.ACTIVE_LAYERS
    configured_heads = os.environ.get("PYRAMIDKV_POLICY_TRACE_HEADS", "0")
    if configured_heads.strip() != "0":
        raise ValueError("v171 requires PYRAMIDKV_POLICY_TRACE_HEADS=0")
    os.environ["PYRAMIDKV_POLICY_TRACE_HEADS"] = "0"


def maybe_filter_smoke_tasks(tasks):
    raw = os.environ.get("V171_SMOKE_PROMPT_INDEX")
    if raw is None or not raw.strip():
        return list(tasks)
    prompt_index = int(raw)
    if not 0 <= prompt_index < PROMPT_COUNT:
        raise ValueError("V171_SMOKE_PROMPT_INDEX must be within [0, 15]")
    selected = [
        task
        for task in tasks
        if task[0].key in NEW_METHODS and int(task[1]) == prompt_index
    ]
    if len(selected) != len(NEW_METHODS):
        raise RuntimeError(
            f"smoke filter expected {len(NEW_METHODS)} tasks, got {len(selected)}"
        )
    return selected


def v170_run_root() -> Path:
    default = ROOT / "runs" / "v170_matched_attribution_moviebench16" / "full8"
    return Path(os.environ.get("V171_REUSE_V170_ROOT", default)).resolve()


def load_v170_source(prompt_manifest: dict, *, pf_runtime: dict) -> dict:
    root = v170_run_root()
    paths = {
        "published_manifest": root / "published_manifest.json",
        "experiment_contract": root / "contracts" / "experiment.json",
        "matched_metrics": root / "analysis" / "v170_matched_metrics.json",
        "mechanism_trace": root / "automated_screen" / "full_layer_trace.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"missing frozen v170 reuse source: {path}")
    published = json.loads(paths["published_manifest"].read_text(encoding="utf-8"))
    experiment = json.loads(paths["experiment_contract"].read_text(encoding="utf-8"))
    metrics = json.loads(paths["matched_metrics"].read_text(encoding="utf-8"))
    mechanism = json.loads(paths["mechanism_trace"].read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods", [])}
    source_method = REUSE_METHODS[contract.V166]
    source_pf = experiment.get("pf", {})
    if (
        published.get("ok") is not True
        or published.get("experiment") != "v170_matched_attribution_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or published.get("experiment_contract_sha256")
        != text_sha256_lf(paths["experiment_contract"])
        or source_method not in rows
        or experiment.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or source_pf.get("config_sha256") != pf_runtime["config_sha256"]
        or int(source_pf.get("checkpoint_size", -1))
        != int(pf_runtime["checkpoint_size"])
        or metrics.get("experiment") != "v170_matched_metric_analysis"
        or mechanism.get("mechanism_gate") is not True
        or metrics.get("development_decision", {}).get("attribution_gate") is not False
    ):
        raise ValueError("v170 source does not match the frozen v171 contract")

    video_dir = root / "published" / source_method
    video_paths = sorted(video_dir.glob("*.mp4"))
    expected_names = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    total_bytes = sum(path.stat().st_size for path in video_paths)
    source_row = rows[source_method]
    if (
        {path.name for path in video_paths} != expected_names
        or any(path.stat().st_size <= 0 for path in video_paths)
        or int(source_row.get("video_count", -1)) != PROMPT_COUNT
        or int(source_row.get("total_bytes", -1)) != total_bytes
    ):
        raise ValueError("incomplete v170 v166 reuse source")
    return {
        "root": str(root),
        "source_method": source_method,
        "video_dir": str(video_dir.resolve()),
        "total_bytes": total_bytes,
        "published_manifest": str(paths["published_manifest"].resolve()),
        "published_manifest_sha256": sha256(paths["published_manifest"]),
        "experiment_contract": str(paths["experiment_contract"].resolve()),
        "experiment_contract_sha256": text_sha256_lf(
            paths["experiment_contract"]
        ),
        "matched_metrics": str(paths["matched_metrics"].resolve()),
        "matched_metrics_sha256": sha256(paths["matched_metrics"]),
        "mechanism_trace": str(paths["mechanism_trace"].resolve()),
        "mechanism_trace_sha256": sha256(paths["mechanism_trace"]),
        "v170_decision": metrics["development_decision"],
        "matched_pf_runtime": pf_runtime,
    }


def run_reused_task(
    args,
    *,
    method,
    prompt_index: int,
    cell,
    gpu: str,
    contract_sha256: str,
) -> dict:
    source = Path(args.v170_source["video_dir"]) / f"{prompt_index:06d}.mp4"
    target = (
        args.out_root / "published" / method.key / runner.published_name(prompt_index)
    )
    indexed = (
        args.out_root
        / "published_indexed"
        / method.key
        / runner.published_name(prompt_index, indexed=True)
    )
    link_mode = runner.link_or_validate(source, target)
    indexed_mode = runner.link_or_validate(source, indexed)
    marker = (
        args.out_root
        / "status"
        / "published"
        / f"{method.key}.p{prompt_index:03d}.json"
    )
    write_frozen(
        marker,
        {
            "version": 1,
            "experiment_contract_sha256": contract_sha256,
            "method": method.key,
            "engine": method.engine,
            "prompt_index": prompt_index,
            "task_cell": cell.name,
            "source_method": args.v170_source["source_method"],
            "source": str(source),
            "target": str(target),
            "indexed_target": str(indexed),
            "size": source.stat().st_size,
            "reuse_manifest_sha256": args.v170_source[
                "published_manifest_sha256"
            ],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v170_v166_a",
        "gpu": str(gpu),
    }


def run_task_with_optional_reuse(args, **kwargs):
    if kwargs["method"].key in REUSE_METHODS:
        return run_reused_task(args, **kwargs)
    return _PARENT_RUN_TASK(args, **kwargs)


def build_contract(
    args,
    *,
    methods,
    prompts: list[str],
    prompt_manifest: dict,
    layer_manifest: dict,
    map_audits: dict,
) -> dict:
    experiment = runner.experiment_contract(
        args,
        methods=methods,
        prompts=prompts,
        map_audit=map_audits["middle10"],
    )
    offline_path = (
        ROOT
        / "runs"
        / EXPERIMENT
        / "offline"
        / "v171_counterfactual.json"
    )
    if not offline_path.is_file():
        raise ValueError(f"missing frozen v171 offline gate: {offline_path}")
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    if offline.get("offline_gate") is not True:
        raise ValueError("v171 offline counterfactual gate did not pass")
    experiment.update(
        {
            "version": 1,
            "prompt_suite": prompt_manifest,
            "layer_map_manifest": layer_manifest,
            "layer_membership": map_audits["middle10"],
            "policy_scope": {
                "meaning": "layer gate, not a semantic head taxonomy",
                "active_layers": list(contract.ACTIVE_LAYERS),
                "active_heads_per_layer": 12,
                "other_layers": "recent-only control",
            },
            "v170_source": args.v170_source,
            "offline_counterfactual": {
                "path": str(offline_path.resolve()),
                "sha256": sha256(offline_path),
                "coverage": offline["coverage"],
                "methods": offline["methods"],
            },
            "cache_contract": {
                "active": "sink1 + reservoir2 + recalled atomic pair2 + recent4",
                "other": "sink1 + recent8",
                "archive_pairs": 4,
                "read_pairs": 1,
                "admission_freshness_horizon": 12,
                "max_read_age": 24,
                "direction_similarity_floor": 0.1,
                "max_full_frame_equivalents": 9,
                "exclusive_dynamic_owner": True,
            },
            "demand_gate": {
                "warmup_updates": 4,
                "local_signal": "norm(z_last - z_previous)",
                "context_signal": "norm(z_last - z_first) / block_steps",
                "baseline": "per-layer online median over previous updates",
                "trigger": "both local and context are below their baselines",
                "dataset_tuned_thresholds": [],
            },
            "selectors": {
                contract.DEFICIT_QUERY: (
                    "v166 when healthy; v170 query-weighted score on deficit"
                ),
                contract.DEFICIT_BASELINE: (
                    "v166 when healthy; current direction plus online-baseline "
                    "magnitude compatibility on deficit"
                ),
                "tie_break": "v166 score then newer atomic pair",
            },
            "design": {
                "primary": contract.DEFICIT_BASELINE,
                "mechanism_control": contract.DEFICIT_QUERY,
                "reference": contract.V166,
                "new_video_count": len(NEW_METHODS) * PROMPT_COUNT,
                "reused_video_count": len(REUSE_METHODS) * PROMPT_COUNT,
                "fixed_prompt_count": PROMPT_COUNT,
                "duration_seconds": 30,
                "manual_review_default": False,
            },
            "trace_contract": {
                "layers": list(contract.ACTIVE_LAYERS),
                "heads": list(contract.TRACE_HEADS),
                "independent_recomputation": True,
            },
            "claim_boundary": (
                "v171 is an adaptive 16-prompt development experiment. Passing "
                "only authorizes matched and held-out confirmation."
            ),
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "v171_demand_gated_contract.py",
        ROOT / "scripts" / "analyze_v171_demand_gated_counterfactual.py",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "role_event.py",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "policy_overrides.py",
        ROOT / "configs" / "head_maps" / v157.MANIFEST_FILENAME,
        ROOT / "configs" / "head_maps" / v157.MAP_FILENAMES["middle10"],
        ROOT / "prompts" / v155.PROMPT_FILENAME,
        ROOT / "prompts" / v155.PROMPT_MANIFEST_FILENAME,
    )
    experiment["implementation_hashes"].update(
        {str(path.relative_to(ROOT)): sha256(path) for path in extra_paths}
    )
    return experiment


def main() -> None:
    configure_parent_runner()
    args = runner.parse_args()
    prompts, prompt_manifest = v155.load_prompt_suite(args)
    pf_runtime = v159.load_pf_runtime_contract(args.pf_config, args.pf_checkpoint)
    args.v170_source = load_v170_source(
        prompt_manifest,
        pf_runtime=pf_runtime,
    )
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != contract.METHODS:
        raise SystemExit(f"v171 requires method order {contract.METHODS}")
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
        "[V171Contract] "
        + canonical_json(
            {
                "methods": [method.key for method in methods],
                "contract_sha256": contract_sha,
                "new_videos": len(NEW_METHODS) * PROMPT_COUNT,
                "reused_videos": len(REUSE_METHODS) * PROMPT_COUNT,
                "trace_layers": list(contract.ACTIVE_LAYERS),
                "trace_heads": list(contract.TRACE_HEADS),
            }
        )
        .decode("utf-8")
        .strip(),
        flush=True,
    )
    tasks = maybe_filter_smoke_tasks(
        runner.selected_tasks(
            methods,
            node_rank=args.node_rank,
            num_nodes=args.num_nodes,
        )
    )
    gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    if args.mode == "preflight":
        new_count = sum(method.key in NEW_METHODS for method, _, _ in tasks)
        print(
            f"[v171-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"total_tasks={len(runner.all_tasks(methods))} "
            f"node_tasks={len(tasks)} new={new_count} "
            f"reused={len(tasks) - new_count} gpus={len(gpus)}",
            flush=True,
        )
        return
    if args.mode == "audit":
        if args.node_rank != 0:
            raise SystemExit("v171 audit requires NODE_RANK=0")
        payload = runner.audit_published(
            args,
            methods=methods,
            contract_sha256=contract_sha,
        )
        print(
            f"[v171-audit] PASS methods={len(payload['methods'])} "
            f"manifest={args.out_root / 'published_manifest.json'}",
            flush=True,
        )
        return

    task_list = list(tasks)
    worker_tasks = [task_list[index :: len(gpus)] for index in range(len(gpus))]
    worker_tasks = [items for items in worker_tasks if items]
    results: list[dict] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, len(worker_tasks))) as executor:
        futures = {
            executor.submit(
                runner.run_worker,
                args,
                gpu=gpus[index],
                tasks=items,
                contract_sha256=contract_sha,
            ): gpus[index]
            for index, items in enumerate(worker_tasks)
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                results.extend(future.result())
            except Exception as error:  # noqa: BLE001 - retain all worker errors
                failures.append(f"gpu={gpu}: {error}")
    failures.extend(
        f"{row.get('name')}: {row.get('error')}"
        for row in results
        if row.get("status") == "failed"
    )
    summary = {
        "version": 1,
        "experiment": EXPERIMENT,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "task_count": len(task_list),
        "result_count": len(results),
        "results": sorted(
            results,
            key=lambda row: (
                str(row.get("method", row.get("name", ""))),
                int(row.get("prompt_index", -1)),
            ),
        ),
        "failures": failures,
        "ok": not failures and len(results) == len(task_list),
    }
    summary_path = args.out_root / "status" / f"node{args.node_rank}.summary.json"
    write_runtime_json(summary_path, summary)
    if not summary["ok"]:
        raise SystemExit("\n".join(failures or ["v171 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
