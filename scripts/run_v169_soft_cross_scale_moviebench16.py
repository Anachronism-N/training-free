#!/usr/bin/env python3
"""Run the v169 soft cross-scale motion-recall experiment."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import analyze_v168_cross_scale_consensus_trace as v168_trace
import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
import run_v157_layer_gated_moviebench16 as v157
import run_v159_motion_coherent_reservoir_moviebench16 as v159
import run_v168_cross_scale_consensus_moviebench16 as v168
import v169_soft_cross_scale_contract as soft_contract
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v169_soft_cross_scale_moviebench16"
PROMPT_COUNT = 16
DIRECTION_MATCH = v168.DIRECTION_MATCH
MULTISCALE_MOTION = v168.MULTISCALE_MOTION
PARETO_MOTION = v168.PARETO_MOTION
QUERY_WEIGHTED = soft_contract.QUERY_WEIGHTED
BOTTLENECK = soft_contract.BOTTLENECK
EXPECTED_METHOD_KEYS = (
    "sf_native",
    DIRECTION_MATCH,
    MULTISCALE_MOTION,
    PARETO_MOTION,
    QUERY_WEIGHTED,
    BOTTLENECK,
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    DIRECTION_MATCH: DIRECTION_MATCH,
    MULTISCALE_MOTION: MULTISCALE_MOTION,
    PARETO_MOTION: PARETO_MOTION,
}
NEW_METHODS = {QUERY_WEIGHTED, BOTTLENECK}
PRIMARY = QUERY_WEIGHTED
_PARENT_RUN_TASK = runner.run_task


V169_CELLS = (
    Cell(
        "v169_middle10_reservoir2_directionmatch1",
        "v169_reference",
        "single",
        support_policy="reservoir2_directionmatch1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v169_middle10_reservoir2_multiscalemotion1",
        "v169_reference",
        "single",
        support_policy="reservoir2_multiscalemotion1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v169_middle10_reservoir2_multiscalepareto1",
        "v169_negative_reference",
        "single",
        support_policy="reservoir2_multiscalepareto1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v169_middle10_reservoir2_multiscalequeryweighted1",
        "v169_soft_cross_scale",
        "single",
        support_policy="reservoir2_multiscalequeryweighted1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v169_middle10_reservoir2_multiscalebottleneck1",
        "v169_soft_cross_scale",
        "single",
        support_policy="reservoir2_multiscalebottleneck1",
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
    runner.TASK_STAGE = "moviebench16"
    runner.PUBLISHED_TAG = "v169"
    runner.RUN_LABEL = "v169"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 5
    runner.DEFAULT_CANDIDATES = (
        "middle10_reservoir2_directionmatch1",
        "middle10_reservoir2_multiscalemotion1",
        "middle10_reservoir2_multiscalepareto1",
        "middle10_reservoir2_multiscalequeryweighted1",
        "middle10_reservoir2_multiscalebottleneck1",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V169_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "middle10_reservoir2_directionmatch1": (
                "v169_middle10_reservoir2_directionmatch1",
                "v166_direction_match_reference",
            ),
            "middle10_reservoir2_multiscalemotion1": (
                "v169_middle10_reservoir2_multiscalemotion1",
                "v166_multiscale_motion_reference",
            ),
            "middle10_reservoir2_multiscalepareto1": (
                "v169_middle10_reservoir2_multiscalepareto1",
                "v168_hard_guard_negative_reference",
            ),
            "middle10_reservoir2_multiscalequeryweighted1": (
                "v169_middle10_reservoir2_multiscalequeryweighted1",
                "query_activity_weighted_cross_scale_recall",
            ),
            "middle10_reservoir2_multiscalebottleneck1": (
                "v169_middle10_reservoir2_multiscalebottleneck1",
                "max_min_cross_scale_recall",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def maybe_filter_smoke_tasks(tasks):
    raw = os.environ.get("V169_SMOKE_PROMPT_INDEX")
    if raw is None or not raw.strip():
        return list(tasks)
    prompt_index = int(raw)
    if not 0 <= prompt_index < PROMPT_COUNT:
        raise ValueError("V169_SMOKE_PROMPT_INDEX must be within [0, 15]")
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


def v168_run_root() -> Path:
    default = ROOT / "runs" / v168.EXPERIMENT / "full8"
    return Path(os.environ.get("V169_REUSE_V168_ROOT", default)).resolve()


def corrected_v168_pareto_trace(root: Path) -> dict:
    trace_dir = root / "traces"
    paths = sorted(trace_dir.glob(f"{PARETO_MOTION}__p*.policy.jsonl"))
    if len(paths) != PROMPT_COUNT:
        raise ValueError("v168 Pareto trace coverage is incomplete")
    prompts = [v168_trace.analyze_prompt(path, method=PARETO_MOTION) for path in paths]
    if [row["prompt_index"] for row in prompts] != list(range(PROMPT_COUNT)):
        raise ValueError("v168 Pareto trace prompt order mismatch")
    aggregate = v168_trace.aggregate_method(prompts, method=PARETO_MOTION)
    if aggregate.get("mechanism_gate") is not True:
        raise ValueError("corrected v168 Pareto trace audit failed")
    return {
        "trace_dir": str(trace_dir.resolve()),
        "trace_count": len(paths),
        "trace_sha256": {path.name: text_sha256_lf(path) for path in paths},
        "aggregate": aggregate,
        "audit_note": (
            "recomputed with the corrected no-passing reason contract; the "
            "checked-in v168 report used a stale reason label"
        ),
    }


def load_v168_source(prompt_manifest: dict, *, pf_runtime: dict) -> dict:
    root = v168_run_root()
    published_path = root / "published_manifest.json"
    experiment_path = root / "contracts" / "experiment.json"
    metrics_path = root / "analysis" / "v168_corrected_metrics.json"
    for path in (published_path, experiment_path, metrics_path):
        if not path.is_file():
            raise ValueError(f"missing frozen v168 reuse source: {path}")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods", [])}
    source_pf = experiment.get("pf", {})
    quality = metrics.get("aggregate_official_quality_score", {})
    required_sources = set(REUSE_METHODS.values())
    if (
        published.get("ok") is not True
        or published.get("experiment") != v168.EXPERIMENT
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("prompt_file_sha256") != prompt_manifest["prompt_file_sha256"]
        or published.get("experiment_contract_sha256")
        != text_sha256_lf(experiment_path)
        or experiment.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or source_pf.get("config_sha256") != pf_runtime["config_sha256"]
        or int(source_pf.get("checkpoint_size", -1))
        != int(pf_runtime["checkpoint_size"])
        or not required_sources.issubset(rows)
        or metrics.get("experiment") != "v168_corrected_metric_analysis"
        or not required_sources.issubset(quality)
    ):
        raise ValueError("v168 source does not match the frozen v169 contract")

    pareto_trace = corrected_v168_pareto_trace(root)
    expected_names = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    sources = {}
    for target, source_method in REUSE_METHODS.items():
        video_dir = root / "published" / source_method
        video_paths = sorted(video_dir.glob("*.mp4"))
        source_row = rows[source_method]
        total_bytes = sum(path.stat().st_size for path in video_paths)
        if (
            {path.name for path in video_paths} != expected_names
            or any(path.stat().st_size <= 0 for path in video_paths)
            or int(source_row.get("video_count", -1)) != PROMPT_COUNT
            or int(source_row.get("total_bytes", -1)) != total_bytes
        ):
            raise ValueError(f"incomplete v168 reuse source: {source_method}")
        sources[target] = {
            "source_method": source_method,
            "video_dir": str(video_dir.resolve()),
            "total_bytes": total_bytes,
        }
    return {
        "root": str(root),
        "published_manifest": str(published_path),
        "published_manifest_sha256": sha256(published_path),
        "experiment_contract": str(experiment_path),
        "experiment_contract_sha256": text_sha256_lf(experiment_path),
        "experiment_contract_raw_sha256": sha256(experiment_path),
        "corrected_metrics": str(metrics_path),
        "corrected_metrics_sha256": text_sha256_lf(metrics_path),
        "multiscale_motion_quality": quality[MULTISCALE_MOTION],
        "pareto_quality": quality[PARETO_MOTION],
        "corrected_pareto_trace": pareto_trace,
        "matched_pf_runtime": pf_runtime,
        "sources": sources,
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
    reuse = args.v168_source["sources"][method.key]
    source = Path(reuse["video_dir"]) / f"{prompt_index:06d}.mp4"
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
            "source_method": reuse["source_method"],
            "source": str(source),
            "target": str(target),
            "indexed_target": str(indexed),
            "size": source.stat().st_size,
            "reuse_manifest_sha256": args.v168_source["published_manifest_sha256"],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v168",
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
    experiment.update(
        {
            "version": 1,
            "prompt_suite": prompt_manifest,
            "head_membership": {
                "meaning": "layer policy gate, not a semantic head class",
                "label10": "Middle10 layer; all 12 heads use tested cache",
                "label11": "other layer; all heads use sink1+recent8",
            },
            "layer_membership": {
                key: map_audits[key] for key in ("interleaved10", "middle10")
            },
            "layer_map_manifest": layer_manifest,
            "v168_source": args.v168_source,
            "cache_contract": {
                "selected_layers": (
                    "sink1 + reservoir2 + one recalled atomic pair + recent4"
                ),
                "other_layers": "sink1 + recent8",
                "motion_archive_pairs": 4,
                "motion_read_pairs": 1,
                "admission_freshness_horizon": 12,
                "read_max_age": 24,
                "direction_similarity_floor": 0.1,
                "no_match_fallback": "newest age-eligible atomic pair",
                "atomic_pair_read": True,
                "max_read_full_frame_equivalents": 9,
                "exclusive_dynamic_owner": True,
            },
            "soft_cross_scale_selection": {
                "local_component": (
                    "local direction cosine times local magnitude match"
                ),
                "context_component": (
                    "context direction cosine times per-step context magnitude match"
                ),
                "query_weighted": (
                    "weighted mean of local/context components; weights are "
                    "their current query displacement norms normalized to sum 1"
                ),
                "bottleneck": (
                    "maximize the smaller available local/context component"
                ),
                "candidate_gate": "frozen v166 mean-direction floor 0.1",
                "tie_break": "v166 score, then newer pair",
                "learned_parameters": [],
                "new_thresholds": [],
                "hard_newest_fallback_on_conflict": False,
            },
            "design": {
                "primary": PRIMARY,
                "new_methods": [QUERY_WEIGHTED, BOTTLENECK],
                "new_video_count": len(NEW_METHODS) * PROMPT_COUNT,
                "reuse_video_count": len(REUSE_METHODS) * PROMPT_COUNT,
                "isolated_test": (
                    "replace only v166 equal cross-scale aggregation; cache "
                    "allocation, candidate set and archive updates are fixed"
                ),
                "offline_gate": (
                    "each selector changes 0-20% of passing v166 decisions, "
                    "changes at least one conflict and preserves old recalls"
                ),
            },
            "evaluation": {
                "mechanism_gate": (
                    "independently recompute primitive scores, both selectors, "
                    "fallback and actual atomic reads"
                ),
                "primary_comparison": MULTISCALE_MOTION,
                "secondary_comparisons": ["sf_native", PARETO_MOTION],
                "promotion": (
                    "relative to v166, aggregate official Quality, identity/"
                    "background, temporal mechanics and dynamic degree must "
                    "all be nonnegative"
                ),
                "automatic_first": (
                    "no manual review unless corrected metrics leave a close "
                    "candidate; review remains capped at two prompts"
                ),
            },
            "claim_boundary": (
                "v169 is a 16-prompt adaptive development experiment and "
                "cannot be reported as held-out paper evidence"
            ),
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "v169_soft_cross_scale_contract.py",
        ROOT / "scripts" / "analyze_v169_offline_counterfactual.py",
        ROOT / "scripts" / "analyze_v169_soft_cross_scale_trace.py",
        ROOT / "scripts" / "run_v100_fast_selection_1video.py",
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
    pf_runtime = v159.load_pf_runtime_contract(
        args.pf_config,
        args.pf_checkpoint,
    )
    args.v168_source = load_v168_source(
        prompt_manifest,
        pf_runtime=pf_runtime,
    )
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            f"v169 requires the frozen six-method order: {EXPECTED_METHOD_KEYS}"
        )
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
    frozen_contract = build_contract(
        args,
        methods=methods,
        prompts=prompts,
        prompt_manifest=prompt_manifest,
        layer_manifest=layer_manifest,
        map_audits=map_audits,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    contract_sha = (
        write_frozen(contract_path, frozen_contract)
        if args.node_rank == 0
        else wait_for_frozen(
            contract_path,
            frozen_contract,
            args.contract_wait_seconds,
        )
    )
    print(
        "[V169Contract] "
        + canonical_json(
            {
                "methods": [method.key for method in methods],
                "contract_sha256": contract_sha,
                "new_videos": len(NEW_METHODS) * PROMPT_COUNT,
                "reused_videos": len(REUSE_METHODS) * PROMPT_COUNT,
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
            f"[v169-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"methods={len(methods)} total_tasks={len(runner.all_tasks(methods))} "
            f"node_tasks={len(tasks)} new={new_count} "
            f"reused={len(tasks) - new_count} gpus={len(gpus)}",
            flush=True,
        )
        return
    if args.mode == "audit":
        payload = runner.audit_published(
            args,
            methods=methods,
            contract_sha256=contract_sha,
        )
        print(
            f"[v169-audit] PASS methods={len(payload['methods'])} "
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
            except Exception as error:
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
        raise SystemExit("\n".join(failures or ["v169 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
