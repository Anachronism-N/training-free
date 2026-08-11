#!/usr/bin/env python3
"""Run the v172 normalized-depth cache dose and placement experiment."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
import run_v159_motion_coherent_reservoir_moviebench16 as v159
import run_v166_multiscale_motion_moviebench16 as v166
from analyze_v152_one_sided_history_critical import audit_binary_map
from build_v172_relative_depth_maps import (
    MANIFEST_FILENAME,
    MAP_FILENAMES,
    MAP_SPECS,
    build_manifest,
)
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v172_relative_depth_moviebench16"
PROMPT_COUNT = 16
POLICY = "reservoir2_multiscalemotion1"
CENTER_1OF6 = "ours_depth_center_1of6_multiscalemotion"
CENTER_1OF4 = "ours_depth_center_1of4_multiscalemotion"
CENTER_1OF3 = "ours_depth_center_1of3_multiscalemotion_reference"
CENTER_1OF2 = "ours_depth_center_1of2_multiscalemotion"
EARLY_1OF3 = "ours_depth_early_1of3_multiscalemotion"
LATE_1OF3 = "ours_depth_late_1of3_multiscalemotion"
INTERLEAVED_1OF3 = "ours_depth_interleaved_1of3_multiscalemotion"
ALL_LAYERS = "ours_depth_all_multiscalemotion"
EXPECTED_METHOD_KEYS = (
    "sf_native",
    CENTER_1OF6,
    CENTER_1OF4,
    CENTER_1OF3,
    CENTER_1OF2,
    EARLY_1OF3,
    LATE_1OF3,
    INTERLEAVED_1OF3,
    ALL_LAYERS,
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    CENTER_1OF3: v166.MULTISCALE_MOTION,
}
NEW_METHODS = set(EXPECTED_METHOD_KEYS) - set(REUSE_METHODS)
_PARENT_RUN_TASK = runner.run_task


V172_CELLS = (
    Cell(
        "v172_depth_center_1of6_multiscalemotion",
        "v172_depth_dose",
        "single",
        support_policy=POLICY,
        suppress_policy="recent8_sink1",
        map_key="center_1of6",
    ),
    Cell(
        "v172_depth_center_1of4_multiscalemotion",
        "v172_depth_dose",
        "single",
        support_policy=POLICY,
        suppress_policy="recent8_sink1",
        map_key="center_1of4",
    ),
    Cell(
        "v172_depth_center_1of3_multiscalemotion",
        "v172_depth_reference",
        "single",
        support_policy=POLICY,
        suppress_policy="recent8_sink1",
        map_key="center_1of3",
    ),
    Cell(
        "v172_depth_center_1of2_multiscalemotion",
        "v172_depth_dose",
        "single",
        support_policy=POLICY,
        suppress_policy="recent8_sink1",
        map_key="center_1of2",
    ),
    Cell(
        "v172_depth_early_1of3_multiscalemotion",
        "v172_depth_placement",
        "single",
        support_policy=POLICY,
        suppress_policy="recent8_sink1",
        map_key="early_1of3",
    ),
    Cell(
        "v172_depth_late_1of3_multiscalemotion",
        "v172_depth_placement",
        "single",
        support_policy=POLICY,
        suppress_policy="recent8_sink1",
        map_key="late_1of3",
    ),
    Cell(
        "v172_depth_interleaved_1of3_multiscalemotion",
        "v172_depth_placement",
        "single",
        support_policy=POLICY,
        suppress_policy="recent8_sink1",
        map_key="interleaved_1of3",
    ),
    Cell(
        "v172_depth_all_multiscalemotion",
        "v172_depth_upper_bound",
        "single",
        support_policy=POLICY,
        suppress_policy=POLICY,
        map_key="center_1of3",
    ),
)


def configure_parent_runner() -> None:
    runner.EXPERIMENT = EXPERIMENT
    runner.PROMPT_COUNT = PROMPT_COUNT
    runner.TASK_STAGE = "moviebench16"
    runner.PUBLISHED_TAG = "v172"
    runner.RUN_LABEL = "v172"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 8
    runner.DEFAULT_CANDIDATES = (
        "depth_center_1of6_multiscalemotion",
        "depth_center_1of4_multiscalemotion",
        "depth_center_1of3_multiscalemotion_reference",
        "depth_center_1of2_multiscalemotion",
        "depth_early_1of3_multiscalemotion",
        "depth_late_1of3_multiscalemotion",
        "depth_interleaved_1of3_multiscalemotion",
        "depth_all_multiscalemotion",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V172_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "depth_center_1of6_multiscalemotion": (
                "v172_depth_center_1of6_multiscalemotion",
                "normalized_center_dose_1of6",
            ),
            "depth_center_1of4_multiscalemotion": (
                "v172_depth_center_1of4_multiscalemotion",
                "normalized_center_dose_1of4",
            ),
            "depth_center_1of3_multiscalemotion_reference": (
                "v172_depth_center_1of3_multiscalemotion",
                "v166_center_third_reference",
            ),
            "depth_center_1of2_multiscalemotion": (
                "v172_depth_center_1of2_multiscalemotion",
                "normalized_center_dose_1of2",
            ),
            "depth_early_1of3_multiscalemotion": (
                "v172_depth_early_1of3_multiscalemotion",
                "count_matched_early_third",
            ),
            "depth_late_1of3_multiscalemotion": (
                "v172_depth_late_1of3_multiscalemotion",
                "count_matched_late_third",
            ),
            "depth_interleaved_1of3_multiscalemotion": (
                "v172_depth_interleaved_1of3_multiscalemotion",
                "count_matched_distributed_third",
            ),
            "depth_all_multiscalemotion": (
                "v172_depth_all_multiscalemotion",
                "all_layer_upper_bound",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def load_depth_maps(args) -> tuple[dict, dict[str, Path], dict]:
    map_dir = ROOT / "configs" / "head_maps"
    manifest_path = map_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != build_manifest():
        raise ValueError("v172 relative-depth manifest is stale")
    paths = {key: map_dir / filename for key, filename in MAP_FILENAMES.items()}
    audits = {}
    for key, path in paths.items():
        audit = audit_binary_map(path, args.pf_labels)
        expected = manifest["maps"][key]
        expected_counts = {
            "10": expected["selected_head_count"],
            "11": 360 - expected["selected_head_count"],
        }
        selected_layers = tuple(
            layer
            for layer, count in enumerate(audit["label10_per_layer"])
            if count == 12
        )
        if (
            audit["sha256"] != expected["sha256"]
            or audit["counts"] != expected_counts
            or audit["label10_per_layer"] != expected["label10_per_layer"]
            or selected_layers != MAP_SPECS[key]
        ):
            raise ValueError(f"v172 depth map violates contract: {key}")
        audits[key] = audit
    return manifest, paths, audits


def v166_run_root() -> Path:
    default = ROOT / "runs" / v166.EXPERIMENT / "full8"
    return Path(os.environ.get("V172_REUSE_V166_ROOT", default)).resolve()


def load_v166_source(prompt_manifest: dict, *, pf_runtime: dict) -> dict:
    root = v166_run_root()
    published_path = root / "published_manifest.json"
    contract_path = root / "contracts" / "experiment.json"
    trace_path = root / "automated_screen" / "multiscale_motion_trace.json"
    for path in (published_path, contract_path, trace_path):
        if not path.is_file():
            raise ValueError(f"missing frozen v166 reuse source: {path}")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods", [])}
    source_pf = contract.get("pf", {})
    if (
        published.get("ok") is not True
        or published.get("experiment") != v166.EXPERIMENT
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or published.get("experiment_contract_sha256")
        != v166.text_sha256_lf(contract_path)
        or contract.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or source_pf.get("config_sha256") != pf_runtime["config_sha256"]
        or int(source_pf.get("checkpoint_size", -1))
        != int(pf_runtime["checkpoint_size"])
        or trace.get("mechanism_gate") is not True
        or not set(REUSE_METHODS.values()).issubset(rows)
    ):
        raise ValueError("v166 source violates the frozen v172 reuse contract")
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
            raise ValueError(f"incomplete v166 reuse source: {source_method}")
        sources[target] = {
            "source_method": source_method,
            "video_dir": str(video_dir.resolve()),
            "total_bytes": total_bytes,
        }
    return {
        "root": str(root),
        "published_manifest": str(published_path),
        "published_manifest_sha256": sha256(published_path),
        "experiment_contract": str(contract_path),
        "experiment_contract_sha256": v166.text_sha256_lf(contract_path),
        "mechanism_trace": str(trace_path),
        "mechanism_trace_sha256": v166.text_sha256_lf(trace_path),
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
    reuse = args.v166_source["sources"][method.key]
    source = Path(reuse["video_dir"]) / f"{prompt_index:06d}.mp4"
    target = args.out_root / "published" / method.key / runner.published_name(
        prompt_index
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
            "reuse_manifest_sha256": args.v166_source[
                "published_manifest_sha256"
            ],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v166",
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
    depth_manifest: dict,
    map_audits: dict,
) -> dict:
    contract = runner.experiment_contract(
        args,
        methods=methods,
        prompts=prompts,
        map_audit=map_audits["center_1of3"],
    )
    contract.update(
        {
            "version": 1,
            "prompt_suite": prompt_manifest,
            "v166_source": args.v166_source,
            "relative_depth_manifest": depth_manifest,
            "relative_depth_audits": map_audits,
            "classification_claim": {
                "type": "cache allocation rule, not semantic head taxonomy",
                "coordinate": "u_l=(l+0.5)/L",
                "current_backbone": {"layers": 30, "heads_per_layer": 12},
                "reference_rule": "center one-third of model depth",
                "reference_current_layers": list(MAP_SPECS["center_1of3"]),
                "absolute_layer_ids_are_not_part_of_the_method": True,
            },
            "cache_contract": {
                "selected_layers": (
                    "sink1 + reservoir2 + one recalled atomic motion pair "
                    "+ recent4 using frozen v166 MultiScaleMotion"
                ),
                "default_layers": "sink1 + recent8",
                "all_layer_control": (
                    "both binary labels use the same selected-layer policy"
                ),
                "max_read_full_frame_equivalents": 9,
                "exclusive_dynamic_owner": True,
            },
            "pre_registered_questions": {
                "dose": (
                    "central 1/6, 1/4, 1/3, and 1/2 depth fractions test "
                    "whether the v166 gain has a plateau or monotonic dose"
                ),
                "placement": (
                    "early, center, late, and interleaved one-third routes "
                    "use equal layer/head counts and the identical operator"
                ),
                "upper_bound": (
                    "all-layer routing estimates maximum operator exposure"
                ),
                "no_posthoc_theory": (
                    "the best development fraction cannot be called a "
                    "universal memory depth without cross-backbone transfer"
                ),
            },
            "evaluation": {
                "automatic_only": True,
                "metrics": "prompt-correct VBench-Long core-9",
                "paired_unit": "prompt",
                "primary_outputs": [
                    "full dose curve",
                    "count-matched placement table",
                    "Pareto set",
                    "bootstrap confidence intervals",
                ],
                "manual_review": "none by default",
                "cross_model_gate": (
                    "repeat the frozen normalized rule on at least one "
                    "different-depth compatible AR video DiT before claiming "
                    "architecture transfer"
                ),
            },
            "design": {
                "method_count": len(EXPECTED_METHOD_KEYS),
                "new_video_count": len(NEW_METHODS) * PROMPT_COUNT,
                "reused_video_count": len(REUSE_METHODS) * PROMPT_COUNT,
                "fixed_factors": (
                    "prompts, seeds, checkpoint, v166 cache contents, "
                    "retrieval selector, read budget, RoPE, and owner"
                ),
            },
            "claim_boundary": (
                "v172 is a 16-prompt adaptive development experiment. It "
                "tests an architecture-normalized allocation hypothesis but "
                "does not establish cross-model generalization."
            ),
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "build_v172_relative_depth_maps.py",
        ROOT / "scripts" / "prepare_v172_vbench_comparison.py",
        ROOT / "scripts" / "run_v172_vbench_long.py",
        ROOT / "scripts" / "analyze_v172_depth_metrics.py",
        ROOT / "configs" / "head_maps" / MANIFEST_FILENAME,
        *(
            ROOT / "configs" / "head_maps" / filename
            for filename in MAP_FILENAMES.values()
        ),
        ROOT / "prompts" / v155.PROMPT_FILENAME,
        ROOT / "prompts" / v155.PROMPT_MANIFEST_FILENAME,
    )
    contract["implementation_hashes"].update(
        {str(path.relative_to(ROOT)): sha256(path) for path in extra_paths}
    )
    return contract


def main() -> None:
    configure_parent_runner()
    args = runner.parse_args()
    prompts, prompt_manifest = v155.load_prompt_suite(args)
    pf_runtime = v159.load_pf_runtime_contract(
        args.pf_config,
        args.pf_checkpoint,
    )
    args.v166_source = load_v166_source(
        prompt_manifest,
        pf_runtime=pf_runtime,
    )
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            "v172 requires the frozen nine-method order: "
            f"{EXPECTED_METHOD_KEYS}"
        )
    depth_manifest, depth_paths, map_audits = load_depth_maps(args)
    args.head_maps = depth_paths
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
    contract = build_contract(
        args,
        methods=methods,
        prompts=prompts,
        prompt_manifest=prompt_manifest,
        depth_manifest=depth_manifest,
        map_audits=map_audits,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    contract_sha = (
        write_frozen(contract_path, contract)
        if args.node_rank == 0
        else wait_for_frozen(contract_path, contract, args.contract_wait_seconds)
    )
    print(
        "[V172Contract] "
        + canonical_json(
            {
                "methods": [method.key for method in methods],
                "contract_sha256": contract_sha,
                "selected_layers": {
                    key: list(value) for key, value in MAP_SPECS.items()
                },
                "new_videos": len(NEW_METHODS) * PROMPT_COUNT,
                "reused_videos": len(REUSE_METHODS) * PROMPT_COUNT,
            }
        )
        .decode("utf-8")
        .strip(),
        flush=True,
    )
    tasks = runner.selected_tasks(
        methods,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
    )
    gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    if args.mode == "preflight":
        new_count = sum(method.key in NEW_METHODS for method, _, _ in tasks)
        print(
            f"[v172-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"total_tasks={len(runner.all_tasks(methods))} "
            f"node_tasks={len(tasks)} new={new_count} "
            f"reused={len(tasks)-new_count} gpus={len(gpus)}",
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
            f"[v172-audit] PASS methods={len(payload['methods'])} "
            f"manifest={args.out_root / 'published_manifest.json'}",
            flush=True,
        )
        return

    task_list = list(tasks)
    worker_tasks = [task_list[index::len(gpus)] for index in range(len(gpus))]
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
        raise SystemExit("\n".join(failures or ["v172 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
