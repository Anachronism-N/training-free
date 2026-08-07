#!/usr/bin/env python3
"""Run the v164 direction-compatibility and freshness experiment."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
import run_v157_layer_gated_moviebench16 as v157
import run_v159_motion_coherent_reservoir_moviebench16 as v159
import run_v161_state_matched_motion_moviebench16 as v161
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v164_direction_freshness_moviebench16"
PROMPT_COUNT = 16
DIRECTION_MATCH = "ours_middle10_reservoir2_directionmatch1"
DIRECTION_FRESH = "ours_middle10_reservoir2_directionfresh1"
STATE_REFERENCE = "ours_middle10_reservoir2_statemotionpair1_reference"
RESERVOIR_REFERENCE = "ours_middle10_reservoir4_reference"
RECENT_REFERENCE = "ours_all_recent8_reference"
EXPECTED_METHOD_KEYS = (
    "sf_native",
    DIRECTION_MATCH,
    DIRECTION_FRESH,
    STATE_REFERENCE,
    RESERVOIR_REFERENCE,
    RECENT_REFERENCE,
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    STATE_REFERENCE: v161.PRIMARY,
    RESERVOIR_REFERENCE: RESERVOIR_REFERENCE,
    RECENT_REFERENCE: RECENT_REFERENCE,
}
NEW_METHODS = {DIRECTION_MATCH, DIRECTION_FRESH}
_PARENT_RUN_TASK = runner.run_task


V164_CELLS = (
    Cell(
        "v164_middle10_reservoir2_directionmatch1",
        "v164_direction_match",
        "single",
        support_policy="reservoir2_directionmatch1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v164_middle10_reservoir2_directionfresh1",
        "v164_direction_fresh",
        "single",
        support_policy="reservoir2_directionfresh1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v164_middle10_reservoir2_statemotionpair1_reference",
        "v164_reference",
        "single",
        support_policy="reservoir2_statemotion1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v164_middle10_reservoir4_reference",
        "v164_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v164_all_recent8_reference",
        "v164_reference",
        "single",
        support_policy="recent8",
        suppress_policy="recent8_sink1",
        map_key="interleaved10",
    ),
)


def configure_parent_runner() -> None:
    runner.EXPERIMENT = EXPERIMENT
    runner.PROMPT_COUNT = PROMPT_COUNT
    runner.TASK_STAGE = "moviebench16"
    runner.PUBLISHED_TAG = "v164"
    runner.RUN_LABEL = "v164"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 5
    runner.DEFAULT_CANDIDATES = (
        "middle10_reservoir2_directionmatch1",
        "middle10_reservoir2_directionfresh1",
        "middle10_reservoir2_statemotionpair1_reference",
        "middle10_reservoir4_reference",
        "all_recent8_reference",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V164_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "middle10_reservoir2_directionmatch1": (
                "v164_middle10_reservoir2_directionmatch1",
                "direction_match_primary",
            ),
            "middle10_reservoir2_directionfresh1": (
                "v164_middle10_reservoir2_directionfresh1",
                "direction_fresh_primary",
            ),
            "middle10_reservoir2_statemotionpair1_reference": (
                "v164_middle10_reservoir2_statemotionpair1_reference",
                "v161_state_motion_reference",
            ),
            "middle10_reservoir4_reference": (
                "v164_middle10_reservoir4_reference",
                "v159_reservoir_reference",
            ),
            "all_recent8_reference": (
                "v164_all_recent8_reference",
                "v159_recent_reference",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def v161_run_root() -> Path:
    default = ROOT / "runs" / v161.EXPERIMENT / "full8"
    return Path(os.environ.get("V164_REUSE_V161_ROOT", default)).resolve()


def load_v161_source(prompt_manifest: dict, *, pf_runtime: dict) -> dict:
    root = v161_run_root()
    published_path = root / "published_manifest.json"
    contract_path = root / "contracts" / "experiment.json"
    trace_path = (
        ROOT
        / "docs"
        / "results"
        / v161.EXPERIMENT
        / "state_motion_trace.json"
    )
    for path in (published_path, contract_path, trace_path):
        if not path.is_file():
            raise ValueError(f"missing frozen v161 reuse source: {path}")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods", [])}
    source_pf = contract.get("pf", {})
    required_sources = set(REUSE_METHODS.values())
    if (
        published.get("ok") is not True
        or published.get("experiment") != v161.EXPERIMENT
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or source_pf.get("config_sha256") != pf_runtime["config_sha256"]
        or int(source_pf.get("checkpoint_size", -1))
        != int(pf_runtime["checkpoint_size"])
        or not required_sources.issubset(rows)
        or trace.get("mechanism_gate") is not True
    ):
        raise ValueError("v161 source does not match the frozen v164 contract")
    expected_names = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    sources = {}
    for target, source_method in REUSE_METHODS.items():
        video_dir = root / "published" / source_method
        if {path.name for path in video_dir.glob("*.mp4")} != expected_names:
            raise ValueError(f"incomplete v161 reuse source: {source_method}")
        sources[target] = {
            "source_method": source_method,
            "video_dir": str(video_dir.resolve()),
        }
    return {
        "root": str(root),
        "published_manifest": str(published_path),
        "published_manifest_sha256": sha256(published_path),
        "experiment_contract": str(contract_path),
        "experiment_contract_sha256": sha256(contract_path),
        "state_motion_trace": str(trace_path),
        "state_motion_trace_sha256": sha256(trace_path),
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
    reuse = args.v164_source["sources"][method.key]
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
            "reuse_manifest_sha256": args.v164_source[
                "published_manifest_sha256"
            ],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v161",
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
    contract = runner.experiment_contract(
        args,
        methods=methods,
        prompts=prompts,
        map_audit=map_audits["middle10"],
    )
    contract.update(
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
            "v161_source": args.v164_source,
            "cache_contract": {
                "selected_layers": (
                    "sink1 + reservoir2 + one recalled motion pair + recent4"
                ),
                "other_layers": "sink1 + recent8",
                "motion_archive_pairs": 4,
                "motion_read_pairs": 1,
                "admission_freshness_horizon": 12,
                "read_max_age": 24,
                "direction_similarity_floor": 0.1,
                "state_similarity_role": "logged only; no gate and no score",
                "direction_match_score": "direction_similarity",
                "direction_fresh_score": (
                    "direction_similarity - 0.25 * age / 24"
                ),
                "no_match_fallback": "newest age-eligible atomic pair",
                "atomic_pair_read": True,
                "max_read_full_frame_equivalents": 9,
                "exclusive_dynamic_owner": True,
            },
            "design": {
                "new_methods": [DIRECTION_MATCH, DIRECTION_FRESH],
                "new_video_count": len(NEW_METHODS) * PROMPT_COUNT,
                "reuse_video_count": len(REUSE_METHODS) * PROMPT_COUNT,
                "isolated_test_1": (
                    "remove the saturated absolute-state cosine from the "
                    "v161 retrieval gate and score"
                ),
                "isolated_test_2": (
                    "add only a fixed freshness penalty to direction ranking"
                ),
                "fixed_factors": (
                    "prompt, seed, Middle10 placement, sink/reservoir/recent "
                    "allocation, archive/read capacity, direction floor, "
                    "admission policy, and maximum attention read budget"
                ),
            },
            "evaluation": {
                "mechanism_gate": (
                    "direction ranking is exercised; reads are atomic and "
                    "age-bounded; fallback preserves the pair-read budget"
                ),
                "automatic_first": (
                    "run temporal and comprehensive metrics before selecting "
                    "any human-review cases"
                ),
                "promotion_requires": (
                    "mechanism pass plus improvement over SF and at least one "
                    "memory reference without motion collapse"
                ),
            },
            "claim_boundary": (
                "v164 is a 16-prompt development experiment. It may select a "
                "generator for later held-out evaluation but is not paper "
                "evidence itself."
            ),
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_v100_fast_selection_1video.py",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "role_event.py",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "factory.py",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "policy_overrides.py",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pipeline"
        / "pyramidkv_config.py",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pipeline"
        / "causal_inference.py",
        ROOT / "configs" / "head_maps" / v157.MANIFEST_FILENAME,
        ROOT / "configs" / "head_maps" / v157.MAP_FILENAMES["middle10"],
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
    args.v164_source = load_v161_source(
        prompt_manifest,
        pf_runtime=pf_runtime,
    )
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            "v164 requires the frozen six-method order: "
            f"{EXPECTED_METHOD_KEYS}"
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
    contract = build_contract(
        args,
        methods=methods,
        prompts=prompts,
        prompt_manifest=prompt_manifest,
        layer_manifest=layer_manifest,
        map_audits=map_audits,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    contract_sha = (
        write_frozen(contract_path, contract)
        if args.node_rank == 0
        else wait_for_frozen(contract_path, contract, args.contract_wait_seconds)
    )
    print(
        "[V164Contract] "
        + canonical_json(
            {
                "methods": [method.key for method in methods],
                "contract_sha256": contract_sha,
                "new_videos": len(NEW_METHODS) * PROMPT_COUNT,
                "reused_videos": len(REUSE_METHODS) * PROMPT_COUNT,
            }
        ).decode("utf-8").strip(),
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
            f"[v164-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"tasks={len(tasks)} new={new_count} reused={len(tasks)-new_count} "
            f"gpus={len(gpus)}",
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
            f"[v164-audit] PASS methods={len(payload['methods'])} "
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
        raise SystemExit("\n".join(failures or ["v164 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
