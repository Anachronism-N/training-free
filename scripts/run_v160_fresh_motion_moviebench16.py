#!/usr/bin/env python3
"""Run the v160 freshness-aware coherent-motion recovery experiment."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
import run_v157_layer_gated_moviebench16 as v157
import run_v159_motion_coherent_reservoir_moviebench16 as v159
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v160_fresh_motion_moviebench16"
PROMPT_COUNT = 16
PRIMARY = "ours_middle10_reservoir2_freshmotionpair1"
EXPECTED_METHOD_KEYS = (
    "sf_native",
    PRIMARY,
    "ours_middle10_reservoir2_motionpair1_reference",
    "ours_middle10_reservoir4_reference",
    "ours_all_recent8_reference",
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    "ours_middle10_reservoir2_motionpair1_reference": (
        "ours_middle10_reservoir2_motionpair1"
    ),
    "ours_middle10_reservoir4_reference": (
        "ours_middle10_reservoir4_reference"
    ),
    "ours_all_recent8_reference": "ours_all_recent8_reference",
}
NEW_METHODS = {PRIMARY}
_PARENT_RUN_TASK = runner.run_task


V160_CELLS = (
    Cell(
        "v160_middle10_reservoir2_freshmotionpair1",
        "v160_fresh_motion",
        "single",
        support_policy="reservoir2_freshmotion1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v160_middle10_reservoir2_motionpair1_reference",
        "v160_reference",
        "single",
        support_policy="reservoir2_motion1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v160_middle10_reservoir4_reference",
        "v160_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v160_all_recent8_reference",
        "v160_reference",
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
    runner.PUBLISHED_TAG = "v160"
    runner.RUN_LABEL = "v160"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 4
    runner.DEFAULT_CANDIDATES = (
        "middle10_reservoir2_freshmotionpair1",
        "middle10_reservoir2_motionpair1_reference",
        "middle10_reservoir4_reference",
        "all_recent8_reference",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V160_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "middle10_reservoir2_freshmotionpair1": (
                "v160_middle10_reservoir2_freshmotionpair1",
                "fresh_motion_primary",
            ),
            "middle10_reservoir2_motionpair1_reference": (
                "v160_middle10_reservoir2_motionpair1_reference",
                "v159_motion_reference",
            ),
            "middle10_reservoir4_reference": (
                "v160_middle10_reservoir4_reference",
                "v159_reservoir_reference",
            ),
            "all_recent8_reference": (
                "v160_all_recent8_reference",
                "v159_recent_reference",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def v159_run_root() -> Path:
    default = ROOT / "runs" / "v159_motion_coherent_reservoir_moviebench16" / "full8"
    return Path(os.environ.get("V160_REUSE_V159_ROOT", default)).resolve()


def _frozen_result_path(name: str) -> Path:
    return (
        ROOT
        / "docs"
        / "results"
        / "v159_motion_coherent_reservoir_moviebench16"
        / name
    )


def load_v159_source(prompt_manifest: dict, *, pf_runtime: dict) -> dict:
    root = v159_run_root()
    published_path = root / "published_manifest.json"
    contract_path = root / "contracts" / "experiment.json"
    diagnosis_path = _frozen_result_path("v159_motion_pair_trace_diagnosis.json")
    diagnostics_tar = _frozen_result_path("v159_diagnostics.tar.gz")
    metric_path = _frozen_result_path("vbench_core9_summary.json")
    for path in (
        published_path,
        contract_path,
        diagnosis_path,
        diagnostics_tar,
        metric_path,
    ):
        if not path.is_file():
            raise ValueError(f"missing frozen v159 source: {path}")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods", [])}
    source_pf = contract.get("pf", {})
    required_sources = set(REUSE_METHODS.values())
    diagnosis_methods = diagnosis.get("methods", {})
    metric_methods = metrics.get("methods", {})
    if (
        not published.get("ok")
        or published.get("experiment") != v159.EXPERIMENT
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or source_pf.get("config_sha256") != pf_runtime["config_sha256"]
        or int(source_pf.get("checkpoint_size", -1))
        != int(pf_runtime["checkpoint_size"])
        or not required_sources.issubset(rows)
        or diagnosis.get("experiment")
        != "v159_motion_pair_trace_diagnosis"
        or diagnosis.get("source", {}).get("sha256") != sha256(diagnostics_tar)
        or diagnosis.get("diagnosis", {}).get("dominant_rejection")
        != "motion_quantile_gate"
        or diagnosis.get("diagnosis", {}).get(
            "max_pair_age_is_not_a_hard_refresh_bound"
        )
        is not True
        or set(v159.NEW_METHODS) - set(diagnosis_methods)
        or not required_sources.issubset(metric_methods)
    ):
        raise ValueError("v159 source does not match the frozen v160 rationale")
    middle_motion = metric_methods[
        "ours_middle10_reservoir2_motionpair1"
    ]
    middle_reservoir = metric_methods["ours_middle10_reservoir4_reference"]
    if (
        abs(float(middle_motion["dynamic_degree"]) - 0.7458333333333333)
        > 1e-12
        or abs(float(middle_reservoir["dynamic_degree"]) - 0.7791666666666667)
        > 1e-12
    ):
        raise ValueError("v159 metric source changed unexpectedly")
    expected_names = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    sources = {}
    for target, source_method in REUSE_METHODS.items():
        video_dir = root / "published" / source_method
        if {path.name for path in video_dir.glob("*.mp4")} != expected_names:
            raise ValueError(f"incomplete v159 reuse source: {source_method}")
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
        "trace_diagnosis": str(diagnosis_path),
        "trace_diagnosis_sha256": sha256(diagnosis_path),
        "vbench_core9": str(metric_path),
        "vbench_core9_sha256": sha256(metric_path),
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
    reuse = args.v160_source["sources"][method.key]
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
            "reuse_manifest_sha256": args.v160_source[
                "published_manifest_sha256"
            ],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v159",
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
                "label10": "selected Middle10 layer; all 12 heads use tested cache",
                "label11": "other layer; all 12 heads use sink1+recent8",
            },
            "layer_membership": {
                key: map_audits[key] for key in ("interleaved10", "middle10")
            },
            "layer_map_manifest": layer_manifest,
            "v159_source": args.v160_source,
            "cache_contract": {
                "primary": (
                    "Middle10 sink1+reservoir2+fresh coherent motion pair1+recent4"
                ),
                "freshness_horizon_frames": 12,
                "stale_refresh": (
                    "eligible positive-motion, semantically coherent stale pair "
                    "bypasses only the rolling motion-quantile gate"
                ),
                "unchanged_gates": [
                    "semantic_floor",
                    "positive_motion",
                    "adjacent_pair",
                    "pair_spacing",
                ],
                "other_layers": "sink1+recent8",
                "max_read_full_frame_equivalents": 9,
                "exclusive_dynamic_owner": True,
            },
            "design": {
                "primary": PRIMARY,
                "new_video_count": len(NEW_METHODS) * PROMPT_COUNT,
                "reuse_video_count": len(REUSE_METHODS) * PROMPT_COUNT,
                "isolated_change": (
                    "v159 Middle10 hybrid plus max_pair_age 24->12 and "
                    "stale-only quantile bypass"
                ),
                "fixed_factors": (
                    "prompt, seed, Middle10 placement, cache allocation, "
                    "read budget, motion quantile and replacement margin"
                ),
            },
            "evaluation": {
                "automatic_screen_role": (
                    "quality-control and adaptive review selection only"
                ),
                "human_review": (
                    "wave1=4 prompts x 3 methods; expand to wave2 only when "
                    "the prespecified decision is inconclusive"
                ),
                "paper_claim_requires": (
                    "held-out prompts and standard metrics after exploratory gate"
                ),
            },
            "claim_boundary": (
                "v160 is an exploratory mechanism-recovery experiment. "
                "Automated triage and adaptive review cannot be reported as a "
                "confirmatory paper result."
            ),
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "analyze_v159_motion_pair_trace.py",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "role_event.py",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "factory.py",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "policy_overrides.py",
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
    args.v160_source = load_v159_source(
        prompt_manifest,
        pf_runtime=pf_runtime,
    )
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            "v160 requires the frozen five-method order: "
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
        "[V160Contract] "
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
            f"[v160-preflight] PASS node={args.node_rank}/{args.num_nodes} "
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
            f"[v160-audit] PASS methods={len(payload['methods'])} "
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
        raise SystemExit("\n".join(failures or ["v160 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
