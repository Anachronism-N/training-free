#!/usr/bin/env python3
"""Run the v159 motion-coherent reservoir recovery experiment."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
import run_v157_layer_gated_moviebench16 as v157
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v159_motion_coherent_reservoir_moviebench16"
PROMPT_COUNT = 16
PRIMARY = "ours_interleaved10_reservoir2_motionpair1"
EXPECTED_METHOD_KEYS = (
    "sf_native",
    PRIMARY,
    "ours_interleaved10_motionpair2",
    "ours_middle10_reservoir2_motionpair1",
    "ours_interleaved10_reservoir4_reference",
    "ours_middle10_reservoir4_reference",
    "ours_all_reservoir4_reference",
    "ours_all_recent8_reference",
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    "ours_interleaved10_reservoir4_reference": (
        "ours_layer_interleaved10_reservoir4"
    ),
    "ours_middle10_reservoir4_reference": "ours_layer_middle10_reservoir4",
    "ours_all_reservoir4_reference": "ours_all_reservoir4_reference",
    "ours_all_recent8_reference": "ours_all_recent8_reference",
}
NEW_METHODS = {
    PRIMARY,
    "ours_interleaved10_motionpair2",
    "ours_middle10_reservoir2_motionpair1",
}
_PARENT_RUN_TASK = runner.run_task


V159_CELLS = (
    Cell(
        "v159_interleaved10_reservoir2_motionpair1",
        "v159_motion_recovery",
        "single",
        support_policy="reservoir2_motion1",
        suppress_policy="recent8_sink1",
        map_key="interleaved10",
    ),
    Cell(
        "v159_interleaved10_motionpair2",
        "v159_motion_recovery",
        "single",
        support_policy="motion_pair",
        suppress_policy="recent8_sink1",
        map_key="interleaved10",
    ),
    Cell(
        "v159_middle10_reservoir2_motionpair1",
        "v159_motion_recovery",
        "single",
        support_policy="reservoir2_motion1",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v159_interleaved10_reservoir4_reference",
        "v159_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="interleaved10",
    ),
    Cell(
        "v159_middle10_reservoir4_reference",
        "v159_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v159_all_reservoir4_reference",
        "v159_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="reservoir",
        map_key="interleaved10",
    ),
    Cell(
        "v159_all_recent8_reference",
        "v159_reference",
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
    runner.PUBLISHED_TAG = "v159"
    runner.RUN_LABEL = "v159"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 7
    runner.DEFAULT_CANDIDATES = (
        "interleaved10_reservoir2_motionpair1",
        "interleaved10_motionpair2",
        "middle10_reservoir2_motionpair1",
        "interleaved10_reservoir4_reference",
        "middle10_reservoir4_reference",
        "all_reservoir4_reference",
        "all_recent8_reference",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V159_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "interleaved10_reservoir2_motionpair1": (
                "v159_interleaved10_reservoir2_motionpair1",
                "dual_timescale_primary",
            ),
            "interleaved10_motionpair2": (
                "v159_interleaved10_motionpair2",
                "motion_only_mechanism_control",
            ),
            "middle10_reservoir2_motionpair1": (
                "v159_middle10_reservoir2_motionpair1",
                "layer_placement_control",
            ),
            "interleaved10_reservoir4_reference": (
                "v159_interleaved10_reservoir4_reference",
                "v157_interleaved_reference",
            ),
            "middle10_reservoir4_reference": (
                "v159_middle10_reservoir4_reference",
                "v157_middle_reference",
            ),
            "all_reservoir4_reference": (
                "v159_all_reservoir4_reference",
                "v157_all_reservoir_reference",
            ),
            "all_recent8_reference": (
                "v159_all_recent8_reference",
                "v157_recent_only_reference",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def v157_run_root() -> Path:
    default = ROOT / "runs" / "v157_layer_gated_moviebench16" / "full8"
    return Path(os.environ.get("V159_REUSE_V157_ROOT", default)).resolve()


def load_pf_runtime_contract(config_path: Path, checkpoint_path: Path) -> dict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PF config is not a mapping: {config_path}")
    block_size = int(payload.get("num_frame_per_block", -1))
    if block_size != 3:
        raise ValueError(
            "v159 requires num_frame_per_block=3 so every committed block "
            "contains two adjacent motion edges"
        )
    return {
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size": checkpoint_path.stat().st_size,
        "num_frame_per_block": block_size,
    }


def load_v157_source(
    prompt_manifest: dict,
    *,
    pf_runtime: dict,
) -> dict:
    root = v157_run_root()
    published_path = root / "published_manifest.json"
    contract_path = root / "contracts" / "experiment.json"
    diagnosis_path = (
        ROOT
        / "docs"
        / "results"
        / "v157_layer_gated_moviebench16"
        / "v157_motion_failure_diagnosis.json"
    )
    human_path = (
        ROOT
        / "docs"
        / "results"
        / "v157_layer_gated_moviebench16"
        / "v157_metric_screened_confirmation_report.json"
    )
    for path in (published_path, contract_path, diagnosis_path, human_path):
        if not path.is_file():
            raise ValueError(f"missing frozen v157 source: {path}")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods", [])}
    deficit = diagnosis.get("primary_minus_all_reservoir_motion", {})
    source_pf = contract.get("pf", {})
    if (
        not published.get("ok")
        or published.get("experiment") != "v157_layer_gated_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or source_pf.get("config_sha256") != pf_runtime["config_sha256"]
        or int(source_pf.get("checkpoint_size", -1))
        != int(pf_runtime["checkpoint_size"])
        or not set(REUSE_METHODS.values()).issubset(rows)
        or diagnosis.get("experiment") != "v157_motion_failure_diagnosis"
        or diagnosis.get("primary")
        != "ours_layer_interleaved10_reservoir4"
        or deficit.get("deficit_prompts") != [0, 1, 5, 6, 7, 15]
        or abs(float(deficit.get("mean", 0.0)) + 0.3125) > 1e-12
        or human.get("metric_screened_confirmation_gate") is not False
        or human.get("confirmation_checks", {}).get(
            "motion_noninferior_to_all_controls"
        )
        is not False
    ):
        raise ValueError("v157 source does not match the frozen v159 rationale")
    expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    sources = {}
    for target, source_method in REUSE_METHODS.items():
        video_dir = Path(rows[source_method]["video_dir"]).resolve()
        if {path.name for path in video_dir.glob("*.mp4")} != expected:
            raise ValueError(f"incomplete v157 reuse source: {source_method}")
        sources[target] = {
            "source_method": source_method,
            "video_dir": str(video_dir),
        }
    return {
        "root": str(root),
        "published_manifest": str(published_path),
        "published_manifest_sha256": sha256(published_path),
        "experiment_contract_sha256": sha256(contract_path),
        "human_report": str(human_path),
        "human_report_sha256": sha256(human_path),
        "motion_diagnosis": str(diagnosis_path),
        "motion_diagnosis_sha256": sha256(diagnosis_path),
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
    reuse = args.v159_source["sources"][method.key]
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
            "version": 2,
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
            "reuse_manifest_sha256": args.v159_source[
                "published_manifest_sha256"
            ],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v157",
        "gpu": str(gpu),
    }


def run_task_with_optional_reuse(args, **kwargs):
    method = kwargs["method"]
    if method.key in REUSE_METHODS:
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
        map_audit=map_audits["interleaved10"],
    )
    contract.update(
        {
            "version": 5,
            "prompt_suite": prompt_manifest,
            "head_membership": {
                "meaning": "layer policy gate, not a semantic head class",
                "label10": "selected layer; all 12 heads use tested middle cache",
                "label11": "other layer; all 12 heads use recent8",
            },
            "layer_membership": {
                key: map_audits[key] for key in ("interleaved10", "middle10")
            },
            "layer_map_manifest": layer_manifest,
            "v157_failure_basis": args.v159_source,
            "cache_contract": {
                "reservoir_reference": "sink1+reservoir4+recent4",
                "motion_only": "sink1+motion_pair2(4 frames)+recent4",
                "dual_timescale": (
                    "sink1+reservoir2+motion_pair1(2 frames)+recent4"
                ),
                "other_layers": "sink1+recent8",
                "max_read_full_frame_equivalents": 9,
                "middle_union_deduplicates_frame_t": True,
                "exclusive_dynamic_owner": True,
            },
            "design": {
                "primary": PRIMARY,
                "new_video_count": len(NEW_METHODS) * PROMPT_COUNT,
                "reuse_video_count": len(REUSE_METHODS) * PROMPT_COUNT,
                "mechanism_contrast": (
                    "interleaved reservoir4 vs motionpair2 vs "
                    "reservoir2+motionpair1"
                ),
                "placement_contrast": (
                    "interleaved10 vs middle10 under identical hybrid cache"
                ),
                "fixed_factors": (
                    "prompt, seed, layer count, heads per selected layer, "
                    "sink/recent/read budget"
                ),
            },
            "evaluation": {
                "vbench_role": (
                    "diagnostic only because v157 Dynamic Degree contradicted "
                    "human motion quality"
                ),
                "human_primary_contrast": (
                    "primary vs interleaved reservoir4 reference"
                ),
                "required_human_dimensions": [
                    "identity_continuity",
                    "background_continuity",
                    "motion_quality",
                    "overall_preference",
                    "severe_failure",
                ],
            },
            "claim_boundary": (
                "v159 is an exploratory recovery experiment designed after "
                "v157 human review. It cannot provide confirmatory evidence "
                "without a later held-out prompt run."
            ),
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "analyze_v157_motion_failure_diagnosis.py",
        ROOT / "scripts" / "run_v157_layer_gated_moviebench16.py",
        ROOT / "third_party" / "Pyramid-Forcing" / "pyramidkv" / "temporal_reservoir.py",
        ROOT / "configs" / "head_maps" / v157.MANIFEST_FILENAME,
        ROOT / "configs" / "head_maps" / v157.MAP_FILENAMES["interleaved10"],
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
    pf_runtime = load_pf_runtime_contract(args.pf_config, args.pf_checkpoint)
    args.v159_source = load_v157_source(
        prompt_manifest,
        pf_runtime=pf_runtime,
    )
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            "v159 requires the frozen eight-method order: "
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
        "[V159Contract] "
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
        methods, node_rank=args.node_rank, num_nodes=args.num_nodes
    )
    gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    if args.mode == "preflight":
        new_count = sum(method.key in NEW_METHODS for method, _, _ in tasks)
        print(
            f"[v159-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"tasks={len(tasks)} new={new_count} reused={len(tasks)-new_count} "
            f"gpus={len(gpus)}",
            flush=True,
        )
        return
    if args.mode == "audit":
        payload = runner.audit_published(
            args, methods=methods, contract_sha256=contract_sha
        )
        print(
            f"[v159-audit] PASS methods={len(payload['methods'])} "
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
        raise SystemExit("\n".join(failures or ["v159 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
