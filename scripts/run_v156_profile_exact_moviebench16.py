#!/usr/bin/env python3
"""Run the v156 exact v152-uniform8 transfer experiment on MovieBench-16."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v156_profile_exact_moviebench16"
PROMPT_COUNT = 16
EXPECTED_METHOD_KEYS = (
    "sf_native",
    "ours_qk_top4_profile_uniform4",
    "ours_qk_bottom4_profile_uniform4_control",
    "ours_qk_random4_profile_uniform4_control",
    "ours_all_profile_uniform4_control",
    "ours_all_recent8_exact_control",
    "ours_qk_top4_reservoir4_reference",
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    "ours_qk_top4_reservoir4_reference": "ours_qk_top4_reservoir4",
}
_PARENT_RUN_TASK = runner.run_task


V156_CELLS = (
    Cell(
        "v156_qk_top4_profile_uniform4_default_recent8",
        "v156_membership",
        "single",
        support_policy="profile_anchor",
        suppress_policy="recent8_exact",
        map_key="qk_top4",
        history_budget_profile="profile_exact8",
        max_full_frame_equivalents=8,
    ),
    Cell(
        "v156_qk_bottom4_profile_uniform4_default_recent8",
        "v156_membership",
        "single",
        support_policy="profile_anchor",
        suppress_policy="recent8_exact",
        map_key="qk_bottom4_control",
        history_budget_profile="profile_exact8",
        max_full_frame_equivalents=8,
    ),
    Cell(
        "v156_qk_random4_profile_uniform4_default_recent8",
        "v156_membership",
        "single",
        support_policy="profile_anchor",
        suppress_policy="recent8_exact",
        map_key="random4_control",
        history_budget_profile="profile_exact8",
        max_full_frame_equivalents=8,
    ),
    Cell(
        "v156_qk_top4_all_profile_uniform4_control",
        "v156_selectivity",
        "single",
        support_policy="profile_anchor",
        suppress_policy="profile_anchor",
        map_key="qk_top4",
        history_budget_profile="profile_exact8",
        max_full_frame_equivalents=8,
    ),
    Cell(
        "v156_qk_top4_all_recent8_exact_control",
        "v156_selectivity",
        "single",
        support_policy="recent8_exact",
        suppress_policy="recent8_exact",
        map_key="qk_top4",
        history_budget_profile="profile_exact8",
        max_full_frame_equivalents=8,
    ),
    Cell(
        "v156_qk_top4_reservoir4_reference",
        "v156_policy_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
        max_full_frame_equivalents=9,
    ),
)


def configure_parent_runner() -> None:
    runner.EXPERIMENT = EXPERIMENT
    runner.PROMPT_COUNT = PROMPT_COUNT
    runner.TASK_STAGE = "moviebench16"
    runner.PUBLISHED_TAG = "v156"
    runner.RUN_LABEL = "v156"
    runner.DEFAULT_PROMPT_PATH = str(
        ROOT / "prompts" / v155.PROMPT_FILENAME
    )
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 6
    runner.DEFAULT_CANDIDATES = (
        "qk_top4_profile_uniform4",
        "qk_bottom4_profile_uniform4_control",
        "qk_random4_profile_uniform4_control",
        "all_profile_uniform4_control",
        "all_recent8_exact_control",
        "qk_top4_reservoir4_reference",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V156_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "qk_top4_profile_uniform4": (
                "v156_qk_top4_profile_uniform4_default_recent8",
                "primary_profile_exact_candidate",
            ),
            "qk_bottom4_profile_uniform4_control": (
                "v156_qk_bottom4_profile_uniform4_default_recent8",
                "inverse_membership_control",
            ),
            "qk_random4_profile_uniform4_control": (
                "v156_qk_random4_profile_uniform4_default_recent8",
                "count_matched_membership_control",
            ),
            "all_profile_uniform4_control": (
                "v156_qk_top4_all_profile_uniform4_control",
                "all_head_profile_policy_control",
            ),
            "all_recent8_exact_control": (
                "v156_qk_top4_all_recent8_exact_control",
                "all_head_recent_policy_control",
            ),
            "qk_top4_reservoir4_reference": (
                "v156_qk_top4_reservoir4_reference",
                "v155_reservoir_reference",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def load_v155_reuse(prompt_manifest: dict) -> dict | None:
    raw_root = os.environ.get("V156_REUSE_V155_ROOT", "").strip()
    if not raw_root:
        return None
    root = Path(raw_root).resolve()
    published_path = root / "published_manifest.json"
    contract_path = root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError(f"missing v155 reuse contracts under {root}")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods", [])}
    if (
        not published.get("ok")
        or published.get("experiment") != "v155_profile_aligned_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or not set(REUSE_METHODS.values()).issubset(rows)
    ):
        raise ValueError("v155 reuse artifacts violate the frozen contract")
    expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    sources = {}
    for target, source_method in REUSE_METHODS.items():
        video_dir = Path(rows[source_method]["video_dir"]).resolve()
        if {path.name for path in video_dir.glob("*.mp4")} != expected:
            raise ValueError(f"incomplete v155 reuse source: {source_method}")
        sources[target] = {
            "source_method": source_method,
            "video_dir": str(video_dir),
        }
    return {
        "root": str(root),
        "published_manifest": str(published_path),
        "published_manifest_sha256": sha256(published_path),
        "experiment_contract_sha256": sha256(contract_path),
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
    reuse = args.v156_reuse["sources"][method.key]
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
            "reuse_manifest_sha256": args.v156_reuse[
                "published_manifest_sha256"
            ],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v155",
        "gpu": str(gpu),
    }


def run_task_with_optional_reuse(args, **kwargs):
    method = kwargs["method"]
    if args.v156_reuse is not None and method.key in REUSE_METHODS:
        return run_reused_task(args, **kwargs)
    return _PARENT_RUN_TASK(args, **kwargs)


def build_contract(
    args,
    *,
    methods,
    prompts: list[str],
    prompt_manifest: dict,
    head_manifest: dict,
    map_audits: dict,
) -> dict:
    contract = runner.experiment_contract(
        args,
        methods=methods,
        prompts=prompts,
        map_audit=map_audits["qk_top4"],
    )
    contract.update(
        {
            "version": 4,
            "prompt_suite": prompt_manifest,
            "head_membership": map_audits,
            "v152_profile_definition": {
                "score": "qk_compatibility(uniform8)-qk_compatibility(recent8)",
                "selection": "top4_per_layer_seed0",
                "history_frames": 117,
                "uniform8_frame_ids": [0, 37, 75, 112, 113, 114, 115, 116],
                "recent8_frame_ids": [109, 110, 111, 112, 113, 114, 115, 116],
                "recurrence": head_manifest[
                    "discovery_validation_recurrence"
                ],
            },
            "cache_contract": {
                "history_critical": (
                    "sink0+fixed profile anchors [0,37,75,112]+recent4"
                ),
                "default": "sink0+recent8",
                "max_read_full_frame_equivalents": 8,
                "max_physical_selected_head_storage_ffe": 8,
                "profile_anchor_pending_storage": 0,
                "early_horizon_boundary": (
                    "the fixed old-history bank is underfilled until each "
                    "target id arrives; exact equivalence is claimed only at "
                    "the frozen v152 frame-117 profiling context"
                ),
                "exclusive_dynamic_owner": True,
            },
            "falsification": {
                "membership": "top must outperform bottom and random",
                "selectivity": (
                    "top routing must outperform both all-profile and "
                    "all-recent exact controls"
                ),
                "policy_fidelity": (
                    "exact top-profile is compared with the reused v155 "
                    "top-reservoir route"
                ),
                "promotion": (
                    "do not scale beyond MovieBench-16 unless objective and "
                    "blind membership gates both pass"
                ),
            },
            "v155_reuse": args.v156_reuse,
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "temporal_reservoir.py",
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "policy_overrides.py",
        ROOT / "scripts" / "run_v100_fast_selection_1video.py",
        ROOT / "prompts" / v155.PROMPT_FILENAME,
        ROOT / "prompts" / v155.PROMPT_MANIFEST_FILENAME,
        ROOT / "configs" / "head_maps" / v155.HEAD_MANIFEST_FILENAME,
        *(
            ROOT / "configs" / "head_maps" / value
            for value in v155.MAP_FILENAMES.values()
        ),
    )
    contract["implementation_hashes"].update(
        {
            str(path.relative_to(ROOT)): sha256(path)
            for path in extra_paths
        }
    )
    return contract


def main() -> None:
    configure_parent_runner()
    args = runner.parse_args()
    prompts, prompt_manifest = v155.load_prompt_suite(args)
    args.v156_reuse = load_v155_reuse(prompt_manifest)
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            "v156 requires the frozen seven-method order: "
            f"{EXPECTED_METHOD_KEYS}"
        )
    head_manifest, args.head_maps, map_audits = v155.load_head_maps(args)
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
        head_manifest=head_manifest,
        map_audits=map_audits,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    contract_sha = (
        write_frozen(contract_path, contract)
        if args.node_rank == 0
        else wait_for_frozen(contract_path, contract, args.contract_wait_seconds)
    )
    print(
        "[V156Contract] "
        + canonical_json(
            {
                "methods": [method.key for method in methods],
                "prompt_sha256": prompt_manifest["prompt_file_sha256"],
                "contract_sha256": contract_sha,
                "reuse": args.v156_reuse is not None,
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
        print(
            f"[v156-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"tasks={len(tasks)} gpus={len(gpus)}",
            flush=True,
        )
        return
    if args.mode == "audit":
        payload = runner.audit_published(
            args, methods=methods, contract_sha256=contract_sha
        )
        print(
            f"[v156-audit] PASS methods={len(payload['methods'])} "
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
        raise SystemExit("\n".join(failures or ["v156 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
