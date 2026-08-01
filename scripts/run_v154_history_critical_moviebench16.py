#!/usr/bin/env python3
"""Run the paired v154 History-Critical MovieBench-16 experiment."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_v120_moviebench32_main as runner
from analyze_v152_one_sided_history_critical import (
    HEADS_PER_LAYER,
    MANIFEST_FILENAME as HEAD_MANIFEST_FILENAME,
    MAP_FILENAMES,
    audit_binary_map,
)
from build_v154_history_critical_suite import (
    MANIFEST_FILENAME as PROMPT_MANIFEST_FILENAME,
    PROMPT_FILENAME,
)
from run_v100_fast_selection_1video import (
    Cell,
    canonical_json,
    sha256,
    wait_for_frozen,
    write_frozen,
    write_runtime_json,
)
from run_v109_legacy_v98_suppressive_cache_1video import (
    EXPECTED_MAP_SHA256,
    validate_frozen_map,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v154_history_critical_moviebench16"
PROMPT_COUNT = 16
EXPECTED_METHOD_KEYS = (
    "sf_native",
    "ours_qk_top4",
    "ours_qk_bottom4_control",
    "ours_qk_random4_control",
    "ours_all_recent8_control",
    "ours_all_prototype4_control",
    "ours_legacy_membership",
    "ours_legacy_reference",
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    "ours_legacy_reference": "ours_prototype_retrieval1_age24",
}
_PARENT_RUN_TASK = runner.run_task


V154_CELLS = (
    Cell(
        "v154_qk_top4_prototype4_default_recent8",
        "v154_membership",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
    ),
    Cell(
        "v154_qk_bottom4_control_prototype4_default_recent8",
        "v154_membership",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="qk_bottom4_control",
    ),
    Cell(
        "v154_qk_random4_control_prototype4_default_recent8",
        "v154_membership",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="random4_control",
    ),
    Cell(
        "v154_qk_top4_all_recent8_control",
        "v154_route_control",
        "single",
        support_policy="recent8",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
    ),
    Cell(
        "v154_qk_top4_all_prototype4_control",
        "v154_route_control",
        "single",
        support_policy="prototype",
        suppress_policy="prototype",
        map_key="qk_top4",
    ),
    Cell(
        "v154_legacy_v98_membership_prototype4_default_recent8",
        "v154_external_membership",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="legacy",
    ),
    Cell(
        "v154_legacy_v98_prototype4_retrieval1_age24_reference",
        "v154_known_reference",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        map_key="legacy",
    ),
)


def configure_parent_runner() -> None:
    runner.EXPERIMENT = EXPERIMENT
    runner.PROMPT_COUNT = PROMPT_COUNT
    runner.TASK_STAGE = "moviebench16"
    runner.PUBLISHED_TAG = "v154"
    runner.RUN_LABEL = "v154"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 7
    runner.DEFAULT_CANDIDATES = (
        "qk_top4",
        "qk_bottom4_control",
        "qk_random4_control",
        "all_recent8_control",
        "all_prototype4_control",
        "legacy_membership",
        "legacy_reference",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V154_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "qk_top4": (
                "v154_qk_top4_prototype4_default_recent8",
                "primary_history_critical_candidate",
            ),
            "qk_bottom4_control": (
                "v154_qk_bottom4_control_prototype4_default_recent8",
                "inverse_membership_control",
            ),
            "qk_random4_control": (
                "v154_qk_random4_control_prototype4_default_recent8",
                "count_matched_membership_control",
            ),
            "all_recent8_control": (
                "v154_qk_top4_all_recent8_control",
                "no_distributed_history_control",
            ),
            "all_prototype4_control": (
                "v154_qk_top4_all_prototype4_control",
                "all_head_distributed_history_control",
            ),
            "legacy_membership": (
                "v154_legacy_v98_membership_prototype4_default_recent8",
                "legacy_v98_membership_reference",
            ),
            "legacy_reference": (
                "v154_legacy_v98_prototype4_retrieval1_age24_reference",
                "v125_known_working_reference",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def load_v125_reuse(prompt_manifest: dict) -> dict | None:
    raw_root = os.environ.get("V154_REUSE_V125_ROOT", "").strip()
    if not raw_root:
        return None
    root = Path(raw_root).resolve()
    manifest_path = root / "published_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing v125 reuse manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = {row["key"]: row for row in manifest.get("methods", [])}
    source_sha = prompt_manifest["source"]["canonical_sha256"]
    if (
        not manifest.get("ok")
        or int(manifest.get("prompt_count", -1)) != 128
        or manifest.get("prompt_file_sha256") != source_sha
        or not set(REUSE_METHODS.values()).issubset(methods)
    ):
        raise ValueError("v125 reuse manifest violates the v154 source contract")
    expected_names = {f"{index:06d}.mp4" for index in range(128)}
    sources = {}
    for current_method, source_method in REUSE_METHODS.items():
        source_dir = Path(methods[source_method]["video_dir"]).resolve()
        actual_names = {path.name for path in source_dir.glob("*.mp4")}
        if actual_names != expected_names:
            raise ValueError(
                f"v125 reuse source is incomplete: {source_method} "
                f"missing={len(expected_names - actual_names)} "
                f"extra={len(actual_names - expected_names)}"
            )
        sources[current_method] = {
            "source_method": source_method,
            "video_dir": str(source_dir),
        }
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "prompt_file_sha256": source_sha,
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
    reuse = args.v154_reuse["sources"][method.key]
    source_index = int(
        args.v154_prompt_manifest["items"][prompt_index]["source_index"]
    )
    source = Path(reuse["video_dir"]) / f"{source_index:06d}.mp4"
    target = (
        args.out_root
        / "published"
        / method.key
        / runner.published_name(prompt_index)
    )
    indexed_target = (
        args.out_root
        / "published_indexed"
        / method.key
        / runner.published_name(prompt_index, indexed=True)
    )
    link_mode = runner.link_or_validate(source, target)
    indexed_mode = runner.link_or_validate(source, indexed_target)
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
            "source_prompt_index": source_index,
            "task_cell": cell.name,
            "source": str(source),
            "target": str(target),
            "indexed_target": str(indexed_target),
            "size": source.stat().st_size,
            "reuse_manifest_sha256": args.v154_reuse["manifest_sha256"],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v125",
        "source_prompt_index": source_index,
        "gpu": str(gpu),
    }


def run_task_with_optional_reuse(args, **kwargs):
    method = kwargs["method"]
    if args.v154_reuse is not None and method.key in REUSE_METHODS:
        return run_reused_task(args, **kwargs)
    return _PARENT_RUN_TASK(args, **kwargs)


def load_prompt_suite(args) -> tuple[list[str], dict]:
    if args.num_nodes <= 0 or not 0 <= args.node_rank < args.num_nodes:
        raise SystemExit("require 0 <= node-rank < num-nodes")
    if args.seed != 0:
        raise SystemExit("the frozen v154 experiment requires seed 0")
    transfer_approved = (
        args.promotion_approved
        or os.environ.get("V153_TRANSFER_APPROVED", "0") == "1"
    )
    if args.mode == "generate" and not transfer_approved:
        raise SystemExit(
            "v154 requires the clean v153 transfer result; set "
            "V153_TRANSFER_APPROVED=1"
        )
    required = (
        args.sf_repo / "inference.py",
        args.sf_config,
        args.sf_checkpoint,
        args.pf_repo / "inference.py",
        args.pf_config,
        args.pf_checkpoint,
        args.pf_labels,
        args.legacy_map,
        args.prompts,
        ROOT / "prompts" / PROMPT_MANIFEST_FILENAME,
        ROOT / "scripts" / "audit_indexed_videos.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required v154 files: {missing}")
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(prompts) != PROMPT_COUNT:
        raise SystemExit(
            f"v154 requires {PROMPT_COUNT} prompts, found {len(prompts)}"
        )
    manifest = json.loads(
        (ROOT / "prompts" / PROMPT_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    canonical = ("\n".join(prompts) + "\n").encode("utf-8")
    if (
        manifest.get("suite") != "v154_qwen_moviebench_diverse16"
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or manifest.get("prompt_file_sha256") != hashlib.sha256(canonical).hexdigest()
        or [row["text"] for row in manifest["items"]] != prompts
    ):
        raise SystemExit("v154 prompt suite manifest does not match the text file")
    return prompts, manifest


def load_head_maps(args) -> tuple[dict, dict, dict]:
    map_dir = ROOT / "configs" / "head_maps"
    head_manifest = json.loads(
        (map_dir / HEAD_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if (
        int(head_manifest.get("version", -1)) < 2
        or not head_manifest["gate_reanalysis"]["one_sided_transfer_candidate"]
    ):
        raise ValueError("v154 requires the portable v152 one-sided manifest")
    paths = {
        name: map_dir / filename for name, filename in MAP_FILENAMES.items()
    }
    paths["legacy"] = args.legacy_map
    audits = {}
    for name in MAP_FILENAMES:
        audit = audit_binary_map(
            paths[name],
            args.pf_labels,
            expected_label10_per_layer=HEADS_PER_LAYER,
        )
        if audit["sha256"] != head_manifest["maps"][name]["sha256"]:
            raise ValueError(f"v154 head-map hash changed: {name}")
        audits[name] = audit
    legacy_audit = validate_frozen_map(args.legacy_map, args.pf_labels)
    if legacy_audit["sha256"] != EXPECTED_MAP_SHA256:
        raise ValueError("legacy v98 map changed")
    audits["legacy"] = legacy_audit
    return head_manifest, paths, audits


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
        map_audit=map_audits["legacy"],
    )
    contract.update(
        {
            "version": 2,
            "prompt_suite": prompt_manifest,
            "head_membership": {
                "qk_top4": map_audits["qk_top4"],
                "qk_bottom4_control": map_audits["qk_bottom4_control"],
                "random4_control": map_audits["random4_control"],
                "legacy": map_audits["legacy"],
            },
            "v152_profiling_gate": head_manifest["gate_reanalysis"],
            "v152_discovery_validation_recurrence": head_manifest[
                "discovery_validation_recurrence"
            ],
            "v153_transfer_evidence": {
                "commit": "bde3e78",
                "cells_completed": 7,
                "structural_failures": 0,
                "claim_boundary": (
                    "v153 established executable clean routing only; v154 tests "
                    "cross-prompt generation utility"
                ),
            },
            "decision_rule": {
                "primary": (
                    "QK-top must beat both bottom and random membership controls "
                    "without a material dynamic-degree reduction"
                ),
                "promotion": (
                    "clean artifacts, paired human preference on at least 10/16 "
                    "prompts, no more than one severe failure, and metric gains "
                    "consistent with the reviewed identity/background effect"
                ),
            },
            "v125_reuse": args.v154_reuse,
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "analyze_v152_one_sided_history_critical.py",
        ROOT / "scripts" / "build_v154_history_critical_suite.py",
        ROOT / "prompts" / PROMPT_FILENAME,
        ROOT / "prompts" / PROMPT_MANIFEST_FILENAME,
        ROOT / "configs" / "head_maps" / HEAD_MANIFEST_FILENAME,
        *(ROOT / "configs" / "head_maps" / value for value in MAP_FILENAMES.values()),
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
    prompts, prompt_manifest = load_prompt_suite(args)
    args.v154_prompt_manifest = prompt_manifest
    args.v154_reuse = load_v125_reuse(prompt_manifest)
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            "v154 requires the frozen eight-method order: "
            f"{EXPECTED_METHOD_KEYS}"
        )
    head_manifest, args.head_maps, map_audits = load_head_maps(args)
    args.head_map_audits = {
        **map_audits,
        "pf": {
            "path": str(args.pf_labels),
            "sha256": sha256(args.pf_labels),
            "kind": "audit_only_pf_labels",
        },
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
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
    if args.node_rank == 0:
        contract_sha = write_frozen(contract_path, contract)
    else:
        contract_sha = wait_for_frozen(
            contract_path, contract, args.contract_wait_seconds
        )
    print(
        "[V154Contract] "
        + canonical_json(
            {
                "method_set_id": args.method_set_id,
                "methods": [method.key for method in methods],
                "prompt_sha256": prompt_manifest["prompt_file_sha256"],
                "contract_sha256": contract_sha,
                "qk_top_map_sha256": map_audits["qk_top4"]["sha256"],
            }
        ).decode("utf-8").strip(),
        flush=True,
    )

    if args.mode == "preflight":
        tasks = runner.selected_tasks(
            methods, node_rank=args.node_rank, num_nodes=args.num_nodes
        )
        gpus = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
        if not gpus or len(gpus) != len(set(gpus)):
            raise SystemExit("--gpu-list must contain unique GPU ids")
        print(
            f"[v154-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"tasks={len(tasks)} gpus={len(gpus)}",
            flush=True,
        )
        return

    if args.mode == "audit":
        payload = runner.audit_published(
            args, methods=methods, contract_sha256=contract_sha
        )
        print(
            f"[v154-audit] PASS methods={len(payload['methods'])} "
            f"manifest={args.out_root / 'published_manifest.json'}",
            flush=True,
        )
        return

    tasks = list(
        runner.selected_tasks(
            methods, node_rank=args.node_rank, num_nodes=args.num_nodes
        )
    )
    gpus = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    worker_tasks = [tasks[index::len(gpus)] for index in range(len(gpus))]
    worker_tasks = [items for items in worker_tasks if items]
    print(
        f"[v154] node={args.node_rank}/{args.num_nodes} tasks={len(tasks)} "
        f"workers={len(worker_tasks)} out={args.out_root}",
        flush=True,
    )

    worker_failures: list[str] = []
    results: list[dict] = []
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
                worker_failures.append(f"gpu={gpu}: {error}")
                print(f"[failed] gpu={gpu}: {error}", flush=True)
    task_failures = [row for row in results if row.get("status") == "failed"]
    failures = [
        *worker_failures,
        *(f"{row.get('name')}: {row.get('error')}" for row in task_failures),
    ]
    summary = {
        "version": 1,
        "experiment": EXPERIMENT,
        "method_set_id": args.method_set_id,
        "node_rank": args.node_rank,
        "num_nodes": args.num_nodes,
        "contract_sha256": contract_sha,
        "task_count": len(tasks),
        "result_count": len(results),
        "results": sorted(
            results,
            key=lambda row: (
                str(row.get("method", row.get("name", ""))),
                int(row.get("prompt_index", -1)),
            ),
        ),
        "failures": failures,
        "ok": not failures and len(results) == len(tasks),
    }
    summary_path = args.out_root / "status" / f"node{args.node_rank}.summary.json"
    write_runtime_json(summary_path, summary)
    if not summary["ok"]:
        raise SystemExit("\n".join(failures or ["v154 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
