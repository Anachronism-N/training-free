#!/usr/bin/env python3
"""Validate profile-aligned dispersed history on MovieBench-16."""
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


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "v155_profile_aligned_moviebench16"
PROMPT_COUNT = 16
EXPECTED_METHOD_KEYS = (
    "sf_native",
    "ours_qk_top4_reservoir4",
    "ours_qk_bottom4_reservoir4_control",
    "ours_qk_random4_reservoir4_control",
    "ours_all_reservoir4_control",
    "ours_qk_top4_prototype4_reference",
    "ours_all_recent8_reference",
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    "ours_qk_top4_prototype4_reference": "ours_qk_top4",
    "ours_all_recent8_reference": "ours_all_recent8_control",
}
_PARENT_RUN_TASK = runner.run_task


V155_CELLS = (
    Cell(
        "v155_qk_top4_reservoir4_default_recent8",
        "v155_membership",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
    ),
    Cell(
        "v155_qk_bottom4_reservoir4_default_recent8",
        "v155_membership",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="qk_bottom4_control",
    ),
    Cell(
        "v155_qk_random4_reservoir4_default_recent8",
        "v155_membership",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="random4_control",
    ),
    Cell(
        "v155_qk_top4_all_reservoir4_control",
        "v155_selectivity",
        "single",
        support_policy="reservoir",
        suppress_policy="reservoir",
        map_key="qk_top4",
    ),
    Cell(
        "v155_qk_top4_prototype4_reference",
        "v155_policy_reference",
        "single",
        support_policy="prototype",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
    ),
    Cell(
        "v155_qk_top4_all_recent8_reference",
        "v155_policy_reference",
        "single",
        support_policy="recent8",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
    ),
)


def configure_parent_runner() -> None:
    runner.EXPERIMENT = EXPERIMENT
    runner.PROMPT_COUNT = PROMPT_COUNT
    runner.TASK_STAGE = "moviebench16"
    runner.PUBLISHED_TAG = "v155"
    runner.RUN_LABEL = "v155"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 6
    runner.DEFAULT_CANDIDATES = (
        "qk_top4_reservoir4",
        "qk_bottom4_reservoir4_control",
        "qk_random4_reservoir4_control",
        "all_reservoir4_control",
        "qk_top4_prototype4_reference",
        "all_recent8_reference",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V155_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "qk_top4_reservoir4": (
                "v155_qk_top4_reservoir4_default_recent8",
                "primary_profile_aligned_candidate",
            ),
            "qk_bottom4_reservoir4_control": (
                "v155_qk_bottom4_reservoir4_default_recent8",
                "inverse_membership_control",
            ),
            "qk_random4_reservoir4_control": (
                "v155_qk_random4_reservoir4_default_recent8",
                "count_matched_membership_control",
            ),
            "all_reservoir4_control": (
                "v155_qk_top4_all_reservoir4_control",
                "all_head_selectivity_control",
            ),
            "qk_top4_prototype4_reference": (
                "v155_qk_top4_prototype4_reference",
                "v154_policy_reference",
            ),
            "all_recent8_reference": (
                "v155_qk_top4_all_recent8_reference",
                "local_history_reference",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def load_prompt_suite(args) -> tuple[list[str], dict]:
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
        raise SystemExit(f"missing required v155 files: {missing}")
    if args.seed != 0:
        raise SystemExit("the frozen v155 experiment requires seed 0")
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    manifest = json.loads(
        (ROOT / "prompts" / PROMPT_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    canonical = ("\n".join(prompts) + "\n").encode("utf-8")
    if (
        len(prompts) != PROMPT_COUNT
        or manifest.get("suite") != "v154_qwen_moviebench_diverse16"
        or int(manifest.get("prompt_count", -1)) != PROMPT_COUNT
        or manifest.get("prompt_file_sha256")
        != hashlib.sha256(canonical).hexdigest()
        or [row["text"] for row in manifest["items"]] != prompts
    ):
        raise SystemExit("v155 prompt suite violates the frozen v154 subset")
    return prompts, manifest


def load_head_maps(args) -> tuple[dict, dict, dict]:
    map_dir = ROOT / "configs" / "head_maps"
    manifest = json.loads(
        (map_dir / HEAD_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if (
        int(manifest.get("version", -1)) < 2
        or not manifest["gate_reanalysis"]["one_sided_transfer_candidate"]
    ):
        raise ValueError("v155 requires the frozen v152 one-sided maps")
    paths = {name: map_dir / filename for name, filename in MAP_FILENAMES.items()}
    audits = {}
    for name, path in paths.items():
        audit = audit_binary_map(
            path,
            args.pf_labels,
            expected_label10_per_layer=HEADS_PER_LAYER,
        )
        if audit["sha256"] != manifest["maps"][name]["sha256"]:
            raise ValueError(f"v155 head map changed: {name}")
        audits[name] = audit
    return manifest, paths, audits


def load_v154_reuse(prompt_manifest: dict) -> dict | None:
    raw_root = os.environ.get("V155_REUSE_V154_ROOT", "").strip()
    if not raw_root:
        return None
    root = Path(raw_root).resolve()
    published_path = root / "published_manifest.json"
    contract_path = root / "contracts" / "experiment.json"
    if not published_path.is_file() or not contract_path.is_file():
        raise ValueError(f"missing v154 reuse contracts under {root}")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in published.get("methods", [])}
    if (
        not published.get("ok")
        or published.get("experiment") != "v154_history_critical_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or not set(REUSE_METHODS.values()).issubset(rows)
    ):
        raise ValueError("v154 reuse artifacts violate the frozen contract")
    expected = {f"{index:06d}.mp4" for index in range(PROMPT_COUNT)}
    sources = {}
    for target, source_method in REUSE_METHODS.items():
        video_dir = Path(rows[source_method]["video_dir"]).resolve()
        if {path.name for path in video_dir.glob("*.mp4")} != expected:
            raise ValueError(f"incomplete v154 reuse source: {source_method}")
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
    reuse = args.v155_reuse["sources"][method.key]
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
            "reuse_manifest_sha256": args.v155_reuse[
                "published_manifest_sha256"
            ],
        },
    )
    return {
        "method": method.key,
        "prompt_index": prompt_index,
        "status": link_mode,
        "indexed_status": indexed_mode,
        "generation_status": "reused_v154",
        "gpu": str(gpu),
    }


def run_task_with_optional_reuse(args, **kwargs):
    method = kwargs["method"]
    if args.v155_reuse is not None and method.key in REUSE_METHODS:
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
            "version": 3,
            "prompt_suite": prompt_manifest,
            "head_membership": map_audits,
            "v152_profile_definition": {
                "score": "qk_compatibility(uniform8)-qk_compatibility(recent8)",
                "selection": "top4_per_layer_seed0",
                "recurrence": head_manifest[
                    "discovery_validation_recurrence"
                ],
            },
            "cache_contract": {
                "history_critical": (
                    "sink1+deterministic TemporalReservoir4+recent4"
                ),
                "default": "sink1+recent8",
                "reservoir_admission": (
                    "exact-frame K/V enters only after leaving recent4; "
                    "bounded Algorithm-R sample with seed 2026"
                ),
                "max_full_frame_equivalents": 9,
                "max_physical_selected_head_storage_ffe": 13,
                "storage_note": (
                    "pending4 duplicates the logical recent4 frames in the "
                    "current Python integration; readout and unique-frame "
                    "budgets remain 9 FFE"
                ),
                "exclusive_dynamic_owner": True,
            },
            "falsification": {
                "membership": "top must outperform bottom and random",
                "selectivity": "top routing is compared with all-head reservoir",
                "policy_fidelity": (
                    "top-reservoir is compared with the v154 top-prototype route"
                ),
                "claim_boundary": (
                    "v155 tests profile-aligned dispersed-history transfer; it "
                    "does not assume the remaining heads prefer recent history"
                ),
            },
            "v154_reuse": args.v155_reuse,
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pyramidkv"
        / "temporal_reservoir.py",
        ROOT / "scripts" / "analyze_v152_one_sided_history_critical.py",
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
    args.v155_reuse = load_v154_reuse(prompt_manifest)
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            "v155 requires the frozen seven-method order: "
            f"{EXPECTED_METHOD_KEYS}"
        )
    head_manifest, args.head_maps, map_audits = load_head_maps(args)
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
        "[V155Contract] "
        + canonical_json(
            {
                "methods": [method.key for method in methods],
                "prompt_sha256": prompt_manifest["prompt_file_sha256"],
                "contract_sha256": contract_sha,
                "reuse": args.v155_reuse is not None,
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
            f"[v155-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"tasks={len(tasks)} gpus={len(gpus)}",
            flush=True,
        )
        return
    if args.mode == "audit":
        payload = runner.audit_published(
            args, methods=methods, contract_sha256=contract_sha
        )
        print(
            f"[v155-audit] PASS methods={len(payload['methods'])} "
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
        raise SystemExit("\n".join(failures or ["v155 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
