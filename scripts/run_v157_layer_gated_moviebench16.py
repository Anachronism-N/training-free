#!/usr/bin/env python3
"""Run the v157 count-matched layer-gated reservoir experiment."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
from analyze_v152_one_sided_history_critical import audit_binary_map
from build_v157_layer_gate_maps import (
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
EXPERIMENT = "v157_layer_gated_moviebench16"
PROMPT_COUNT = 16
EXPECTED_METHOD_KEYS = (
    "sf_native",
    "ours_layer_early10_reservoir4",
    "ours_layer_middle10_reservoir4",
    "ours_layer_late10_reservoir4",
    "ours_layer_interleaved10_reservoir4",
    "ours_all_reservoir4_reference",
    "ours_qk_top4_reservoir4_reference",
    "ours_all_recent8_reference",
)
REUSE_METHODS = {
    "sf_native": "sf_native",
    "ours_all_reservoir4_reference": "ours_all_reservoir4_control",
    "ours_qk_top4_reservoir4_reference": "ours_qk_top4_reservoir4",
    "ours_all_recent8_reference": "ours_all_recent8_reference",
}
_PARENT_RUN_TASK = runner.run_task


V157_CELLS = (
    Cell(
        "v157_layer_early10_reservoir4",
        "v157_layer_placement",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="early10",
    ),
    Cell(
        "v157_layer_middle10_reservoir4",
        "v157_layer_placement",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v157_layer_late10_reservoir4",
        "v157_layer_placement",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="late10",
    ),
    Cell(
        "v157_layer_interleaved10_reservoir4",
        "v157_layer_placement",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="interleaved10",
    ),
    Cell(
        "v157_all_reservoir4_reference",
        "v157_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="reservoir",
        map_key="qk_top4",
    ),
    Cell(
        "v157_qk_top4_reservoir4_reference",
        "v157_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="qk_top4",
    ),
    Cell(
        "v157_all_recent8_reference",
        "v157_reference",
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
    runner.PUBLISHED_TAG = "v157"
    runner.RUN_LABEL = "v157"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 7
    runner.DEFAULT_CANDIDATES = (
        "layer_early10_reservoir4",
        "layer_middle10_reservoir4",
        "layer_late10_reservoir4",
        "layer_interleaved10_reservoir4",
        "all_reservoir4_reference",
        "qk_top4_reservoir4_reference",
        "all_recent8_reference",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V157_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "layer_early10_reservoir4": (
                "v157_layer_early10_reservoir4",
                "early_layer_gate",
            ),
            "layer_middle10_reservoir4": (
                "v157_layer_middle10_reservoir4",
                "middle_layer_gate",
            ),
            "layer_late10_reservoir4": (
                "v157_layer_late10_reservoir4",
                "late_layer_gate",
            ),
            "layer_interleaved10_reservoir4": (
                "v157_layer_interleaved10_reservoir4",
                "depth_distributed_layer_gate",
            ),
            "all_reservoir4_reference": (
                "v157_all_reservoir4_reference",
                "v155_all_reservoir_reference",
            ),
            "qk_top4_reservoir4_reference": (
                "v157_qk_top4_reservoir4_reference",
                "v155_qk_membership_reference",
            ),
            "all_recent8_reference": (
                "v157_all_recent8_reference",
                "v155_recent_only_reference",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def load_layer_maps(args) -> tuple[dict, dict[str, Path], dict]:
    map_dir = ROOT / "configs" / "head_maps"
    manifest_path = map_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != build_manifest():
        raise ValueError("v157 layer-map manifest is stale")
    paths = {key: map_dir / filename for key, filename in MAP_FILENAMES.items()}
    audits = {}
    for key, path in paths.items():
        audit = audit_binary_map(path, args.pf_labels)
        expected = manifest["maps"][key]
        if (
            audit["sha256"] != expected["sha256"]
            or audit["counts"] != {"10": 120, "11": 240}
            or audit["label10_per_layer"] != expected["label10_per_layer"]
        ):
            raise ValueError(f"v157 layer map violates frozen contract: {key}")
        selected_layers = tuple(
            layer
            for layer, count in enumerate(audit["label10_per_layer"])
            if count == 12
        )
        if selected_layers != MAP_SPECS[key]:
            raise ValueError(f"v157 selected layers changed: {key}")
        audits[key] = audit
    return manifest, paths, audits


def load_v155_reuse(prompt_manifest: dict) -> dict | None:
    raw_root = os.environ.get("V157_REUSE_V155_ROOT", "").strip()
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
    reuse = args.v157_reuse["sources"][method.key]
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
            "reuse_manifest_sha256": args.v157_reuse[
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
    if args.v157_reuse is not None and method.key in REUSE_METHODS:
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
    v155_summary_path: Path,
) -> dict:
    contract = runner.experiment_contract(
        args,
        methods=methods,
        prompts=prompts,
        map_audit=map_audits["early10"],
    )
    contract.update(
        {
            "version": 4,
            "prompt_suite": prompt_manifest,
            "layer_membership": map_audits,
            "layer_map_manifest": layer_manifest,
            "v155_decision_basis": {
                "result": "cache useful, classifier unsupported",
                "core9_summary": str(v155_summary_path),
                "core9_summary_sha256": sha256(v155_summary_path),
                "all_reservoir_dynamic_degree": 0.8333333333333334,
                "all_reservoir_temporal_flickering": 0.954679480415154,
                "sf_dynamic_degree": 0.6416666666666667,
                "sf_temporal_flickering": 0.968037148164768,
                "qk_membership_gate": False,
            },
            "cache_contract": {
                "selected_layers": "sink1+TemporalReservoir4+recent4",
                "other_layers": "sink1+recent8",
                "max_read_full_frame_equivalents": 9,
                "selected_head_count_per_method": 120,
                "exclusive_dynamic_owner": True,
            },
            "falsification": {
                "placement": (
                    "early, middle, late, and interleaved routes use the same "
                    "120-head count and identical cache policy"
                ),
                "pareto_goal": (
                    "retain a material fraction of the all-reservoir motion "
                    "gain while recovering temporal stability"
                ),
                "claim_boundary": (
                    "v157 tests layer placement, not QK or semantic head roles"
                ),
                "promotion": (
                    "do not scale unless one layer route passes frozen metric "
                    "and paired human-review gates"
                ),
            },
            "metric_gate": {
                "min_dynamic_gain_over_recent": 0.02,
                "min_temporal_recovery_over_all_reservoir": 0.003,
                "min_history_delta_over_recent": -0.002,
                "min_temporal_delta_over_recent": -0.004,
                "min_visual_delta_over_recent": -0.01,
            },
            "blind_review": {
                "predeclared_primary": (
                    "ours_layer_interleaved10_reservoir4"
                ),
                "required_controls": [
                    "ours_layer_early10_reservoir4",
                    "ours_layer_middle10_reservoir4",
                    "ours_layer_late10_reservoir4",
                    "ours_all_reservoir4_reference",
                    "ours_all_recent8_reference",
                ],
            },
            "v155_reuse": args.v157_reuse,
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "build_v157_layer_gate_maps.py",
        ROOT / "scripts" / "analyze_v157_vbench.py",
        ROOT / "scripts" / "analyze_v157_blind_review.py",
        ROOT / "scripts" / "prepare_v157_vbench_comparison.py",
        ROOT / "configs" / "head_maps" / MANIFEST_FILENAME,
        *(ROOT / "configs" / "head_maps" / value for value in MAP_FILENAMES.values()),
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
    args.v157_reuse = load_v155_reuse(prompt_manifest)
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(
            "v157 requires the frozen eight-method order: "
            f"{EXPECTED_METHOD_KEYS}"
        )
    layer_manifest, layer_paths, map_audits = load_layer_maps(args)
    _, qk_paths, _ = v155.load_head_maps(args)
    args.head_maps = {**layer_paths, **qk_paths}
    args.head_map_audits = map_audits
    v155_summary_path = (
        ROOT
        / "docs"
        / "results"
        / "v155_profile_aligned_moviebench16"
        / "vbench_core9_summary.json"
    )
    if not v155_summary_path.is_file():
        raise SystemExit("v157 requires the frozen v155 core-9 summary")
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
        v155_summary_path=v155_summary_path,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    contract_sha = (
        write_frozen(contract_path, contract)
        if args.node_rank == 0
        else wait_for_frozen(contract_path, contract, args.contract_wait_seconds)
    )
    print(
        "[V157Contract] "
        + canonical_json(
            {
                "methods": [method.key for method in methods],
                "contract_sha256": contract_sha,
                "reuse": args.v157_reuse is not None,
                "new_videos": 64,
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
            f"[v157-preflight] PASS node={args.node_rank}/{args.num_nodes} "
            f"tasks={len(tasks)} gpus={len(gpus)}",
            flush=True,
        )
        return
    if args.mode == "audit":
        payload = runner.audit_published(
            args, methods=methods, contract_sha256=contract_sha
        )
        print(
            f"[v157-audit] PASS methods={len(payload['methods'])} "
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
        raise SystemExit("\n".join(failures or ["v157 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
