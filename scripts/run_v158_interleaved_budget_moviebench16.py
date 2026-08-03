#!/usr/bin/env python3
"""Run the conditional v158 nested interleaved layer-budget sweep."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import analyze_v157_vbench as v157_analysis
import analyze_v157_blind_review as v157_human_analysis
import run_v120_moviebench32_main as runner
import run_v155_profile_aligned_moviebench16 as v155
from analyze_v152_one_sided_history_critical import audit_binary_map
from build_v157_layer_gate_maps import MAP_FILENAMES as V157_MAP_FILENAMES
from build_v158_interleaved_budget_maps import (
    MANIFEST_FILENAME,
    MAP_FILENAMES,
    MAP_SPECS,
    build_manifest,
)
from prepare_v157_metric_screened_review import (
    EXPERIMENT as SCREENED_REVIEW_EXPERIMENT,
    METHODS as SCREENED_REVIEW_METHODS,
    PRIMARY as V157_PRIMARY,
    source_evidence as screened_review_source_evidence,
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
EXPERIMENT = "v158_interleaved_budget_moviebench16"
PROMPT_COUNT = 16
PRIMARY = "ours_interleaved8_reservoir4"
EXPECTED_METHOD_KEYS = (
    "sf_native",
    "ours_interleaved6_reservoir4",
    PRIMARY,
    "ours_interleaved10_reservoir4_reference",
    "ours_interleaved12_reservoir4",
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
    "ours_interleaved6_reservoir4",
    PRIMARY,
    "ours_interleaved12_reservoir4",
}
_PARENT_RUN_TASK = runner.run_task


V158_CELLS = tuple(
    Cell(
        f"v158_{key}_reservoir4",
        "v158_interleaved_budget",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key=key,
    )
    for key in MAP_SPECS
) + (
    Cell(
        "v158_middle10_reservoir4_reference",
        "v158_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="recent8_sink1",
        map_key="middle10",
    ),
    Cell(
        "v158_all_reservoir4_reference",
        "v158_reference",
        "single",
        support_policy="reservoir",
        suppress_policy="reservoir",
        map_key="interleaved10",
    ),
    Cell(
        "v158_all_recent8_reference",
        "v158_reference",
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
    runner.PUBLISHED_TAG = "v158"
    runner.RUN_LABEL = "v158"
    runner.DEFAULT_PROMPT_PATH = str(ROOT / "prompts" / v155.PROMPT_FILENAME)
    runner.INCLUDE_PF_BASELINE = False
    runner.ALLOW_PARTIAL_SCOPE = False
    runner.MAX_CANDIDATES = 7
    runner.DEFAULT_CANDIDATES = (
        "interleaved6_reservoir4",
        "interleaved8_reservoir4",
        "interleaved10_reservoir4_reference",
        "interleaved12_reservoir4",
        "middle10_reservoir4_reference",
        "all_reservoir4_reference",
        "all_recent8_reference",
    )
    runner._CELLS_BY_NAME.update({cell.name: cell for cell in V158_CELLS})
    runner._CANDIDATE_SPECS.update(
        {
            "interleaved6_reservoir4": (
                "v158_interleaved6_reservoir4",
                "distributed_budget6_exploratory",
            ),
            "interleaved8_reservoir4": (
                "v158_interleaved8_reservoir4",
                "distributed_budget8_primary",
            ),
            "interleaved10_reservoir4_reference": (
                "v158_interleaved10_reservoir4",
                "v157_interleaved10_reference",
            ),
            "interleaved12_reservoir4": (
                "v158_interleaved12_reservoir4",
                "distributed_budget12_exploratory",
            ),
            "middle10_reservoir4_reference": (
                "v158_middle10_reservoir4_reference",
                "v157_contiguous_middle_reference",
            ),
            "all_reservoir4_reference": (
                "v158_all_reservoir4_reference",
                "v157_all_reservoir_reference",
            ),
            "all_recent8_reference": (
                "v158_all_recent8_reference",
                "v157_recent_only_reference",
            ),
        }
    )
    runner.run_task = run_task_with_optional_reuse


def load_budget_maps(args) -> tuple[dict, dict[str, Path], dict]:
    map_dir = ROOT / "configs" / "head_maps"
    manifest_path = map_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != build_manifest():
        raise ValueError("v158 budget-map manifest is stale")
    paths = {key: map_dir / filename for key, filename in MAP_FILENAMES.items()}
    audits = {}
    for key, path in paths.items():
        audit = audit_binary_map(path, args.pf_labels)
        selected_count = len(MAP_SPECS[key]) * 12
        expected = manifest["maps"][key]
        if (
            audit["sha256"] != expected["sha256"]
            or audit["counts"]
            != {"10": selected_count, "11": 360 - selected_count}
            or audit["label10_per_layer"] != expected["label10_per_layer"]
        ):
            raise ValueError(f"v158 budget map violates contract: {key}")
        audits[key] = audit
    return manifest, paths, audits


def v157_run_root() -> Path:
    default_root = ROOT / "runs" / "v157_layer_gated_moviebench16" / "full8"
    return Path(os.environ.get("V158_REUSE_V157_ROOT", default_root)).resolve()


def load_v157_source(prompt_manifest: dict) -> dict:
    root = v157_run_root()
    published_path = root / "published_manifest.json"
    contract_path = root / "contracts" / "experiment.json"
    summary_path = (
        ROOT
        / "docs"
        / "results"
        / "v157_layer_gated_moviebench16"
        / "vbench_core9_summary.json"
    )
    for path in (published_path, contract_path, summary_path):
        if not path.is_file():
            raise ValueError(f"missing frozen v157 source: {path}")
    published = json.loads(published_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = v157_analysis.analyze(summary)
    rows = {row["key"]: row for row in published.get("methods", [])}
    primary_gate = analysis["candidate_gates"][
        "ours_layer_interleaved10_reservoir4"
    ]
    if (
        not published.get("ok")
        or published.get("experiment") != "v157_layer_gated_moviebench16"
        or int(published.get("prompt_count", -1)) != PROMPT_COUNT
        or published.get("experiment_contract_sha256") != sha256(contract_path)
        or contract.get("prompt_suite", {}).get("prompt_file_sha256")
        != prompt_manifest["prompt_file_sha256"]
        or not primary_gate.get("passes")
        or not set(REUSE_METHODS.values()).issubset(rows)
    ):
        raise ValueError("v157 source violates the frozen v158 reuse contract")
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
        "vbench_core9_summary": str(summary_path),
        "vbench_core9_summary_sha256": sha256(summary_path),
        "primary_metric_gate": primary_gate,
        "sources": sources,
    }


def load_blind_authorization() -> dict:
    analysis_root = v157_run_root() / "analysis"
    screened_path = (
        analysis_root / "v157_metric_screened_confirmation_report.json"
    )
    full_path = analysis_root / "v157_blind_review_report.json"
    explicit_path = os.environ.get("V158_V157_REVIEW_REPORT") or os.environ.get(
        "V158_V157_BLIND_REPORT"
    )
    if explicit_path:
        path = Path(explicit_path).resolve()
    elif screened_path.is_file():
        path = screened_path.resolve()
    elif full_path.is_file():
        path = full_path.resolve()
    else:
        path = screened_path.resolve()
    if not path.is_file():
        return {
            "ready": False,
            "path": str(path),
            "candidates": [str(screened_path), str(full_path)],
            "reason": "missing",
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("experiment") == SCREENED_REVIEW_EXPERIMENT:
        evidence_matches = False
        try:
            evidence_matches = report.get(
                "source_evidence"
            ) == screened_review_source_evidence(v157_run_root())
        except (KeyError, TypeError, ValueError):
            pass
        ready = bool(
            report.get("protocol_amendment") is True
            and report.get("primary") == V157_PRIMARY
            and int(report.get("prompt_count", -1)) == PROMPT_COUNT
            and int(report.get("video_count", -1)) == 64
            and tuple(report.get("methods_reviewed", []))
            == SCREENED_REVIEW_METHODS
            and report.get("metric_screened_confirmation_gate") is True
            and evidence_matches
        )
        authorization_type = "metric_screened_confirmation64"
        gate = report.get("metric_screened_confirmation_gate")
        reason = "passed" if ready else "gate_or_evidence_failed"
    else:
        review_contract = report.get("review_contract", {})
        review_sheet = Path(review_contract.get("review_sheet", ""))
        blind_key = Path(review_contract.get("blind_key", ""))
        reproducible = False
        try:
            hashes_match = bool(
                review_sheet.is_file()
                and blind_key.is_file()
                and review_contract.get("review_sheet_sha256")
                == sha256(review_sheet)
                and review_contract.get("blind_key_sha256") == sha256(blind_key)
            )
            if hashes_match:
                completed_rows = v157_human_analysis.load_completed_rows(
                    review_sheet, blind_key
                )
                reproducible = report == v157_human_analysis.analyze(
                    completed_rows, review_sheet, blind_key
                )
        except (KeyError, TypeError, ValueError):
            pass
        ready = bool(
            report.get("experiment")
            == "v157_layer_gated_moviebench16_blind_review"
            and report.get("primary") == V157_PRIMARY
            and int(report.get("prompt_count", -1)) == PROMPT_COUNT
            and int(report.get("video_count", -1)) == 128
            and tuple(report.get("methods_reviewed", []))
            == tuple(v157_human_analysis.METHODS)
            and report.get("human_promotion_gate") is True
            and reproducible
        )
        authorization_type = "full_blind_review128"
        gate = report.get("human_promotion_gate")
        reason = "passed" if ready else "gate_or_provenance_failed"
    return {
        "ready": ready,
        "path": str(path),
        "sha256": sha256(path),
        "authorization_type": authorization_type,
        "gate": gate,
        "reason": reason,
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
    reuse = args.v158_source["sources"][method.key]
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
            "launch_authorization_sha256": args.launch_authorization_sha256,
            "method": method.key,
            "engine": method.engine,
            "prompt_index": prompt_index,
            "task_cell": cell.name,
            "source_method": reuse["source_method"],
            "source": str(source),
            "target": str(target),
            "indexed_target": str(indexed),
            "size": source.stat().st_size,
            "reuse_manifest_sha256": args.v158_source[
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
    if kwargs["method"].key in REUSE_METHODS:
        return run_reused_task(args, **kwargs)
    return _PARENT_RUN_TASK(args, **kwargs)


def build_contract(
    args,
    *,
    methods,
    prompts: list[str],
    prompt_manifest: dict,
    budget_manifest: dict,
    map_audits: dict,
) -> dict:
    contract = runner.experiment_contract(
        args,
        methods=methods,
        prompts=prompts,
        map_audit=map_audits["interleaved8"],
    )
    contract.update(
        {
            "version": 1,
            "prompt_suite": prompt_manifest,
            "budget_membership": map_audits,
            "budget_map_manifest": budget_manifest,
            "v157_source": args.v158_source,
            "launch_requirement": {
                "required_primary": V157_PRIMARY,
                "accepted_authorizations": [
                    {
                        "experiment": (
                            "v157_layer_gated_moviebench16_blind_review"
                        ),
                        "gate": "human_promotion_gate",
                        "video_count": 128,
                    },
                    {
                        "experiment": SCREENED_REVIEW_EXPERIMENT,
                        "gate": "metric_screened_confirmation_gate",
                        "video_count": 64,
                        "methods": list(SCREENED_REVIEW_METHODS),
                        "scope": "v158_16_prompt_budget_pilot_only",
                    },
                ],
                "authorization_is_separate_frozen_runtime_artifact": True,
            },
            "cache_contract": {
                "selected_layers": "sink1+TemporalReservoir4+recent4",
                "other_layers": "sink1+recent8",
                "max_read_full_frame_equivalents": 9,
                "selected_head_counts": {
                    key: len(layers) * 12 for key, layers in MAP_SPECS.items()
                },
                "exclusive_dynamic_owner": True,
            },
            "hypotheses": {
                "primary": (
                    "interleaved8 retains the v157 Pareto improvement with "
                    "20 percent fewer reservoir layers than interleaved10"
                ),
                "exploratory": (
                    "interleaved6 and interleaved12 bound the layer-count dose"
                ),
                "claim_boundary": (
                    "v158 studies layer budget, not head membership or a new "
                    "placement search"
                ),
            },
            "metric_gate": {
                "primary_method": PRIMARY,
                "min_dynamic_gain_over_recent": 0.02,
                "min_temporal_recovery_over_all_reservoir": 0.003,
                "min_history_delta_over_recent": -0.002,
                "min_temporal_delta_over_recent": -0.004,
                "min_visual_delta_over_recent": -0.01,
                "min_dynamic_delta_over_interleaved10": -0.02,
                "min_temporal_delta_over_interleaved10": -0.002,
                "min_history_delta_over_interleaved10": -0.002,
                "min_visual_delta_over_interleaved10": -0.005,
            },
            "blind_review": {
                "predeclared_primary": PRIMARY,
                "promotion_controls": [
                    "ours_interleaved10_reservoir4_reference",
                    "ours_all_recent8_reference",
                ],
                "contextual_controls": [
                    "ours_middle10_reservoir4_reference",
                    "ours_all_reservoir4_reference",
                ],
                "exploratory_methods": [
                    "ours_interleaved6_reservoir4",
                    "ours_interleaved12_reservoir4",
                ],
                "thresholds": {
                    "max_primary_severe_failures": 1,
                    "min_overall_noninferior_prompts": 10,
                    "min_mean_identity_background_delta": -0.125,
                    "min_motion_mean_delta": -0.25,
                },
            },
        }
    )
    extra_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "build_v158_interleaved_budget_maps.py",
        ROOT / "scripts" / "analyze_v158_vbench.py",
        ROOT / "scripts" / "analyze_v158_blind_review.py",
        ROOT / "scripts" / "prepare_v158_vbench_comparison.py",
        ROOT / "scripts" / "prepare_v157_metric_screened_review.py",
        ROOT / "scripts" / "analyze_v157_metric_screened_review.py",
        ROOT / "scripts" / "analyze_v157_blind_review.py",
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
    args.v158_source = load_v157_source(prompt_manifest)
    authorization = load_blind_authorization()
    args.launch_authorization_sha256 = None
    methods = runner.methods_for(args.candidate_keys, scope=args.method_scope)
    if tuple(method.key for method in methods) != EXPECTED_METHOD_KEYS:
        raise SystemExit(f"v158 requires frozen method order: {EXPECTED_METHOD_KEYS}")
    budget_manifest, budget_paths, map_audits = load_budget_maps(args)
    middle_path = ROOT / "configs" / "head_maps" / V157_MAP_FILENAMES["middle10"]
    args.head_maps = {**budget_paths, "middle10": middle_path}
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
        budget_manifest=budget_manifest,
        map_audits=map_audits,
    )
    contract_path = args.out_root / "contracts" / "experiment.json"
    contract_sha = (
        write_frozen(contract_path, contract)
        if args.node_rank == 0
        else wait_for_frozen(contract_path, contract, args.contract_wait_seconds)
    )
    tasks = runner.selected_tasks(
        methods, node_rank=args.node_rank, num_nodes=args.num_nodes
    )
    gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpu-list must contain unique GPU ids")
    print(
        "[V158Contract] "
        + canonical_json(
            {
                "methods": [method.key for method in methods],
                "contract_sha256": contract_sha,
                "new_videos": len(NEW_METHODS) * PROMPT_COUNT,
                "reused_videos": len(REUSE_METHODS) * PROMPT_COUNT,
                "launch_ready": authorization["ready"],
            }
        ).decode("utf-8").strip(),
        flush=True,
    )
    if args.mode == "preflight":
        state = "PASS" if authorization["ready"] else "HOLD"
        print(
            f"[v158-preflight] {state} node={args.node_rank}/{args.num_nodes} "
            f"tasks={len(tasks)} gpus={len(gpus)} "
            f"human_confirmation={authorization['reason']}",
            flush=True,
        )
        return
    authorization_path = (
        args.out_root / "contracts" / "v157_human_authorization.json"
    )
    if args.mode == "generate":
        if not authorization["ready"]:
            raise SystemExit(
                "v158 generation is blocked until a frozen v157 human "
                f"authorization passes: {authorization['path']}"
            )
        authorization_payload = {
            "version": 1,
            "experiment": EXPERIMENT,
            "contract_sha256": contract_sha,
            "authorization": authorization,
        }
        args.launch_authorization_sha256 = (
            write_frozen(authorization_path, authorization_payload)
            if args.node_rank == 0
            else wait_for_frozen(
                authorization_path,
                authorization_payload,
                args.contract_wait_seconds,
            )
        )
    elif not authorization_path.is_file():
        raise SystemExit("v158 audit requires the frozen launch authorization")
    else:
        args.launch_authorization_sha256 = sha256(authorization_path)

    if args.mode == "audit":
        payload = runner.audit_published(
            args, methods=methods, contract_sha256=contract_sha
        )
        print(
            f"[v158-audit] PASS methods={len(payload['methods'])} "
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
        "launch_authorization_sha256": args.launch_authorization_sha256,
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
        raise SystemExit("\n".join(failures or ["v158 task count mismatch"]))
    print(f"[complete] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
