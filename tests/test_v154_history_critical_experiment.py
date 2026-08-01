from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v152_one_sided_history_critical as head_analysis  # noqa: E402
import analyze_v154_blind_review as blind_analysis  # noqa: E402
import analyze_v154_vbench as vbench_analysis  # noqa: E402
import build_v154_history_critical_suite as suite  # noqa: E402
import prepare_v154_blind_review as blind  # noqa: E402
import prepare_v154_vbench_comparison as vbench_prepare  # noqa: E402
import run_v120_moviebench32_main as parent  # noqa: E402
import run_v154_history_critical_moviebench16 as v154  # noqa: E402
import run_v154_vbench_long as vbench_runner  # noqa: E402
from run_v100_fast_selection_1video import expected_policy  # noqa: E402


PF_ROOT = ROOT / "third_party" / "Pyramid-Forcing"
PF_LABELS = PF_ROOT / "configs" / "head_configs" / "best_labels.csv"
MAP_DIR = ROOT / "configs" / "head_maps"


def test_v154_prompt_suite_is_reproducible() -> None:
    source = ROOT / "prompts" / "moviegen_128_qwen_v129.txt"
    if not source.is_file():
        source = Path(suite.SERVER_SOURCE)
    if not source.is_file():
        pytest.skip("Qwen MovieBench-128 source is not available")
    payloads = suite.build_payloads(source)

    assert set(payloads) == {suite.PROMPT_FILENAME, suite.MANIFEST_FILENAME}
    for filename, payload in payloads.items():
        assert (ROOT / "prompts" / filename).read_bytes() == payload
    manifest = json.loads(payloads[suite.MANIFEST_FILENAME])
    assert manifest["prompt_count"] == 16
    assert [row["source_index"] for row in manifest["items"]] == [
        value[0] for value in suite.SELECTIONS
    ]


def test_text_and_matrix_hashes_are_newline_portable(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"1,2\n3,4\n")
    crlf.write_bytes(b"1,2\r\n3,4\r\n")

    assert head_analysis.sha256_normalized_text_file(
        lf
    ) == head_analysis.sha256_normalized_text_file(crlf)
    committed = json.loads(
        (MAP_DIR / head_analysis.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert committed["version"] == 2
    assert committed["source"]["hash_contract"]["text"].endswith(
        "normalized_to_lf"
    )


def test_v154_method_grid_and_cache_routes_are_frozen() -> None:
    v154.configure_parent_runner()
    methods = parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")

    assert tuple(method.key for method in methods) == v154.EXPECTED_METHOD_KEYS
    assert len(parent.all_tasks(methods)) == 128
    assert [
        len(parent.selected_tasks(methods, node_rank=rank, num_nodes=4))
        for rank in range(4)
    ] == [32, 32, 32, 32]

    primary = v154.V154_CELLS[0]
    assert expected_policy(primary, 10) == (
        ("TemporalPrototypeStrategy",),
        1,
        4,
        "temporal_prototype",
    )
    assert expected_policy(primary, 11) == ((), 1, 8, "stride")
    assert v154.V154_CELLS[0].map_key == "qk_top4"
    assert v154.V154_CELLS[1].map_key == "qk_bottom4_control"
    assert v154.V154_CELLS[2].map_key == "random4_control"


def test_v154_loads_all_frozen_head_maps() -> None:
    args = SimpleNamespace(
        legacy_map=MAP_DIR / "legacy_v98_absolute_sign_304_56.csv",
        pf_labels=PF_LABELS,
    )
    manifest, paths, audits = v154.load_head_maps(args)

    assert manifest["version"] == 2
    assert set(paths) == {
        "qk_top4",
        "qk_bottom4_control",
        "random4_control",
        "legacy",
    }
    assert audits["qk_top4"]["counts"] == {"10": 120, "11": 240}
    assert audits["qk_top4"]["label10_per_layer"] == [4] * 30


def test_blind_review_is_balanced_and_gate_is_paired(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for method in blind.METHODS:
        method_dir = run_root / "published" / method
        method_dir.mkdir(parents=True)
        for prompt in range(blind.PROMPT_COUNT):
            (method_dir / f"{prompt:06d}.mp4").write_bytes(
                f"{method}:{prompt}".encode("ascii")
            )
    prompt_manifest = json.loads(
        (ROOT / "prompts" / suite.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    _, key_rows = blind.build_rows(
        run_root=run_root, prompt_manifest=prompt_manifest
    )
    assert len(key_rows) == 128
    for prompt in range(blind.PROMPT_COUNT):
        assert {
            row["method"]
            for row in key_rows
            if int(row["prompt_index"]) == prompt
        } == set(blind.METHODS)

    completed = []
    for row in key_rows:
        value = 1.0 if row["method"] == blind_analysis.PRIMARY else 0.0
        completed.append(
            {
                **row,
                **{column: value for column in blind_analysis.RATING_COLUMNS},
                blind_analysis.SEVERE_COLUMN: 0,
            }
        )
    report = blind_analysis.analyze(completed)
    assert report["human_promotion_gate"] is True
    assert report["paired_primary_minus_comparator"][
        "ours_qk_random4_control"
    ]["overall_preference_-2_to_2"]["wins"] == 16


def test_vbench_gate_requires_history_gain_without_motion_collapse() -> None:
    rows = {}
    for method in vbench_analysis.METHODS:
        value = 0.8 if method == vbench_analysis.PRIMARY else 0.7
        rows[method] = {
            dimension: value for dimension in vbench_analysis.DIMENSIONS
        }
    for control in vbench_analysis.MEMBERSHIP_CONTROLS:
        rows[control]["dynamic_degree"] = 0.61
    rows[vbench_analysis.PRIMARY]["dynamic_degree"] = 0.60
    payload = {
        "methods": rows,
        "dimensions": list(vbench_analysis.DIMENSIONS),
        "missing": [],
    }

    report = vbench_analysis.analyze(payload)
    assert report["metric_promotion_gate"] is True

    rows[vbench_analysis.PRIMARY]["dynamic_degree"] = 0.50
    report = vbench_analysis.analyze(payload)
    assert report["metric_promotion_gate"] is False


def test_vbench_jobs_and_results_are_prompt_complete(tmp_path: Path) -> None:
    jobs = vbench_runner.all_jobs()
    assert len(jobs) == 64
    assert [len(jobs[rank::4]) for rank in range(4)] == [16] * 4
    assert vbench_prepare.comparison_name(7) == "000007-0.mp4"

    output = tmp_path / "result"
    output.mkdir()
    dimension = "subject_consistency"
    payload = {
        dimension: {
            "score": 0.75,
            "video_results": [
                {"video_path": f"/split/{index:06d}-0/clip.mp4"}
                for index in range(vbench_prepare.PROMPT_COUNT)
            ],
        }
    }
    (output / "candidate_eval_results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    normalized = vbench_runner.normalize_result(output, dimension)
    assert vbench_runner.finite_score(normalized[dimension]) == 0.75
    assert (output / "results.json").is_file()

    payload[dimension]["video_results"].pop()
    (output / "results.json").unlink()
    (output / "candidate_eval_results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="prompt-complete"):
        vbench_runner.normalize_result(output, dimension)


def test_vbench_comparison_materializes_indexed_inputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    contract_path = run_root / "contracts" / "experiment.json"
    contract_path.parent.mkdir(parents=True)
    prompt_manifest_path = ROOT / "prompts" / suite.MANIFEST_FILENAME
    prompt_manifest = json.loads(
        prompt_manifest_path.read_text(encoding="utf-8")
    )
    contract = {
        "experiment": vbench_prepare.EXPERIMENT,
        "prompt_suite": prompt_manifest,
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
        },
        "seed": 0,
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    methods = []
    for method in vbench_prepare.METHODS:
        video_dir = run_root / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(vbench_prepare.PROMPT_COUNT):
            (video_dir / f"{index:06d}.mp4").write_bytes(
                f"{method}:{index}".encode("ascii")
            )
        methods.append(
            {"key": method, "role": "test", "video_dir": str(video_dir)}
        )
    (run_root / "published_manifest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "experiment": vbench_prepare.EXPERIMENT,
                "prompt_count": vbench_prepare.PROMPT_COUNT,
                "experiment_contract_sha256": vbench_prepare.sha256(
                    contract_path
                ),
                "methods": methods,
            }
        ),
        encoding="utf-8",
    )

    comparison_root = run_root / "vbench_comparison"
    report = vbench_prepare.prepare(
        run_root, comparison_root, prompt_manifest_path
    )
    assert report["videos"] == 128
    manifest = json.loads(
        (comparison_root / "comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["key"] for row in manifest["methods"]] == list(
        vbench_prepare.METHODS
    )
    target = comparison_root / "published" / "sf_native" / "000004-0.mp4"
    source = run_root / "published" / "sf_native" / "000004.mp4"
    assert target.samefile(source)


def test_v125_reuse_preserves_source_prompt_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "v125"
    method_rows = []
    for source_method in v154.REUSE_METHODS.values():
        video_dir = source_root / "published" / source_method
        video_dir.mkdir(parents=True)
        for index in range(128):
            (video_dir / f"{index:06d}.mp4").write_bytes(
                f"{source_method}:{index}".encode("ascii")
            )
        method_rows.append({"key": source_method, "video_dir": str(video_dir)})
    prompt_manifest = json.loads(
        (ROOT / "prompts" / suite.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    (source_root / "published_manifest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "prompt_count": 128,
                "prompt_file_sha256": prompt_manifest["source"][
                    "canonical_sha256"
                ],
                "methods": method_rows,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("V154_REUSE_V125_ROOT", str(source_root))
    reuse = v154.load_v125_reuse(prompt_manifest)
    assert reuse is not None

    v154.configure_parent_runner()
    method = next(
        item
        for item in parent.methods_for(parent.DEFAULT_CANDIDATES, scope="all")
        if item.key == "sf_native"
    )
    cell = parent.task_cell(method, 2)
    args = SimpleNamespace(
        out_root=tmp_path / "v154",
        v154_reuse=reuse,
        v154_prompt_manifest=prompt_manifest,
    )
    result = v154.run_reused_task(
        args,
        method=method,
        prompt_index=2,
        cell=cell,
        gpu="0",
        contract_sha256="contract",
    )

    assert result["generation_status"] == "reused_v125"
    assert result["source_prompt_index"] == 4
    target = args.out_root / "published" / "sf_native" / "000002.mp4"
    source = source_root / "published" / "sf_native" / "000004.mp4"
    assert target.samefile(source)
