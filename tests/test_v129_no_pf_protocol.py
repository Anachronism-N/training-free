from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_v100_fast_selection_1video import Cell, inference_command


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load(
    "v129_comparison_for_tests",
    SCRIPTS / "prepare_v129_paper_comparison.py",
)
external = _load(
    "v129_external_for_tests",
    SCRIPTS / "run_v129_external_baselines.py",
)
prompt_aware = _load(
    "v129_prompt_aware_for_tests",
    SCRIPTS / "eval_vbench_long_prompt_aware.py",
)
splits = _load(
    "v129_splits_for_tests",
    SCRIPTS / "prepare_v129_vbench_splits.py",
)
table = _load(
    "v129_table_for_tests",
    SCRIPTS / "build_v129_paper_table.py",
)
analyzer = _load(
    "v129_gate_analyzer_for_tests",
    SCRIPTS / "analyze_v129_retrieval_gate.py",
)


def test_v129_frozen_comparison_has_eight_methods_and_no_pf():
    keys = [source.final_key for source in comparison.SOURCES]
    assert keys == [
        "sf_native",
        "deep_forcing",
        "rolling_forcing",
        "longlive",
        "ours_prototype_retrieval_age24",
        "ours_confidence_recent",
        "ours_prototype_retrieval_motion",
        "ours_confidence_motion",
    ]
    assert not any("pf" in key.lower() for key in keys)
    roles = {
        source.final_key: source.role for source in comparison.SOURCES
    }
    assert roles["deep_forcing"] == "same_checkpoint_external_method"
    assert roles["rolling_forcing"] == "external_trained_system"
    assert roles["longlive"] == "external_trained_system"
    sources = {
        source.final_key: source.source_key
        for source in comparison.SOURCES
    }
    # v125 intentionally renamed the raw "...retrieval1..." source while
    # assembling comparison_quality8; v129 consumes that published alias.
    assert (
        sources["ours_prototype_retrieval_age24"]
        == "ours_prototype_retrieval_age24"
    )
    assert comparison.CORE_DIMENSIONS == (
        "subject_consistency",
        "background_consistency",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        "overall_consistency",
    )


def test_v129_internal_runner_source_freezes_two_gate_candidates():
    source = (
        SCRIPTS / "run_v129_ours128_main.py"
    ).read_text(encoding="utf-8")
    assert 'runner.INCLUDE_PF_BASELINE = False' in source
    assert 'runner.PROMPT_COUNT = 128' in source
    assert '"prototype_retrieval_conf_recent"' in source
    assert '"prototype_retrieval_conf_motion"' in source
    assert "retrieval_abstain=True" in source
    assert "CONFIDENCE_MIN_SIMILARITY = 0.55" in source
    assert "CONFIDENCE_MIN_MARGIN = 0.005" in source


def test_v129_gate_flags_reach_inference_cli(tmp_path):
    cell = Cell(
        "confidence_gate",
        "test",
        "single",
        support_policy="prototype",
        suppress_policy="retrieval1_age24",
        retrieval_abstain=True,
        retrieval_min_similarity=0.55,
        retrieval_min_margin=0.005,
    )
    args = SimpleNamespace(
        single_prompts=tmp_path / "moviebench.txt",
        aba_prompts=tmp_path / "aba.txt",
        single_prompt_index=0,
        aba_prompt_index=0,
        pf_labels=tmp_path / "pf.csv",
        legacy_map=tmp_path / "legacy.csv",
        pf_config=tmp_path / "config.yaml",
        pf_checkpoint=tmp_path / "model.pt",
        seed=0,
        pf_repo=tmp_path,
        num_output_frames=120,
    )
    command, _, _, _ = inference_command(
        args,
        cell=cell,
        output=tmp_path / "videos",
        transition_trace=tmp_path / "transition.jsonl",
        scene_trace=tmp_path / "scene.jsonl",
    )
    assert "--pyramidkv_semantic_retrieval_abstain" in command
    assert command[
        command.index("--pyramidkv_semantic_retrieval_min_similarity") + 1
    ] == "0.55"
    assert command[
        command.index("--pyramidkv_semantic_retrieval_min_margin") + 1
    ] == "0.005"
    assert command[command.index("--num_output_frames") + 1] == "120"


def test_v129_master_script_has_no_pf_or_aba_task():
    source = (
        SCRIPTS / "run_v129_no_pf_10h.sh"
    ).read_text(encoding="utf-8")
    assert "pf_native" not in source
    assert "generate-aba" not in source
    assert "deep_forcing,rolling_forcing,longlive" in source
    assert "V129_DURATION_SECONDS=30" in source


def test_external_worker_intervals_cover_128_in_four_prompt_batches():
    intervals = [
        external.interval_for_worker(rank, 32)
        for rank in range(32)
    ]
    assert intervals[0] == (0, 4)
    assert intervals[-1] == (124, 128)
    assert all(end - start == 4 for start, end in intervals)
    covered = [
        index
        for start, end in intervals
        for index in range(start, end)
    ]
    assert covered == list(range(128))
    assert external.parse_method_keys(
        "longlive,deep_forcing,rolling_forcing"
    ) == ("deep_forcing", "rolling_forcing", "longlive")


def test_external_resume_removes_owned_publication_for_corrupt_raw(
    tmp_path,
    monkeypatch,
):
    args = SimpleNamespace(
        out_root=tmp_path,
        duration=30,
        expected_video_frames=477,
    )
    method = external.Method(
        "deep_forcing",
        tmp_path,
        tmp_path / "config.yaml",
        (),
        tmp_path / "wan",
    )
    raw_dir = tmp_path / "raw" / method.key / "worker000"
    raw_dir.mkdir(parents=True)
    source = raw_dir / "0-0_ema.mp4"
    source.write_bytes(b"corrupt")
    target, indexed = external.publication_paths(args, method, 0)
    target.parent.mkdir(parents=True)
    indexed.parent.mkdir(parents=True)
    target.write_bytes(b"old")
    indexed.write_bytes(b"old")
    marker = external.prompt_marker(args, method, 0)
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "experiment_contract_sha256": "contract",
                "worker_config_sha256": "worker",
                "method": method.key,
                "prompt_index": 0,
                "source": str(source),
                "target": str(target),
                "indexed_target": str(indexed),
            }
        ),
        encoding="utf-8",
    )

    def reject_video(*_args, **_kwargs):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(external, "validate_video", reject_video)
    repairs = external.repair_partial_raw_videos(
        args,
        method,
        raw_dir,
        start=0,
        end=1,
        contract_sha="contract",
        worker_config_sha="worker",
    )
    assert len(repairs) == 1
    assert len(repairs[0]["publication_removed"]) == 3
    assert not source.exists()
    assert not target.exists()
    assert not indexed.exists()
    assert not marker.exists()


def test_prompt_mapping_replaces_numeric_folder_names(tmp_path):
    full_info = tmp_path / "full_info.json"
    full_info.write_text(
        json.dumps(
            [
                {
                    "prompt_en": "000000-0",
                    "video_list": [
                        str(
                            tmp_path
                            / "split_clip"
                            / f"{index:06d}-0"
                            / f"{index:06d}-0_000.mp4"
                        )
                    ],
                }
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )
    prompts = ["first prompt", "second prompt", "third prompt"]
    report = prompt_aware.rewrite_full_info_prompts(full_info, prompts)
    payload = json.loads(full_info.read_text(encoding="utf-8"))
    assert report["mapped_count"] == 3
    assert [row["prompt_en"] for row in payload] == prompts
    assert [row["v129_prompt_index"] for row in payload] == [0, 1, 2]


def test_prompt_mapping_rejects_incomplete_coverage(tmp_path):
    full_info = tmp_path / "full_info.json"
    full_info.write_text(
        json.dumps(
            [
                {
                    "video_list": [
                        str(
                            tmp_path
                            / "split_clip"
                            / "000000-0"
                            / "000000-0_000.mp4"
                        )
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        prompt_aware.rewrite_full_info_prompts(
            full_info,
            ["first", "second"],
        )


def test_v129_split_cache_is_bound_to_source_video_sizes(tmp_path):
    video_dir = tmp_path / "published" / "method"
    split_root = video_dir / "split_clip"
    split_root.mkdir(parents=True)
    source_rows = []
    for index in range(2):
        stem = f"{index:06d}-0"
        source = video_dir / f"{stem}.mp4"
        source.write_bytes(b"source")
        source_rows.append({"name": source.name, "size": source.stat().st_size})
        folder = split_root / stem
        folder.mkdir()
        for clip_index in range(2):
            (folder / f"{stem}_{clip_index:03d}.mp4").write_bytes(b"clip")
    (split_root / ".v129_split_manifest.json").write_text(
        json.dumps(
            {
                "comparison_manifest_sha256": "manifest",
                "vbench_commit": "commit",
                "video_dir": str(video_dir.resolve()),
                "prompt_count": 2,
                "clips_per_video": 2,
                "source_videos": source_rows,
            }
        ),
        encoding="utf-8",
    )
    assert splits.validate_split(
        split_root,
        comparison_manifest_sha256="manifest",
        vbench_commit="commit",
        prompt_count=2,
        clips_per_video=2,
    ) is not None
    (video_dir / "000001-0.mp4").write_bytes(b"changed-size")
    assert splits.validate_split(
        split_root,
        comparison_manifest_sha256="manifest",
        vbench_commit="commit",
        prompt_count=2,
        clips_per_video=2,
    ) is None


def test_official_composite_requires_every_dimension():
    keys = {
        dimension.replace("_", " ")
        for dimension in table.QUALITY_DIMS + table.SEMANTIC_DIMS
    }
    constants = SimpleNamespace(
        NORMALIZE_DIC={
            key: {"Min": 0.0, "Max": 1.0} for key in keys
        },
        DIM_WEIGHT={key: 1.0 for key in keys},
        QUALITY_LIST=[
            dimension.replace("_", " ")
            for dimension in table.QUALITY_DIMS
        ],
        SEMANTIC_LIST=[
            dimension.replace("_", " ")
            for dimension in table.SEMANTIC_DIMS
        ],
    )
    row = {dimension: 0.5 for dimension in table.QUALITY_DIMS}
    score, missing = table.official_composite(
        row,
        table.QUALITY_DIMS,
        constants=constants,
    )
    assert score == pytest.approx(0.5)
    assert missing == []
    row.pop("dynamic_degree")
    score, missing = table.official_composite(
        row,
        table.QUALITY_DIMS,
        constants=constants,
    )
    assert score is None
    assert missing == ["dynamic_degree"]


def test_gate_analyzer_reports_selected_and_abstained_reads(tmp_path):
    trace = tmp_path / "trace.policy.jsonl"

    def event(reason: str, selected: list[dict], top1: float):
        return {
            "event": "middle_selection",
            "layer": 7,
            "head": 2,
            "branch": "conditional",
            "cache_contract_violations": [],
            "strategies": [
                {
                    "name": "SemanticRetrievalStrategy",
                    "state": {
                        "last_retrieval": {
                            "reason": reason,
                            "selected": selected,
                            "top1_similarity": top1,
                            "top2_similarity": 0.50,
                            "margin": top1 - 0.50,
                            "min_similarity": 0.55,
                            "min_margin": 0.005,
                            "abstain_on_low_confidence": True,
                        }
                    },
                }
            ],
        }

    trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                event(
                    "selected",
                    [{"t": 4, "age": 8, "similarity": 0.8}],
                    0.8,
                ),
                event("similarity_gate", [], 0.4),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    summary, sweep = analyzer.summarize_method(
        "ours_prototype_retrieval_conf_recent",
        [(0, trace)],
    )
    assert summary["retrieval_record_count"] == 2
    assert summary["scored_candidate_count"] == 2
    assert summary["selected_rate"] == pytest.approx(0.5)
    assert summary["selected_rate_when_scored"] == pytest.approx(0.5)
    assert summary["low_confidence_abstain_rate"] == pytest.approx(0.5)
    assert summary[
        "low_confidence_abstain_rate_when_scored"
    ] == pytest.approx(0.5)
    assert summary["selected_age"]["p50"] == pytest.approx(8.0)
    assert len(sweep) == (
        len(analyzer.SIMILARITY_SWEEP) * len(analyzer.MARGIN_SWEEP)
    )
