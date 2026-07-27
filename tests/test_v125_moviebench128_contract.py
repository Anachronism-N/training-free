from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v125 = _load(
    "v125_moviebench128_runner_no_torch",
    SCRIPTS / "run_v125_moviebench128_main.py",
)
prepare = _load(
    "v125_comparison_preparer",
    SCRIPTS / "prepare_v125_moviebench128_comparison.py",
)
merge = _load(
    "v125_vbench_merger",
    SCRIPTS / "merge_v125_vbench_long_parts.py",
)
splits = _load(
    "v125_vbench_split_preparer",
    SCRIPTS / "prepare_v125_vbench_splits.py",
)


def test_v125_protocol_and_default_candidates_are_frozen():
    runner = v125.runner
    assert runner.EXPERIMENT == "v125_moviebench128_main"
    assert runner.PROMPT_COUNT == 128
    assert runner.TASK_STAGE == "moviebench128"
    assert runner.PUBLISHED_TAG == "v125"
    assert runner.RUN_LABEL == "v125"
    assert runner.ALLOW_PARTIAL_SCOPE is False
    assert runner.DEFAULT_CANDIDATES == (
        "landmark_retrieval1_age24",
        "landmark_retrieval_motion",
    )

    methods = runner.methods_for(
        runner.DEFAULT_CANDIDATES,
        scope="ours",
    )
    assert [method.key for method in methods] == [
        "ours_landmark_retrieval1_age24",
        "ours_landmark_retrieval_motion",
    ]
    assert methods[0].source_cell.suppress_policy == "retrieval1_age24"
    assert (
        methods[1].source_cell.suppress_policy
        == "retrieval1_motion1_age24"
    )


def test_v125_four_node_partition_has_sixteen_videos_per_gpu():
    runner = v125.runner
    methods = runner.methods_for(runner.DEFAULT_CANDIDATES)
    shards = [
        runner.selected_tasks(methods, node_rank=rank, num_nodes=4)
        for rank in range(4)
    ]
    identities = [
        (method.key, prompt_index)
        for shard in shards
        for method, prompt_index, _ in shard
    ]
    assert [len(shard) for shard in shards] == [128, 128, 128, 128]
    assert len(identities) == len(set(identities)) == 4 * 128
    assert 128 // 8 == 16
    assert {
        prompt_index for _, prompt_index in identities
    } == set(range(128))


def test_v125_parser_uses_rewritten_128_prompt_file(monkeypatch):
    runner = v125.runner
    monkeypatch.delenv("V120_BASELINE_ONLY", raising=False)
    monkeypatch.delenv("V120_OURS_ONLY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v125_moviebench128_main.py",
            "--promotion-approved",
        ],
    )
    args = runner.parse_args()
    assert args.prompts.name == "MovieGen_128_qwen.txt"
    assert runner.DEFAULT_PROMPT_PATH == prepare.QWEN_PROMPT_PATH
    assert prepare.REWRITE_SCRIPT_PATH.endswith(
        "RollingForcing/scripts/prompt_refine_qwen.py"
    )
    assert args.method_scope == "all"
    assert args.out_root.parent.name == "v125_moviebench128_main"


def test_v125_comparison_names_remain_pairwise_parseable():
    assert prepare.comparison_name(0) == "000000-0.mp4"
    assert prepare.comparison_name(127) == "000127-0.mp4"
    assert prepare.SOURCE_METHODS == {
        "sf_native": "sf_native",
        "pf_native": "pf_native",
        "ours_retrieval_age24": "ours_landmark_retrieval1_age24",
        "ours_retrieval_motion": "ours_landmark_retrieval_motion",
    }


def test_blind_review_accepts_official_prompt_sample_names(tmp_path):
    import prepare_blind_review

    method_dir = tmp_path / "method"
    method_dir.mkdir()
    (method_dir / "000000-0.mp4").write_bytes(b"video")
    (method_dir / "000001-0.mp4").write_bytes(b"video")
    assert [path.name for path in prepare_blind_review._videos(method_dir, 2)] == [
        "000000-0.mp4",
        "000001-0.mp4",
    ]


def test_v125_source_root_is_full_sf_pf_ours_table(tmp_path):
    source = prepare.expected_source_root(tmp_path)
    assert source.parent == tmp_path / "runs" / "v125_moviebench128_main"
    assert source.name.startswith("ours2_")


def test_v125_source_contract_requires_exact_rewritten_prompt_items(tmp_path):
    source_root = tmp_path / "source"
    contract_dir = source_root / "contracts"
    contract_dir.mkdir(parents=True)
    prompts = [f"rewritten prompt {index}" for index in range(128)]
    prompt_path = tmp_path / "MovieGen_128_qwen.txt"
    prompt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    prompt_sha = prepare.sha256(prompt_path)
    method_keys = list(prepare.SOURCE_METHODS.values())
    manifest = {
        "ok": True,
        "prompt_count": 128,
        "prompt_file_sha256": prompt_sha,
        "methods": [{"key": key} for key in method_keys],
    }
    contract = {
        "experiment": "v125_moviebench128_main",
        "prompt_count": 128,
        "candidate_keys": list(prepare.OURS_CANDIDATES),
        "baseline_only": False,
        "num_output_frames": 120,
        "decoded_video_contract": {
            "frames": 477,
            "fps": 16,
            "duration_seconds": 29.8125,
        },
        "seed": 0,
        "methods": [{"key": key} for key in method_keys],
        "prompts": {
            "path": str(prompt_path.resolve()),
            "sha256": prompt_sha,
            "items": [
                {"index": index, "text": text}
                for index, text in enumerate(prompts)
            ],
        },
    }
    (source_root / "published_manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    contract_path = contract_dir / "experiment.json"
    contract_path.write_text(
        json.dumps(contract) + "\n",
        encoding="utf-8",
    )

    prepare.validate_source_contract(
        source_root,
        prompt_path=prompt_path.resolve(),
        prompt_sha256=prompt_sha,
        prompts=prompts,
    )

    contract["prompts"]["items"][17]["text"] = "wrong prompt"
    contract_path.write_text(
        json.dumps(contract) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="prompt items differ"):
        prepare.validate_source_contract(
            source_root,
            prompt_path=prompt_path.resolve(),
            prompt_sha256=prompt_sha,
            prompts=prompts,
        )


def test_v125_vbench_merger_extracts_scores_and_video_paths():
    value = [
        0.75,
        [
            {
                "video_path": "/run/000000-0.mp4",
                "video_results": 0.7,
            },
            {
                "video_path": "/run/000001-0.mp4",
                "video_results": 0.8,
            },
        ],
    ]
    assert merge.finite_score(value) == 0.75
    assert merge.collect_video_paths(value) == {
        "/run/000000-0.mp4",
        "/run/000001-0.mp4",
    }
    assert merge.collect_prompt_indices(value) == {0, 1}


def test_v125_vbench_shell_shards_dimensions_and_checks_raft():
    text = (SCRIPTS / "run_v125_vbench_long.sh").read_text(
        encoding="utf-8"
    )
    assert "dynamic_degree" in text
    assert "raft-things.pth" in text
    assert "amt-s.pth" in text
    assert "prepare_v125_vbench_splits.py" in text
    assert "index % NUM_NODES == NODE_RANK" in text
    assert 'run_worker "$slot" "$gpu"' in text
    assert "done_marker_valid" in text
    assert "job_contract_sha256" in text
    assert "merge_v125_vbench_long_parts.py" in text


def test_v125_split_validation_is_contract_and_clip_complete(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(splits, "PROMPT_COUNT", 2)
    monkeypatch.setattr(splits, "CLIPS_PER_VIDEO", 3)
    split_root = tmp_path / "split_clip"
    split_root.mkdir()
    for prompt_index in range(2):
        stem = f"{prompt_index:06d}-0"
        folder = split_root / stem
        folder.mkdir()
        for clip_index in range(3):
            (folder / f"{stem}_{clip_index:03d}.mp4").write_bytes(b"clip")
    (split_root / ".v125_split_manifest.json").write_text(
        json.dumps(
            {
                "comparison_manifest_sha256": "manifest",
                "vbench_commit": "commit",
                "prompt_count": 2,
                "clips_per_video": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert splits.validate_split(
        split_root,
        comparison_manifest_sha256="manifest",
        vbench_commit="commit",
    )["clip_count"] == 6
    (split_root / "000001-0" / "000001-0_002.mp4").unlink()
    assert (
        splits.validate_split(
            split_root,
            comparison_manifest_sha256="manifest",
            vbench_commit="commit",
        )
        is None
    )


def test_v125_comparison_manifest_has_four_default_methods():
    assert list(prepare.SOURCE_METHODS) == [
        "sf_native",
        "pf_native",
        "ours_retrieval_age24",
        "ours_retrieval_motion",
    ]


def test_v125_merger_requires_and_preserves_all_128_prompts(
    tmp_path,
    monkeypatch,
):
    comparison = tmp_path / "comparison"
    methods = ["sf_native", "pf_native", "ours_a", "ours_b"]
    dimensions = [
        "subject_consistency",
        "background_consistency",
        "aesthetic_quality",
        "imaging_quality",
        "motion_smoothness",
        "dynamic_degree",
    ]
    manifest = {
        "experiment": "v125_moviebench128_comparison",
        "prompt_count": 128,
        "num_output_frames": 120,
        "seed": 0,
        "methods": [{"key": method} for method in methods],
        "vbench_long_dimensions": dimensions,
    }
    manifest_path = comparison / "comparison_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = merge.sha256(manifest_path)

    for method in methods:
        video_dir = comparison / "published" / method
        video_dir.mkdir(parents=True)
        for index in range(128):
            (video_dir / f"{index:06d}-0.mp4").write_bytes(b"video")
        for dimension in dimensions:
            part_dir = (
                comparison
                / "metrics"
                / "vbench_long_parts"
                / method
                / dimension
            )
            part_dir.mkdir(parents=True)
            result = {
                dimension: [
                    0.75,
                    [
                        {
                            "video_path": f"/run/{index:06d}-0.mp4",
                            "video_results": 0.75,
                        }
                        for index in range(128)
                    ],
                ]
            }
            result_path = part_dir / "results.json"
            result_path.write_text(
                json.dumps(result) + "\n",
                encoding="utf-8",
            )
            contract_path = part_dir / "job_contract.json"
            contract_path.write_text(
                json.dumps(
                    {
                        "comparison_manifest_sha256": manifest_sha,
                        "method": method,
                        "dimension": dimension,
                        "vbench_commit": "test-commit",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (part_dir / "done.json").write_text(
                json.dumps(
                    {
                        "comparison_manifest_sha256": manifest_sha,
                        "method": method,
                        "dimension": dimension,
                        "result_sha256": merge.sha256(result_path),
                        "job_contract_sha256": merge.sha256(contract_path),
                        "vbench_commit": "test-commit",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_v125_vbench_long_parts.py",
            "--comparison-root",
            str(comparison),
        ],
    )
    merge.main()

    summary = json.loads(
        (comparison / "metrics" / "vbench_long_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["prompt_count"] == 128
    assert summary["methods"]["ours_a"]["dynamic_degree"] == 0.75
    coverage = json.loads(
        (comparison / "metrics" / "vbench_long_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(coverage["jobs"]) == len(methods) * len(dimensions)
    assert {
        row["reported_prompt_indices"] for row in coverage["jobs"]
    } == {128}
