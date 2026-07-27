from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys


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


v116 = _load(
    "v116_role_memory_runner_no_torch",
    SCRIPTS / "run_v116_role_memory_diverse16.py",
)


def test_diverse16_manifest_is_frozen_to_moviebench_128():
    manifest_path = ROOT / "prompts" / "moviegenbench_diverse16.json"
    source_path = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "prompts"
        / "MovieGenVideoBench_num128.txt"
    )
    items, payload = v116.load_prompt_items(source_path, manifest_path)

    assert payload["source_prompt_count"] == 128
    assert len(items) == 16
    assert [item.subset_index for item in items] == list(range(16))
    assert len({item.source_index for item in items}) == 16
    covered_tags = {tag for item in items for tag in item.tags}
    assert {
        "human_identity",
        "multi_subject",
        "fast_motion",
        "scene_transition",
        "transformation",
    }.issubset(covered_tags)


def test_v116_default_is_nine_methods_times_sixteen_prompts():
    manifest_path = ROOT / "prompts" / "moviegenbench_diverse16.json"
    source_path = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "prompts"
        / "MovieGenVideoBench_num128.txt"
    )
    prompts, _ = v116.load_prompt_items(source_path, manifest_path)
    methods = v116.methods_for(v116.DEFAULT_METHODS)
    tasks = v116.all_tasks(methods, prompts)

    assert v116.DEFAULT_METHODS == (
        "landmark_recent8",
        "landmark_motion2",
        "landmark_motion1",
        "landmark_prototype2",
        "landmark_snapshot2",
        "landmark_retrieval2",
        "landmark_sparse75",
        "support_prototype_recent",
        "prototype_motion1",
    )
    assert [method.key for method in methods] == list(v116.DEFAULT_METHODS)
    assert len(tasks) == 144
    assert len({cell.name for _, _, cell in tasks}) == 144
    assert {
        prompt.source_index for _, prompt, _ in tasks
    } == {item.source_index for item in prompts}
    method_csv = ",".join(v116.DEFAULT_METHODS)
    assert (
        f"m9_{hashlib.sha256(method_csv.encode()).hexdigest()[:12]}"
        == "m9_7a14c511d500"
    )
    for name in ("run_v116_vbench_long.sh", "run_v116_aux_metrics.sh"):
        assert method_csv in (SCRIPTS / name).read_text(encoding="utf-8")


def test_v116_four_node_partition_is_complete_and_nonoverlapping():
    manifest_path = ROOT / "prompts" / "moviegenbench_diverse16.json"
    source_path = (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "prompts"
        / "MovieGenVideoBench_num128.txt"
    )
    prompts, _ = v116.load_prompt_items(source_path, manifest_path)
    methods = v116.methods_for(v116.DEFAULT_METHODS)
    shards = [
        v116.selected_tasks(
            methods,
            prompts,
            node_rank=rank,
            num_nodes=4,
        )
        for rank in range(4)
    ]
    names = [cell.name for shard in shards for _, _, cell in shard]

    assert [len(shard) for shard in shards] == [36, 36, 36, 36]
    assert len(names) == len(set(names)) == 144


def test_v116_default_factorizes_suppressive_cache_under_landmark_support():
    methods = v116.methods_for(v116.DEFAULT_METHODS)
    landmark_methods = [
        method for method in methods
        if method.source_cell.support_policy == "landmark"
    ]
    assert {
        method.source_cell.suppress_policy for method in landmark_methods
    } == {
        "recent8_sink1",
        "motion_pair",
        "motion_pair1",
        "prototype2",
        "snapshot2",
        "retrieval2",
        "sparse75",
    }
    assert all(
        not method.role.endswith("control")
        for method in methods
    )


def test_v116_method_parser_rejects_unknown_and_duplicate_keys():
    for raw in (
        "unknown",
        "prototype_motion1,prototype_motion1",
        "control_landmark_recent,landmark_recent8",
    ):
        try:
            v116.parse_method_keys(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid method list: {raw}")


def test_v116_publishes_metric_specific_names_without_prompt_drift():
    assert v116.published_name(0) == "000000.mp4"
    assert v116.published_name(15) == "000015.mp4"
    assert v116.published_name(0, indexed=True) == "000000-0_v116.mp4"
    assert v116.published_name(15, indexed=True) == "000015-0_v116.mp4"


def test_v116_audit_requires_both_publication_views(tmp_path):
    method = v116.methods_for(("prototype_motion1",))[0]
    prompt = v116.PromptItem(
        subset_index=0,
        source_index=7,
        source_number=8,
        tags=("test",),
        text="test prompt",
    )
    source = tmp_path / "videos" / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not-a-real-video")
    target = (
        tmp_path
        / "published"
        / method.key
        / v116.published_name(0)
    )
    indexed_target = (
        tmp_path
        / "published_indexed"
        / method.key
        / v116.published_name(0, indexed=True)
    )
    v116.link_or_validate(source, target)
    v116.link_or_validate(source, indexed_target)

    marker = (
        tmp_path
        / "status"
        / "published"
        / f"{method.key}.s00.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "experiment_contract_sha256": "contract",
                "method": method.key,
                "subset_index": 0,
                "source_index": 7,
                "source": str(source),
                "target": str(target),
                "indexed_target": str(indexed_target),
                "size": source.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    prompt_file = tmp_path / "prompts" / "moviegenbench_diverse16.txt"
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("test prompt\n", encoding="utf-8")

    payload = v116.audit_published(
        SimpleNamespace(out_root=tmp_path, method_set_id="test"),
        methods=(method,),
        prompts=[prompt],
        contract_sha256="contract",
    )

    assert payload["ok"] is True
    assert payload["methods"][0]["video_count"] == 1
    assert payload["methods"][0]["indexed_video_count"] == 1
