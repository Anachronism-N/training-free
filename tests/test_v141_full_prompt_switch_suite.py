import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_v141_full_prompt_switch_suite.py"
)
SPEC = importlib.util.spec_from_file_location("v141_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v141_suite_has_controlled_aba_contract(tmp_path):
    metadata = MODULE.write_suite(tmp_path, seed=0)
    jobs = [
        json.loads(line)
        for line in (
            tmp_path / "v141_full_prompt_switch_32.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    prompts = (
        tmp_path / "v141_full_prompt_switch_32.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert metadata["job_count"] == 32
    assert len(jobs) == len(prompts) == 32
    assert [job["dataset_index"] for job in jobs] == list(range(32))
    assert {job["switch_type"] for job in jobs} == {
        "scene_action",
        "identity_scene",
    }
    assert all(job["switch_frames"] == [39, 78] for job in jobs)
    assert all(job["segment_labels"] == ["A1", "B", "A2"] for job in jobs)
    assert all(job["base_prompt"] == prompt for job, prompt in zip(jobs, prompts))
    for job in jobs:
        prompt_a, prompt_b, prompt_a2 = job["schedule_prompts"]
        assert prompt_a == prompt_a2
        assert prompt_a != prompt_b
        assert job["base_prompt"] == f"{prompt_a} || {prompt_b} || {prompt_a}"
        assert set(job["shadow_prompts"]) == {
            "exact_a",
            "exact_b",
            "paraphrase_a",
            "paraphrase_b",
        }
        assert job["shadow_prompts"]["exact_a"] == prompt_a
        assert job["shadow_prompts"]["exact_b"] == prompt_b
