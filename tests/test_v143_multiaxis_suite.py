import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_v143_multiaxis_profile_suite.py"
)
SPEC = importlib.util.spec_from_file_location("v143_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v143_suite_builds_natural_and_ab_jobs(tmp_path):
    source = tmp_path / "natural.txt"
    source.write_text(
        "\n".join(f"natural prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    metadata = MODULE.write_suite(
        tmp_path / "suite", natural_prompts=source, seed=0
    )
    natural = [
        json.loads(line)
        for line in (
            tmp_path / "suite" / "v143_natural_128.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    ab = [
        json.loads(line)
        for line in (
            tmp_path / "suite" / "v143_ab_32.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert metadata["natural_count"] == len(natural) == 128
    assert metadata["ab_count"] == len(ab) == 32
    assert [job["dataset_index"] for job in ab] == list(range(32))
    assert all(job["switch_frames"] == [57] for job in ab)
    assert all(job["segment_labels"] == ["A", "B"] for job in ab)
    assert all(
        job["persistent_capture_frames"] == [0, 18, 36, 54]
        for job in ab
    )
    for job in ab:
        prompt_a, prompt_b = job["schedule_prompts"]
        assert prompt_a != prompt_b
        assert job["base_prompt"] == f"{prompt_a} || {prompt_b}"
