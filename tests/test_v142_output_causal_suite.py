import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_v142_output_causal_suite.py"
)
SPEC = importlib.util.spec_from_file_location("v142_suite", SCRIPT)
V142 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V142)


def test_v142_builds_natural_and_persistent_aba_suites(tmp_path):
    prompts = tmp_path / "natural.txt"
    prompts.write_text(
        "\n".join(f"Natural prompt {index}" for index in range(128)) + "\n",
        encoding="utf-8",
    )
    metadata = V142.write_suite(
        tmp_path / "suite",
        natural_prompts=prompts,
        seed=0,
    )
    assert metadata["natural_count"] == 128
    assert metadata["aba_count"] == 32
    assert metadata["persistent_capture_frames"] == [0, 18, 36]

    natural = [
        json.loads(line)
        for line in (
            tmp_path / "suite" / "v142_natural_128.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    aba = [
        json.loads(line)
        for line in (
            tmp_path / "suite" / "v142_persistent_aba_32.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["dataset_index"] for row in natural] == list(range(128))
    assert {row["kind"] for row in natural} == {"output_causal_natural"}
    assert [row["dataset_index"] for row in aba] == list(range(32))
    assert {row["kind"] for row in aba} == {
        "output_causal_persistent_aba"
    }
    assert {tuple(row["switch_frames"]) for row in aba} == {(39, 78)}
    assert all(row["persistent_capture_frames"] == [0, 18, 36] for row in aba)
    assert all(len(row["shadow_prompts"]) == 4 for row in aba)
