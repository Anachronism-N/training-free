import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_v138_history_intervention_suite.py"
)
SPEC = importlib.util.spec_from_file_location("v138_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_suite_uses_same_seed_and_preserves_prompt_order():
    prompts = [f"Long prompt {index}" for index in range(128)]
    jobs = MODULE.build_jobs(prompts, seed=17)
    assert len(jobs) == 128
    assert [job["dataset_index"] for job in jobs] == list(range(128))
    assert [job["base_prompt"] for job in jobs] == prompts
    assert {job["seed"] for job in jobs} == {17}
    assert {job["kind"] for job in jobs} == {"history_intervention"}
