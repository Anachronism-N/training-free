import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_v134_head_discovery_suite.py"
)
SPEC = importlib.util.spec_from_file_location("v134_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_counterfactual_suite_has_balanced_single_factor_jobs():
    jobs = MODULE.build_counterfactual_jobs()
    assert len(jobs) == 128
    assert [job["dataset_index"] for job in jobs] == list(range(128))
    assert len({job["job_id"] for job in jobs}) == 128
    assert {
        factor: sum(job["factor"] == factor for job in jobs)
        for factor in MODULE.FACTORS
    } == {factor: 16 for factor in MODULE.FACTORS}
    for job in jobs:
        assert job["changed_fields"] == [job["factor"]]
        assert job["base_prompt"] != job["semantic_prompt"]
        assert job["base_prompt"] != job["null_prompt"]
        assert len(job["base_prompt"].split()) >= 45
    for family_index in range(16):
        family = jobs[family_index * 8 : (family_index + 1) * 8]
        assert len({job["base_prompt"] for job in family}) == 1
        assert len({job["null_prompt"] for job in family}) == 1
        assert {job["seed"] for job in family} == {family_index}


def test_observational_suite_preserves_moviebench_order():
    prompts = [f"Complex prompt number {index}" for index in range(128)]
    jobs = MODULE.build_observational_jobs(prompts)
    assert [job["base_prompt"] for job in jobs] == prompts
    assert jobs[17]["dataset_index"] == 17
    assert jobs[17]["kind"] == "observational"
