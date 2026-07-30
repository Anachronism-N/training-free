import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_v144_factorized_mechanism_suite.py"
)
SPEC = importlib.util.spec_from_file_location("v144_factor_suite_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_v144_suite_is_factor_matched_and_seed_controlled():
    jobs = MODULE.build_jobs(seed_base=700)
    assert len(jobs) == 128
    assert [job["dataset_index"] for job in jobs] == list(range(128))
    assert len({job["job_id"] for job in jobs}) == 128
    for family in range(16):
        rows = {
            job["variant"]: job
            for job in jobs
            if job["family_index"] == family
        }
        assert tuple(rows) == MODULE.VARIANTS
        base = rows["base"]
        assert rows["seed_control"]["base_prompt"] == base["base_prompt"]
        assert rows["seed_control"]["seed"] == base["seed"] + 10000
        assert rows["seed_control"]["same_seed_as_base"] is False
        assert rows["paraphrase"]["changed_factors"] == []
        assert rows["paraphrase"]["surface_rewrite"] is True
        for factor in ("identity", "scene", "action", "camera"):
            assert rows[factor]["changed_factors"] == [factor]
            assert rows[factor]["seed"] == base["seed"]
            assert rows[factor]["normalized_token_edit_distance"] > 0
        assert set(rows["full_semantic"]["changed_factors"]) == set(
            MODULE.V134.FACTORS
        )


def test_v144_suite_writes_reproducible_artifacts(tmp_path):
    first = MODULE.write_suite(tmp_path / "first", seed_base=123)
    second = MODULE.write_suite(tmp_path / "second", seed_base=123)
    assert first["prompts_sha256"] == second["prompts_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["job_count"] == 128
