import importlib.util
import sys
from collections import Counter
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_v145_crossed_seed_head_suite.py"
)
SPEC = importlib.util.spec_from_file_location("v145_suite_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_crossed_suite_pairs_every_variant_with_both_seeds():
    jobs = MODULE.build_jobs(seed_base=145000)
    assert len(jobs) == 160
    assert [job["dataset_index"] for job in jobs] == list(range(160))
    assert Counter(job["variant"] for job in jobs) == {
        variant: 32 for variant in MODULE.VARIANTS
    }
    assert Counter(job["seed_replicate"] for job in jobs) == {0: 80, 1: 80}

    for family in range(16):
        rows = [job for job in jobs if job["family_index"] == family]
        assert len(rows) == 10
        for replicate in MODULE.SEED_REPLICATES:
            by_variant = {
                job["variant"]: job
                for job in rows
                if job["seed_replicate"] == replicate
            }
            assert set(by_variant) == set(MODULE.VARIANTS)
            assert len({job["seed"] for job in by_variant.values()}) == 1
            assert all(
                job["reference_seed"] == job["seed"]
                for job in by_variant.values()
            )
            assert (
                by_variant["base"]["base_prompt"]
                == by_variant["paraphrase"]["reference_prompt"]
            )
        seed0 = next(
            job["seed"] for job in rows
            if job["seed_replicate"] == 0
        )
        seed1 = next(
            job["seed"] for job in rows
            if job["seed_replicate"] == 1
        )
        assert seed1 - seed0 == 10000


def test_crossed_suite_writes_reproducible_contract(tmp_path):
    first = MODULE.write_suite(tmp_path / "first", seed_base=145000)
    second = MODULE.write_suite(tmp_path / "second", seed_base=145000)
    assert first["job_count"] == 160
    assert first["prompts_sha256"] == second["prompts_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
