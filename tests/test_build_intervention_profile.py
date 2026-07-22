import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_intervention_profile.py"
SPEC = importlib.util.spec_from_file_location("build_intervention_profile", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_profile_pairs_native_and_expands_groups():
    rows = []
    for seed in ("0", "1", "2"):
        rows.append(
            {
                "cell": "sf_native",
                "prompt_id": "p0",
                "seed": seed,
                "layer_start": "0",
                "layer_end": "0",
                "head_start": "0",
                "head_end": "0",
                "dino": "0.5",
                "motion": "0.5",
                "loop": "0.2",
            }
        )
        rows.append(
            {
                "cell": "group_a",
                "prompt_id": "p0",
                "seed": seed,
                "layer_start": "4",
                "layer_end": "5",
                "head_start": "2",
                "head_end": "3",
                "dino": "0.6",
                "motion": "0.55",
                "loop": "0.1",
            }
        )
    profile = MODULE.build_profile(
        rows,
        baseline_cell="sf_native",
        num_layers=30,
        num_heads=12,
        metrics=["dino", "motion", "loop"],
        min_samples=3,
    )

    assert profile["paired_observations"] == 3
    assert len(profile["entries"]) == 4
    assert all(entry["reliability"] == 1.0 for entry in profile["entries"])
