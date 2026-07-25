import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_v96_qk_head_thresholds.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_v96_qk_head_thresholds", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gmm_threshold_recovers_two_prompt_response_groups():
    values = [
        -2.4,
        -2.2,
        -2.0,
        -1.8,
        -1.6,
        -1.5,
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
    ]
    model = MODULE.fit_gmm_1d(values, 2)
    threshold = MODULE.gmm_threshold(model)

    assert model["means"][0] < threshold < model["means"][1]
    assert -1.0 < threshold < 1.0


def test_threshold_map_does_not_copy_pf_class_count():
    scores = {
        (0, head): value
        for head, value in enumerate(
            [-2.5, -2.0, -1.8, -1.5, 1.2, 1.5, 1.8, 2.2]
        )
    }
    matrix, report = MODULE.threshold_map(
        "synthetic", scores, num_layers=1, num_heads=8
    )

    assert matrix == [[1, 1, 1, 1, -1, -1, -1, -1]]
    assert report["stable_count"] == 4
    assert report["responsive_count"] == 4


def test_pf_overlap_is_posthoc_membership_analysis():
    candidate = [[1, 1, -1, -1]]
    pf = [[1, -1, 1, 2]]
    overlap = MODULE.pf_overlap(candidate, pf)

    assert overlap["stable_anchor_precision"] == 0.5
    assert overlap["anchor_recall"] == 0.5
    assert overlap["cross_tab"]["-1"] == {
        "stable": 1,
        "responsive": 0,
    }
