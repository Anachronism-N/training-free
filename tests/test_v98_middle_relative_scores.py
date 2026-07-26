import math
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import extract_v98_middle_relative_scores as scores


INFERENCE = ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"


def _record(logits):
    return {
        "query_frame": 10,
        "key_frames": list(range(10)),
        "logits": logits,
    }


def test_middle_relative_margin_is_invariant_to_common_logit_shift():
    base = _record(
        [
            [0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    shifted = _record(
        [[value + 37.5 for value in row] for row in base["logits"]]
    )

    left = scores.record_middle_relative_margins(
        base, num_heads=2, sink_frames=3, recent_frames=4
    )
    right = scores.record_middle_relative_margins(
        shifted, num_heads=2, sink_frames=3, recent_frames=4
    )

    assert left is not None and right is not None
    assert left[0] > 0.0
    assert left[1] < 0.0
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(left, right))


def test_middle_relative_margin_excludes_sink_and_uses_latest_distinct_recent():
    record = {
        "query_frame": 12,
        "key_frames": [0, 1, 2, 3, 6, 7, 8, 9, 10, 11],
        # Sink logits are deliberately huge; they must not affect the result.
        "logits": [[1000.0, 1000.0, 1000.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    }
    margin = scores.record_middle_relative_margins(
        record, num_heads=1, sink_frames=3, recent_frames=4
    )
    assert margin is not None
    assert margin[0] > 0.0


def test_middle_relative_margin_requires_both_intervention_groups():
    record = {
        "query_frame": 5,
        "key_frames": [0, 1, 2, 3, 4],
        "logits": [[0.0] * 5],
    }
    assert (
        scores.record_middle_relative_margins(
            record, num_heads=1, sink_frames=3, recent_frames=4
        )
        is None
    )


def test_bootstrap_sign_agreement_is_deterministic_and_sensitive():
    stable = scores.bootstrap_sign_agreement(
        [1.0, 1.1, 0.9, 1.2], rounds=200, seed=7
    )
    unstable = scores.bootstrap_sign_agreement(
        [-1.0, -0.25, 0.5, 1.0], rounds=200, seed=7
    )
    assert stable == 1.0
    assert 0.0 < unstable < stable


def test_bootstrap_treats_exact_zero_effect_as_unstable():
    assert scores.bootstrap_sign_agreement(
        [0.0, 0.0, 0.0, 0.0], rounds=50, seed=7
    ) == 0.0
    assert scores.balanced_bootstrap_sign_agreement(
        {
            "uniform_stride": [1.0, 1.0],
            "uniform_merge": [-1.0, -1.0],
        },
        rounds=50,
        seed=7,
    ) == 0.0
    samples = {
        ("pair_a", "a", 0): 1.0,
        ("pair_a", "b", 0): 1.0,
    }
    assert scores.paired_cluster_bootstrap_sign_agreement(
        {
            "uniform_stride": samples,
            "uniform_merge": {
                key: -value for key, value in samples.items()
            },
        },
        rounds=50,
        seed=7,
    ) == 0.0


def test_balanced_bootstrap_gives_each_probe_policy_equal_weight():
    # A large number of stride observations must not outweigh the merge
    # topology: the estimator is the median of the two policy medians.
    values = {
        "uniform_stride": [4.0] * 100,
        "uniform_merge": [-2.0, -2.0, -2.0],
    }
    agreement = scores.balanced_bootstrap_sign_agreement(
        values, rounds=100, seed=11
    )
    assert agreement == 1.0
    assert scores.median(scores.median(v) for v in values.values()) == 1.0


def test_balanced_bootstrap_requires_every_policy_to_have_observations():
    try:
        scores.balanced_bootstrap_sign_agreement(
            {"uniform_stride": [1.0], "uniform_merge": []},
            rounds=10,
            seed=3,
        )
    except ValueError as error:
        assert "every policy" in str(error)
    else:
        raise AssertionError("empty probe policy should be rejected")


def test_inference_accepts_middle_relative_profile_kind():
    source = INFERENCE.read_text(encoding="utf-8")
    assert 'choices=("prompt", "temporal", "middle_relative")' in source


def test_uniform_probe_map_validator_rejects_mixed_policy(tmp_path):
    path = tmp_path / "probe.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([[1, 1], [1, 2]])
    try:
        scores.validate_uniform_probe_map(
            path, num_layers=2, num_heads=2, expected_label=1
        )
    except ValueError as error:
        assert "uniform label-1" in str(error)
    else:
        raise AssertionError("mixed probe map should be rejected")


def test_paired_cluster_bootstrap_preserves_probe_sample_matching():
    samples = {
        ("pair_a", "a", 0): 1.0,
        ("pair_a", "b", 0): 1.2,
        ("pair_b", "a", 0): 0.8,
        ("pair_b", "b", 0): 1.1,
    }
    agreement = scores.paired_cluster_bootstrap_sign_agreement(
        {
            "uniform_stride": samples,
            "uniform_merge": {key: value * 0.5 for key, value in samples.items()},
        },
        rounds=100,
        seed=5,
    )
    assert agreement == 1.0

    try:
        scores.paired_cluster_bootstrap_sign_agreement(
            {
                "uniform_stride": samples,
                "uniform_merge": {
                    ("pair_a", "a", 0): 1.0,
                },
            },
            rounds=10,
            seed=5,
        )
    except ValueError as error:
        assert "identical policy samples" in str(error)
    else:
        raise AssertionError("unmatched probe samples should be rejected")
