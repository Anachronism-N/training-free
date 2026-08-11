from fractions import Fraction

from scripts.build_v172_relative_depth_maps import (
    contiguous_layers,
    interleaved_layers,
    relative_depth_specs,
    rounded_quota,
)


def test_current_30_layer_rules_reproduce_middle10_and_v157_interleaved10():
    specs = relative_depth_specs(30)

    assert specs["center_1of3"] == tuple(range(10, 20))
    assert specs["early_1of3"] == tuple(range(10))
    assert specs["late_1of3"] == tuple(range(20, 30))
    assert specs["interleaved_1of3"] == (
        1,
        4,
        7,
        10,
        13,
        16,
        19,
        22,
        25,
        28,
    )


def test_fractional_quotas_are_architecture_relative():
    assert rounded_quota(30, Fraction(1, 6)) == 5
    assert rounded_quota(30, Fraction(1, 4)) == 8
    assert rounded_quota(30, Fraction(1, 3)) == 10
    assert rounded_quota(30, Fraction(1, 2)) == 15
    assert rounded_quota(24, Fraction(1, 3)) == 8
    assert rounded_quota(32, Fraction(1, 3)) == 11


def test_custom_depth_specs_are_complete_unique_and_in_bounds():
    for num_layers in (12, 24, 30, 32, 40):
        specs = relative_depth_specs(num_layers)
        for layers in specs.values():
            assert len(layers) == len(set(layers))
            assert all(0 <= layer < num_layers for layer in layers)


def test_contiguous_and_interleaved_rules_have_equal_count():
    for num_layers, count in ((24, 8), (30, 10), (32, 11)):
        early = contiguous_layers(num_layers, count, "early")
        center = contiguous_layers(num_layers, count, "center")
        late = contiguous_layers(num_layers, count, "late")
        interleaved = interleaved_layers(num_layers, count)

        assert all(len(item) == count for item in (early, center, late, interleaved))
        assert early[0] == 0
        assert late[-1] == num_layers - 1
