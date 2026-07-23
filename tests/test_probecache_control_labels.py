import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONTROLS = _load(
    "build_probecache_control_labels",
    "scripts/build_probecache_control_labels.py",
)
COMPARE = _load(
    "compare_probecache_head_profiles",
    "scripts/compare_probecache_head_profiles.py",
)


def test_control_maps_preserve_counts_and_define_negative_controls():
    learned = [[1, -1, 1], [-1, 1, -1]]
    pf = [[1, 2, -1], [-1, 1, 2]]
    entries = []
    for layer in range(2):
        for head in range(3):
            index = layer * 3 + head
            entries.append(
                {
                    "layer": layer,
                    "head": head,
                    "remote_z": float(index),
                    "prompt_z": float(5 - index),
                }
            )
    controls = CONTROLS.build_controls(
        learned,
        {"entries": entries},
        pf,
        random_seeds=[7],
    )

    assert controls["inverse"] == [[-1, 1, -1], [1, -1, 1]]
    assert controls["pf_binary"] == [[1, -1, -1], [-1, 1, -1]]
    assert sum(value == 1 for row in controls["remote_only"] for value in row) == 3
    assert sum(value == 1 for row in controls["prompt_only"] for value in row) == 3
    assert [
        row.count(1) for row in controls["random_7"]
    ] == [
        row.count(1) for row in learned
    ]


def test_profile_comparison_reports_agreement_and_kappa():
    reference = [[1, -1], [1, -1]]
    candidate = [[1, -1], [-1, 1]]
    report = COMPARE.compare_profiles(reference, candidate)
    assert report["agreement"] == 0.5
    assert report["cohen_kappa"] == 0.0
    assert report["persistent_jaccard"] == 1 / 3
