import numpy as np
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "lifecycle_kv"
    / "head_taxonomy.py"
)
SPEC = importlib.util.spec_from_file_location("v143_head_taxonomy", MODULE_PATH)
TAXONOMY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = TAXONOMY
SPEC.loader.exec_module(TAXONOMY)

CLUSTER_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "cluster_v143_multiaxis_head_taxonomy.py"
)
CLUSTER_SPEC = importlib.util.spec_from_file_location(
    "v143_cluster_taxonomy_test", CLUSTER_PATH
)
CLUSTER = importlib.util.module_from_spec(CLUSTER_SPEC)
assert CLUSTER_SPEC.loader is not None
sys.modules[CLUSTER_SPEC.name] = CLUSTER
CLUSTER_SPEC.loader.exec_module(CLUSTER)

PF_ANCHOR = TAXONOMY.PF_ANCHOR
PF_VEIL = TAXONOMY.PF_VEIL
PF_WAVE = TAXONOMY.PF_WAVE
adjusted_rand_index = TAXONOMY.adjusted_rand_index
align_labels = TAXONOMY.align_labels
deterministic_kmeans = TAXONOMY.deterministic_kmeans
forcing_kv_labels = TAXONOMY.forcing_kv_labels
head_forcing_labels = TAXONOMY.head_forcing_labels
normalized_mutual_information = TAXONOMY.normalized_mutual_information
pf_tri_pattern_labels = TAXONOMY.pf_tri_pattern_labels
robust_fit_transform = TAXONOMY.robust_fit_transform
silhouette_score = TAXONOMY.silhouette_score


def test_published_threshold_reimplementations_have_expected_roles():
    time = np.arange(72)
    logits = np.stack(
        (
            np.ones(72),
            -np.ones(72),
            np.sin(2 * np.pi * time / 6.0),
        )
    )
    labels, diagnostics = pf_tri_pattern_labels(logits)
    assert labels.tolist() == [PF_ANCHOR, PF_VEIL, PF_WAVE]
    assert diagnostics["dominant_period"][2] < 6.4

    forcing = forcing_kv_labels(
        np.asarray([0.9, 0.2]), np.asarray([1.0, 1.0])
    )
    assert forcing.tolist() == ["static", "dynamic"]

    head_forcing = head_forcing_labels(
        np.arange(20, dtype=float),
        np.arange(20, dtype=float)[::-1],
    )
    assert (head_forcing == "anchor").sum() == 5
    assert (head_forcing == "local").sum() == 3
    assert (head_forcing == "memory").sum() == 12


def test_deterministic_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(11)
    left = rng.normal(-3, 0.1, size=(40, 3))
    right = rng.normal(3, 0.1, size=(40, 3))
    values = np.concatenate((left, right), axis=0)
    scaled, _, _, _ = robust_fit_transform(values)
    result = deterministic_kmeans(scaled, 2, restarts=8, seed=5)
    truth = np.asarray([0] * 40 + [1] * 40)
    aligned = align_labels(truth, result.labels)
    assert adjusted_rand_index(truth, aligned) == 1.0
    assert normalized_mutual_information(truth, aligned) > 0.999
    assert (
        normalized_mutual_information(
            np.zeros(4, dtype=int),
            np.asarray([0, 0, 1, 1]),
        )
        == 0.0
    )
    assert silhouette_score(scaled, aligned) > 0.9


def test_cluster_splits_controlled_jobs_by_family_not_factor():
    assert CLUSTER._independent_unit("cf_00_action", "v136_prompt") == 0
    assert CLUSTER._independent_unit("cf_00_style", "v136_prompt") == 0
    assert CLUSTER._independent_unit("cf_01_action", "v136_prompt") == 1
    assert (
        CLUSTER._independent_unit(
            "moviebench_history_intervention_127", "v138_local"
        )
        == 127
    )
    assert (
        CLUSTER._feature_group("v136_prompt.cphi_score")
        == "prompt_modulation"
    )
    assert (
        CLUSTER._feature_group("v143_natural.policy_need")
        == "output_policy"
    )
    assert (
        CLUSTER._feature_group("v143_ab.persistent_output")
        == "episodic_compatibility"
    )
