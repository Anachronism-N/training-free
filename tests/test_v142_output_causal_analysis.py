import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_v142_output_causal_profiles.py"
)
SPEC = importlib.util.spec_from_file_location("v142_analysis", SCRIPT)
V142 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V142)


def _policy_record(values):
    return {
        "causal_policy_metrics": {
            name: {
                "projected_relative_error": torch.tensor(
                    [values[index], values[index] * 2]
                )
            }
            for index, name in enumerate(V142.POLICIES)
        }
    }


def test_policy_vectors_preserve_argmin_and_normalize():
    record = _policy_record([0.3, 0.1, 0.2])
    raw = V142._raw_policy_vector(record)
    normalized = V142._normalized_policy_vector(record)
    assert raw.shape == (2, 3)
    assert np.array_equal(raw.argmin(axis=-1), np.array([1, 1]))
    assert np.allclose(normalized.sum(axis=-1), 1.0)


def test_persistent_selectivity_uses_a_minus_b_and_paraphrase_noise():
    branches = {
        "exact_a": {
            "persistent_probe_metrics": {
                "content_top1_cosine": torch.tensor([0.8, 0.4])
            }
        },
        "exact_b": {
            "persistent_probe_metrics": {
                "content_top1_cosine": torch.tensor([0.5, 0.6])
            }
        },
        "paraphrase_a": {
            "persistent_probe_metrics": {
                "content_top1_cosine": torch.tensor([0.75, 0.42])
            }
        },
        "paraphrase_b": {
            "persistent_probe_metrics": {
                "content_top1_cosine": torch.tensor([0.52, 0.58])
            }
        },
    }
    selectivity, noise = V142._persistent_selectivity(
        branches, "content_top1_cosine"
    )
    assert np.allclose(selectivity, [0.3, -0.2])
    assert np.allclose(noise, [0.035, 0.02])


def test_recommendation_prefers_context_conditioned_routing():
    natural = {
        "correctness_gate": True,
        "static_policy_gate": False,
        "online_policy_opportunity": True,
    }
    aba = {
        "correctness_gate": True,
        "prompt_policy_modulation_gate": False,
        "persistent_a_selectivity_gate": True,
    }
    assert (
        V142._recommendation(natural, aba)
        == "online_context_conditioned_output_causal_routing"
    )
