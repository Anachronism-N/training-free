import math
import sys
import types

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from scripts.analyze_v152_online_policy_profiles import (
    _qualifies,
    _summarize_effects,
)


def test_preferred_policy_gate_uses_paired_seeds_and_effect_size():
    rows = []
    for prompt in range(32):
        for replicate in (0, 1):
            rows.append(
                {
                    "prompt_slot": prompt,
                    "seed_replicate": replicate,
                    "preferred_advantage_x0": math.log(1.08)
                    + prompt * 1e-4
                    + replicate * 1e-5,
                }
            )
    summary = _summarize_effects(
        rows,
        value_field="preferred_advantage_x0",
        label="qk:preferred",
        group="qk_uniform4",
        context="noisy_f117_t500",
        seed=152,
    )
    assert summary["positive_fraction"] == 1.0
    assert summary["seed_replicate_spearman"] > 0.99
    assert _qualifies(summary)


def test_policy_gate_rejects_small_but_reproducible_effect():
    rows = [
        {
            "prompt_slot": prompt,
            "seed_replicate": replicate,
            "preferred_advantage_x0": math.log(1.005) + prompt * 1e-6,
        }
        for prompt in range(32)
        for replicate in (0, 1)
    ]
    summary = _summarize_effects(
        rows,
        value_field="preferred_advantage_x0",
        label="qk:preferred",
        group="qk_uniform4",
        context="noisy_f117_t500",
        seed=153,
    )
    assert not _qualifies(summary)
