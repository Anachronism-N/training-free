from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_horizon_selector_recovers_crossfit_time_conditioning() -> None:
    module = load_module(
        "v200_horizon_positive",
        SCRIPTS / "analyze_v200_head_phase_horizon.py",
    )
    prompts, calls, layers, heads, positions = 16, 2, 2, 6, 6
    gain = np.zeros((prompts, calls, layers, heads, positions), dtype=np.float64)
    horizon = np.linspace(-1.0, 1.0, positions)
    flat_cells = calls * layers * heads
    for cell in range(flat_cells):
        call, remainder = divmod(cell, layers * heads)
        layer, head = divmod(remainder, heads)
        direction = 1.0 if cell % 2 == 0 else -1.0
        gain[:, call, layer, head, :] = 0.08 * direction * horizon
    prompt_offsets = np.linspace(-0.002, 0.002, prompts)
    gain += prompt_offsets[:, None, None, None, None]
    tensors = {
        "gain": gain,
        "energy": np.ones_like(gain),
        "full_budget": np.ones_like(gain),
    }
    original_fractions = module.SPARSITY_FRACTIONS
    original_primary = module.PRIMARY_FRACTION
    module.SPARSITY_FRACTIONS = (0.25, 0.50, 0.75)
    module.PRIMARY_FRACTION = 0.50
    try:
        report = module.analyze_operator_tensor(
            tensors,
            discovery=list(range(8)),
            validation=list(range(8, 16)),
            operator_index=0,
            bootstrap_samples=500,
            permutations=800,
        )
    finally:
        module.SPARSITY_FRACTIONS = original_fractions
        module.PRIMARY_FRACTION = original_primary
    primary = next(row for row in report["selector_tests"] if row["fraction"] == 0.50)
    assert primary["equal_exposure_verified"] is True
    assert primary["paired_delta_ci95"][0] > 0.0
    assert primary["paired_win_fraction"] == 1.0
    assert primary["time_assignment_permutation_p"] <= 0.05
    assert report["horizon_conditioning_gate"] is True


def test_horizon_selector_rejects_time_invariant_gain() -> None:
    module = load_module(
        "v200_horizon_null",
        SCRIPTS / "analyze_v200_head_phase_horizon.py",
    )
    rng = np.random.default_rng(2026)
    static = rng.normal(size=(2, 2, 8))
    gain = np.repeat(static[None, ..., None], 16, axis=0)
    gain = np.repeat(gain, 6, axis=-1)
    report = module.selector_test(
        gain,
        list(range(8)),
        list(range(8, 16)),
        fraction=0.25,
        seed=7,
        bootstrap_samples=500,
        permutations=500,
    )
    assert report["equal_exposure_verified"] is True
    assert abs(report["paired_delta_mean"]) < 1e-12
    assert report["paired_delta_ci95"] == [0.0, 0.0]
    assert report["time_assignment_permutation_p"] == 1.0


def test_prompt_slopes_are_centered_and_length_agnostic() -> None:
    module = load_module(
        "v200_horizon_slopes",
        SCRIPTS / "analyze_v200_head_phase_horizon.py",
    )
    positions = np.linspace(-1.0, 1.0, 7)
    gain = 0.25 + 0.4 * positions
    tensor = np.broadcast_to(gain, (5, 2, 3, 4, 7)).copy()
    slopes = module.prompt_slopes(tensor)
    assert slopes.shape == (5, 2, 3, 4)
    assert np.allclose(slopes, 0.4)


def test_v200_runner_is_zero_gpu_and_does_not_generate_video() -> None:
    runner = (SCRIPTS / "run_v200_head_phase_horizon_audit.sh").read_text(
        encoding="utf-8"
    )
    assert "preflight|analyze|show|package" in runner
    assert "CUDA_VISIBLE_DEVICES" not in runner
    assert "inference.py" not in runner
    assert "ffmpeg" not in runner
    assert 'tar -czf "$archive" -C "$OUT_ROOT" analysis' in runner
    assert "manual_review_required=false" in runner


def test_v200_contract_keeps_generation_holdout_unused() -> None:
    source = (SCRIPTS / "analyze_v200_head_phase_horizon.py").read_text(
        encoding="utf-8"
    )
    assert '"generation_holdout_used": False' in source
    assert '"changes_v189_frozen_map": False' in source
    assert '"new_video_generation_required": False' in source
    assert "equal_exposure_verified" in source
    assert "time_assignment_permutation_p" in source
