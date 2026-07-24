from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_v90_priority_factorization_16gpu.sh"
POST_SCRIPT = ROOT / "scripts" / "postprocess_v90_priority_factorization.sh"


def _launches() -> list[tuple[str, int]]:
    script = RUN_SCRIPT.read_text()
    return [
        (method, int(gpu))
        for method, gpu in re.findall(
            r"^launch run_(?:pf|transition) (\S+) \"\$\{GPUS\[(\d+)\]\}\"",
            script,
            re.MULTILINE,
        )
    ]


def test_v90_assigns_16_unique_methods_and_gpus() -> None:
    launches = _launches()

    assert len(launches) == 16
    assert len({method for method, _ in launches}) == 16
    assert {gpu for _, gpu in launches} == set(range(16))


def test_v90_contains_three_matched_pf_v78_seed_pairs() -> None:
    script = RUN_SCRIPT.read_text()

    for seed in (1, 2, 3):
        assert re.search(
            rf"launch run_pf pf_s{seed} .* {seed}$",
            script,
            re.MULTILINE,
        )
        assert re.search(
            rf"launch run_transition v78_s{seed} .* {seed} \"\"",
            script,
            re.MULTILINE,
        )


def test_v90_postprocess_matches_generation_methods() -> None:
    methods = [method for method, _ in _launches()]
    script = POST_SCRIPT.read_text()
    match = re.search(r"METHODS=\(\s*(.*?)\s*\)", script, re.DOTALL)

    assert match is not None
    assert match.group(1).split() == methods
    assert "BASELINE_METHODS=(pf v78 pf_binary_balanced learned_balanced)" in script
    assert "motion_smoothness" not in re.search(
        r'VBENCH_DIMS="\$\{VBENCH_DIMS:-(.*?)\}"',
        script,
    ).group(1)
    assert '[[ "${#TRACES[@]}" -eq 13 ]]' in script
    assert "--skip_m3" in script
    assert "analyze_v90_metrics.py" in script


def test_v90_builds_and_uses_pf_factorized_maps() -> None:
    script = RUN_SCRIPT.read_text()

    assert "build_pf_transition_controls.py" in script
    assert "pf_binary.csv" in script
    assert "wave_only.csv" in script
    assert "veil_only.csv" in script
    assert "pf_age_only" in script
    assert "pf_novelty_only" in script
