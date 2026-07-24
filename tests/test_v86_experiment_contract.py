from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_v86_screen_has_16_complex_prompts() -> None:
    prompt_path = ROOT / "prompts" / "v86_single_long_complex_16.txt"
    prompts = [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]

    assert len(prompts) == 16
    assert all(len(prompt) >= 500 for prompt in prompts)


def test_v86_screen_assigns_16_unique_methods_and_gpus() -> None:
    script = (ROOT / "scripts" / "run_v86_role_transition_16gpu.sh").read_text()
    screen = script.split("launch_screen() {", 1)[1].split("\n}", 1)[0]
    launches = re.findall(
        r"launch run_(?:sf|pf|echo|transition) (\S+) \"\$\{GPUS\[(\d+)\]\}\"",
        screen,
    )

    assert len(launches) == 16
    assert len({method for method, _ in launches}) == 16
    assert {int(gpu) for _, gpu in launches} == set(range(16))
    assert {"sf_native", "pf", "echo_pc", "v78"}.issubset(
        {method for method, _ in launches}
    )
    assert "use a clean OUT_ROOT" in script
    assert '-eq "$PROMPT_COUNT"' in script


def test_v86_postprocess_defaults_to_parallel_vbench_long() -> None:
    script = (ROOT / "scripts" / "postprocess_v86_role_transition.sh").read_text()
    run_script = (ROOT / "scripts" / "run_v86_role_transition_16gpu.sh").read_text()
    screen = run_script.split("launch_screen() {", 1)[1].split("\n}", 1)[0]
    run_methods = re.findall(
        r"launch run_(?:sf|pf|echo|transition) (\S+) \"\$\{GPUS\[\d+\]\}\"",
        screen,
    )
    postprocess_screen = script.split("screen)", 1)[1].split(";;", 1)[0]
    postprocess_methods = re.search(
        r"METHODS=\(\s*(.*?)\s*\)", postprocess_screen, re.DOTALL
    )

    assert 'RUN_VBENCH="${RUN_VBENCH:-1}"' in script
    assert "v86_single_long_complex_16.txt" in script
    assert 'CUDA_VISIBLE_DEVICES="$gpu"' in script
    assert "--mode long_custom_input" in script
    assert postprocess_methods is not None
    assert postprocess_methods.group(1).split() == run_methods
    assert '[[ "$count" -eq "$EXPECTED" ]]' in script
