from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "scripts" / "run_v93_moviebench_main_16gpu.sh"
HEAD = ROOT / "scripts" / "run_v93_moviebench_head32_16gpu.sh"
POST = ROOT / "scripts" / "postprocess_v93_moviebench.sh"
QUEUE = ROOT / "scripts" / "run_v93_moviebench_10h.sh"

MAIN_METHODS = (
    "sf_native",
    "pf",
    "echo_pc",
    "v78",
    "pf_binary_read_v78",
    "prompt_pfcount_read_v78",
    "prompt_kmeans_read_v78",
    "veil_priority_b005",
)
HEAD_METHODS = (
    "pf",
    "pf_binary_read",
    "prompt_pfcount_read",
    "prompt_kmeans_read",
    "v78",
    "pf_binary_read_v78",
    "prompt_pfcount_read_v78",
    "prompt_kmeans_read_v78",
    "prompt_replica_read_v78",
    "prompt_consensus_read_v78",
    "prompt_inverse_read_v78",
    "prompt_random_read_v78",
    "remote_read_v78",
    "role_score_read_v78",
    "pf_read_prompt_priority",
    "prompt_read_prompt_priority",
)


def test_main_is_eight_methods_two_shards_and_128_prompts():
    text = MAIN.read_text(encoding="utf-8")
    launched = re.findall(r"^launch_pair ([a-z0-9_]+) ", text, re.MULTILINE)
    assert tuple(launched) == MAIN_METHODS
    assert 'PROMPT_COUNT" -eq 128' in text
    assert "0 0 64" in text
    assert "1 64 128" in text
    assert "--reseed_per_prompt" in text
    assert "expected 10 transition traces" in text


def test_head_screen_is_exact_16_cell_causal_contract():
    text = HEAD.read_text(encoding="utf-8")
    launched = re.findall(r"^launch ([a-z0-9_]+) ", text, re.MULTILINE)
    assert tuple(launched) == HEAD_METHODS
    assert len(set(launched)) == 16
    assert 'PROMPT_COUNT" -eq 32' in text
    assert "--reseed_per_prompt" in text
    assert "expected 12 transition traces" in text
    assert "prompt_inverse_pfcount.csv" in text
    assert "prompt_random_pfcount.csv" in text
    assert "prompt_replica_pfcount.csv" in text


def test_postprocess_runs_multiple_metrics_and_blind_review():
    text = POST.read_text(encoding="utf-8")
    assert "prepare_blind_review.py" in text
    assert "evaluate_comprehensive.py" in text
    assert "compute_temporal_jump_diagnostic.py" in text
    assert "collect_vbench_long_results.py" in text
    assert "subject_consistency background_consistency" in text
    assert "motion_smoothness" not in text
    for method in (*MAIN_METHODS, *HEAD_METHODS):
        assert method in text


def test_queue_runs_generation_before_corresponding_metrics():
    text = QUEUE.read_text(encoding="utf-8")
    assert text.index("main-generation") < text.index("main-metrics")
    assert text.index("head32-generation") < text.index("head32-metrics")


def test_all_three_inference_engines_support_global_shards_and_reseeding():
    for relative in (
        "third_party/Self-Forcing/inference.py",
        "third_party/Pyramid-Forcing/inference.py",
        "third_party/Echo-Forcing/inference.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "--start_idx" in text
        assert "--end_idx" in text
        assert "--reseed_per_prompt" in text
