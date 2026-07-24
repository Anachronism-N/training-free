from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "run_v92_prompt_binary_cache_16gpu.sh"
POST_SCRIPT = ROOT / "scripts" / "postprocess_v92_prompt_binary_cache.sh"
ECHO_SCRIPT = ROOT / "scripts" / "run_v92_echo_unique_snapshot_4gpu.sh"

METHODS = (
    "pf_binary_read",
    "pf_binary_read_v78",
    "prompt_pfcount_read",
    "prompt_pfcount_read_v78",
    "prompt_kmeans_read",
    "prompt_kmeans_read_v78",
    "prompt_replica_read_v78",
    "prompt_consensus_read_v78",
    "prompt_inverse_read_v78",
    "prompt_random_read_v78",
    "remote_read_v78",
    "role_score_read_v78",
    "pf_read_prompt_priority",
    "prompt_read_prompt_priority",
    "prompt_read_v78_coverage",
    "pf_binary_read_v78_coverage",
)


def test_v92_launches_exact_16_method_contract():
    text = RUN_SCRIPT.read_text(encoding="utf-8")
    launched = re.findall(r"^launch run_cell ([a-z0-9_]+) ", text, re.MULTILINE)
    assert tuple(launched) == METHODS
    assert len(set(launched)) == 16
    assert "v92 requires exactly 16 GPU ids" in text
    assert 'PROMPT_COUNT" -eq 16' in text
    assert "pyramidkv_head_config_path" in text
    assert "PYRAMIDKV_HEAD_MAP_DEBUG=1" in text
    assert r"\[PyramidKVHeadMap\]" in text
    assert "prompt_contrastive_manifest.json" not in text


def test_v92_holds_budget_and_read_policy_fixed_for_prompt_control():
    text = RUN_SCRIPT.read_text(encoding="utf-8")
    assert "prompt_pfcount.csv" in text
    assert "prompt_inverse_pfcount.csv" in text
    assert "prompt_random_pfcount.csv" in text
    assert "read_policy_label_1=stable_stride_sink3_recent4" in text
    assert "read_policy_label_-1=responsive_cyclic_sink1_recent4" in text
    assert "--pyramidkv_cache_transition_min_reliability .55" in text
    assert "--pyramidkv_cache_transition_max_commit_fraction .75" in text


def test_v92_archive_is_bounded_and_not_the_only_candidate():
    text = RUN_SCRIPT.read_text(encoding="utf-8")
    assert "--pyramidkv_structured_memory_archive_policy coverage" in text
    assert "--pyramidkv_structured_memory_archive_max_frames 24" in text
    assert "--pyramidkv_structured_memory_head_labels 1" in text
    assert "--pyramidkv_structured_memory_readout_gate .05" in text
    assert text.count(" 1\n") > 0


def test_v92_postprocess_is_review_first_and_has_13_traces():
    text = POST_SCRIPT.read_text(encoding="utf-8")
    assert 'HUMAN_REVIEW_DONE:-0' in text
    assert "expected 13 transition traces" in text
    assert "--skip_m3" in text
    assert "motion_smoothness" not in text
    for method in METHODS:
        assert method in text


def test_echo_snapshot_screen_is_isolated_and_emits_diagnostics():
    text = ECHO_SCRIPT.read_text(encoding="utf-8")
    assert "requires exactly 4 GPU ids" in text
    assert 'PROMPT_COUNT" -eq 3' in text
    assert "echo_score_weighted" in text
    assert "echo_token_select" in text
    assert "echo_coherent_u015" in text
    assert "echo_coherent_u030" in text
    assert "ECHO_COMPRESS_MODE" in text
    assert r"\[EchoUnique\]" in text
