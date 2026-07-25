from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_v96_cache_screen_has_16_distinct_cells():
    script = (
        ROOT / "scripts" / "run_v96_binary_cache_16gpu.sh"
    ).read_text(encoding="utf-8")
    launches = [
        line.strip().split()[1]
        for line in script.splitlines()
        if line.strip().startswith("launch ")
    ]

    assert len(launches) == 16
    assert len(set(launches)) == 16
    assert "pf_binary_cyclic" in launches
    assert "pf_binary_merge" in launches
    assert "consensus_cyclic" in launches
    assert "consensus_merge" in launches
    assert "random_merge" in launches
    assert "inverse_merge" in launches


def test_v96_profile_does_not_reuse_pf_membership():
    script = (
        ROOT / "scripts" / "run_v96_qk_head_profile_16gpu.sh"
    ).read_text(encoding="utf-8")

    assert "uniform_stride_all_heads.csv" in script
    assert '--pyramidkv_head_config_path "$UNIFORM_LABELS"' in script
    assert "--few_step_cfg_enabled" in script
    assert "--head_qk_profile_branches cond,uncond" in script
    assert 'branches != {"cond", "uncond"}' in script


def test_v96_binary_policy_is_explicit_at_runtime():
    inference = (
        ROOT / "third_party" / "Pyramid-Forcing" / "inference.py"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts" / "run_v96_binary_cache_16gpu.sh"
    ).read_text(encoding="utf-8")

    assert "--pyramidkv_binary_responsive_policy" in inference
    assert "[BinaryPolicyOverride]" in inference
    assert "[PyramidKVRuntimePolicy]" in (
        ROOT
        / "third_party"
        / "Pyramid-Forcing"
        / "pipeline"
        / "causal_inference.py"
    ).read_text(encoding="utf-8")
    assert 'grep -q "\\[BinaryPolicyOverride\\]' in runner
