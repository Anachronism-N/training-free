from pathlib import Path


ROOT = Path(__file__).parents[1]
MODEL = (
    ROOT
    / "third_party"
    / "Self-Forcing"
    / "wan"
    / "modules"
    / "causal_model.py"
)
PIPELINE = (
    ROOT
    / "third_party"
    / "Self-Forcing"
    / "pipeline"
    / "causal_inference.py"
)


def test_shadow_attention_returns_before_native_cache_mutation():
    source = MODEL.read_text(encoding="utf-8")
    branch_start = source.index("if profile_read_only:")
    branch_end = source.index("lifecache_layer_enabled = (", branch_start)
    native_mutation = source.index("kv_cache_size = kv_cache[\"k\"].shape[1]")
    branch = source[branch_start:branch_end]
    assert branch_start < branch_end < native_mutation
    assert "kv_cache[\"k\"][:," not in branch
    assert "kv_cache[\"v\"][:," not in branch
    assert "global_end_index\"].fill_" not in branch
    assert "local_end_index\"].fill_" not in branch
    assert "return x" in branch


def test_shadow_runs_before_noisy_input_advances_and_checks_indices():
    source = PIPELINE.read_text(encoding="utf-8")
    loop_start = source.index("profile_capture = False")
    shadow_call = source.index(
        "self._run_head_profile_shadows(", loop_start
    )
    next_noise = source.index("next_noise = torch.randn_like", loop_start)
    assert shadow_call < next_noise
    helper_start = source.index("def _run_head_profile_shadows(")
    helper_end = source.index("\n    def inference(", helper_start)
    helper = source[helper_start:helper_end]
    assert "cache_indices_before" in helper
    assert "cache_indices_after != cache_indices_before" in helper
    assert "cpu_rng_state = torch.random.get_rng_state()" in helper
    assert "torch.cuda.set_rng_state(" in helper
    assert "_mark_head_profile_read_only(self.kv_cache1, False)" in helper
