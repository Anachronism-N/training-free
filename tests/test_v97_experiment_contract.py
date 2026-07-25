from pathlib import Path
import re


ROOT = Path(__file__).parents[1]


def _array(script: str, name: str) -> list[str]:
    match = re.search(rf"{name}=\(\n(.*?)\n\)", script, flags=re.DOTALL)
    assert match is not None
    return [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_v97_generation_has_16_matching_cells():
    script = (
        ROOT / "scripts" / "run_v97_threshold_pf_merge_16gpu.sh"
    ).read_text(encoding="utf-8")
    methods = _array(script, "METHODS")
    cells = [
        line.strip('"').split("|", 1)[0]
        for line in _array(script, "ALL_CELLS")
    ]

    assert len(methods) == 16
    assert len(set(methods)) == 16
    assert cells == methods
    assert "pf_ar_stride_merge" in methods
    assert "pf_aw_stride_merge" in methods
    assert "pf_native" in methods
    assert "pf_anchor_extended_recent" in methods
    assert "pf_wave_extended_recent" in methods
    assert "pf_veil_extended_recent" in methods
    assert "prompt_tau_1p0_random_merge" in methods
    assert "prompt_tau_1p0_reversed_merge" in methods


def test_v97_profile_requires_explicit_complete_layers_and_separate_scores():
    script = (
        ROOT / "scripts" / "run_v97_qk_head_profile_16gpu.sh"
    ).read_text(encoding="utf-8")

    assert 'sources != {"kv_cache.layer_idx"}' in script
    assert "layers != expected_layers" in script
    assert "extract_v97_qk_head_scores.py" in script
    assert "classify_v97_qk_head_scores.py" in script
    assert "qk_head_score_artifact.json" in script


def test_v97_postprocess_matches_generation_methods():
    generation = (
        ROOT / "scripts" / "run_v97_threshold_pf_merge_16gpu.sh"
    ).read_text(encoding="utf-8")
    postprocess = (
        ROOT / "scripts" / "postprocess_v97_threshold_pf_merge.sh"
    ).read_text(encoding="utf-8")

    assert _array(generation, "METHODS") == _array(postprocess, "METHODS")
    assert "summarize_v97_policy_traces.py" in postprocess
    assert "--strict" in postprocess
    assert "analyze_v97_threshold_pf_merge.py" in postprocess
