from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_v132_ablation_comparison as comparison


def test_v132_ablation_dimension_profiles_are_complete():
    assert len(comparison.CORE_DIMENSIONS) == 8
    assert len(comparison.SEMANTIC_DIMENSIONS) == 8
    assert set(comparison.CORE_DIMENSIONS).isdisjoint(
        comparison.SEMANTIC_DIMENSIONS
    )


def test_v132_ablation_sources_are_no_pf():
    source = (
        SCRIPTS / "run_v132_ablation_vbench.sh"
    ).read_text(encoding="utf-8")
    assert "v132_binary_memory_ablation_comparison_30s" in source
    assert "VBENCH_EXPECTED_METHOD_COUNT" in source


def test_v132_ablation_collector_accepts_dynamic_contract():
    source = (
        SCRIPTS / "merge_v129_vbench_long_parts.py"
    ).read_text(encoding="utf-8")
    assert "--expected-experiment" in source
    assert "--expected-method-count" in source
    assert "--expected-num-output-frames" in source
