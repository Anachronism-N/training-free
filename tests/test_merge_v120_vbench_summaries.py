from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_merge():
    path = SCRIPTS / "merge_v120_vbench_summaries.py"
    spec = importlib.util.spec_from_file_location("merge_v120_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(methods):
    return {
        "methods": methods,
        "dimensions": ["subject_consistency", "dynamic_degree"],
        "sources": {key: f"{key}.json" for key in methods},
        "missing": [],
    }


def test_merge_v120_vbench_keeps_baseline_then_ours_order():
    merge = _load_merge().merge_summaries
    payload = merge(
        _summary({"sf_native": {}, "pf_native": {}}),
        _summary({"ours_landmark_motion1": {}}),
    )
    assert list(payload["methods"]) == [
        "sf_native",
        "pf_native",
        "ours_landmark_motion1",
    ]


def test_merge_v120_vbench_rejects_non_ours_method():
    merge = _load_merge().merge_summaries
    with pytest.raises(ValueError, match="ours summary"):
        merge(
            _summary({"sf_native": {}, "pf_native": {}}),
            _summary({"pf_copy": {}}),
        )
