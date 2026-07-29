from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_v132_blind_review as blind
import run_v132_existing_paired_analysis as paired


def test_v132_existing_evidence_keeps_main_and_motion_decision_pair():
    assert "ours_prototype_retrieval_age24" in paired.CANDIDATES
    assert "ours_confidence_motion" in paired.CANDIDATES
    assert blind.DEFAULT_METHODS == (
        "sf_native",
        "deep_forcing",
        "ours_prototype_retrieval_age24",
        "ours_confidence_motion",
    )
