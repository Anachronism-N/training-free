from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_v213_evidence import checked_hash, summarize


def test_uploaded_summary_receipts_and_quality_decomposition():
    result = summarize(ROOT / "artifacts/experiment_results/v213_generation_a39f503a_eval_45e79ec1")
    assert len(result["compact_hash_checks"]) == 239
    assert result["all_official_ci_include_zero"]
    assert not result["paper_ready_positive_superiority_claim"]
    assert result["candidates"]["fifo_full_a002"]["official_quality"]["mean_delta"] == pytest.approx(.1331382672)


def test_corrupted_hash_is_not_normalized_away(tmp_path):
    path = tmp_path / "data.json"
    path.write_bytes(b'{"changed":true}\r\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_hash(path, "0" * 64)
