import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_v215_vbench_provenance as export


def test_snapshot_exports_sources_not_weights(tmp_path, monkeypatch):
    code, weight = tmp_path / "dynamic_degree.py", tmp_path / "raft.pth"
    code.write_bytes(b"threshold = 6\n")
    weight.write_bytes(b"fake-weight")
    patch = b"fake patch"
    fp = {"diff_binary_head_sha256": hashlib.sha256(patch).hexdigest(), "runtime_path_sha256": {
        code.name: export.sha256(code), weight.name: export.sha256(weight)}}
    monkeypatch.setattr(export, "vbench_checkout_fingerprint", lambda _: fp)
    monkeypatch.setattr(export, "_git_output", lambda *a: patch)
    result = export.snapshot(tmp_path, {"vbench_fingerprint": fp}, weight)
    assert result["modified_runtime_sources"] == {code.name: "threshold = 6\n"}
    assert result["omitted_paths"] == [weight.name]
    assert result["operator_declared_raft_checkpoint"]["sha256"] == export.sha256(weight)
    with pytest.raises(ValueError, match="evaluator drift"):
        export.snapshot(tmp_path, {"vbench_fingerprint": {}}, weight)
