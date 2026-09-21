import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_cf_v1_checkpoint as cf


def test_immutable_download():
    command = cf.download_command(Path("weights"))
    assert command[command.index("--revision") + 1] == "373037a987c3e06eaab3ec7b2fc2f5c9c296b649"
    assert command[3] == "chunkwise/causal_forcing.pt"
    assert "main" not in command
    assert cf.EXPECTED_BYTES == 5676282643


def test_verify_content_and_size(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    assert cf.verify_file(path, expected_bytes=8, expected_sha256=digest)["sha256"] == digest
    with pytest.raises(ValueError, match="size mismatch"):
        cf.verify_file(path, expected_bytes=9, expected_sha256=digest)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cf.verify_file(path, expected_bytes=8, expected_sha256="0" * 64)


def test_reject_lfs_pointer(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="size mismatch"):
        cf.verify_file(path)
