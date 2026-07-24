from pathlib import Path

from scripts.audit_indexed_videos import audit_interval


def test_audit_accepts_complete_shard_in_shared_directory(tmp_path: Path):
    for index in range(8):
        (tmp_path / f"{index}-0_ema.mp4").write_bytes(b"video")

    first = audit_interval(tmp_path, start_idx=0, end_idx=4)
    second = audit_interval(tmp_path, start_idx=4, end_idx=8)

    assert first["ok"]
    assert first["found"] == 4
    assert second["ok"]
    assert second["found"] == 4


def test_audit_reports_missing_and_empty(tmp_path: Path):
    (tmp_path / "0-0_ema.mp4").write_bytes(b"video")
    (tmp_path / "1-0_ema.mp4").write_bytes(b"")

    result = audit_interval(tmp_path, start_idx=0, end_idx=3)

    assert not result["ok"]
    assert result["missing"] == [2]
    assert result["empty"] == ["1-0_ema.mp4"]


def test_audit_rejects_non_indexed_video_names(tmp_path: Path):
    (tmp_path / "0-0_ema.mp4").write_bytes(b"video")
    (tmp_path / "prompt_named.mp4").write_bytes(b"video")

    result = audit_interval(tmp_path, start_idx=0, end_idx=1)

    assert not result["ok"]
    assert result["malformed"] == ["prompt_named.mp4"]
