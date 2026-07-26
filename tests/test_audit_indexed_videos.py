import json
from pathlib import Path

from scripts.audit_indexed_videos import (
    audit_interval,
    reuse_audit_report,
    stage_verified_videos,
)


def _valid_probe(_path: Path, *, decode: bool = True) -> dict[str, object]:
    return {
        "codec": "h264",
        "frames": 477,
        "fps": 16.0,
        "width": 832,
        "height": 480,
        "duration": 29.8125,
        "fully_decoded": decode,
    }


def test_audit_accepts_complete_shard_in_shared_directory(tmp_path: Path):
    for index in range(8):
        (tmp_path / f"{index}-0_ema.mp4").write_bytes(b"video")

    first = audit_interval(
        tmp_path,
        start_idx=0,
        end_idx=4,
        allow_outside_interval=True,
        probe_video=_valid_probe,
    )
    second = audit_interval(
        tmp_path,
        start_idx=4,
        end_idx=8,
        allow_outside_interval=True,
        probe_video=_valid_probe,
    )

    assert first["ok"]
    assert first["found"] == 4
    assert second["ok"]
    assert second["found"] == 4


def test_audit_reports_missing_and_empty(tmp_path: Path):
    (tmp_path / "0-0_ema.mp4").write_bytes(b"video")
    (tmp_path / "1-0_ema.mp4").write_bytes(b"")

    result = audit_interval(
        tmp_path,
        start_idx=0,
        end_idx=3,
        probe_video=_valid_probe,
    )

    assert not result["ok"]
    assert result["missing"] == [2]
    assert result["empty"] == ["1-0_ema.mp4"]


def test_audit_rejects_non_indexed_video_names(tmp_path: Path):
    (tmp_path / "0-0_ema.mp4").write_bytes(b"video")
    (tmp_path / "prompt_named.mp4").write_bytes(b"video")

    result = audit_interval(
        tmp_path,
        start_idx=0,
        end_idx=1,
        probe_video=_valid_probe,
    )

    assert not result["ok"]
    assert result["malformed"] == ["prompt_named.mp4"]


def test_audit_rejects_extra_indices_samples_and_wrong_media(tmp_path: Path):
    (tmp_path / "0-0_ema.mp4").write_bytes(b"video")
    (tmp_path / "1-0_ema.mp4").write_bytes(b"video")
    (tmp_path / "0-1_ema.mp4").write_bytes(b"video")

    def wrong_probe(_path: Path, *, decode: bool = True) -> dict[str, object]:
        return {
            **_valid_probe(_path, decode=decode),
            "frames": 476,
            "fps": 15.0,
            "width": 480,
            "height": 832,
        }

    result = audit_interval(
        tmp_path,
        start_idx=0,
        end_idx=1,
        expected_frames=477,
        expected_fps=16,
        expected_width=832,
        expected_height=480,
        probe_video=wrong_probe,
    )

    assert not result["ok"]
    assert result["extra_indices"] == [
        {"file": "1-0_ema.mp4", "prompt_idx": 1}
    ]
    assert result["unexpected_samples"] == [
        {"file": "0-1_ema.mp4", "prompt_idx": 0, "sample_idx": 1}
    ]
    assert set(result["media_errors"]["0-0_ema.mp4"]) == {
        "frame count mismatch: expected=477 actual=476",
        "fps mismatch: expected=16 actual=15.0 tolerance=0.01",
        "width mismatch: expected=832 actual=480",
        "height mismatch: expected=480 actual=832",
    }


def test_audit_stages_zero_padded_numeric_inputs(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(12):
        (source / f"{index}-0_ema.mp4").write_bytes(f"video-{index}".encode())
    payload = audit_interval(
        source,
        start_idx=0,
        end_idx=12,
        probe_video=_valid_probe,
    )
    destination = tmp_path / "metrics" / "method"

    stage_verified_videos(payload, destination, replace=False)

    assert sorted(path.name for path in destination.glob("*.mp4")) == [
        f"{index:06d}-0_ema.mp4" for index in range(12)
    ]
    assert (
        json.loads(
            (destination / ".video_input.json").read_text(encoding="utf-8")
        )["input_fingerprint"]
        == payload["input_fingerprint"]
    )


def test_reuse_report_requires_identical_source_bytes(tmp_path: Path):
    for index in range(2):
        (tmp_path / f"{index}-0_ema.mp4").write_bytes(f"video-{index}".encode())
    payload = audit_interval(
        tmp_path,
        start_idx=0,
        end_idx=2,
        expected_frames=477,
        expected_fps=16,
        expected_width=832,
        expected_height=480,
        probe_video=_valid_probe,
    )
    report = tmp_path / "audit.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    kwargs = {
        "video_dir": tmp_path,
        "start_idx": 0,
        "end_idx": 2,
        "sample_idx": 0,
        "expected_frames": 477,
        "expected_fps": 16,
        "expected_width": 832,
        "expected_height": 480,
        "fps_tolerance": 0.01,
        "allow_outside_interval": False,
        "decode": True,
    }

    assert reuse_audit_report(report, **kwargs) == payload

    (tmp_path / "1-0_ema.mp4").write_bytes(b"changed")
    assert reuse_audit_report(report, **kwargs) is None
