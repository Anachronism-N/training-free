from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from vbench_long_split_cache import (  # noqa: E402
    CLEAN_MANIFEST,
    LEGACY_MANIFEST,
    prepare_clean_split,
    validate_clean_split,
)


def _contract(video_dir: Path) -> dict:
    return {
        "version": 1,
        "comparison_manifest_sha256": "comparison",
        "vbench_commit": "commit",
        "method": "method",
        "video_dir": str(video_dir.resolve()),
        "prompt_count": 2,
        "clip_seconds": 2,
        "clips_per_video": 3,
        "source_videos": [
            {
                "name": f"{index:06d}-0.mp4",
                "size": (video_dir / f"{index:06d}-0.mp4").stat().st_size,
            }
            for index in range(2)
        ],
    }


def _tree(tmp_path: Path, *, legacy: bool) -> Path:
    video_dir = tmp_path / "method"
    video_dir.mkdir()
    for index in range(2):
        (video_dir / f"{index:06d}-0.mp4").write_bytes(b"video")
        folder = video_dir / "split_clip" / f"{index:06d}-0"
        folder.mkdir(parents=True)
        for clip in range(3):
            (folder / f"{index:06d}-0_{clip:03d}.mp4").write_bytes(b"clip")
    target = (
        video_dir / "split_clip" / LEGACY_MANIFEST
        if legacy
        else video_dir / CLEAN_MANIFEST
    )
    target.write_text(json.dumps(_contract(video_dir)), encoding="utf-8")
    return video_dir


def _validate(video_dir: Path):
    return validate_clean_split(
        video_dir,
        comparison_manifest_sha256="comparison",
        vbench_commit="commit",
        prompt_count=2,
        clips_per_video=3,
    )


def test_clean_split_requires_directory_only_vbench_root(tmp_path: Path) -> None:
    video_dir = _tree(tmp_path, legacy=False)
    assert _validate(video_dir) is not None

    contaminant = video_dir / "split_clip" / "manifest.json"
    contaminant.write_text("{}", encoding="utf-8")
    assert _validate(video_dir) is None


def test_legacy_manifest_is_migrated_without_resplitting(tmp_path: Path) -> None:
    video_dir = _tree(tmp_path, legacy=True)

    def should_not_split(*args, **kwargs):
        raise AssertionError("valid legacy split should only migrate metadata")

    result = prepare_clean_split(
        method="method",
        video_dir=video_dir,
        manifest_sha="comparison",
        vbench_commit="commit",
        prompt_count=2,
        clips_per_video=3,
        split_video=should_not_split,
    )

    assert result["status"] == "migrated"
    assert (video_dir / CLEAN_MANIFEST).is_file()
    assert not (video_dir / "split_clip" / LEGACY_MANIFEST).exists()
    assert all(path.is_dir() for path in (video_dir / "split_clip").iterdir())
    assert _validate(video_dir) is not None
