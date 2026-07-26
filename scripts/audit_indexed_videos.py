#!/usr/bin/env python3
"""Strictly audit index-named inference videos for a prompt interval.

The audit is intentionally media-aware: ffprobe counts decoded frames and
ffmpeg performs a full error-on-decode pass.  A successful report can also be
materialized as a clean, zero-padded evaluator input directory so lexical
filename order is identical to numeric prompt order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Callable


INDEX_PATTERN = re.compile(r"^(\d+)-(\d+)_([^.]+)\.mp4$")
AUDIT_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(value: object) -> float:
    text = str(value or "0/0")
    try:
        rate = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        rate = 0.0
    return rate


def _probe_video(path: Path, *, decode: bool = True) -> dict[str, object]:
    """Return strict first-video-stream metadata and optionally fully decode it."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,nb_read_frames,duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"ffprobe failed ({completed.returncode}): {detail}")
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        stream = streams[0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("ffprobe returned no readable video stream") from error

    raw_frames = stream.get("nb_read_frames")
    if raw_frames in (None, "", "N/A"):
        raw_frames = stream.get("nb_frames")
    try:
        frames = int(raw_frames)
    except (TypeError, ValueError) as error:
        raise ValueError(f"ffprobe did not report a frame count: {raw_frames!r}") from error
    if frames <= 0:
        raise ValueError(f"invalid decoded frame count: {frames}")

    fps = _rate(stream.get("avg_frame_rate"))
    if fps <= 0:
        fps = _rate(stream.get("r_frame_rate"))
    if fps <= 0:
        raise ValueError("ffprobe did not report a positive frame rate")

    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ffprobe did not report a valid resolution") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid resolution: {width}x{height}")

    if decode:
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if decoded.returncode != 0:
            detail = decoded.stderr.strip() or decoded.stdout.strip()
            raise ValueError(f"ffmpeg decode failed ({decoded.returncode}): {detail}")

    duration = stream.get("duration")
    try:
        parsed_duration: float | None = float(duration)
    except (TypeError, ValueError):
        parsed_duration = None
    return {
        "codec": stream.get("codec_name"),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": parsed_duration,
        "fully_decoded": decode,
    }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_interval(
    video_dir: Path,
    *,
    start_idx: int,
    end_idx: int,
    sample_idx: int = 0,
    expected_frames: int | None = None,
    expected_fps: float | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    fps_tolerance: float = 0.01,
    allow_outside_interval: bool = False,
    decode: bool = True,
    probe_video: Callable[..., dict[str, object]] | None = None,
) -> dict[str, object]:
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError("invalid half-open prompt interval")
    if sample_idx < 0:
        raise ValueError("sample index must be non-negative")
    if fps_tolerance < 0:
        raise ValueError("fps tolerance must be non-negative")
    for label, value in (
        ("expected frames", expected_frames),
        ("expected width", expected_width),
        ("expected height", expected_height),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{label} must be positive")
    if expected_fps is not None and expected_fps <= 0:
        raise ValueError("expected fps must be positive")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {video_dir}")

    media_probe = probe_video or _probe_video
    matched: dict[int, list[tuple[Path, str]]] = {}
    malformed: list[str] = []
    empty: list[str] = []
    extra_indices: list[dict[str, object]] = []
    unexpected_samples: list[dict[str, object]] = []
    for path in sorted(video_dir.glob("*.mp4"), key=lambda item: item.name):
        match = INDEX_PATTERN.fullmatch(path.name)
        if match is None:
            malformed.append(path.name)
            continue
        prompt_idx = int(match.group(1))
        current_sample = int(match.group(2))
        suffix = match.group(3)
        if current_sample != sample_idx:
            unexpected_samples.append(
                {
                    "file": path.name,
                    "prompt_idx": prompt_idx,
                    "sample_idx": current_sample,
                }
            )
            continue
        if not (start_idx <= prompt_idx < end_idx):
            if not allow_outside_interval:
                extra_indices.append(
                    {"file": path.name, "prompt_idx": prompt_idx}
                )
            continue
        matched.setdefault(prompt_idx, []).append((path, suffix))
        if path.stat().st_size <= 0:
            empty.append(path.name)

    expected = set(range(start_idx, end_idx))
    actual = set(matched)
    missing = sorted(expected - actual)
    duplicates = {
        str(index): [path.name for path, _ in entries]
        for index, entries in matched.items()
        if len(entries) != 1
    }

    media_errors: dict[str, list[str]] = {}
    videos: list[dict[str, object]] = []
    for prompt_idx in sorted(matched):
        for path, suffix in matched[prompt_idx]:
            if path.stat().st_size <= 0:
                continue
            errors: list[str] = []
            try:
                metadata = media_probe(path, decode=decode)
            except Exception as error:  # audit must report every bad file
                metadata = None
                errors.append(str(error))
            if metadata is not None:
                if (
                    expected_frames is not None
                    and metadata.get("frames") != expected_frames
                ):
                    errors.append(
                        "frame count mismatch: "
                        f"expected={expected_frames} actual={metadata.get('frames')}"
                    )
                actual_fps = metadata.get("fps")
                if expected_fps is not None and (
                    not isinstance(actual_fps, (int, float))
                    or abs(float(actual_fps) - expected_fps) > fps_tolerance
                ):
                    errors.append(
                        "fps mismatch: "
                        f"expected={expected_fps} actual={actual_fps} "
                        f"tolerance={fps_tolerance}"
                    )
                if (
                    expected_width is not None
                    and metadata.get("width") != expected_width
                ):
                    errors.append(
                        "width mismatch: "
                        f"expected={expected_width} actual={metadata.get('width')}"
                    )
                if (
                    expected_height is not None
                    and metadata.get("height") != expected_height
                ):
                    errors.append(
                        "height mismatch: "
                        f"expected={expected_height} actual={metadata.get('height')}"
                    )
            if errors:
                media_errors[path.name] = errors
            videos.append(
                {
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "suffix": suffix,
                    "file": path.name,
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                    "metadata": metadata,
                }
            )

    constraints = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "sample_idx": sample_idx,
        "expected_frames": expected_frames,
        "expected_fps": expected_fps,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "fps_tolerance": fps_tolerance,
        "allow_outside_interval": allow_outside_interval,
        "decode": decode,
    }
    fingerprint_payload = {
        "audit_version": AUDIT_VERSION,
        "constraints": constraints,
        "videos": videos,
    }
    ok = not any(
        (
            missing,
            duplicates,
            empty,
            malformed,
            extra_indices,
            unexpected_samples,
            media_errors,
        )
    )
    return {
        "audit_version": AUDIT_VERSION,
        "video_dir": str(video_dir.resolve()),
        **constraints,
        "expected": end_idx - start_idx,
        "found": len(actual),
        "missing": missing,
        "duplicates": duplicates,
        "empty": empty,
        "malformed": malformed,
        "extra_indices": extra_indices,
        "unexpected_samples": unexpected_samples,
        "media_errors": media_errors,
        "videos": videos,
        "input_fingerprint": _canonical_sha256(fingerprint_payload),
        "ok": ok,
    }


def _safe_remove_tree(path: Path) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor).resolve()
    if resolved == anchor or resolved == Path.home().resolve():
        raise ValueError(f"refusing to remove unsafe staging path: {resolved}")
    shutil.rmtree(resolved)


def stage_verified_videos(
    payload: dict[str, object],
    destination: Path,
    *,
    replace: bool,
) -> None:
    if not payload.get("ok"):
        raise ValueError("cannot stage videos from a failed audit")
    source = Path(str(payload["video_dir"])).resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("staging directory must differ from the source directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    backup = destination.parent / (
        f".{destination.name}.backup.{os.getpid()}.{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    moved_old = False
    try:
        for item in payload["videos"]:
            assert isinstance(item, dict)
            prompt_idx = int(item["prompt_idx"])
            sample_idx = int(item["sample_idx"])
            suffix = str(item["suffix"])
            source_path = source / str(item["file"])
            staged_name = f"{prompt_idx:06d}-{sample_idx}_{suffix}.mp4"
            destination_path = temporary / staged_name
            try:
                os.link(source_path, destination_path)
            except OSError:
                shutil.copy2(source_path, destination_path)
        stage_manifest = {
            "version": 1,
            "source": str(source),
            "input_fingerprint": payload["input_fingerprint"],
            "videos": payload["videos"],
        }
        (temporary / ".video_input.json").write_text(
            json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            if not replace:
                raise FileExistsError(
                    f"{destination} already exists; pass --replace-stage"
                )
            os.replace(destination, backup)
            moved_old = True
        try:
            os.replace(temporary, destination)
        except Exception:
            if moved_old and backup.exists():
                os.replace(backup, destination)
            raise
        if moved_old:
            _safe_remove_tree(backup)
    finally:
        if temporary.exists():
            _safe_remove_tree(temporary)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def reuse_audit_report(
    report_path: Path,
    *,
    video_dir: Path,
    start_idx: int,
    end_idx: int,
    sample_idx: int,
    expected_frames: int | None,
    expected_fps: float | None,
    expected_width: int | None,
    expected_height: int | None,
    fps_tolerance: float,
    allow_outside_interval: bool,
    decode: bool,
) -> dict[str, object] | None:
    """Reuse a successful decode audit only when every source byte is unchanged."""

    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_contract = {
        "audit_version": AUDIT_VERSION,
        "video_dir": str(video_dir.resolve()),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "sample_idx": sample_idx,
        "expected_frames": expected_frames,
        "expected_fps": expected_fps,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "fps_tolerance": fps_tolerance,
        "allow_outside_interval": allow_outside_interval,
        "decode": decode,
        "ok": True,
    }
    if any(payload.get(key) != value for key, value in expected_contract.items()):
        return None
    videos = payload.get("videos")
    if not isinstance(videos, list):
        return None
    expected_count = end_idx - start_idx
    if (
        payload.get("expected") != expected_count
        or payload.get("found") != expected_count
        or len(videos) != expected_count
    ):
        return None
    prompt_indices = [item.get("prompt_idx") for item in videos if isinstance(item, dict)]
    if prompt_indices != list(range(start_idx, end_idx)):
        return None
    if any(item.get("sample_idx") != sample_idx for item in videos):
        return None
    for key in (
        "missing",
        "duplicates",
        "empty",
        "malformed",
        "extra_indices",
        "unexpected_samples",
        "media_errors",
    ):
        if payload.get(key):
            return None
    constraints = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "sample_idx": sample_idx,
        "expected_frames": expected_frames,
        "expected_fps": expected_fps,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "fps_tolerance": fps_tolerance,
        "allow_outside_interval": allow_outside_interval,
        "decode": decode,
    }
    expected_fingerprint = _canonical_sha256(
        {
            "audit_version": AUDIT_VERSION,
            "constraints": constraints,
            "videos": videos,
        }
    )
    if payload.get("input_fingerprint") != expected_fingerprint:
        return None
    expected_names = {str(item.get("file")) for item in videos}
    actual_names = {path.name for path in video_dir.glob("*.mp4")}
    if actual_names != expected_names:
        return None
    for item in videos:
        if not isinstance(item, dict):
            return None
        path = video_dir / str(item.get("file"))
        if not path.is_file() or path.stat().st_size != item.get("size"):
            return None
        if _sha256(path) != item.get("sha256"):
            return None
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--start-idx", required=True, type=int)
    parser.add_argument("--end-idx", required=True, type=int)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--fps-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--allow-outside-interval",
        action="store_true",
        help="Permit valid sample-index videos outside the audited shard interval.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip the full ffmpeg decode pass (ffprobe frame counting still runs).",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--reuse-valid-report",
        action="store_true",
        help=(
            "Reuse --output-json media metadata only after re-hashing and "
            "confirming the exact source video set is unchanged."
        ),
    )
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--replace-stage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = None
    if args.reuse_valid_report and args.output_json is not None:
        payload = reuse_audit_report(
            args.output_json,
            video_dir=args.video_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            sample_idx=args.sample_idx,
            expected_frames=args.expected_frames,
            expected_fps=args.expected_fps,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            fps_tolerance=args.fps_tolerance,
            allow_outside_interval=args.allow_outside_interval,
            decode=not args.skip_decode,
        )
    reused = payload is not None
    if payload is None:
        payload = audit_interval(
            args.video_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            sample_idx=args.sample_idx,
            expected_frames=args.expected_frames,
            expected_fps=args.expected_fps,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            fps_tolerance=args.fps_tolerance,
            allow_outside_interval=args.allow_outside_interval,
            decode=not args.skip_decode,
        )
    if args.output_json is not None:
        _write_json_atomic(args.output_json, payload)
    if payload["ok"] and args.stage_dir is not None:
        stage_verified_videos(
            payload,
            args.stage_dir,
            replace=args.replace_stage,
        )
    print(
        "[VideoAudit] "
        f"dir={args.video_dir} interval=[{args.start_idx},{args.end_idx}) "
        f"found={payload['found']}/{payload['expected']} "
        f"fingerprint={payload['input_fingerprint']} reused={reused} "
        f"ok={payload['ok']}",
        flush=True,
    )
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
