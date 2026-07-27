"""Fail-closed cache-layout checks that do not depend on torch."""

from __future__ import annotations


def validate_exclusive_opening_partition(
    *,
    sink_frames: int,
    recent_frames: int,
    opening_frames: int,
) -> int:
    """Return available opening recent frames or reject an all-sink block.

    Explicit role compositions move sink frames out of the dynamic store.
    Capturing the whole opening block as a time-synchronised static sink
    removes every temporally distinct recent frame and is unsafe.
    """

    sink_frames = max(0, int(sink_frames))
    recent_frames = max(0, int(recent_frames))
    opening_frames = max(0, int(opening_frames))
    available_recent = min(
        recent_frames,
        max(0, opening_frames - sink_frames),
    )
    if opening_frames > 0 and recent_frames > 0 and available_recent == 0:
        raise ValueError(
            "unsafe exclusive cache opening: "
            f"sink_frames={sink_frames} captures all "
            f"opening_frames={opening_frames}, leaving zero recent frames"
        )
    return available_recent
