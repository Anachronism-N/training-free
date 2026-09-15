"""Frame-level sidecar for observing dense KV writes without changing tensors."""

from __future__ import annotations


def observe_write(
    cache: dict, *, current_start: int, write_start: int, write_end: int,
    capacity: int, sink: int, read_start: int = 0,
) -> list[int]:
    """All offsets are latent frames; repeated noisy/clean calls overwrite in place."""
    if not (0 <= sink < capacity and 0 <= read_start <= write_start < write_end <= capacity):
        raise ValueError("invalid dense cache trace bounds")
    if current_start == 0:
        cache["_parity_frame_ids"] = []
        cache["_parity_frame_end"] = 0
    if "_parity_frame_ids" not in cache:
        raise ValueError("dense trace must start at frame zero")
    ids = list(cache["_parity_frame_ids"])
    count = write_end - write_start
    current_end = current_start + count
    if current_end > cache["_parity_frame_end"]:
        overflow = max(0, len(ids) + count - capacity)
        if overflow:
            del ids[sink:sink + overflow]
    if write_start > len(ids):
        raise ValueError("dense trace observed a write gap")
    ids[write_start:write_end] = range(current_start, current_end)
    if len(ids) != write_end or len(ids) != len(set(ids)):
        raise ValueError("dense trace frame count or uniqueness drift")
    cache["_parity_frame_ids"] = ids
    cache["_parity_frame_end"] = current_end
    return ids[read_start:write_end]
