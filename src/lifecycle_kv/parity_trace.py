"""Opt-in numerical traces for Self-Forcing runtime parity checks.

The tracer is deliberately inert unless ``SF_PARITY_TRACE_DIR`` is set.  It
stores full pipeline tensors (small for short parity trajectories) and compact,
deterministically sampled attention/cache tensors.  Both vendored runtimes use
this module so event identities and numerical summaries are directly aligned.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import threading
from pathlib import Path
from typing import Any, Iterable

import torch


_LOCK = threading.Lock()
_CONTEXT = threading.local()
_STATE: dict[str, Any] = {
    "initialized": False,
    "directory": None,
    "counter": 0,
    "layers": None,
    "sample_values": 4096,
}

_FULL_EVENTS = {
    "input_noise",
    "noisy_input",
    "flow_prediction",
    "denoised_prediction",
    "scheduler_noise",
    "scheduler_output",
    "clean_refresh_input",
    "committed_latent",
    "final_latents",
}


def _parse_layers(raw: str) -> frozenset[int] | None:
    raw = raw.strip().lower()
    if not raw or raw in {"all", "*"}:
        return None
    values: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("SF_PARITY_TRACE_LAYERS ranges must ascend")
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    return frozenset(values)


def _initialize() -> None:
    if _STATE["initialized"]:
        return
    _STATE["initialized"] = True
    raw_directory = os.environ.get("SF_PARITY_TRACE_DIR", "").strip()
    if not raw_directory:
        return
    directory = Path(raw_directory)
    directory.mkdir(parents=True, exist_ok=True)
    _STATE["directory"] = directory
    _STATE["layers"] = _parse_layers(
        os.environ.get("SF_PARITY_TRACE_LAYERS", "all")
    )
    _STATE["sample_values"] = max(
        32, int(os.environ.get("SF_PARITY_SAMPLE_VALUES", "4096"))
    )
    metadata = {
        "version": 1,
        "run_kind": os.environ.get("SF_PARITY_RUN_KIND", "unknown"),
        "contract_sha256": os.environ.get("SF_PARITY_CONTRACT_SHA256"),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "trace_layers": (
            "all" if _STATE["layers"] is None else sorted(_STATE["layers"])
        ),
        "sample_values": _STATE["sample_values"],
        "full_events": sorted(_FULL_EVENTS),
    }
    (directory / "trace_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "events.jsonl").write_text("", encoding="utf-8")
    print(
        "[SFParityTrace] "
        f"enabled kind={metadata['run_kind']} dir={directory} "
        f"layers={metadata['trace_layers']} samples={metadata['sample_values']}",
        flush=True,
    )


def enabled() -> bool:
    _initialize()
    return _STATE["directory"] is not None


def set_context(**values: Any) -> None:
    if not enabled():
        return
    current = dict(getattr(_CONTEXT, "values", {}))
    current.update(values)
    _CONTEXT.values = current


def clear_context() -> None:
    if hasattr(_CONTEXT, "values"):
        delattr(_CONTEXT, "values")


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "event"


def _sample_indices(numel: int, count: int, device: torch.device) -> torch.Tensor:
    if numel <= count:
        return torch.arange(numel, device=device, dtype=torch.long)
    # Integer arithmetic gives stable, duplicate-free endpoints on every device.
    return torch.div(
        torch.arange(count, device=device, dtype=torch.long) * (numel - 1),
        count - 1,
        rounding_mode="floor",
    )


def _tensor_payload(tensor: torch.Tensor, *, keep_full: bool) -> dict[str, Any]:
    detached = tensor.detach()
    value = detached.float()
    flat = value.reshape(-1)
    indices = _sample_indices(
        flat.numel(), min(flat.numel(), int(_STATE["sample_values"])), flat.device
    )
    sample = flat.index_select(0, indices).cpu()
    stats_value = value if keep_full else sample
    stats_flat = stats_value.reshape(-1)
    sample_bytes = sample.contiguous().numpy().tobytes()
    result: dict[str, Any] = {
        "shape": tuple(int(item) for item in detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "stats_scope": "full" if keep_full else "deterministic_sample",
        "mean": float(stats_value.mean().item()) if stats_value.numel() else 0.0,
        "std": (
            float(stats_value.std(unbiased=False).item())
            if stats_value.numel()
            else 0.0
        ),
        "rms": (
            float(torch.linalg.vector_norm(stats_flat).item())
            / max(1, stats_flat.numel()) ** 0.5
            if stats_flat.numel()
            else 0.0
        ),
        "min": float(stats_value.min().item()) if stats_value.numel() else 0.0,
        "max": float(stats_value.max().item()) if stats_value.numel() else 0.0,
        "sample_indices": indices.cpu(),
        "sample": sample,
        "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
    }
    if keep_full:
        result["full"] = detached.cpu().clone()
    return result


def record_event(
    event: str,
    tensors: dict[str, torch.Tensor],
    *,
    metadata: dict[str, Any] | None = None,
    keep_full: bool | None = None,
) -> None:
    if not enabled():
        return
    metadata = dict(metadata or {})
    layer = metadata.get("layer")
    layers = _STATE["layers"]
    if layer is not None and layers is not None and int(layer) not in layers:
        return
    if keep_full is None:
        keep_full = event in _FULL_EVENTS
    context = dict(getattr(_CONTEXT, "values", {}))
    with _LOCK:
        counter = int(_STATE["counter"])
        _STATE["counter"] = counter + 1
    payload = {
        "version": 1,
        "counter": counter,
        "event": str(event),
        "context": context,
        "metadata": metadata,
        "tensors": {
            name: _tensor_payload(tensor, keep_full=bool(keep_full))
            for name, tensor in sorted(tensors.items())
            if isinstance(tensor, torch.Tensor)
        },
    }
    directory: Path = _STATE["directory"]
    filename = f"{counter:06d}_{_sanitize(event)}.pt"
    path = directory / filename
    torch.save(payload, path)
    manifest_row = {
        "counter": counter,
        "event": str(event),
        "context": context,
        "metadata": metadata,
        "file": filename,
        "tensor_shapes": {
            name: list(value["shape"])
            for name, value in payload["tensors"].items()
        },
        "tensor_sample_sha256": {
            name: value["sample_sha256"]
            for name, value in payload["tensors"].items()
        },
    }
    with _LOCK, (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest_row, sort_keys=True) + "\n")


def record_dense_cache_readout(
    *,
    layer: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    frame_ids: Iterable[int],
    backend: str,
) -> None:
    ids = [int(item) for item in frame_ids]
    record_event(
        "cache_readout",
        {"query": query, "key": key, "value": value},
        metadata={
            "layer": int(layer),
            "layout": "dense_b_l_h_d",
            "backend": str(backend),
            "frame_ids_per_sequence": [ids],
            "sequence_count": int(query.shape[0] * query.shape[2]),
        },
        keep_full=False,
    )


def _compress_frame_sequences(
    frame_ids: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    frame_token_count: int,
) -> list[list[int]] | None:
    if frame_ids is None:
        return None
    cu_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    sample_indices: list[int] = []
    row_lengths: list[int] = []
    frame_token_count = max(1, int(frame_token_count))
    for index in range(max(0, cu_cpu.numel() - 1)):
        start, end = int(cu_cpu[index]), int(cu_cpu[index + 1])
        indices = list(range(start, end, frame_token_count))
        sample_indices.extend(indices)
        row_lengths.append(len(indices))
    if sample_indices:
        index_tensor = torch.tensor(
            sample_indices, device=frame_ids.device, dtype=torch.long
        )
        sampled_ids = frame_ids.detach().reshape(-1).index_select(
            0, index_tensor
        ).to(device="cpu", dtype=torch.long)
    else:
        sampled_ids = torch.empty(0, dtype=torch.long)
    rows: list[list[int]] = []
    cursor = 0
    for length in row_lengths:
        sequence = sampled_ids[cursor:cursor + length].tolist()
        cursor += length
        rows.append(list(dict.fromkeys(int(item) for item in sequence)))
    return rows


def record_varlen_cache_readout(
    *,
    layer: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    frame_ids: torch.Tensor | None,
    rope_positions: torch.Tensor | None = None,
    backend: str,
    subcall: str,
    query_chunk_starts: Iterable[int] | None = None,
    frame_token_count: int = 1,
) -> None:
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "cu_seqlens": cu_seqlens,
    }
    if frame_ids is not None:
        tensors["frame_ids"] = frame_ids
    if rope_positions is not None:
        tensors["rope_positions"] = rope_positions
    record_event(
        "cache_readout",
        tensors,
        metadata={
            "layer": int(layer),
            "layout": "head_varlen",
            "backend": str(backend),
            "subcall": str(subcall),
            "query_chunk_starts": (
                None
                if query_chunk_starts is None
                else [int(item) for item in query_chunk_starts]
            ),
            "frame_ids_per_sequence": _compress_frame_sequences(
                frame_ids, cu_seqlens, frame_token_count
            ),
            "sequence_count": max(0, int(cu_seqlens.numel()) - 1),
        },
        keep_full=False,
    )


def reset_for_tests() -> None:
    """Reset process-global state; intended only for isolated unit tests."""
    clear_context()
    _STATE.update(
        initialized=False,
        directory=None,
        counter=0,
        layers=None,
        sample_values=4096,
    )
