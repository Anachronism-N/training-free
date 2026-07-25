"""Frame-level QK profiling for offline head-role discovery."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from .capture import FRAME_ATTENTION_CAPTURE


class HeadQKProfileRecorder:
    """Collect bounded frame-level pre-softmax QK traces on CPU.

    The attention implementation already computes frame-level statistics in
    chunks. This recorder keeps only the last query frame against strictly
    older key frames, which is the temporal sequence used by PF-style
    sign-rate and periodicity analysis.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.prompt_id = 0
        self.max_calls_per_location = 4
        self.max_records_per_layer_branch = 256
        self.allowed_update_modes = {"noisy", "clean"}
        self.allowed_branches = {"cond", "uncond"}
        self.expected_num_layers = 0
        self.expected_num_heads = 0
        self.records: list[dict] = []
        self._location_calls: dict[tuple, int] = defaultdict(int)
        self._layer_branch_records: dict[tuple, int] = defaultdict(int)

    def reset(
        self,
        *,
        max_calls_per_location: int = 4,
        max_records_per_layer_branch: int = 256,
        update_modes: Iterable[str] = ("noisy", "clean"),
        branches: Iterable[str] = ("cond", "uncond"),
        expected_num_layers: int = 0,
        expected_num_heads: int = 0,
    ) -> None:
        self.enabled = True
        self.prompt_id = 0
        self.max_calls_per_location = max(1, int(max_calls_per_location))
        self.max_records_per_layer_branch = max(
            1, int(max_records_per_layer_branch)
        )
        self.allowed_update_modes = {
            str(value).strip() for value in update_modes if str(value).strip()
        }
        self.allowed_branches = {
            str(value).strip() for value in branches if str(value).strip()
        }
        self.expected_num_layers = max(0, int(expected_num_layers))
        self.expected_num_heads = max(0, int(expected_num_heads))
        self.records = []
        self._location_calls = defaultdict(int)
        self._layer_branch_records = defaultdict(int)

    def set_prompt_id(self, prompt_id: int) -> None:
        self.prompt_id = int(prompt_id)

    def __call__(
        self,
        *,
        layer_idx: int,
        frame_attn_logits: torch.Tensor | None,
        frame_attn_prob: torch.Tensor | None,
        q_frame_indices: list[int] | None = None,
        k_frame_indices: list[int] | None = None,
        current_start: int | None = None,
        cache_update_mode: str = "default",
        cfg_branch: str = "unknown",
        layer_index_source: str = "unknown",
        **_: object,
    ) -> None:
        if not self.enabled or frame_attn_logits is None:
            return
        if frame_attn_logits.ndim != 3 or frame_attn_logits.shape[1] == 0:
            return
        if self.expected_num_layers and not (
            0 <= int(layer_idx) < self.expected_num_layers
        ):
            raise ValueError(
                f"QK profile received invalid layer {layer_idx}; "
                f"expected [0, {self.expected_num_layers})"
            )
        if (
            self.expected_num_heads
            and int(frame_attn_logits.shape[0]) != self.expected_num_heads
        ):
            raise ValueError(
                "QK profile head dimension mismatch: "
                f"expected {self.expected_num_heads}, "
                f"found {frame_attn_logits.shape[0]}"
            )

        update_mode = str(cache_update_mode)
        branch = str(cfg_branch)
        if update_mode not in self.allowed_update_modes:
            return
        if branch not in self.allowed_branches:
            return

        q_indices = list(q_frame_indices or [])
        k_indices = list(k_frame_indices or [])
        if not q_indices:
            q_indices = list(range(int(frame_attn_logits.shape[1])))
        if not k_indices:
            k_indices = list(range(int(frame_attn_logits.shape[2])))
        if len(q_indices) != int(frame_attn_logits.shape[1]):
            return
        if len(k_indices) != int(frame_attn_logits.shape[2]):
            return

        last_q = int(q_indices[-1])
        historical_columns = [
            index for index, frame_id in enumerate(k_indices)
            if int(frame_id) < last_q
        ]
        if not historical_columns:
            return

        location_key = (
            self.prompt_id,
            int(layer_idx),
            int(current_start or 0),
            update_mode,
            branch,
        )
        call_index = int(self._location_calls[location_key])
        self._location_calls[location_key] = call_index + 1
        if call_index >= self.max_calls_per_location:
            return

        layer_branch_key = (self.prompt_id, int(layer_idx), branch)
        if (
            self._layer_branch_records[layer_branch_key]
            >= self.max_records_per_layer_branch
        ):
            return
        self._layer_branch_records[layer_branch_key] += 1

        columns = torch.as_tensor(
            historical_columns,
            device=frame_attn_logits.device,
            dtype=torch.long,
        )
        logits = frame_attn_logits[:, -1].index_select(-1, columns)
        probabilities = None
        if (
            frame_attn_prob is not None
            and frame_attn_prob.shape == frame_attn_logits.shape
        ):
            probabilities = frame_attn_prob[:, -1].index_select(-1, columns)

        self.records.append(
            {
                "prompt_id": self.prompt_id,
                "layer": int(layer_idx),
                "layer_index_source": str(layer_index_source),
                "current_start": int(current_start or 0),
                "cache_update_mode": update_mode,
                "cfg_branch": branch,
                "call_index": call_index,
                "query_frame": last_q,
                "key_frames": torch.tensor(
                    [int(k_indices[index]) for index in historical_columns],
                    dtype=torch.int32,
                ),
                "logits": logits.detach().to(
                    device="cpu", dtype=torch.float32
                ),
                "probabilities": (
                    None
                    if probabilities is None
                    else probabilities.detach().to(
                        device="cpu", dtype=torch.float16
                    )
                ),
            }
        )

    def save(self, output_path: str | os.PathLike[str], metadata: dict) -> None:
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        observed_layers = sorted(
            {int(record["layer"]) for record in self.records}
        )
        observed_branches = sorted(
            {str(record["cfg_branch"]) for record in self.records}
        )
        layer_sources = sorted(
            {str(record["layer_index_source"]) for record in self.records}
        )
        records_per_layer_branch = {
            f"{layer}:{branch}": sum(
                1
                for record in self.records
                if int(record["layer"]) == layer
                and str(record["cfg_branch"]) == branch
            )
            for layer in observed_layers
            for branch in observed_branches
        }
        audit = {
            "expected_num_layers": self.expected_num_layers,
            "expected_num_heads": self.expected_num_heads,
            "observed_layers": observed_layers,
            "observed_branches": observed_branches,
            "layer_index_sources": layer_sources,
            "records_per_layer_branch": records_per_layer_branch,
        }
        payload = {
            "version": 2,
            "method": "frame_level_pre_softmax_qk_last_query",
            "metadata": dict(metadata),
            "audit": audit,
            "records": list(self.records),
        }
        torch.save(payload, path)
        print(
            f"[HeadQKProfile] records={len(self.records)} "
            f"layers={observed_layers} branches={observed_branches} "
            f"layer_sources={layer_sources} output={path}",
            flush=True,
        )


HEAD_QK_PROFILE_RECORDER = HeadQKProfileRecorder()


def enable_head_qk_profile(
    *,
    num_layers: int,
    frame_seq_length: int,
    num_heads: int,
    max_calls_per_location: int,
    max_records_per_layer_branch: int,
    update_modes: Iterable[str],
    branches: Iterable[str],
) -> None:
    HEAD_QK_PROFILE_RECORDER.reset(
        max_calls_per_location=max_calls_per_location,
        max_records_per_layer_branch=max_records_per_layer_branch,
        update_modes=update_modes,
        branches=branches,
        expected_num_layers=num_layers,
        expected_num_heads=num_heads,
    )
    FRAME_ATTENTION_CAPTURE.enable(
        layer_indices=None,
        num_layers=int(num_layers),
        frame_seq_length=int(frame_seq_length),
        num_heads=int(num_heads),
        on_frame_attention=HEAD_QK_PROFILE_RECORDER,
        capture_mode="logits_mean",
        pyramid_only=True,
    )
    print(
        "[HeadQKProfile] enabled "
        f"layers={num_layers} heads={num_heads} "
        f"calls_per_location={max_calls_per_location} "
        f"records_per_layer_branch={max_records_per_layer_branch}",
        flush=True,
    )


def set_head_qk_profile_prompt_id(prompt_id: int) -> None:
    HEAD_QK_PROFILE_RECORDER.set_prompt_id(prompt_id)


def save_head_qk_profile(
    output_path: str | os.PathLike[str],
    metadata: dict,
) -> None:
    HEAD_QK_PROFILE_RECORDER.save(output_path, metadata)
