from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


def _parse_ints(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value.strip():
        return default
    return tuple(
        sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    )


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem[:80] or "job"


@dataclass(frozen=True)
class HeadProfileConfig:
    enabled: bool
    output_dir: Path
    manifest_path: Path | None
    noisy_frames: tuple[int, ...]
    noisy_timesteps: tuple[int, ...]
    clean_frames: tuple[int, ...]
    recent_frames: int
    spatial_samples: int
    strict: bool

    @classmethod
    def from_env(cls) -> "HeadProfileConfig":
        enabled = os.environ.get("HEAD_PROFILE_ENABLE", "0") == "1"
        output = Path(
            os.environ.get("HEAD_PROFILE_OUTPUT_DIR", "runs/v134_head_profile/raw")
        )
        manifest_text = os.environ.get("HEAD_PROFILE_JOB_MANIFEST", "").strip()
        return cls(
            enabled=enabled,
            output_dir=output,
            manifest_path=Path(manifest_text) if manifest_text else None,
            noisy_frames=_parse_ints(
                os.environ.get("HEAD_PROFILE_AR_FRAMES", ""),
                (3, 21, 42, 63, 84, 117),
            ),
            noisy_timesteps=_parse_ints(
                os.environ.get("HEAD_PROFILE_TIMESTEPS", ""),
                (250, 500, 750, 1000),
            ),
            clean_frames=_parse_ints(
                os.environ.get("HEAD_PROFILE_CLEAN_AR_FRAMES", ""),
                (21, 63, 117),
            ),
            recent_frames=max(
                1, int(os.environ.get("HEAD_PROFILE_RECENT_FRAMES", "4"))
            ),
            spatial_samples=max(
                1, int(os.environ.get("HEAD_PROFILE_SPATIAL_SAMPLES", "16"))
            ),
            strict=os.environ.get("HEAD_PROFILE_STRICT", "1") == "1",
        )


class HeadProfileSession:
    """Process-local recorder for read-only counterfactual head profiling."""

    VERSION = 2

    def __init__(self, config: HeadProfileConfig) -> None:
        self.config = config
        self.jobs = self._load_jobs(config.manifest_path)
        self.active_job: dict | None = None
        self.records: list[dict] = []
        self.context: dict | None = None
        self.call_summaries: list[dict] = []
        self._call_index = 0
        self._recorded_layers: dict[int, set[int]] = {}
        self._video_metadata: dict = {}

    @staticmethod
    def _load_jobs(path: Path | None) -> dict[int, dict]:
        if path is None:
            return {}
        jobs: dict[int, dict] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                index = int(row.get("dataset_index", len(jobs)))
                if index in jobs:
                    raise ValueError(
                        f"duplicate dataset_index={index} in {path}:{line_number}"
                    )
                jobs[index] = row
        if not jobs:
            raise ValueError(f"head-profile manifest is empty: {path}")
        return jobs

    def begin_video(
        self,
        *,
        dataset_index: int,
        text_prompts: list[str],
        num_frames: int,
        frame_seq_length: int,
        num_frame_per_block: int,
        local_attn_size: int,
    ) -> dict:
        if self.active_job is not None:
            raise RuntimeError("previous head-profile video was not finalized")
        job = dict(self.jobs.get(dataset_index) or {})
        if self.jobs and not job:
            raise KeyError(
                f"dataset_index={dataset_index} is missing from head-profile manifest"
            )
        base_prompt = str(job.get("base_prompt") or text_prompts[0])
        if self.config.strict and base_prompt.strip() != text_prompts[0].strip():
            raise ValueError(
                "head-profile manifest prompt does not match inference data at "
                f"dataset_index={dataset_index}"
            )
        job.setdefault("dataset_index", dataset_index)
        job.setdefault("job_id", f"prompt_{dataset_index:05d}")
        job.setdefault("kind", "observational")
        job["base_prompt"] = base_prompt
        self.active_job = job
        self.records = []
        self.context = None
        self.call_summaries = []
        self._call_index = 0
        self._recorded_layers = {}
        self._video_metadata = {
            "run_commit": os.environ.get("HEAD_PROFILE_RUN_COMMIT"),
            "seed": int(
                job.get(
                    "seed", os.environ.get("HEAD_PROFILE_SEED", "0")
                )
            ),
            "num_frames": int(num_frames),
            "frame_seq_length": int(frame_seq_length),
            "num_frame_per_block": int(num_frame_per_block),
            "local_attn_size": int(local_attn_size),
            "noisy_frames": list(self.config.noisy_frames),
            "noisy_timesteps": list(self.config.noisy_timesteps),
            "clean_frames": list(self.config.clean_frames),
            "recent_frames": int(self.config.recent_frames),
            "spatial_samples": int(self.config.spatial_samples),
        }
        print(
            "[HeadProfile] begin "
            f"index={dataset_index} job={job['job_id']} kind={job['kind']} "
            f"frames={num_frames}",
            flush=True,
        )
        return job

    def seed_for_job(self, dataset_index: int, default: int) -> int:
        job = self.jobs.get(int(dataset_index))
        if job is None:
            return int(default)
        return int(job.get("seed", default))

    def alternate_prompts(self) -> list[tuple[str, str]]:
        if self.active_job is None:
            return []
        prompts = []
        for branch, key in (
            ("semantic", "semantic_prompt"),
            ("null", "null_prompt"),
        ):
            prompt = str(self.active_job.get(key) or "").strip()
            if prompt:
                prompts.append((branch, prompt))
        return prompts

    def should_capture(
        self, *, mode: str, current_frame: int, nominal_timestep: int
    ) -> bool:
        if self.active_job is None:
            return False
        if mode == "noisy":
            return (
                int(current_frame) in self.config.noisy_frames
                and int(nominal_timestep) in self.config.noisy_timesteps
            )
        if mode == "clean":
            return int(current_frame) in self.config.clean_frames
        return False

    def set_call_context(
        self,
        *,
        branch: str,
        mode: str,
        current_frame: int,
        nominal_timestep: int,
        actual_timestep: float,
    ) -> bool:
        capture = self.should_capture(
            mode=mode,
            current_frame=current_frame,
            nominal_timestep=nominal_timestep,
        )
        self.context = {
            "branch": str(branch),
            "mode": str(mode),
            "current_frame": int(current_frame),
            "nominal_timestep": int(nominal_timestep),
            "actual_timestep": float(actual_timestep),
            "capture": bool(capture),
            "call_index": None,
        }
        if capture:
            call_index = self._call_index
            self._call_index += 1
            self.context["call_index"] = call_index
            summary = {
                key: value
                for key, value in self.context.items()
                if key != "capture"
            }
            self.call_summaries.append(summary)
            self._recorded_layers[call_index] = set()
        return capture

    def disable_call_context(self) -> None:
        self.context = None

    @staticmethod
    def _signature(tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach().float()
        feature_dim = value.shape[-1]
        groups = min(32, feature_dim)
        if feature_dim % groups != 0:
            groups = 1
        grouped = value.reshape(
            value.shape[0],
            value.shape[1],
            value.shape[2],
            groups,
            feature_dim // groups,
        )
        mean = grouped.mean(dim=(0, 1, 4))
        rms = grouped.square().mean(dim=(0, 1, 4)).clamp_min(0).sqrt()
        return torch.cat((mean, rms), dim=-1).to(dtype=torch.float16)

    def _temporal_profile(
        self,
        query: torch.Tensor,
        history_key: torch.Tensor,
        *,
        frame_seq_length: int,
        current_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_frames = query.shape[1] // frame_seq_length
        history_frames = history_key.shape[1] // frame_seq_length
        if query_frames <= 0 or history_frames <= 0:
            raise ValueError("temporal profile requires non-empty complete frames")
        q = query[:, : query_frames * frame_seq_length].reshape(
            query.shape[0],
            query_frames,
            frame_seq_length,
            query.shape[2],
            query.shape[3],
        )
        k = history_key[:, : history_frames * frame_seq_length].reshape(
            history_key.shape[0],
            history_frames,
            frame_seq_length,
            history_key.shape[2],
            history_key.shape[3],
        )
        sample_count = min(self.config.spatial_samples, frame_seq_length)
        spatial_index = torch.linspace(
            0,
            frame_seq_length - 1,
            steps=sample_count,
            device=query.device,
        ).round().long().unique()
        q_sample = q.index_select(2, spatial_index).float()
        k_sample = k.index_select(2, spatial_index).float()
        logits = torch.einsum(
            "bqshd,bkshd->bhqks", q_sample, k_sample
        )
        logits = logits.mean(dim=(0, 2, 4)) / math.sqrt(query.shape[-1])
        probs = logits.softmax(dim=-1)
        frame_ids = torch.arange(
            int(current_frame) - history_frames,
            int(current_frame),
            dtype=torch.int32,
        )
        return (
            logits.to(dtype=torch.float16),
            probs.to(dtype=torch.float16),
            frame_ids,
        )

    def record_attention(
        self,
        *,
        layer: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        history_key: torch.Tensor,
        history_value: torch.Tensor,
        native_output: torch.Tensor,
        frame_seq_length: int,
        attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        context = self.context
        if not context or not context["capture"]:
            return
        call_index = int(context["call_index"])
        if layer in self._recorded_layers[call_index]:
            raise RuntimeError(
                f"duplicate layer={layer} for head-profile call={call_index}"
            )
        if history_key.shape != history_value.shape:
            raise ValueError("history K/V shapes differ")
        if history_key.shape[1] < frame_seq_length:
            raise ValueError(
                "selected head-profile call has less than one history frame"
            )
        if history_key.shape[1] % frame_seq_length != 0:
            raise ValueError("history cache is not frame aligned")

        history_frames = history_key.shape[1] // frame_seq_length
        recent_tokens = min(
            history_key.shape[1],
            self.config.recent_frames * frame_seq_length,
        )
        recent_key = history_key[:, -recent_tokens:]
        recent_value = history_value[:, -recent_tokens:]
        full_output = attention_fn(query, history_key, history_value)
        recent_output = attention_fn(query, recent_key, recent_value)
        residual = full_output - recent_output
        temporal_logits, temporal_probs, history_frame_ids = (
            self._temporal_profile(
                query,
                history_key,
                frame_seq_length=frame_seq_length,
                current_frame=int(context["current_frame"]),
            )
        )
        record = {
            "branch": context["branch"],
            "mode": context["mode"],
            "current_frame": int(context["current_frame"]),
            "nominal_timestep": int(context["nominal_timestep"]),
            "actual_timestep": float(context["actual_timestep"]),
            "call_index": call_index,
            "layer": int(layer),
            "history_frames": int(history_frames),
            "recent_frames": int(recent_tokens // frame_seq_length),
            "residual_signature": self._signature(residual),
            "native_signature": self._signature(native_output),
            "query_signature": self._signature(query),
            "current_key_signature": self._signature(current_key),
            "temporal_logits": temporal_logits,
            "temporal_probs": temporal_probs,
            "history_frame_ids": history_frame_ids,
        }
        self.records.append(record)
        self._recorded_layers[call_index].add(int(layer))

    def end_video(self, *, expected_layers: int) -> Path:
        if self.active_job is None:
            raise RuntimeError("no active head-profile video")
        incomplete = {
            call: sorted(set(range(expected_layers)) - layers)
            for call, layers in self._recorded_layers.items()
            if len(layers) != expected_layers
        }
        if incomplete and self.config.strict:
            sample = {
                call: missing[:5]
                for call, missing in list(incomplete.items())[:3]
            }
            raise RuntimeError(
                "incomplete head-profile layer coverage: "
                f"{sample} ({len(incomplete)} calls)"
            )
        job = dict(self.active_job)
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_index = int(job["dataset_index"])
        output = output_dir / (
            f"{dataset_index:05d}_{_safe_stem(str(job['job_id']))}.pt"
        )
        for record in self.records:
            for key, value in list(record.items()):
                if isinstance(value, torch.Tensor):
                    record[key] = value.detach().cpu()
        payload = {
            "version": self.VERSION,
            "job": job,
            "metadata": {
                **self._video_metadata,
                "expected_layers": int(expected_layers),
                "record_count": len(self.records),
                "captured_calls": len(self.call_summaries),
                "incomplete_calls": incomplete,
                "manifest_path": (
                    str(self.config.manifest_path)
                    if self.config.manifest_path is not None
                    else None
                ),
            },
            "calls": self.call_summaries,
            "records": self.records,
        }
        temporary = output.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
        print(
            "[HeadProfile] end "
            f"job={job['job_id']} calls={len(self.call_summaries)} "
            f"records={len(self.records)} output={output}",
            flush=True,
        )
        self.active_job = None
        self.records = []
        self.context = None
        self.call_summaries = []
        self._recorded_layers = {}
        return output


_UNSET = object()
_SESSION: HeadProfileSession | None | object = _UNSET


def get_head_profile_session() -> HeadProfileSession | None:
    global _SESSION
    if _SESSION is _UNSET:
        config = HeadProfileConfig.from_env()
        _SESSION = HeadProfileSession(config) if config.enabled else None
    return _SESSION if isinstance(_SESSION, HeadProfileSession) else None


def reset_head_profile_session_for_tests() -> None:
    global _SESSION
    _SESSION = _UNSET
