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
    history_interventions: bool = False
    projection_dim: int = 16
    allow_prompt_schedule: bool = False
    causal_policy_metrics: bool = False
    policy_budget_frames: int = 8
    region_attention_metrics: bool = False
    persistent_probe: bool = False
    persistent_capture_frames: tuple[int, ...] = (0, 18, 36)
    persistent_probe_frames: tuple[int, ...] = (39, 42, 75, 78, 81, 117)
    persistent_spatial_samples: int = 16
    descriptor_export: bool = False
    spatial_topology_metrics: bool = False

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
            history_interventions=(
                os.environ.get(
                    "HEAD_PROFILE_HISTORY_INTERVENTIONS", "0"
                )
                == "1"
            ),
            projection_dim=max(
                4,
                int(os.environ.get("HEAD_PROFILE_PROJECTION_DIM", "16")),
            ),
            allow_prompt_schedule=(
                os.environ.get("HEAD_PROFILE_ALLOW_PROMPT_SCHEDULE", "0")
                == "1"
            ),
            causal_policy_metrics=(
                os.environ.get("HEAD_PROFILE_CAUSAL_POLICY_METRICS", "0")
                == "1"
            ),
            policy_budget_frames=max(
                2,
                int(os.environ.get("HEAD_PROFILE_POLICY_BUDGET_FRAMES", "8")),
            ),
            region_attention_metrics=(
                os.environ.get("HEAD_PROFILE_REGION_METRICS", "0") == "1"
            ),
            persistent_probe=(
                os.environ.get("HEAD_PROFILE_PERSISTENT_PROBE", "0") == "1"
            ),
            persistent_capture_frames=_parse_ints(
                os.environ.get(
                    "HEAD_PROFILE_PERSISTENT_CAPTURE_FRAMES", ""
                ),
                (0, 18, 36),
            ),
            persistent_probe_frames=_parse_ints(
                os.environ.get(
                    "HEAD_PROFILE_PERSISTENT_PROBE_FRAMES", ""
                ),
                (39, 42, 75, 78, 81, 117),
            ),
            persistent_spatial_samples=max(
                1,
                int(
                    os.environ.get(
                        "HEAD_PROFILE_PERSISTENT_SPATIAL_SAMPLES", "16"
                    )
                ),
            ),
            descriptor_export=(
                os.environ.get("HEAD_PROFILE_DESCRIPTOR_EXPORT", "0") == "1"
            ),
            spatial_topology_metrics=(
                os.environ.get("HEAD_PROFILE_SPATIAL_TOPOLOGY", "0") == "1"
            ),
        )


class HeadProfileSession:
    """Process-local recorder for read-only counterfactual head profiling."""

    VERSION = 2
    HISTORY_INTERVENTION_VERSION = 4
    SCHEDULE_PROFILE_VERSION = 5
    CAUSAL_POLICY_PROFILE_VERSION = 6
    MECHANISM_PROFILE_VERSION = 7
    PROJECTION_SEED = 20260729

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
        self._projection_cache: dict[
            tuple[int, int, str, int | None], torch.Tensor
        ] = {}
        self._output_gram_cache: dict[
            tuple[int, int, int, str, int | None], torch.Tensor
        ] = {}
        self._persistent_archive: dict[int, list[dict]] = {}
        self._persistent_capture_seen: set[tuple[int, int]] = set()
        self._persistent_probe_logged: set[int] = set()
        self._causal_policy_logged: set[int] = set()
        self._spatial_topology_logged: set[tuple[str, int, int]] = set()

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
        self._persistent_archive = {}
        self._persistent_capture_seen = set()
        self._persistent_probe_logged = set()
        self._causal_policy_logged = set()
        self._spatial_topology_logged = set()
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
            "history_interventions": bool(
                self.config.history_interventions
            ),
            "projection_dim": int(self.config.projection_dim),
            "projection_seed": int(self.PROJECTION_SEED),
            "allow_prompt_schedule": bool(
                self.config.allow_prompt_schedule
            ),
            "causal_policy_metrics": bool(
                self.config.causal_policy_metrics
            ),
            "policy_budget_frames": int(
                self.config.policy_budget_frames
            ),
            "region_attention_metrics": bool(
                self.config.region_attention_metrics
            ),
            "region_attention_method": (
                "sampled_token_softmax_cartesian"
                if self.config.region_attention_metrics
                else None
            ),
            "region_attention_spatial_samples": int(
                self.config.spatial_samples
            ),
            "persistent_probe": bool(self.config.persistent_probe),
            "persistent_capture_frames": list(
                self.config.persistent_capture_frames
            ),
            "persistent_probe_frames": list(
                self.config.persistent_probe_frames
            ),
            "persistent_spatial_samples": int(
                self.config.persistent_spatial_samples
            ),
            "descriptor_export": bool(self.config.descriptor_export),
            "spatial_topology_metrics": bool(
                self.config.spatial_topology_metrics
            ),
        }
        print(
            "[HeadProfile] begin "
            f"index={dataset_index} job={job['job_id']} kind={job['kind']} "
            f"frames={num_frames} seed={self._video_metadata['seed']} "
            f"descriptors={int(self.config.descriptor_export)} "
            f"topology={int(self.config.spatial_topology_metrics)} "
            f"policy={int(self.config.causal_policy_metrics)}",
            flush=True,
        )
        return job

    def seed_for_job(self, dataset_index: int, default: int) -> int:
        job = self.jobs.get(int(dataset_index))
        if job is None:
            return int(default)
        return int(job.get("seed", default))

    def register_prompt_schedule(
        self, *, prompts: list[str], switch_frames: list[int]
    ) -> None:
        if self.active_job is None:
            raise RuntimeError("cannot register a schedule without an active job")
        declared_prompts = self.active_job.get("schedule_prompts")
        declared_switches = self.active_job.get("switch_frames")
        if (
            self.config.strict
            and declared_prompts is not None
            and [str(value).strip() for value in declared_prompts]
            != [str(value).strip() for value in prompts]
        ):
            raise ValueError("runtime prompt schedule differs from manifest")
        if (
            self.config.strict
            and declared_switches is not None
            and [int(value) for value in declared_switches]
            != [int(value) for value in switch_frames]
        ):
            raise ValueError(
                "runtime prompt-switch frames differ from manifest: "
                f"runtime={switch_frames} manifest={declared_switches}"
            )
        self._video_metadata["schedule_prompts"] = list(prompts)
        self._video_metadata["switch_frames"] = [
            int(value) for value in switch_frames
        ]
        print(
            "[HeadProfile] schedule "
            f"segments={len(prompts)} switches={switch_frames}",
            flush=True,
        )

    def alternate_prompts(self) -> list[tuple[str, str]]:
        if self.active_job is None:
            return []
        declared = self.active_job.get("shadow_prompts")
        if declared is not None:
            if not isinstance(declared, dict):
                raise ValueError("shadow_prompts must be a JSON object")
            prompts = []
            for branch, value in declared.items():
                branch = str(branch).strip()
                prompt = str(value or "").strip()
                if not branch or branch == "base":
                    raise ValueError(
                        "shadow prompt branch must be non-empty and not 'base'"
                    )
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", branch):
                    raise ValueError(
                        f"invalid shadow prompt branch name: {branch!r}"
                    )
                if not prompt:
                    raise ValueError(
                        f"shadow prompt {branch!r} is empty"
                    )
                prompts.append((branch, prompt))
            if len(prompts) != len({branch for branch, _ in prompts}):
                raise ValueError("shadow prompt branch names are not unique")
            return prompts
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
        switch_frames = tuple(
            int(value)
            for value in (self.active_job.get("switch_frames") or ())
        )
        if tuple(sorted(switch_frames)) != switch_frames:
            raise ValueError("switch_frames must be sorted")
        episode_index = sum(
            int(current_frame) >= boundary for boundary in switch_frames
        )
        segment_labels = self.active_job.get("segment_labels") or ()
        episode_label = (
            str(segment_labels[episode_index])
            if episode_index < len(segment_labels)
            else str(episode_index)
        )
        self.context = {
            "branch": str(branch),
            "mode": str(mode),
            "current_frame": int(current_frame),
            "nominal_timestep": int(nominal_timestep),
            "actual_timestep": float(actual_timestep),
            "capture": bool(capture),
            "call_index": None,
            "episode_index": int(episode_index),
            "episode_label": episode_label,
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

    def wants_history_interventions(self) -> bool:
        return bool(
            self.config.history_interventions
            and self.context
            and self.context.get("capture")
            and self.context.get("branch") == "base"
        )

    def wants_persistent_capture(self) -> bool:
        context = self.context
        return bool(
            self.config.persistent_probe
            and context
            and context.get("branch") == "base"
            and context.get("mode") == "clean"
            and int(context.get("current_frame", -1))
            in self.config.persistent_capture_frames
        )

    @staticmethod
    def _sample_complete_frames(
        tensor: torch.Tensor,
        *,
        frame_seq_length: int,
        sample_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 4:
            raise ValueError("sampled head-profile tensor must be [B,T,H,D]")
        if tensor.shape[1] % frame_seq_length != 0:
            raise ValueError("sampled head-profile tensor is not frame aligned")
        frames = tensor.shape[1] // frame_seq_length
        spatial_index = torch.linspace(
            0,
            frame_seq_length - 1,
            steps=min(sample_count, frame_seq_length),
            device=tensor.device,
        ).round().long().unique()
        sampled = (
            tensor.reshape(
                tensor.shape[0],
                frames,
                frame_seq_length,
                tensor.shape[2],
                tensor.shape[3],
            )
            .index_select(2, spatial_index)
            .reshape(
                tensor.shape[0],
                frames * spatial_index.numel(),
                tensor.shape[2],
                tensor.shape[3],
            )
        )
        return sampled, spatial_index

    def capture_persistent_tokens(
        self,
        *,
        layer: int,
        raw_current_key: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        frame_seq_length: int,
    ) -> None:
        """Keep a bounded, read-only sample of A-episode K/V for later probes."""
        if not self.wants_persistent_capture():
            return
        context = self.context or {}
        current_frame = int(context["current_frame"])
        capture_key = (int(layer), current_frame)
        if capture_key in self._persistent_capture_seen:
            raise RuntimeError(
                "duplicate persistent head-profile capture: "
                f"layer={layer} frame={current_frame}"
            )
        if (
            raw_current_key.shape != current_key.shape
            or current_key.shape != current_value.shape
        ):
            raise ValueError("persistent current K/V shapes differ")
        raw_sample, spatial_index = self._sample_complete_frames(
            raw_current_key,
            frame_seq_length=frame_seq_length,
            sample_count=self.config.persistent_spatial_samples,
        )
        post_sample, post_spatial_index = self._sample_complete_frames(
            current_key,
            frame_seq_length=frame_seq_length,
            sample_count=self.config.persistent_spatial_samples,
        )
        value_sample, value_spatial_index = self._sample_complete_frames(
            current_value,
            frame_seq_length=frame_seq_length,
            sample_count=self.config.persistent_spatial_samples,
        )
        if not (
            torch.equal(spatial_index, post_spatial_index)
            and torch.equal(spatial_index, value_spatial_index)
        ):
            raise RuntimeError("persistent K/V spatial samples differ")
        frame_count = current_key.shape[1] // frame_seq_length
        self._persistent_archive.setdefault(int(layer), []).append(
            {
                "capture_frame": current_frame,
                "frame_count": int(frame_count),
                "spatial_index": spatial_index.detach().clone(),
                "raw_key": raw_sample.detach().clone(),
                "post_key": post_sample.detach().clone(),
                "value": value_sample.detach().clone(),
            }
        )
        self._persistent_capture_seen.add(capture_key)
        if int(layer) == 0:
            print(
                "[HeadProfile] persistent-capture "
                f"frame={current_frame} frames={frame_count} "
                f"spatial={spatial_index.numel()} "
                f"tokens={post_sample.shape[1]}",
                flush=True,
            )

    @staticmethod
    def _per_head_rms(value: torch.Tensor) -> torch.Tensor:
        return value.float().square().mean(dim=(0, 1, 3)).clamp_min(0).sqrt()

    @staticmethod
    def _per_head_cosine(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        left = left.float()
        right = right.float()
        numerator = (left * right).mean(dim=(0, 1, 3))
        denominator = (
            HeadProfileSession._per_head_rms(left)
            * HeadProfileSession._per_head_rms(right)
        ).clamp_min(1e-8)
        return (numerator / denominator).clamp(-1, 1)

    @staticmethod
    def _output_projection_gram(
        output_projection_weight: torch.Tensor,
        *,
        heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        weight = output_projection_weight.detach()
        if weight.ndim != 2 or weight.shape[1] != heads * head_dim:
            raise ValueError(
                "output projection does not match attention head geometry"
            )
        weight_f = weight.float().reshape(weight.shape[0], heads, head_dim)
        return torch.einsum("ohd,ohe->hde", weight_f, weight_f)

    def _cached_output_projection_gram(
        self,
        *,
        layer: int,
        output_projection_weight: torch.Tensor,
        heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        key = (
            int(layer),
            int(heads),
            int(head_dim),
            output_projection_weight.device.type,
            output_projection_weight.device.index,
        )
        gram = self._output_gram_cache.get(key)
        if gram is None:
            gram = self._output_projection_gram(
                output_projection_weight,
                heads=heads,
                head_dim=head_dim,
            )
            self._output_gram_cache[key] = gram
        return gram

    @staticmethod
    def _projected_head_geometry(
        value: torch.Tensor,
        output_projection_weight: torch.Tensor,
        projection_gram: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError("attention output must have shape [B,T,H,D]")
        heads = value.shape[2]
        head_dim = value.shape[3]
        if projection_gram.shape != (heads, head_dim, head_dim):
            raise ValueError("cached output-projection Gram shape differs")
        value = value.float()
        energy = torch.einsum(
            "bthd,hde,bthe->h", value, projection_gram, value
        ).clamp_min(0)
        normalizer = max(
            1,
            value.shape[0]
            * value.shape[1]
            * output_projection_weight.shape[0],
        )
        return (energy / normalizer).sqrt()

    @staticmethod
    def _output_error_metrics(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        output_projection_weight: torch.Tensor,
        projection_gram: torch.Tensor,
        projection_reference: torch.Tensor | None = None,
        projection_candidate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if reference.shape != candidate.shape:
            raise ValueError("causal policy outputs have different shapes")
        delta = reference.float() - candidate.float()
        direct_reference = HeadProfileSession._per_head_rms(reference)
        direct_delta = HeadProfileSession._per_head_rms(delta)
        projected_reference_value = (
            reference
            if projection_reference is None
            else projection_reference
        )
        projected_candidate_value = (
            candidate
            if projection_candidate is None
            else projection_candidate
        )
        if projected_reference_value.shape != projected_candidate_value.shape:
            raise ValueError("projected policy samples have different shapes")
        projected_delta_value = (
            projected_reference_value.float()
            - projected_candidate_value.float()
        )
        projected_reference = HeadProfileSession._projected_head_geometry(
            projected_reference_value,
            output_projection_weight,
            projection_gram,
        )
        projected_candidate = HeadProfileSession._projected_head_geometry(
            projected_candidate_value,
            output_projection_weight,
            projection_gram,
        )
        projected_delta = HeadProfileSession._projected_head_geometry(
            projected_delta_value,
            output_projection_weight,
            projection_gram,
        )
        reference_f = projected_reference_value.float()
        candidate_f = projected_candidate_value.float()
        projected_inner = torch.einsum(
            "bthd,hde,bthe->h",
            reference_f,
            projection_gram,
            candidate_f,
        )
        weight_rows = output_projection_weight.shape[0]
        normalizer = max(
            1, reference_f.shape[0] * reference_f.shape[1] * weight_rows
        )
        projected_inner = projected_inner / normalizer
        projected_cosine = projected_inner / (
            projected_reference * projected_candidate
        ).clamp_min(1e-8)
        return {
            "direct_delta_rms": direct_delta,
            "direct_relative_error": direct_delta
            / direct_reference.clamp_min(1e-8),
            "direct_cosine": HeadProfileSession._per_head_cosine(
                reference, candidate
            ),
            "projected_delta_rms": projected_delta,
            "projected_relative_error": projected_delta
            / projected_reference.clamp_min(1e-8),
            "projected_cosine": projected_cosine.clamp(-1, 1),
        }

    @staticmethod
    def _select_frame_tokens(
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        frame_seq_length: int,
        frame_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = key.shape[1] // frame_seq_length
        if key.shape != value.shape or key.shape[1] % frame_seq_length:
            raise ValueError("policy candidate history is not frame aligned")
        frame_indices = frame_indices.to(device=key.device, dtype=torch.long)
        if frame_indices.numel() == 0:
            return key[:, :0], value[:, :0]
        if (
            int(frame_indices.min().item()) < 0
            or int(frame_indices.max().item()) >= frames
        ):
            raise ValueError("policy candidate contains an invalid frame")
        key_frames = key.reshape(
            key.shape[0],
            frames,
            frame_seq_length,
            key.shape[2],
            key.shape[3],
        )
        value_frames = value.reshape_as(key_frames)
        selected_key = key_frames.index_select(1, frame_indices).flatten(1, 2)
        selected_value = value_frames.index_select(
            1, frame_indices
        ).flatten(1, 2)
        return selected_key, selected_value

    def _policy_frame_sets(self, history_frames: int) -> dict[str, torch.Tensor]:
        device = torch.device("cpu")
        budget = min(self.config.policy_budget_frames, history_frames)
        recent_reference = min(self.config.recent_frames, budget)
        recent8 = torch.arange(
            history_frames - budget, history_frames, device=device
        )
        boundary_old = min(3, budget - 1)
        boundary_recent = budget - boundary_old
        boundary = torch.cat(
            (
                torch.arange(boundary_old, device=device),
                torch.arange(
                    history_frames - boundary_recent,
                    history_frames,
                    device=device,
                ),
            )
        ).unique(sorted=True)
        uniform_old_count = budget - recent_reference
        old_end = history_frames - recent_reference
        if uniform_old_count > 0:
            uniform_old = torch.linspace(
                0,
                max(0, old_end - 1),
                steps=uniform_old_count,
                device=device,
            ).round().long().unique(sorted=True)
        else:
            uniform_old = torch.empty(0, dtype=torch.long, device=device)
        uniform_recent = torch.arange(
            history_frames - recent_reference,
            history_frames,
            device=device,
        )
        uniform = torch.cat((uniform_old, uniform_recent)).unique(sorted=True)
        if uniform.numel() < budget:
            used = set(int(value) for value in uniform.tolist())
            fill = [
                index
                for index in range(history_frames - 1, -1, -1)
                if index not in used
            ][: budget - uniform.numel()]
            uniform = torch.tensor(
                sorted([*used, *fill]), dtype=torch.long, device=device
            )
        recent_small = torch.arange(
            history_frames - min(self.config.recent_frames, history_frames),
            history_frames,
            device=device,
        )
        return {
            "current_only": torch.empty(0, dtype=torch.long, device=device),
            "recent4": recent_small,
            "recent_budget": recent8,
            "boundary_recent": boundary,
            "uniform_recent": uniform,
        }

    def _causal_policy_probe(
        self,
        *,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        history_key: torch.Tensor,
        history_value: torch.Tensor,
        native_output: torch.Tensor,
        frame_seq_length: int,
        attention_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        output_projection_weight: torch.Tensor,
        projection_gram: torch.Tensor,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
        history_frames = history_key.shape[1] // frame_seq_length
        full_key = torch.cat((history_key, current_key), dim=1)
        full_value = torch.cat((history_value, current_value), dim=1)
        reconstructed = attention_fn(query, full_key, full_value)
        reconstruction_delta = reconstructed.float() - native_output.float()
        relative_max = (
            reconstruction_delta.abs().max()
            / native_output.float().abs().max().clamp_min(1e-8)
        )
        relative_rms = (
            reconstruction_delta.square().mean().sqrt()
            / native_output.float().square().mean().sqrt().clamp_min(1e-8)
        )
        frame_sets = self._policy_frame_sets(history_frames)
        projected_native, _ = self._sample_complete_frames(
            native_output,
            frame_seq_length=frame_seq_length,
            sample_count=self.config.spatial_samples,
        )
        metrics: dict[str, dict[str, torch.Tensor]] = {}
        frame_indices: dict[str, torch.Tensor] = {}
        for name, indices in frame_sets.items():
            selected_key, selected_value = self._select_frame_tokens(
                history_key,
                history_value,
                frame_seq_length=frame_seq_length,
                frame_indices=indices,
            )
            candidate = attention_fn(
                query,
                torch.cat((selected_key, current_key), dim=1),
                torch.cat((selected_value, current_value), dim=1),
            )
            metrics[name] = self._output_error_metrics(
                native_output,
                candidate,
                output_projection_weight,
                projection_gram,
                projection_reference=projected_native,
                projection_candidate=self._sample_complete_frames(
                    candidate,
                    frame_seq_length=frame_seq_length,
                    sample_count=self.config.spatial_samples,
                )[0],
            )
            frame_indices[name] = indices.to(torch.int16)
        metadata = {
            "history_frames": int(history_frames),
            "budget_frames": int(
                min(self.config.policy_budget_frames, history_frames)
            ),
            "eligible_budget_comparison": bool(
                history_frames > self.config.policy_budget_frames
            ),
            "native_reconstruction_relative_max": float(relative_max),
            "native_reconstruction_relative_rms": float(relative_rms),
            "frame_indices": frame_indices,
        }
        return metrics, metadata

    @staticmethod
    def _qk_probe_metrics(
        query: torch.Tensor, key: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        query_f = torch.nn.functional.normalize(query.float(), dim=-1)
        key_f = torch.nn.functional.normalize(key.float(), dim=-1)
        cosine = torch.einsum("bqhd,bkhd->bhqk", query_f, key_f)
        top_count = min(2, cosine.shape[-1])
        top = cosine.topk(top_count, dim=-1).values
        top1 = top[..., 0].mean(dim=(0, 2))
        margin = (
            (top[..., 0] - top[..., 1]).mean(dim=(0, 2))
            if top_count > 1
            else torch.zeros_like(top1)
        )
        logits = torch.einsum(
            "bqhd,bkhd->bhqk", query.float(), key.float()
        ) / math.sqrt(query.shape[-1])
        probabilities = logits.softmax(dim=-1)
        entropy = -(
            probabilities
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        entropy = entropy / max(1.0, math.log(max(2, key.shape[1])))
        return {
            "top1_cosine": top1,
            "top1_margin": margin,
            "normalized_entropy": entropy.mean(dim=(0, 2)),
            "mean_logsumexp": logits.logsumexp(dim=-1).mean(dim=(0, 2)),
        }

    def _persistent_probe_metrics(
        self,
        *,
        layer: int,
        raw_query: torch.Tensor,
        query: torch.Tensor,
        native_output: torch.Tensor,
        frame_seq_length: int,
        attention_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        output_projection_weight: torch.Tensor,
        projection_gram: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict] | None:
        context = self.context or {}
        current_frame = int(context.get("current_frame", -1))
        if (
            not self.config.persistent_probe
            or current_frame not in self.config.persistent_probe_frames
        ):
            return None
        entries = sorted(
            self._persistent_archive.get(int(layer), []),
            key=lambda row: int(row["capture_frame"]),
        )
        if not entries:
            if self.config.strict:
                raise RuntimeError(
                    f"persistent probe has no archive for layer={layer}"
                )
            return None
        expected = set(self.config.persistent_capture_frames)
        captured = {int(row["capture_frame"]) for row in entries}
        if self.config.strict and captured != expected:
            raise RuntimeError(
                "persistent probe archive is incomplete: "
                f"layer={layer} expected={sorted(expected)} "
                f"captured={sorted(captured)}"
            )
        raw_key = torch.cat(
            [row["raw_key"].to(raw_query.device) for row in entries], dim=1
        ).type_as(raw_query)
        post_key = torch.cat(
            [row["post_key"].to(query.device) for row in entries], dim=1
        ).type_as(query)
        value = torch.cat(
            [row["value"].to(query.device) for row in entries], dim=1
        ).type_as(query)
        raw_query_sample, _ = self._sample_complete_frames(
            raw_query,
            frame_seq_length=frame_seq_length,
            sample_count=self.config.persistent_spatial_samples,
        )
        query_sample, _ = self._sample_complete_frames(
            query,
            frame_seq_length=frame_seq_length,
            sample_count=self.config.persistent_spatial_samples,
        )
        persistent_output = attention_fn(query, post_key, value)
        content = self._qk_probe_metrics(raw_query_sample, raw_key)
        positioned = self._qk_probe_metrics(query_sample, post_key)
        output = self._output_error_metrics(
            native_output,
            persistent_output,
            output_projection_weight,
            projection_gram,
            projection_reference=self._sample_complete_frames(
                native_output,
                frame_seq_length=frame_seq_length,
                sample_count=self.config.spatial_samples,
            )[0],
            projection_candidate=self._sample_complete_frames(
                persistent_output,
                frame_seq_length=frame_seq_length,
                sample_count=self.config.spatial_samples,
            )[0],
        )
        metrics = {
            **{f"content_{key}": value for key, value in content.items()},
            **{
                f"positioned_{key}": value
                for key, value in positioned.items()
            },
            **{f"output_{key}": value for key, value in output.items()},
            "output_rms": self._per_head_rms(persistent_output),
            "output_native_cosine": self._per_head_cosine(
                native_output, persistent_output
            ),
        }
        metadata = {
            "archive_tokens": int(post_key.shape[1]),
            "capture_frames": sorted(captured),
            "capture_blocks": int(len(entries)),
        }
        return metrics, metadata

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

    def _region_attention_profile(
        self,
        query: torch.Tensor,
        history_key: torch.Tensor,
        current_key: torch.Tensor,
        *,
        frame_seq_length: int,
        current_frame: int,
    ) -> dict[str, torch.Tensor | bool]:
        """Estimate token-softmax frame mass over history plus current block."""
        current_frames = current_key.shape[1] // frame_seq_length
        full_key = torch.cat((history_key, current_key), dim=1)
        query_frames = query.shape[1] // frame_seq_length
        key_frames = full_key.shape[1] // frame_seq_length
        if query_frames <= 0 or key_frames <= 0:
            raise ValueError(
                "region attention profile requires complete query/key frames"
            )
        q = query[:, : query_frames * frame_seq_length].reshape(
            query.shape[0],
            query_frames,
            frame_seq_length,
            query.shape[2],
            query.shape[3],
        )
        k = full_key[:, : key_frames * frame_seq_length].reshape(
            full_key.shape[0],
            key_frames,
            frame_seq_length,
            full_key.shape[2],
            full_key.shape[3],
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
        sampled_logits = torch.einsum(
            "bqshd,bkthd->bhqkst", q_sample, k_sample
        ) / math.sqrt(query.shape[-1])
        logits = sampled_logits.mean(dim=(0, 2, 4, 5))
        token_probabilities = sampled_logits.permute(
            0, 1, 2, 4, 3, 5
        ).reshape(
            sampled_logits.shape[0],
            sampled_logits.shape[1],
            query_frames,
            spatial_index.numel(),
            key_frames * spatial_index.numel(),
        ).softmax(dim=-1)
        probabilities = token_probabilities.reshape(
            sampled_logits.shape[0],
            sampled_logits.shape[1],
            query_frames,
            spatial_index.numel(),
            key_frames,
            spatial_index.numel(),
        ).sum(dim=-1).mean(dim=(0, 2, 3))
        frame_ids = torch.arange(
            int(current_frame) - (key_frames - current_frames),
            int(current_frame) + int(current_frames),
            dtype=torch.int32,
        )
        frame_ids_device = frame_ids.to(device=probabilities.device)
        current_mask = frame_ids_device >= int(current_frame)
        history_mask = ~current_mask
        recent_start = int(current_frame) - int(self.config.recent_frames)
        recent_mask = history_mask & (frame_ids_device >= recent_start)
        last4_start = int(current_frame) + int(current_frames) - 4
        last4_mask = frame_ids_device >= last4_start
        oldest_history = int(current_frame) - (
            int(key_frames) - int(current_frames)
        )
        oldest1_mask = history_mask & (frame_ids_device == oldest_history)
        oldest3_mask = history_mask & (
            frame_ids_device < oldest_history + 3
        )
        global_sink1_mask = history_mask & (frame_ids_device == 0)
        global_sink3_mask = history_mask & (frame_ids_device < 3)
        middle_mask = history_mask & ~recent_mask & ~oldest3_mask

        def mass(mask: torch.Tensor) -> torch.Tensor:
            return probabilities[:, mask].float().sum(dim=-1)

        non_oldest_mass = mass(~oldest1_mask)
        return {
            "frame_ids": frame_ids,
            "frame_logits": logits.to(dtype=torch.float16),
            "frame_probabilities": probabilities.to(dtype=torch.float16),
            "oldest1_mass": mass(oldest1_mask).to(dtype=torch.float16),
            "oldest3_mass": mass(oldest3_mask).to(dtype=torch.float16),
            "global_sink1_mass": mass(global_sink1_mask).to(
                dtype=torch.float16
            ),
            "global_sink3_mass": mass(global_sink3_mask).to(
                dtype=torch.float16
            ),
            "middle_mass": mass(middle_mask).to(dtype=torch.float16),
            "recent4_mass": mass(recent_mask).to(dtype=torch.float16),
            "last4_mass": mass(last4_mask).to(dtype=torch.float16),
            "current_mass": mass(current_mask).to(dtype=torch.float16),
            "recent4_non_oldest_ratio": (
                mass(recent_mask) / non_oldest_mass.clamp_min(1e-8)
            ).to(dtype=torch.float16),
            "global_sink_available": bool(
                (frame_ids == 0).any().item()
            ),
        }

    def _projection(
        self,
        feature_dim: int,
        projection_dim: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        key = (
            int(feature_dim),
            int(projection_dim),
            device.type,
            device.index,
        )
        projection = self._projection_cache.get(key)
        if projection is None:
            generator = torch.Generator()
            generator.manual_seed(self.PROJECTION_SEED)
            projection = torch.randn(
                feature_dim,
                projection_dim,
                generator=generator,
                dtype=torch.float32,
            ) / math.sqrt(projection_dim)
            projection = projection.to(device=device)
            self._projection_cache[key] = projection
        return projection

    def _projected_qk_descriptors(
        self,
        query: torch.Tensor,
        history_key: torch.Tensor,
        *,
        frame_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_frames = query.shape[1] // frame_seq_length
        history_frames = history_key.shape[1] // frame_seq_length
        if query_frames <= 0 or history_frames <= 0:
            raise ValueError("Q/K descriptors require complete frames")
        sample_count = min(self.config.spatial_samples, frame_seq_length)
        spatial_index = torch.linspace(
            0,
            frame_seq_length - 1,
            steps=sample_count,
            device=query.device,
        ).round().long().unique()
        q = query.reshape(
            query.shape[0],
            query_frames,
            frame_seq_length,
            query.shape[2],
            query.shape[3],
        ).index_select(2, spatial_index)
        k = history_key.reshape(
            history_key.shape[0],
            history_frames,
            frame_seq_length,
            history_key.shape[2],
            history_key.shape[3],
        ).index_select(2, spatial_index)
        q = q.float().mean(dim=(0, 1)).permute(1, 0, 2)
        k = k.float().mean(dim=0).permute(2, 0, 1, 3)
        projection = self._projection(
            query.shape[-1],
            self.config.projection_dim,
            device=query.device,
        )
        q = torch.einsum("hsd,dp->hsp", q, projection)
        k = torch.einsum("hfsd,dp->hfsp", k, projection)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        return q.to(torch.float16), k.to(torch.float16)

    def _projected_history_value_descriptor(
        self,
        history_value: torch.Tensor,
        *,
        frame_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_frames = history_value.shape[1] // frame_seq_length
        if history_frames <= 0 or (
            history_value.shape[1] % frame_seq_length
        ):
            raise ValueError("V descriptors require complete history frames")
        sample_count = min(self.config.spatial_samples, frame_seq_length)
        spatial_index = torch.linspace(
            0,
            frame_seq_length - 1,
            steps=sample_count,
            device=history_value.device,
        ).round().long().unique()
        value = history_value.reshape(
            history_value.shape[0],
            history_frames,
            frame_seq_length,
            history_value.shape[2],
            history_value.shape[3],
        ).index_select(2, spatial_index)
        value = value.float().mean(dim=0).permute(2, 0, 1, 3)
        value_rms = value.square().mean(dim=-1).sqrt()
        projection = self._projection(
            history_value.shape[-1],
            self.config.projection_dim,
            device=history_value.device,
        )
        value = torch.einsum("hfsd,dp->hfsp", value, projection)
        value = torch.nn.functional.normalize(value, dim=-1)
        return value.to(torch.float16), value_rms.to(torch.float16)

    def _spatial_topology_profile(
        self,
        query: torch.Tensor,
        history_key: torch.Tensor,
        *,
        frame_seq_length: int,
        spatial_grid_shape: tuple[int, int],
    ) -> dict[str, torch.Tensor | int | list[int]]:
        """Measure recent cross-frame correspondence without a motion label.

        The first frame of the current AR block is compared with the latest
        completed history frame. Metrics describe attention topology only;
        they must not be called optical flow or motion fidelity.
        """
        height, width = (int(value) for value in spatial_grid_shape)
        if height <= 0 or width <= 0 or height * width != frame_seq_length:
            raise ValueError(
                "spatial topology grid does not match frame token count: "
                f"grid={height}x{width} tokens={frame_seq_length}"
            )
        query_frames = query.shape[1] // frame_seq_length
        history_frames = history_key.shape[1] // frame_seq_length
        if (
            query_frames <= 0
            or history_frames <= 0
            or query.shape[1] % frame_seq_length
            or history_key.shape[1] % frame_seq_length
        ):
            raise ValueError(
                "spatial topology requires complete query/history frames"
            )
        sample_count = min(self.config.spatial_samples, frame_seq_length)
        spatial_index = torch.linspace(
            0,
            frame_seq_length - 1,
            steps=sample_count,
            device=query.device,
        ).round().long().unique()
        query_frame = query.reshape(
            query.shape[0],
            query_frames,
            frame_seq_length,
            query.shape[2],
            query.shape[3],
        )[:, 0].index_select(1, spatial_index)
        key_frame = history_key.reshape(
            history_key.shape[0],
            history_frames,
            frame_seq_length,
            history_key.shape[2],
            history_key.shape[3],
        )[:, -1].index_select(1, spatial_index)
        logits = torch.einsum(
            "bshd,bthd->bhst",
            query_frame.float(),
            key_frame.float(),
        ) / math.sqrt(query.shape[-1])
        probabilities = torch.softmax(logits, dim=-1)
        rows = torch.div(
            spatial_index, width, rounding_mode="floor"
        ).float()
        columns = (spatial_index % width).float()
        coordinates = torch.stack(
            (
                rows / max(1, height - 1),
                columns / max(1, width - 1),
            ),
            dim=-1,
        )
        displacement = (
            coordinates[None, :, :] - coordinates[:, None, :]
        )
        expected = torch.einsum(
            "bhst,std->bhsd", probabilities, displacement
        )
        expected_norm = expected.square().sum(dim=-1).sqrt()
        mean_vector = expected.mean(dim=(0, 2))
        mean_norm = expected_norm.mean(dim=(0, 2))
        entropy = -(
            probabilities
            * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        entropy = entropy / math.log(max(2, spatial_index.numel()))
        diagonal_mass = probabilities.diagonal(dim1=-2, dim2=-1).mean(
            dim=(0, 2)
        )
        top_index = probabilities.argmax(dim=-1)
        displacement_by_batch = displacement.unsqueeze(0).expand(
            query.shape[0], -1, -1, -1
        )
        gather_index = top_index.permute(0, 2, 1).unsqueeze(-1).expand(
            -1, -1, -1, 2
        )
        top_displacement = torch.gather(
            displacement_by_batch.unsqueeze(2).expand(
                -1, -1, query.shape[2], -1, -1
            ),
            dim=3,
            index=gather_index.unsqueeze(3),
        ).squeeze(3)
        top_norm = top_displacement.square().sum(dim=-1).sqrt().mean(
            dim=(0, 1)
        )
        return {
            "normalized_entropy": entropy.mean(dim=(0, 2)).to(
                torch.float16
            ),
            "diagonal_mass": diagonal_mass.to(torch.float16),
            "expected_displacement": mean_norm.to(torch.float16),
            "directional_coherence": (
                mean_vector.square().sum(dim=-1).sqrt()
                / mean_norm.clamp_min(1e-8)
            ).clamp(0, 1).to(torch.float16),
            "top1_displacement": top_norm.to(torch.float16),
            "sample_count": int(spatial_index.numel()),
            "spatial_indices": spatial_index.to(torch.int32),
            "grid_shape": [height, width],
            "query_block_frame_offset": 0,
            "history_frame_offset": -1,
        }

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
        history_intervention_outputs: dict[str, torch.Tensor] | None = None,
        history_intervention_metadata: dict[str, float] | None = None,
        raw_query: torch.Tensor | None = None,
        raw_current_key: torch.Tensor | None = None,
        current_value: torch.Tensor | None = None,
        output_projection_weight: torch.Tensor | None = None,
        spatial_grid_shape: tuple[int, int] | None = None,
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
            "episode_index": int(context.get("episode_index", 0)),
            "episode_label": str(context.get("episode_label", "0")),
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
        if self.config.region_attention_metrics:
            record["region_attention_metrics"] = (
                self._region_attention_profile(
                    query,
                    history_key,
                    current_key,
                    frame_seq_length=frame_seq_length,
                    current_frame=int(context["current_frame"]),
                )
            )
        if self.config.spatial_topology_metrics:
            if spatial_grid_shape is None:
                raise RuntimeError(
                    "spatial topology profiling requires the latent grid shape"
                )
            topology = self._spatial_topology_profile(
                query,
                history_key,
                frame_seq_length=frame_seq_length,
                spatial_grid_shape=spatial_grid_shape,
            )
            record["spatial_topology_metrics"] = topology
            topology_log_key = (
                str(context["mode"]),
                int(context["current_frame"]),
                int(context["nominal_timestep"]),
            )
            if (
                int(layer) == 0
                and topology_log_key not in self._spatial_topology_logged
            ):
                self._spatial_topology_logged.add(topology_log_key)
                print(
                    "[HeadProfile] spatial-topology "
                    f"mode={topology_log_key[0]} "
                    f"frame={topology_log_key[1]} "
                    f"timestep={topology_log_key[2]} "
                    f"samples={topology['sample_count']} "
                    "entropy_mean="
                    f"{float(topology['normalized_entropy'].float().mean()):.4f} "
                    "diagonal_mean="
                    f"{float(topology['diagonal_mass'].float().mean()):.4f} "
                    "displacement_mean="
                    f"{float(topology['expected_displacement'].float().mean()):.4f}",
                    flush=True,
                )
        projection_gram = None
        if self.config.causal_policy_metrics or self.config.persistent_probe:
            if output_projection_weight is None:
                raise RuntimeError(
                    "output-causal profiling requires output projection weight"
                )
            projection_gram = self._cached_output_projection_gram(
                layer=int(layer),
                output_projection_weight=output_projection_weight,
                heads=native_output.shape[2],
                head_dim=native_output.shape[3],
            )
        if self.config.causal_policy_metrics:
            if current_value is None or output_projection_weight is None:
                raise RuntimeError(
                    "causal policy profiling requires current V and output "
                    "projection weight"
                )
            policy_metrics, policy_metadata = self._causal_policy_probe(
                query=query,
                current_key=current_key,
                current_value=current_value,
                history_key=history_key,
                history_value=history_value,
                native_output=native_output,
                frame_seq_length=frame_seq_length,
                attention_fn=attention_fn,
                output_projection_weight=output_projection_weight,
                projection_gram=projection_gram,
            )
            record["causal_policy_metrics"] = policy_metrics
            record["causal_policy_metadata"] = policy_metadata
            current_frame = int(context["current_frame"])
            if int(layer) == 0 and current_frame not in self._causal_policy_logged:
                self._causal_policy_logged.add(current_frame)
                print(
                    "[HeadProfile] causal-policy "
                    f"frame={current_frame} branch={context['branch']} "
                    f"history={history_frames} "
                    f"budget={policy_metadata['budget_frames']} "
                    "candidates="
                    f"{sorted(policy_metrics)} "
                    "parity_rms="
                    f"{policy_metadata['native_reconstruction_relative_rms']:.3g}",
                    flush=True,
                )
        if self.config.persistent_probe:
            if (
                raw_query is None
                or raw_current_key is None
                or current_value is None
                or output_projection_weight is None
            ):
                raise RuntimeError(
                    "persistent profiling requires raw Q/K, current V, and "
                    "output projection weight"
                )
            persistent = self._persistent_probe_metrics(
                layer=int(layer),
                raw_query=raw_query,
                query=query,
                native_output=native_output,
                frame_seq_length=frame_seq_length,
                attention_fn=attention_fn,
                output_projection_weight=output_projection_weight,
                projection_gram=projection_gram,
            )
            if persistent is not None:
                persistent_metrics, persistent_metadata = persistent
                record["persistent_probe_metrics"] = persistent_metrics
                record["persistent_probe_metadata"] = persistent_metadata
                current_frame = int(context["current_frame"])
                if (
                    int(layer) == 0
                    and current_frame not in self._persistent_probe_logged
                ):
                    self._persistent_probe_logged.add(current_frame)
                    print(
                        "[HeadProfile] persistent-probe "
                        f"frame={current_frame} branch={context['branch']} "
                        f"archive_tokens="
                        f"{persistent_metadata['archive_tokens']} "
                        f"captures="
                        f"{persistent_metadata['capture_frames']}",
                        flush=True,
                    )
        if self.config.history_interventions or self.config.descriptor_export:
            q_descriptor, k_descriptor = self._projected_qk_descriptors(
                query,
                history_key,
                frame_seq_length=frame_seq_length,
            )
            v_descriptor, v_rms = self._projected_history_value_descriptor(
                history_value,
                frame_seq_length=frame_seq_length,
            )
            record.update(
                {
                    "query_projection": q_descriptor,
                    "history_key_projection": k_descriptor,
                    "history_value_projection": v_descriptor,
                    "history_value_rms": v_rms,
                }
            )
        if self.config.history_interventions:
            if context["branch"] != "base":
                raise RuntimeError(
                    "history interventions are defined for base profiles only"
                )
            expected = {
                "reverse",
                "phase_shift",
                "freeze_latest",
                "value_mismatch",
            }
            provided = set(history_intervention_outputs or {})
            if provided != expected:
                raise RuntimeError(
                    "history intervention outputs mismatch: "
                    f"expected={sorted(expected)} provided={sorted(provided)}"
                )
            record.update(
                {
                    "full_history_signature": self._signature(full_output),
                    "recent_history_signature": self._signature(
                        recent_output
                    ),
                }
            )
            for name, output in history_intervention_outputs.items():
                record[f"history_{name}_signature"] = self._signature(output)
            for name, value in (
                history_intervention_metadata or {}
            ).items():
                record[f"history_intervention_{name}"] = float(value)
        self.records.append(record)
        self._recorded_layers[call_index].add(int(layer))

    @classmethod
    def _detach_to_cpu(cls, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {
                key: cls._detach_to_cpu(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._detach_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._detach_to_cpu(item) for item in value)
        return value

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
        persistent_missing: list[tuple[int, int]] = []
        if self.config.persistent_probe:
            persistent_missing = [
                (layer, frame)
                for layer in range(expected_layers)
                for frame in self.config.persistent_capture_frames
                if (layer, frame) not in self._persistent_capture_seen
            ]
            if persistent_missing and self.config.strict:
                raise RuntimeError(
                    "incomplete persistent head-profile archive: "
                    f"{persistent_missing[:8]} "
                    f"({len(persistent_missing)} missing)"
                )
        job = dict(self.active_job)
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_index = int(job["dataset_index"])
        output = output_dir / (
            f"{dataset_index:05d}_{_safe_stem(str(job['job_id']))}.pt"
        )
        self.records = [
            self._detach_to_cpu(record) for record in self.records
        ]
        if (
            self.config.descriptor_export
            or self.config.spatial_topology_metrics
        ):
            profile_version = self.MECHANISM_PROFILE_VERSION
        elif (
            self.config.causal_policy_metrics
            or self.config.persistent_probe
        ):
            profile_version = self.CAUSAL_POLICY_PROFILE_VERSION
        elif self.config.history_interventions:
            profile_version = self.HISTORY_INTERVENTION_VERSION
        elif (
            job.get("shadow_prompts") is not None
            or "||" in str(job.get("base_prompt", ""))
        ):
            profile_version = self.SCHEDULE_PROFILE_VERSION
        else:
            profile_version = self.VERSION
        payload = {
            "version": profile_version,
            "job": job,
            "metadata": {
                **self._video_metadata,
                "expected_layers": int(expected_layers),
                "record_count": len(self.records),
                "captured_calls": len(self.call_summaries),
                "incomplete_calls": incomplete,
                "persistent_capture_count": len(
                    self._persistent_capture_seen
                ),
                "persistent_capture_missing": persistent_missing,
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
        self._persistent_archive = {}
        self._persistent_capture_seen = set()
        self._persistent_probe_logged = set()
        self._causal_policy_logged = set()
        self._spatial_topology_logged = set()
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
