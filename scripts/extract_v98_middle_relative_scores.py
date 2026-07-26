#!/usr/bin/env python3
"""Extract shift-invariant, intervention-aligned v98 head routing scores.

The legacy v98 draft classified heads by the absolute sign of historical
pre-softmax logits.  That sign is not invariant to adding a per-query constant
and mixed sink/recent frames with the middle history that the experiment
actually changes.

This extractor instead compares the mean logit of *middle* keys with the mean
logit of the latest recent-history keys.  Both groups come from the same query,
so their difference is invariant to a common logit shift.  Per-record margins
are standardized by the pooled logit RMS, aggregated within each independently
generated profile, and finally aggregated across profiles with a median so that
profiles with more captured calls cannot dominate the frozen map.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METHOD = "v98_middle_relative_qk_head_scores"
PRIMARY_FIELD = "middle_relative_logit_margin"
FROZEN_NUM_LAYERS = 30
FROZEN_NUM_HEADS = 12
FROZEN_SINK_FRAMES = 3
FROZEN_RECENT_FRAMES = 4
FROZEN_BRANCH = "cond"
FROZEN_UPDATE_MODE = "noisy"
FROZEN_BOOTSTRAP_ROUNDS = 500
FROZEN_BOOTSTRAP_SEED = 20260726
FROZEN_MIN_PROFILES_PER_POLICY_HEAD = 32
FROZEN_MIN_STABLE_HEAD_FRACTION = 0.80
FROZEN_MIN_HEAD_BOOTSTRAP_AGREEMENT = 0.75
FROZEN_MIN_TOPOLOGY_SIGN_AGREEMENT_FRACTION = 0.80
FROZEN_MIN_MINORITY_FRACTION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--sink-frames", type=int, default=3)
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--branch", default="cond")
    parser.add_argument("--update-mode", default="noisy")
    parser.add_argument("--bootstrap-rounds", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--min-profiles-per-policy-head", type=int, default=32)
    parser.add_argument("--min-stable-head-fraction", type=float, default=0.80)
    parser.add_argument("--min-head-bootstrap-agreement", type=float, default=0.75)
    parser.add_argument(
        "--min-topology-sign-agreement-fraction", type=float, default=0.80
    )
    parser.add_argument("--min-minority-fraction", type=float, default=0.05)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot take the median of an empty sequence")
    return float(statistics.median(materialized))


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take the mean of an empty sequence")
    return float(sum(values) / len(values))


def _as_int_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _as_float_rows(values: Any) -> list[list[float]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [[float(value) for value in row] for row in values]


def record_middle_relative_margins(
    record: dict[str, Any],
    *,
    num_heads: int,
    sink_frames: int,
    recent_frames: int,
) -> list[float] | None:
    """Return one standardized middle-vs-recent margin per head.

    Recent history is defined as the latest ``recent_frames`` distinct key
    frame ids that are strictly older than the recorded query.  Middle history
    excludes both that recent set and the immutable absolute sink frames.
    """

    query_frame = int(record["query_frame"])
    key_frames = _as_int_list(record["key_frames"])
    logits = _as_float_rows(record["logits"])
    if len(logits) != num_heads:
        raise ValueError(
            f"expected {num_heads} logit rows, found {len(logits)}"
        )
    if any(len(row) != len(key_frames) for row in logits):
        raise ValueError("logit rows are not aligned with key_frames")

    historical_ids = sorted(
        {frame for frame in key_frames if frame < query_frame}
    )
    if len(historical_ids) <= recent_frames:
        return None
    recent_ids = set(historical_ids[-recent_frames:])
    recent_indices = [
        index
        for index, frame in enumerate(key_frames)
        if frame < query_frame and frame in recent_ids
    ]
    middle_indices = [
        index
        for index, frame in enumerate(key_frames)
        if sink_frames <= frame < query_frame and frame not in recent_ids
    ]
    if not recent_indices or not middle_indices:
        return None

    margins: list[float] = []
    for row in logits:
        middle_values = [row[index] for index in middle_indices]
        recent_values = [row[index] for index in recent_indices]
        pooled = middle_values + recent_values
        pooled_center = mean(pooled)
        pooled_rms = math.sqrt(
            mean([(value - pooled_center) ** 2 for value in pooled])
        )
        delta = mean(middle_values) - mean(recent_values)
        margin = delta / max(pooled_rms, 1e-6)
        if not math.isfinite(margin):
            raise ValueError("non-finite middle-relative margin")
        margins.append(float(margin))
    return margins


def bootstrap_sign_agreement(
    values: list[float],
    *,
    rounds: int,
    seed: int,
) -> float:
    if not values:
        raise ValueError("bootstrap requires observations")
    reference = median(values)
    if reference == 0.0:
        return 0.0
    reference_positive = reference > 0.0
    rng = random.Random(seed)
    agreements = 0
    for _ in range(max(1, int(rounds))):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimate = median(sample)
        agreements += estimate != 0.0 and (estimate > 0.0) == reference_positive
    return agreements / max(1, int(rounds))


def balanced_bootstrap_sign_agreement(
    values_by_policy: dict[str, list[float]],
    *,
    rounds: int,
    seed: int,
) -> float:
    """Bootstrap the equally weighted policy-median estimator.

    Sampling is performed independently within every probe policy.  This keeps
    one cache topology from gaining extra weight merely because it emitted more
    usable recorder calls or profiles.
    """

    if not values_by_policy or any(not values for values in values_by_policy.values()):
        raise ValueError("balanced bootstrap requires observations for every policy")
    reference = median(
        median(values) for _, values in sorted(values_by_policy.items())
    )
    if reference == 0.0:
        return 0.0
    reference_positive = reference > 0.0
    rng = random.Random(seed)
    agreements = 0
    for _ in range(max(1, int(rounds))):
        policy_medians = []
        for _, values in sorted(values_by_policy.items()):
            sample = [values[rng.randrange(len(values))] for _ in values]
            policy_medians.append(median(sample))
        estimate = median(policy_medians)
        agreements += estimate != 0.0 and (estimate > 0.0) == reference_positive
    return agreements / max(1, int(rounds))


def paired_cluster_bootstrap_sign_agreement(
    values_by_policy: dict[str, dict[tuple[str, str, int], float]],
    *,
    rounds: int,
    seed: int,
) -> float:
    """Bootstrap matched probe observations at the prompt-pair level."""

    if not values_by_policy:
        raise ValueError("paired bootstrap requires probe policies")
    sample_sets = [set(values) for values in values_by_policy.values()]
    if any(not samples for samples in sample_sets) or any(
        samples != sample_sets[0] for samples in sample_sets[1:]
    ):
        raise ValueError("paired bootstrap requires identical policy samples")
    pair_ids = sorted({pair_id for pair_id, _, _ in sample_sets[0]})
    if not pair_ids:
        raise ValueError("paired bootstrap requires counterfactual pairs")
    grouped = {
        policy: {
            pair_id: [
                value
                for sample, value in sorted(values.items())
                if sample[0] == pair_id
            ]
            for pair_id in pair_ids
        }
        for policy, values in sorted(values_by_policy.items())
    }

    def estimate(selected_pairs: list[str]) -> float:
        policy_medians = []
        for _, pair_values in sorted(grouped.items()):
            selected = [
                value
                for pair_id in selected_pairs
                for value in pair_values[pair_id]
            ]
            policy_medians.append(median(selected))
        return median(policy_medians)

    reference = estimate(pair_ids)
    if reference == 0.0:
        return 0.0
    reference_positive = reference > 0.0
    rng = random.Random(seed)
    agreements = 0
    for _ in range(max(1, int(rounds))):
        selected_pairs = [
            pair_ids[rng.randrange(len(pair_ids))] for _ in pair_ids
        ]
        bootstrap_estimate = estimate(selected_pairs)
        agreements += (
            bootstrap_estimate != 0.0
            and (bootstrap_estimate > 0.0) == reference_positive
        )
    return agreements / max(1, int(rounds))


def load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected key=value")
        key, value = line.split("=", 1)
        result[key] = value
    return result


def validate_manifest_file(
    manifest: dict[str, str],
    *,
    path_key: str,
    hash_key: str,
) -> Path:
    raw_path = manifest.get(path_key)
    expected_hash = manifest.get(hash_key)
    if not raw_path or not expected_hash:
        raise ValueError(f"run manifest is missing {path_key}/{hash_key}")
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError(f"frozen run input is missing: {path}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"frozen run input hash mismatch for {path_key}: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    return path.resolve()


def validate_uniform_probe_map(
    path: Path,
    *,
    num_layers: int,
    num_heads: int,
    expected_label: int,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    expected = [[expected_label] * num_heads for _ in range(num_layers)]
    if rows != expected:
        raise ValueError(
            f"{path}: expected an exact {num_layers}x{num_heads} "
            f"uniform label-{expected_label} probe map"
        )


def _load_profiles(paths: list[Path]) -> list[dict[str, Any]]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required to load QK profiles") from error

    profiles = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata") or {})
        profiles.append(
            {
                "path": path.resolve(),
                "version": int(payload.get("version", 0)),
                "method": str(payload.get("method") or ""),
                "audit": dict(payload.get("audit") or {}),
                "metadata": metadata,
                "records": list(payload.get("records") or []),
            }
        )
    return profiles


def validate_profile(
    profile: dict[str, Any],
    *,
    num_layers: int,
    num_heads: int,
    branch: str,
    update_mode: str,
    expected_frames: int,
    expected_head_config_paths: dict[str, Path],
    expected_config_path: Path,
    expected_checkpoint_path: Path,
    expected_prompts: dict[tuple[str, str], str],
) -> dict[str, Any]:
    path = Path(profile["path"])
    if int(profile["version"]) < 3:
        raise ValueError(
            f"{path}: profile version must be at least 3 with frozen protocol metadata"
        )
    if profile.get("method") != "frame_level_pre_softmax_qk_last_query":
        raise ValueError(f"{path}: unexpected profile method {profile.get('method')!r}")
    records = list(profile["records"])
    if not records:
        raise ValueError(f"{path}: profile contains no records")

    observed_layers = sorted({int(record["layer"]) for record in records})
    observed_branches = sorted(
        {str(record["cfg_branch"]) for record in records}
    )
    observed_modes = sorted(
        {str(record["cache_update_mode"]) for record in records}
    )
    observed_sources = sorted(
        {str(record.get("layer_index_source", "")) for record in records}
    )
    observed_prompt_ids = sorted(
        {int(record.get("prompt_id", -1)) for record in records}
    )
    if observed_layers != list(range(num_layers)):
        raise ValueError(f"{path}: incomplete layer coverage")
    if observed_branches != [branch]:
        raise ValueError(
            f"{path}: expected only branch={branch}, found {observed_branches}"
        )
    if observed_modes != [update_mode]:
        raise ValueError(
            f"{path}: expected only update_mode={update_mode}, "
            f"found {observed_modes}"
        )
    if observed_sources != ["kv_cache.layer_idx"]:
        raise ValueError(f"{path}: invalid layer index sources {observed_sources}")
    if observed_prompt_ids != [0]:
        raise ValueError(f"{path}: expected exactly prompt_id=0")

    audit = dict(profile.get("audit") or {})
    if int(audit.get("expected_num_layers", -1)) != num_layers:
        raise ValueError(f"{path}: recorder layer contract mismatch")
    if int(audit.get("expected_num_heads", -1)) != num_heads:
        raise ValueError(f"{path}: recorder head contract mismatch")

    metadata = dict(profile.get("metadata") or {})
    metadata_contract = {
        "kind": "middle_relative",
        "update_modes": update_mode,
        "branches": branch,
        "num_output_frames": expected_frames,
        "few_step_cfg_enabled": False,
    }
    metadata_mismatches = {
        key: (expected, metadata.get(key))
        for key, expected in metadata_contract.items()
        if metadata.get(key) != expected
    }
    if metadata_mismatches:
        raise ValueError(
            f"{path}: profile metadata mismatch: {metadata_mismatches}"
        )
    observed_config = Path(str(metadata.get("config_path") or ""))
    observed_checkpoint = Path(str(metadata.get("checkpoint_path") or ""))
    if observed_config.resolve() != expected_config_path.resolve():
        raise ValueError(f"{path}: profile config path is not the frozen config")
    if observed_checkpoint.resolve() != expected_checkpoint_path.resolve():
        raise ValueError(
            f"{path}: profile checkpoint path is not the frozen checkpoint"
        )
    observed_head_config = Path(str(metadata.get("head_config_path") or ""))
    matching_policies = [
        policy
        for policy, expected_path in expected_head_config_paths.items()
        if observed_head_config.resolve() == expected_path.resolve()
    ]
    if len(matching_policies) != 1:
        raise ValueError(
            f"{path}: head config {observed_head_config} does not match "
            f"exactly one frozen probe map {expected_head_config_paths}"
        )
    probe_policy = matching_policies[0]
    pair_id = str(metadata.get("pair_id") or "")
    side = str(metadata.get("side") or "")
    if not pair_id or side not in {"a", "b"}:
        raise ValueError(f"{path}: missing counterfactual pair metadata")
    expected_prompt = expected_prompts.get((pair_id, side))
    if expected_prompt is None:
        raise ValueError(f"{path}: unknown counterfactual sample {pair_id}/{side}")
    prompt_path = Path(str(metadata.get("data_path") or ""))
    if not prompt_path.is_file():
        raise ValueError(f"{path}: missing per-profile prompt file {prompt_path}")
    observed_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if observed_prompt != expected_prompt.strip():
        raise ValueError(
            f"{path}: per-profile prompt does not match {pair_id}/{side}"
        )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "pair_id": pair_id,
        "side": side,
        "seed": int(metadata.get("seed", 0)),
        "probe_policy": probe_policy,
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": sha256(prompt_path),
        "record_count": len(records),
    }


def main() -> None:
    args = parse_args()
    frozen_arguments = {
        "num_layers": (FROZEN_NUM_LAYERS, args.num_layers),
        "num_heads": (FROZEN_NUM_HEADS, args.num_heads),
        "sink_frames": (FROZEN_SINK_FRAMES, args.sink_frames),
        "recent_frames": (FROZEN_RECENT_FRAMES, args.recent_frames),
        "branch": (FROZEN_BRANCH, args.branch),
        "update_mode": (FROZEN_UPDATE_MODE, args.update_mode),
        "bootstrap_rounds": (
            FROZEN_BOOTSTRAP_ROUNDS,
            args.bootstrap_rounds,
        ),
        "bootstrap_seed": (FROZEN_BOOTSTRAP_SEED, args.bootstrap_seed),
        "min_profiles_per_policy_head": (
            FROZEN_MIN_PROFILES_PER_POLICY_HEAD,
            args.min_profiles_per_policy_head,
        ),
        "min_stable_head_fraction": (
            FROZEN_MIN_STABLE_HEAD_FRACTION,
            args.min_stable_head_fraction,
        ),
        "min_head_bootstrap_agreement": (
            FROZEN_MIN_HEAD_BOOTSTRAP_AGREEMENT,
            args.min_head_bootstrap_agreement,
        ),
        "min_topology_sign_agreement_fraction": (
            FROZEN_MIN_TOPOLOGY_SIGN_AGREEMENT_FRACTION,
            args.min_topology_sign_agreement_fraction,
        ),
        "min_minority_fraction": (
            FROZEN_MIN_MINORITY_FRACTION,
            args.min_minority_fraction,
        ),
    }
    frozen_mismatches = {
        key: {"required": required, "actual": actual}
        for key, (required, actual) in frozen_arguments.items()
        if actual != required
    }
    if frozen_mismatches:
        raise ValueError(
            "corrected v98 calibration arguments are frozen: "
            f"{frozen_mismatches}"
        )
    if args.sink_frames < 0 or args.recent_frames < 1:
        raise ValueError("sink/recent frame counts must be non-negative/positive")
    if not 0.0 <= args.min_stable_head_fraction <= 1.0:
        raise ValueError("min stable-head fraction must be in [0, 1]")
    if not 0.0 <= args.min_head_bootstrap_agreement <= 1.0:
        raise ValueError("min head bootstrap agreement must be in [0, 1]")
    if not 0.0 <= args.min_topology_sign_agreement_fraction <= 1.0:
        raise ValueError("min topology sign-agreement fraction must be in [0, 1]")
    if not 0.0 <= args.min_minority_fraction <= 0.5:
        raise ValueError("min minority fraction must be in [0, 0.5]")
    if not args.run_manifest.is_file():
        raise ValueError(f"missing run manifest {args.run_manifest}")

    manifest = load_env(args.run_manifest)
    required_manifest = {
        "EXPERIMENT": "v98_middle_relative_scores",
        "PROFILE_FRAMES": "120",
        "FEW_STEP_CFG_ENABLED": "0",
        "PROFILE_BRANCHES": args.branch,
        "PROFILE_UPDATE_MODES": args.update_mode,
        "PROBE_POLICIES": "uniform_stride,uniform_merge",
        "PAIR_COUNT": "8",
        "PROFILE_COUNT_PER_POLICY": "32",
        "PROFILE_COUNT": "64",
        "SEEDS": "0 1",
        "SINK_FRAMES": str(args.sink_frames),
        "RECENT_FRAMES": str(args.recent_frames),
    }
    mismatches = {
        key: (expected, manifest.get(key))
        for key, expected in required_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"run manifest protocol mismatch: {mismatches}")
    if not manifest.get("RUN_COMMIT"):
        raise ValueError("run manifest is missing RUN_COMMIT")
    if manifest.get("TRACKED_WORKTREE_DIRTY") != "0":
        raise ValueError("score extraction requires a clean tracked worktree")

    expected_config_path = validate_manifest_file(
        manifest, path_key="CONFIG", hash_key="CONFIG_SHA256"
    )
    expected_checkpoint_path = validate_manifest_file(
        manifest, path_key="CHECKPOINT", hash_key="CHECKPOINT_SHA256"
    )
    pair_json_path = validate_manifest_file(
        manifest, path_key="PAIR_JSON", hash_key="PAIR_SHA256"
    )
    pair_payload = json.loads(pair_json_path.read_text(encoding="utf-8"))
    pair_rows = list(pair_payload.get("prompt_pairs") or [])
    pair_ids = [str(row.get("id") or "") for row in pair_rows]
    if (
        len(pair_ids) != int(manifest["PAIR_COUNT"])
        or any(not pair_id for pair_id in pair_ids)
        or len(set(pair_ids)) != len(pair_ids)
    ):
        raise ValueError(
            "counterfactual pair file must contain exactly eight unique ids"
        )
    expected_prompts = {
        (str(row["id"]), side): str(row[side])
        for row in pair_rows
        for side in ("a", "b")
    }
    expected_probe_maps = {
        "uniform_stride": validate_manifest_file(
            manifest,
            path_key="PROBE_MAP_STRIDE",
            hash_key="PROBE_MAP_STRIDE_SHA256",
        ),
        "uniform_merge": validate_manifest_file(
            manifest,
            path_key="PROBE_MAP_MERGE",
            hash_key="PROBE_MAP_MERGE_SHA256",
        ),
    }
    validate_uniform_probe_map(
        expected_probe_maps["uniform_stride"],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        expected_label=1,
    )
    validate_uniform_probe_map(
        expected_probe_maps["uniform_merge"],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        expected_label=2,
    )

    profiles = _load_profiles(args.profiles)
    if len(profiles) != int(manifest["PROFILE_COUNT"]):
        raise ValueError(
            f"expected exactly {manifest['PROFILE_COUNT']} profiles, "
            f"found {len(profiles)}"
        )
    profile_audit = [
        validate_profile(
            profile,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            branch=args.branch,
            update_mode=args.update_mode,
            expected_frames=int(manifest["PROFILE_FRAMES"]),
            expected_head_config_paths=expected_probe_maps,
            expected_config_path=expected_config_path,
            expected_checkpoint_path=expected_checkpoint_path,
            expected_prompts=expected_prompts,
        )
        for profile in profiles
    ]
    pair_sides: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for item in profile_audit:
        pair_sides[
            (item["probe_policy"], item["pair_id"], item["seed"])
        ].append(item["side"])
    invalid_pairs = {
        f"{policy}:{pair_id}:seed{seed}": sorted(sides)
        for (policy, pair_id, seed), sides in pair_sides.items()
        if sorted(sides) != ["a", "b"]
    }
    if invalid_pairs:
        raise ValueError(f"incomplete counterfactual profile pairs: {invalid_pairs}")
    samples_by_policy: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    for item in profile_audit:
        samples_by_policy[item["probe_policy"]].add(
            (item["pair_id"], item["side"], item["seed"])
        )
    if set(samples_by_policy) != set(expected_probe_maps):
        raise ValueError(
            f"probe-policy coverage mismatch: {sorted(samples_by_policy)}"
        )
    reference_samples = samples_by_policy["uniform_stride"]
    if len(reference_samples) != int(manifest["PROFILE_COUNT_PER_POLICY"]):
        raise ValueError(
            "unexpected profiles per policy: "
            f"{len(reference_samples)}"
        )
    for policy, samples in samples_by_policy.items():
        if samples != reference_samples:
            raise ValueError(
                f"unbalanced counterfactual samples for {policy}: "
                f"missing={sorted(reference_samples - samples)} "
                f"extra={sorted(samples - reference_samples)}"
            )
    if {pair_id for pair_id, _, _ in reference_samples} != set(pair_ids):
        raise ValueError("profile pair ids do not match the frozen pair file")
    if {seed for _, _, seed in reference_samples} != {0, 1}:
        raise ValueError("profile seeds must be exactly {0, 1}")

    # Aggregate records within each independently generated profile first.
    per_head_policy_profiles: dict[
        tuple[int, int],
        dict[str, dict[tuple[str, str, int], float]],
    ] = defaultdict(lambda: defaultdict(dict))
    per_head_record_counts: dict[tuple[int, int], int] = defaultdict(int)
    observation_payload: dict[str, dict[str, dict[str, float]]] = {}
    for profile, audit_item in zip(profiles, profile_audit, strict=True):
        probe_policy = str(audit_item["probe_policy"])
        sample_key = (
            str(audit_item["pair_id"]),
            str(audit_item["side"]),
            int(audit_item["seed"]),
        )
        within: dict[tuple[int, int], list[float]] = defaultdict(list)
        for record in profile["records"]:
            if str(record["cfg_branch"]) != args.branch:
                continue
            if str(record["cache_update_mode"]) != args.update_mode:
                continue
            margins = record_middle_relative_margins(
                record,
                num_heads=args.num_heads,
                sink_frames=args.sink_frames,
                recent_frames=args.recent_frames,
            )
            if margins is None:
                continue
            layer = int(record["layer"])
            for head, margin in enumerate(margins):
                within[(layer, head)].append(margin)

        expected_keys = {
            (layer, head)
            for layer in range(args.num_layers)
            for head in range(args.num_heads)
        }
        missing = sorted(expected_keys - set(within))
        if missing:
            raise ValueError(
                f"{profile['path']}: no eligible middle/recent records for "
                f"{missing[:12]} (total={len(missing)})"
            )
        for key in sorted(expected_keys):
            profile_value = median(within[key])
            policy_profiles = per_head_policy_profiles[key][probe_policy]
            if sample_key in policy_profiles:
                raise ValueError(
                    f"duplicate head observation for {probe_policy}/{sample_key}"
                )
            policy_profiles[sample_key] = profile_value
            per_head_record_counts[key] += len(within[key])
            observation_payload.setdefault(
                f"L{key[0]}H{key[1]}", {}
            ).setdefault(probe_policy, {})[
                f"{sample_key[0]}|{sample_key[1]}|{sample_key[2]}"
            ] = profile_value

    entries: list[dict[str, Any]] = []
    bootstrap_values: list[float] = []
    topology_agreements: list[bool] = []
    labels: list[int] = []
    for layer in range(args.num_layers):
        for head in range(args.num_heads):
            key = (layer, head)
            values_by_policy = dict(per_head_policy_profiles[key])
            if set(values_by_policy) != set(expected_probe_maps):
                raise ValueError(
                    f"L{layer}H{head}: incomplete policy coverage "
                    f"{sorted(values_by_policy)}"
                )
            for policy, sample_values in values_by_policy.items():
                if len(sample_values) < args.min_profiles_per_policy_head:
                    raise ValueError(
                        f"L{layer}H{head}/{policy}: only {len(sample_values)} profile "
                        "observations; need "
                        f"{args.min_profiles_per_policy_head}"
                    )
            policy_scores = {
                policy: median(sample_values.values())
                for policy, sample_values in sorted(values_by_policy.items())
            }
            score = median(policy_scores.values())
            agreement = paired_cluster_bootstrap_sign_agreement(
                values_by_policy,
                rounds=args.bootstrap_rounds,
                seed=args.bootstrap_seed + layer * args.num_heads + head,
            )
            positive_fraction = mean(
                [
                    sum(value > 0.0 for value in sample_values.values())
                    / len(sample_values)
                    for _, sample_values in sorted(values_by_policy.items())
                ]
            )
            stride_score = policy_scores["uniform_stride"]
            merge_score = policy_scores["uniform_merge"]
            topology_agreement = (
                stride_score != 0.0
                and merge_score != 0.0
                and (stride_score > 0.0) == (merge_score > 0.0)
            )
            profile_count = sum(
                len(sample_values) for sample_values in values_by_policy.values()
            )
            entries.append(
                {
                    "layer": layer,
                    "head": head,
                    PRIMARY_FIELD: score,
                    "uniform_stride_margin": policy_scores["uniform_stride"],
                    "uniform_merge_margin": policy_scores["uniform_merge"],
                    "topology_sign_agreement": int(topology_agreement),
                    "profile_observation_count": profile_count,
                    "record_observation_count": per_head_record_counts[key],
                    "profile_positive_fraction": positive_fraction,
                    "bootstrap_sign_agreement": agreement,
                }
            )
            bootstrap_values.append(agreement)
            topology_agreements.append(topology_agreement)
            labels.append(10 if score >= 0.0 else 11)

    head_count = args.num_layers * args.num_heads
    stable_fraction = (
        sum(
            value >= args.min_head_bootstrap_agreement
            for value in bootstrap_values
        )
        / head_count
    )
    support_count = labels.count(10)
    suppress_count = labels.count(11)
    minority_fraction = min(support_count, suppress_count) / head_count
    topology_agreement_fraction = (
        sum(topology_agreements) / head_count
    )
    gates = {
        "complete_head_grid": {
            "observed": len(entries),
            "required": head_count,
            "passed": len(entries) == head_count,
        },
        "bootstrap_stable_head_fraction": {
            "observed": stable_fraction,
            "required": args.min_stable_head_fraction,
            "per_head_threshold": args.min_head_bootstrap_agreement,
            "passed": stable_fraction >= args.min_stable_head_fraction,
        },
        "topology_sign_agreement_fraction": {
            "observed": topology_agreement_fraction,
            "required": args.min_topology_sign_agreement_fraction,
            "passed": (
                topology_agreement_fraction
                >= args.min_topology_sign_agreement_fraction
            ),
        },
        "minority_role_fraction": {
            "observed": minority_fraction,
            "required": args.min_minority_fraction,
            "passed": minority_fraction >= args.min_minority_fraction,
        },
    }
    accepted = all(bool(item["passed"]) for item in gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_csv = args.output_dir / "qk_head_scores.csv"
    with score_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)

    observations = args.output_dir / "qk_head_observations.json"
    observations.write_text(
        json.dumps(
            {
                "version": 3,
                "method": METHOD,
                "primary_field": PRIMARY_FIELD,
                "per_head_policy_profile_margins": observation_payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = {
        "version": 2,
        "method": METHOD,
        "accepted": accepted,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "head_count": len(entries),
        "score_definition": {
            "primary_field": PRIMARY_FIELD,
            "formula": (
                "(mean(middle_logits)-mean(recent_logits))/"
                "rms_centered(middle_logits+recent_logits)"
            ),
            "aggregation": (
                "median_records_within_profile_then_median_profiles_per_policy_"
                "then_equal_policy_median"
            ),
            "branch": args.branch,
            "update_mode": args.update_mode,
            "sink_frames_excluded": args.sink_frames,
            "recent_distinct_key_frames": args.recent_frames,
            "common_logit_shift_invariant": True,
            "pf_labels_used": False,
            "probe_policy_balanced": True,
            "probe_policies": ["uniform_stride", "uniform_merge"],
            "bootstrap_unit": "counterfactual_prompt_pair",
        },
        "bootstrap_protocol": {
            "rounds": args.bootstrap_rounds,
            "seed": args.bootstrap_seed,
            "zero_effect_is_stable": False,
        },
        "acceptance_protocol": {
            "min_profiles_per_policy_head": args.min_profiles_per_policy_head,
            "min_stable_head_fraction": args.min_stable_head_fraction,
            "min_head_bootstrap_agreement": args.min_head_bootstrap_agreement,
            "min_topology_sign_agreement_fraction": (
                args.min_topology_sign_agreement_fraction
            ),
            "min_minority_fraction": args.min_minority_fraction,
        },
        "profile_protocol": manifest,
        "profile_audit": profile_audit,
        "acceptance_gates": gates,
        "label_counts_at_zero": {
            "10": support_count,
            "11": suppress_count,
        },
        "files": {
            "score_csv": score_csv.name,
            "score_csv_sha256": sha256(score_csv),
            "observations": observations.name,
            "observations_sha256": sha256(observations),
            "run_manifest": os.path.relpath(
                args.run_manifest.resolve(), args.output_dir.resolve()
            ),
            "run_manifest_sha256": sha256(args.run_manifest),
        },
    }
    artifact_path = args.output_dir / "qk_head_score_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[V98MiddleRelativeScores] "
        f"profiles={len(profiles)} heads={len(entries)} "
        f"support={support_count} suppress={suppress_count} "
        f"stable_fraction={stable_fraction:.4f} "
        f"topology_agreement={topology_agreement_fraction:.4f} "
        f"accepted={accepted} "
        f"score_sha256={artifact['files']['score_csv_sha256']}",
        flush=True,
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
