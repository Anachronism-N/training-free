#!/usr/bin/env python3
"""Analyze full-prompt A-B-A counterfactual head profiles."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch


V140_PATH = Path(__file__).with_name(
    "analyze_v140_prompt_threshold_robustness.py"
)
V140_SPEC = importlib.util.spec_from_file_location("v140_threshold", V140_PATH)
V140 = importlib.util.module_from_spec(V140_SPEC)
assert V140_SPEC.loader is not None
V140_SPEC.loader.exec_module(V140)

EPS = 1e-4
EXPECTED_LAYERS = 30
EXPECTED_HEADS = 12
EXPECTED_BRANCHES = {
    "base",
    "exact_a",
    "exact_b",
    "paraphrase_a",
    "paraphrase_b",
}
SIGNATURES = (
    "residual_signature",
    "native_signature",
    "query_signature",
    "current_key_signature",
)
SWITCH_FRAMES = (39, 78)


def _relative(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("paired signatures must share [heads, features]")
    left = left.float()
    right = right.float()
    numerator = (left - right).norm(dim=-1)
    denominator = 0.5 * (
        left.norm(dim=-1) + right.norm(dim=-1)
    ).clamp_min(1e-6)
    return numerator / denominator


def _normalize_probabilities(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp_min(0)
    return value / value.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _js_per_head(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = _normalize_probabilities(left)
    right = _normalize_probabilities(right)
    middle = 0.5 * (left + right)
    left_kl = (
        left
        * (
            left.clamp_min(1e-12).log()
            - middle.clamp_min(1e-12).log()
        )
    ).sum(dim=-1)
    right_kl = (
        right
        * (
            right.clamp_min(1e-12).log()
            - middle.clamp_min(1e-12).log()
        )
    ).sum(dim=-1)
    return 0.5 * (left_kl + right_kl)


def _w1_per_head(
    left: torch.Tensor, right: torch.Tensor, frame_ids: torch.Tensor
) -> torch.Tensor:
    left = _normalize_probabilities(left)
    right = _normalize_probabilities(right)
    order = torch.argsort(frame_ids.float())
    left = left.index_select(-1, order)
    right = right.index_select(-1, order)
    positions = frame_ids.float().index_select(0, order)
    if positions.numel() <= 1:
        return torch.zeros(left.shape[0])
    gaps = positions[1:] - positions[:-1]
    return (
        (left.cumsum(-1)[:, :-1] - right.cumsum(-1)[:, :-1]).abs()
        * gaps.unsqueeze(0)
    ).sum(-1)


def _state_key(record: dict) -> tuple:
    return (
        str(record["mode"]),
        int(record["current_frame"]),
        int(record["nominal_timestep"]),
        int(record["layer"]),
    )


def _episode(current_frame: int) -> tuple[int, str]:
    if current_frame < SWITCH_FRAMES[0]:
        return 0, "A1"
    if current_frame < SWITCH_FRAMES[1]:
        return 1, "B"
    return 2, "A2"


def _boundary_phase(current_frame: int) -> str:
    mapping = {
        36: "pre_b",
        39: "switch_b",
        42: "post_b",
        75: "pre_a2",
        78: "switch_a2",
        81: "post_a2",
        117: "late_a2",
    }
    return mapping.get(current_frame, f"frame_{current_frame}")


def load_profiles(directory: Path) -> list[dict]:
    profiles = []
    for path in sorted(directory.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", 0)) != 5:
            raise ValueError(f"v141 requires profile version 5: {path}")
        payload["_path"] = str(path)
        profiles.append(payload)
    return profiles


def audit_profiles(
    profiles: list[dict],
    *,
    expected_count: int,
    expected_states: int,
) -> list[dict]:
    if len(profiles) != expected_count:
        raise ValueError(
            f"profile count mismatch: {len(profiles)} != {expected_count}"
        )
    expected_layers = set(range(EXPECTED_LAYERS))
    expected_calls = expected_states * len(EXPECTED_BRANCHES)
    expected_records = expected_calls * EXPECTED_LAYERS
    run_commits = set()
    seen_indices = set()
    audit = []
    failures = []
    for profile in profiles:
        job = profile.get("job", {})
        metadata = profile.get("metadata", {})
        job_id = str(job.get("job_id", ""))
        index = int(job.get("dataset_index", -1))
        if index in seen_indices:
            failures.append(f"{job_id}: duplicate dataset_index={index}")
        seen_indices.add(index)
        run_commit = str(metadata.get("run_commit") or "")
        if run_commit:
            run_commits.add(run_commit)
        groups = defaultdict(lambda: defaultdict(set))
        tensor_failures = []
        for record in profile.get("records", []):
            state = _state_key(record)[:3]
            branch = str(record["branch"])
            groups[state][branch].add(int(record["layer"]))
            episode_index, episode_label = _episode(
                int(record["current_frame"])
            )
            if (
                int(record.get("episode_index", -1)) != episode_index
                or str(record.get("episode_label", "")) != episode_label
            ):
                tensor_failures.append("episode_metadata_mismatch")
            for name in SIGNATURES:
                value = record.get(name)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 2
                    or value.shape[0] != EXPECTED_HEADS
                ):
                    tensor_failures.append(
                        f"{name}:{getattr(value, 'shape', None)}"
                    )
                elif not bool(torch.isfinite(value.float()).all()):
                    tensor_failures.append(f"{name}:non_finite")
            probs = record.get("temporal_probs")
            frame_ids = record.get("history_frame_ids")
            if (
                not isinstance(probs, torch.Tensor)
                or probs.ndim != 2
                or probs.shape[0] != EXPECTED_HEADS
                or not isinstance(frame_ids, torch.Tensor)
                or frame_ids.ndim != 1
                or probs.shape[1] != frame_ids.numel()
            ):
                tensor_failures.append("temporal_shape_mismatch")
            elif not bool(torch.isfinite(probs.float()).all()):
                tensor_failures.append("temporal_non_finite")
        bad_groups = []
        for state, branches in groups.items():
            if set(branches) != EXPECTED_BRANCHES:
                bad_groups.append((state, "branches"))
                continue
            if any(layers != expected_layers for layers in branches.values()):
                bad_groups.append((state, "layers"))
        calls = profile.get("calls", [])
        call_branch_counts = Counter(
            str(call.get("branch", "")) for call in calls
        )
        passed = (
            str(job.get("kind")) == "full_prompt_switch"
            and set(job.get("shadow_prompts", {}))
            == EXPECTED_BRANCHES - {"base"}
            and [int(value) for value in job.get("switch_frames", [])]
            == list(SWITCH_FRAMES)
            and bool(metadata.get("allow_prompt_schedule", False))
            and [int(value) for value in metadata.get("switch_frames", [])]
            == list(SWITCH_FRAMES)
            and bool(run_commit)
            and int(job.get("seed", -1)) == 0
            and len(calls) == expected_calls
            and call_branch_counts
            == Counter(
                {
                    branch: expected_states
                    for branch in EXPECTED_BRANCHES
                }
            )
            and int(metadata.get("captured_calls", -1)) == expected_calls
            and int(metadata.get("record_count", -1)) == expected_records
            and len(profile.get("records", [])) == expected_records
            and len(groups) == expected_states
            and not bad_groups
            and not tensor_failures
        )
        row = {
            "dataset_index": index,
            "job_id": job_id,
            "switch_type": str(job.get("switch_type", "")),
            "states": len(groups),
            "captured_calls": int(metadata.get("captured_calls", -1)),
            "records": len(profile.get("records", [])),
            "bad_groups": len(bad_groups),
            "tensor_failures": len(tensor_failures),
            "run_commit": run_commit,
            "passed": int(passed),
        }
        audit.append(row)
        if not passed:
            failures.append(
                f"{job_id}: {row}; groups={bad_groups[:2]}; "
                f"tensors={tensor_failures[:3]}"
            )
    if seen_indices != set(range(expected_count)):
        failures.append("dataset indices are not exactly 0..expected_count-1")
    if len(run_commits) != 1:
        failures.append(f"expected one run commit, found {run_commits}")
    if failures:
        raise ValueError("v141 profile audit failed:\n" + "\n".join(failures[:8]))
    return audit


def collect_observations(profiles: list[dict]) -> list[dict]:
    observations = []
    for profile in profiles:
        job = profile["job"]
        grouped = defaultdict(dict)
        for record in profile["records"]:
            grouped[_state_key(record)][str(record["branch"])] = record
        for state, branches in grouped.items():
            if set(branches) != EXPECTED_BRANCHES:
                continue
            base = branches["base"]
            episode_index, episode_label = _episode(
                int(base["current_frame"])
            )
            is_a = episode_label in {"A1", "A2"}
            matched_name = "exact_a" if is_a else "exact_b"
            opposite_name = "exact_b" if is_a else "exact_a"
            paraphrase_name = (
                "paraphrase_a" if is_a else "paraphrase_b"
            )
            matched = branches[matched_name]
            opposite = branches[opposite_name]
            paraphrase = branches[paraphrase_name]
            distances = {}
            for name in SIGNATURES:
                distances[f"parity_{name}"] = _relative(
                    matched[name], base[name]
                )
                distances[f"switch_{name}"] = _relative(
                    opposite[name], base[name]
                )
                distances[f"paraphrase_{name}"] = _relative(
                    paraphrase[name], base[name]
                )
            switch_js = _js_per_head(
                opposite["temporal_probs"], base["temporal_probs"]
            )
            paraphrase_js = _js_per_head(
                paraphrase["temporal_probs"], base["temporal_probs"]
            )
            switch_w1 = _w1_per_head(
                opposite["temporal_probs"],
                base["temporal_probs"],
                base["history_frame_ids"],
            )
            paraphrase_w1 = _w1_per_head(
                paraphrase["temporal_probs"],
                base["temporal_probs"],
                base["history_frame_ids"],
            )
            base_probs = _normalize_probabilities(base["temporal_probs"])
            frame_ids = base["history_frame_ids"].long()
            frame_episodes = torch.tensor(
                [_episode(int(frame))[0] for frame in frame_ids],
                dtype=torch.long,
            )
            previous_mask = frame_episodes != episode_index
            same_a_mask = (
                (frame_episodes == 0)
                if episode_index == 2
                else torch.zeros_like(frame_episodes, dtype=torch.bool)
            )
            previous_mass = (
                base_probs[:, previous_mask].sum(-1)
                if bool(previous_mask.any())
                else torch.zeros(EXPECTED_HEADS)
            )
            returned_a_mass = (
                base_probs[:, same_a_mask].sum(-1)
                if bool(same_a_mask.any())
                else torch.zeros(EXPECTED_HEADS)
            )
            for head in range(EXPECTED_HEADS):
                switch_residual = float(
                    distances["switch_residual_signature"][head]
                )
                paraphrase_residual = float(
                    distances["paraphrase_residual_signature"][head]
                )
                residual_log_ratio = math.log(
                    (switch_residual + EPS)
                    / (paraphrase_residual + EPS)
                )
                switch_query = float(
                    distances["switch_query_signature"][head]
                )
                paraphrase_query = float(
                    distances["paraphrase_query_signature"][head]
                )
                query_log_ratio = math.log(
                    (switch_query + EPS) / (paraphrase_query + EPS)
                )
                row = {
                    "dataset_index": int(job["dataset_index"]),
                    "job_id": str(job["job_id"]),
                    "family_index": int(job["family_index"]),
                    "switch_type": str(job["switch_type"]),
                    "mode": str(base["mode"]),
                    "current_frame": int(base["current_frame"]),
                    "nominal_timestep": int(base["nominal_timestep"]),
                    "layer": int(base["layer"]),
                    "head": head,
                    "episode_index": episode_index,
                    "episode_label": episode_label,
                    "boundary_phase": _boundary_phase(
                        int(base["current_frame"])
                    ),
                    "matched_branch": matched_name,
                    "opposite_branch": opposite_name,
                    "paraphrase_branch": paraphrase_name,
                    "switch_residual": switch_residual,
                    "paraphrase_residual": paraphrase_residual,
                    "parity_residual": float(
                        distances["parity_residual_signature"][head]
                    ),
                    "residual_log_ratio": residual_log_ratio,
                    "query_log_ratio": query_log_ratio,
                    "prompt_history_excess": (
                        residual_log_ratio - query_log_ratio
                    ),
                    "switch_temporal_js": float(switch_js[head]),
                    "paraphrase_temporal_js": float(paraphrase_js[head]),
                    "temporal_js_excess": float(
                        switch_js[head] - paraphrase_js[head]
                    ),
                    "switch_temporal_w1": float(switch_w1[head]),
                    "paraphrase_temporal_w1": float(paraphrase_w1[head]),
                    "temporal_w1_excess": float(
                        switch_w1[head] - paraphrase_w1[head]
                    ),
                    "previous_episode_mass": float(previous_mass[head]),
                    "returned_a1_mass": float(returned_a_mass[head]),
                }
                for name in SIGNATURES[1:]:
                    short = name.removesuffix("_signature")
                    switch_value = float(distances[f"switch_{name}"][head])
                    paraphrase_value = float(
                        distances[f"paraphrase_{name}"][head]
                    )
                    row[f"switch_{short}"] = switch_value
                    row[f"paraphrase_{short}"] = paraphrase_value
                    row[f"parity_{short}"] = float(
                        distances[f"parity_{name}"][head]
                    )
                    row[f"{short}_log_ratio"] = math.log(
                        (switch_value + EPS)
                        / (paraphrase_value + EPS)
                    )
                observations.append(row)
    return observations


AGGREGATE_FIELDS = (
    "switch_residual",
    "paraphrase_residual",
    "parity_residual",
    "residual_log_ratio",
    "query_log_ratio",
    "prompt_history_excess",
    "native_log_ratio",
    "current_key_log_ratio",
    "switch_temporal_js",
    "paraphrase_temporal_js",
    "temporal_js_excess",
    "switch_temporal_w1",
    "paraphrase_temporal_w1",
    "temporal_w1_excess",
    "previous_episode_mass",
    "returned_a1_mass",
)


def _aggregate(
    rows: list[dict],
    keys: tuple[str, ...],
    fields: tuple[str, ...] = AGGREGATE_FIELDS,
) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[name] for name in keys)
        for field in fields:
            grouped[key][field].append(float(row[field]))
    output = []
    for key, values in sorted(grouped.items()):
        row = {name: value for name, value in zip(keys, key)}
        for field in fields:
            row[field] = V140._median(values[field])
            row[f"{field}_samples"] = len(values[field])
        output.append(row)
    return output


def _head_map(rows: list[dict]) -> dict[tuple[int, int], float]:
    aggregated = _aggregate(
        rows, ("layer", "head"), ("prompt_history_excess",)
    )
    return {
        (int(row["layer"]), int(row["head"])): float(
            row["prompt_history_excess"]
        )
        for row in aggregated
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    profiles: list[dict],
    *,
    output_dir: Path,
    expected_count: int,
    expected_states: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_profiles(
        profiles,
        expected_count=expected_count,
        expected_states=expected_states,
    )
    observations = collect_observations(profiles)
    if not observations:
        raise ValueError("no complete v141 branch groups found")
    primary_rows = [
        row
        for row in observations
        if int(row["layer"]) > 0
        and row["boundary_phase"]
        in {"switch_b", "post_b", "switch_a2", "post_a2"}
    ]
    discovery_rows = [
        row for row in primary_rows if int(row["family_index"]) % 2 == 0
    ]
    validation_rows = [
        row for row in primary_rows if int(row["family_index"]) % 2 == 1
    ]
    full_map = _head_map(primary_rows)
    discovery_map = _head_map(discovery_rows)
    validation_map = _head_map(validation_rows)
    active_keys = [
        (layer, head)
        for layer in range(1, EXPECTED_LAYERS)
        for head in range(EXPECTED_HEADS)
    ]
    discovery_values = [discovery_map[key] for key in active_keys]
    validation_values = [validation_map[key] for key in active_keys]
    split_spearman = V140._spearman(discovery_values, validation_values)
    discovery_positive = {
        key for key in active_keys if discovery_map[key] > 0
    }
    validation_positive = {
        key for key in active_keys if validation_map[key] > 0
    }
    label_agreement = sum(
        (key in discovery_positive) == (key in validation_positive)
        for key in active_keys
    ) / len(active_keys)
    union = discovery_positive | validation_positive
    label_jaccard = (
        len(discovery_positive & validation_positive) / len(union)
        if union
        else 1.0
    )
    validation_minority = min(
        len(validation_positive),
        len(active_keys) - len(validation_positive),
    ) / len(active_keys)
    iqr = V140._quantile(discovery_values, 0.75) - V140._quantile(
        discovery_values, 0.25
    )
    boundary_fraction = sum(
        abs(validation_map[key]) <= 0.1 * max(iqr, 1e-12)
        for key in active_keys
    ) / len(active_keys)
    otsu_threshold = V140._otsu_threshold(discovery_values)
    gmm_threshold, gmm = V140._gmm_threshold(discovery_values)

    head_aggregate = {
        (int(row["layer"]), int(row["head"])): row
        for row in _aggregate(primary_rows, ("layer", "head"))
    }
    head_rows = []
    for layer in range(EXPECTED_LAYERS):
        for head in range(EXPECTED_HEADS):
            key = (layer, head)
            aggregate = head_aggregate.get(key, {})
            full_score = full_map.get(key, 0.0)
            discovery_score = discovery_map.get(key, 0.0)
            validation_score = validation_map.get(key, 0.0)
            head_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    **{
                        field: aggregate.get(field, 0.0)
                        for field in AGGREGATE_FIELDS
                    },
                    "discovery_prompt_history_excess": discovery_score,
                    "validation_prompt_history_excess": validation_score,
                    "zero_label": (
                        "switch_responsive"
                        if layer > 0 and full_score > 0
                        else "switch_stable"
                    ),
                    "discovery_zero_label": int(
                        layer > 0 and discovery_score > 0
                    ),
                    "validation_zero_label": int(
                        layer > 0 and validation_score > 0
                    ),
                    "otsu_label": int(
                        layer > 0 and full_score > otsu_threshold
                    ),
                    "gmm2_label": int(
                        layer > 0 and full_score > gmm_threshold
                    ),
                }
            )

    switch_type_rows = _aggregate(
        primary_rows, ("switch_type", "layer", "head")
    )
    episode_rows = _aggregate(
        observations, ("episode_label", "boundary_phase", "layer", "head")
    )
    parity_values = [
        float(row["parity_residual"])
        for row in observations
        if int(row["layer"]) > 0
    ]
    switch_values = [
        float(row["switch_residual"]) for row in primary_rows
    ]
    paraphrase_values = [
        float(row["paraphrase_residual"]) for row in primary_rows
    ]
    global_switch = V140._median(switch_values)
    global_paraphrase = V140._median(paraphrase_values)
    parity_median = V140._median(parity_values)
    parity_p99 = V140._quantile(parity_values, 0.99)
    parity_gate = parity_median <= 1e-5 and parity_p99 <= 1e-3
    magnitude_gate = global_switch > global_paraphrase
    stability_gate = split_spearman >= 0.60 and label_agreement >= 0.80
    split_gate = validation_minority >= 0.05 and boundary_fraction <= 0.20
    prompt_axis_gate = (
        parity_gate and magnitude_gate and stability_gate and split_gate
    )
    counts = Counter(row["zero_label"] for row in head_rows)
    report = {
        "method": "v141_full_prompt_aba_counterfactual_profiling",
        "profile_count": len(profiles),
        "observation_count": len(observations),
        "primary_observation_count": len(primary_rows),
        "profile_contract_passed": all(bool(row["passed"]) for row in audit),
        "gates": {
            "exact_prompt_shadow_parity": parity_gate,
            "full_switch_exceeds_local_paraphrase": magnitude_gate,
            "held_out_rank_and_label_stability": stability_gate,
            "nondegenerate_zero_split": split_gate,
            "full_prompt_switch_axis": prompt_axis_gate,
        },
        "global_effects": {
            "switch_residual_median": global_switch,
            "paraphrase_residual_median": global_paraphrase,
            "exact_parity_median": parity_median,
            "exact_parity_p99": parity_p99,
        },
        "held_out": {
            "split_spearman": split_spearman,
            "zero_label_agreement": label_agreement,
            "zero_positive_jaccard": label_jaccard,
            "validation_positive": len(validation_positive),
            "active_heads": len(active_keys),
            "validation_minority_fraction": validation_minority,
            "zero_boundary_fraction_0p1_iqr": boundary_fraction,
            "discovery_iqr": iqr,
        },
        "thresholds": {
            "primary_zero": 0.0,
            "otsu_discovery": otsu_threshold,
            "gmm2_discovery": gmm_threshold,
            "gmm2": gmm,
            "note": "Otsu and GMM are discovery-only diagnostics.",
        },
        "class_counts_including_forced_layer0": dict(counts),
        "recommendation": (
            "full_prompt_switch_axis_candidate"
            if prompt_axis_gate
            else "retain_continuous_diagnostic_or_redesign_prompt_axis"
        ),
    }
    _write_csv(output_dir / "profile_contract_audit.csv", audit)
    _write_csv(output_dir / "head_axes.csv", head_rows)
    _write_csv(output_dir / "head_switch_type_axes.csv", switch_type_rows)
    _write_csv(output_dir / "head_episode_axes.csv", episode_rows)
    _write_csv(output_dir / "state_observations.csv", observations)
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        "# v141 Full-Prompt A-B-A Head Profiling",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Full prompt-switch gate: `{prompt_axis_gate}`",
        (
            "- Exact-shadow parity median / p99: "
            f"`{parity_median:.6g}` / `{parity_p99:.6g}`"
        ),
        (
            "- Switch / local-paraphrase residual median: "
            f"`{global_switch:.6g}` / `{global_paraphrase:.6g}`"
        ),
        (
            "- Discovery-validation Spearman / label agreement: "
            f"`{split_spearman:.4f}` / `{label_agreement:.4f}`"
        ),
        (
            "- Validation positive / active heads: "
            f"`{len(validation_positive)}` / `{len(active_keys)}`"
        ),
        "",
        "## Interpretation",
        "",
        "The base trajectory executes A-B-A and preserves native self-attention "
        "history. Exact-A/B and local-paraphrase shadows change only current "
        "conditioning on the same latent and history.",
        "The primary score is residual switch/paraphrase log-ratio minus the "
        "corresponding query log-ratio. Zero therefore asks whether prompt "
        "switching changes history use beyond its direct effect on Q.",
        "Layer 0 is structurally prompt-blind at self-attention and is forced "
        "to the stable class. Otsu/GMM thresholds are diagnostics only.",
        "",
    ]
    (output_dir / "analysis_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    parser.add_argument("--expected-states", type=int, default=21)
    args = parser.parse_args()
    profiles = load_profiles(args.profile_dir)
    analyze(
        profiles,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        expected_states=args.expected_states,
    )


if __name__ == "__main__":
    main()
