#!/usr/bin/env python3
"""Re-analyze v152 as a one-sided history-critical head experiment.

The original v152 gate required two symmetric groups: heads preferring
uniform history and heads preferring recent history.  This script keeps that
negative result intact, but separately audits the supported one-sided claim:
high QK policy-margin heads benefit from distributed history.  Seed replicate
0 is the discovery split; replicate 1 is used only to audit map recurrence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


LAYERS = 30
HEADS = 12
HEADS_PER_LAYER = 4
DISCOVERY_REPLICATE = 0
VALIDATION_REPLICATE = 1
RANDOM_SEED = 2026
MIN_EFFECT = math.log(1.03)
MIN_RANDOM_EFFECT = math.log(1.01)
MIN_POSITIVE_FRACTION = 0.65
MIN_SEED_SPEARMAN = 0.30
MIN_ALIGNMENT_SPEARMAN = 0.30
EXPECTED_CONTEXTS = (
    "noisy_f117_t1000",
    "noisy_f117_t750",
    "noisy_f117_t500",
    "noisy_f117_t250",
)
MAP_FILENAMES = {
    "qk_top4": "v152_qk_history_critical_top4_seed0.csv",
    "qk_bottom4_control": (
        "v152_qk_history_critical_bottom4_seed0_control.csv"
    ),
    "random4_control": (
        "v152_qk_history_critical_random4_seed2026_control.csv"
    ),
}
MANIFEST_FILENAME = "v152_qk_history_critical_manifest.json"
PF_NAMES = {-1: "wave", 1: "anchor", 2: "veil"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_text_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(normalized.encode("utf-8"))


def read_matrix(path: Path, allowed: set[int]) -> list[list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            [int(value.strip()) for value in row]
            for row in csv.reader(handle)
            if row
        ]
    if len(rows) != LAYERS or any(len(row) != HEADS for row in rows):
        raise ValueError(f"{path}: expected a complete 30x12 matrix")
    observed = {value for row in rows for value in row}
    if not observed.issubset(allowed):
        raise ValueError(
            f"{path}: labels {sorted(observed)} are not within {sorted(allowed)}"
        )
    return rows


def audit_binary_matrix(
    matrix: list[list[int]],
    pf: list[list[int]],
    *,
    expected_label10_per_layer: int | None = None,
) -> dict[str, object]:
    counts = Counter(value for row in matrix for value in row)
    if set(counts) != {10, 11}:
        raise ValueError(f"binary map must contain labels 10 and 11: {counts}")
    layer_counts = [row.count(10) for row in matrix]
    if expected_label10_per_layer is not None and layer_counts != [
        expected_label10_per_layer
    ] * LAYERS:
        raise ValueError(
            "label-10 layer counts changed: "
            f"expected={expected_label10_per_layer} actual={layer_counts}"
        )
    cross_tab: dict[str, dict[str, int]] = {}
    for pf_label, name in PF_NAMES.items():
        role_counts = Counter(
            matrix[layer][head]
            for layer in range(LAYERS)
            for head in range(HEADS)
            if pf[layer][head] == pf_label
        )
        cross_tab[name] = {
            "10": int(role_counts[10]),
            "11": int(role_counts[11]),
        }
    return {
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "label10_per_layer": layer_counts,
        "pf_cross_tab": cross_tab,
    }


def audit_binary_map(
    path: Path,
    pf_labels: Path,
    *,
    expected_label10_per_layer: int | None = None,
) -> dict[str, object]:
    matrix = read_matrix(path, {10, 11})
    pf = read_matrix(pf_labels, set(PF_NAMES))
    return {
        "path": str(path),
        "sha256": sha256_bytes(_matrix_bytes(matrix)),
        "hash_contract": "parsed_30x12_matrix_with_lf",
        **audit_binary_matrix(
            matrix,
            pf,
            expected_label10_per_layer=expected_label10_per_layer,
        ),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _qualifies(row: dict[str, str], *, minimum_effect: float) -> bool:
    return bool(
        float(row["median_effect"]) >= minimum_effect
        and (
            float(row["prompt_bootstrap_mean_ci_low"]) > 0
            or float(row["positive_fraction"]) >= MIN_POSITIVE_FRACTION
        )
        and float(row["seed_replicate_spearman"]) >= MIN_SEED_SPEARMAN
    )


def _matching_rows(
    rows: Iterable[dict[str, str]], comparison: str
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["comparison"] == comparison]
    if {row["context"] for row in selected} != set(EXPECTED_CONTEXTS):
        raise ValueError(f"{comparison}: incomplete v152 context grid")
    return sorted(selected, key=lambda row: EXPECTED_CONTEXTS.index(row["context"]))


def analyze_one_sided_gates(result_root: Path) -> dict[str, object]:
    pair_rows = _read_csv(result_root / "policy_pair_summary.csv")
    random_rows = _read_csv(result_root / "random_control_summary.csv")
    alignment_rows = _read_csv(result_root / "selector_alignment_summary.csv")
    report = json.loads((result_root / "report.json").read_text(encoding="utf-8"))

    qk_high = _matching_rows(
        pair_rows, "qk_uniform4:uniform_over_recent:x0"
    )
    qk_low = _matching_rows(
        pair_rows, "qk_recent4:preferred_policy:x0"
    )
    qk_random = _matching_rows(
        random_rows, "qk_uniform4:preferred_minus_random"
    )
    qk_contexts = [
        row["context"]
        for row in qk_high
        if _qualifies(row, minimum_effect=MIN_EFFECT)
    ]
    random_contexts = [
        row["context"]
        for row in qk_random
        if _qualifies(row, minimum_effect=MIN_RANDOM_EFFECT)
    ]
    low_recent_contexts = [
        row["context"]
        for row in qk_low
        if _qualifies(row, minimum_effect=MIN_EFFECT)
    ]
    alignment = {
        row["context"]: float(row["oracle_qk_score_spearman"])
        for row in alignment_rows
    }
    if set(alignment) != set(EXPECTED_CONTEXTS):
        raise ValueError("selector alignment summary has an incomplete context grid")
    alignment_contexts = [
        context
        for context in EXPECTED_CONTEXTS
        if alignment[context] >= MIN_ALIGNMENT_SPEARMAN
    ]
    original_gates = report.get("gates") or report.get("gate_results") or {}
    one_sided_pass = bool(
        len(qk_contexts) >= 3
        and len(random_contexts) >= 3
        and len(alignment_contexts) == len(EXPECTED_CONTEXTS)
        and not low_recent_contexts
    )
    return {
        "one_sided_transfer_candidate": one_sided_pass,
        "qk_high_uniform_qualifying_contexts": qk_contexts,
        "qk_high_beats_random_qualifying_contexts": random_contexts,
        "qk_low_recent_qualifying_contexts": low_recent_contexts,
        "qk_oracle_alignment_contexts": alignment_contexts,
        "qk_oracle_spearman": alignment,
        "original_symmetric_gates": original_gates,
        "claim_boundary": (
            "The symmetric uniform-vs-recent binary taxonomy remains rejected. "
            "The passing claim is only that the high-QK tail is a stable, "
            "one-sided candidate for distributed-history retention; trajectory-"
            "level utility must be tested by generation."
        ),
    }


def load_qk_scores(
    snapshots_path: Path,
) -> tuple[dict[tuple[int, int, int], list[float]], dict[str, object]]:
    observations: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    coordinates: set[tuple[int, int, str, int]] = set()
    prompt_slots: set[int] = set()
    with gzip.open(
        snapshots_path, "rt", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["group"] != "qk_uniform4":
                continue
            if row["selector_type"] != "qk_policy_margin" or row["direction"] != "high":
                raise ValueError("qk_uniform4 selector contract changed")
            replicate = int(row["seed_replicate"])
            prompt = int(row["prompt_slot"])
            context = row["context"]
            layer = int(row["layer"])
            coordinate = (replicate, prompt, context, layer)
            if coordinate in coordinates:
                raise ValueError(f"duplicate selector snapshot {coordinate}")
            coordinates.add(coordinate)
            prompt_slots.add(prompt)
            scores = [float(value) for value in json.loads(row["scores"])]
            if len(scores) != HEADS or not all(math.isfinite(value) for value in scores):
                raise ValueError(f"invalid selector scores at {coordinate}")
            for head, score in enumerate(scores):
                observations[(replicate, layer, head)].append(score)

    expected_prompts = set(range(64))
    expected_coordinates = {
        (replicate, prompt, context, layer)
        for replicate in (DISCOVERY_REPLICATE, VALIDATION_REPLICATE)
        for prompt in expected_prompts
        for context in EXPECTED_CONTEXTS
        for layer in range(LAYERS)
    }
    if prompt_slots != expected_prompts or coordinates != expected_coordinates:
        raise ValueError("qk selector snapshots have an incomplete v152 grid")
    expected_observations = len(expected_prompts) * len(EXPECTED_CONTEXTS)
    if any(len(values) != expected_observations for values in observations.values()):
        raise ValueError("qk score aggregation has an incomplete prompt/context grid")
    return observations, {
        "selector_rows": len(coordinates),
        "prompt_count": len(prompt_slots),
        "seed_replicates": [DISCOVERY_REPLICATE, VALIDATION_REPLICATE],
        "contexts": list(EXPECTED_CONTEXTS),
        "observations_per_head_per_replicate": expected_observations,
    }


def _aggregate_scores(
    observations: dict[tuple[int, int, int], list[float]], replicate: int
) -> dict[tuple[int, int], float]:
    return {
        (layer, head): statistics.median(observations[(replicate, layer, head)])
        for layer in range(LAYERS)
        for head in range(HEADS)
    }


def _ranked_map(
    scores: dict[tuple[int, int], float], *, high: bool
) -> dict[int, list[int]]:
    result = {}
    for layer in range(LAYERS):
        if high:
            ranked = sorted(
                range(HEADS), key=lambda head: (-scores[(layer, head)], head)
            )
        else:
            ranked = sorted(
                range(HEADS), key=lambda head: (scores[(layer, head)], head)
            )
        result[layer] = sorted(ranked[:HEADS_PER_LAYER])
    return result


def _random_map(
    high_map: dict[int, list[int]],
    low_map: dict[int, list[int]],
    *,
    seed: int,
) -> dict[int, list[int]]:
    rng = random.Random(seed)
    result = {}
    for layer in range(LAYERS):
        forbidden = {frozenset(high_map[layer]), frozenset(low_map[layer])}
        while True:
            candidate = sorted(rng.sample(range(HEADS), HEADS_PER_LAYER))
            if frozenset(candidate) not in forbidden:
                result[layer] = candidate
                break
    return result


def _binary_matrix(selected: dict[int, list[int]]) -> list[list[int]]:
    return [
        [10 if head in set(selected[layer]) else 11 for head in range(HEADS)]
        for layer in range(LAYERS)
    ]


def _matrix_bytes(matrix: list[list[int]]) -> bytes:
    return (
        "".join(",".join(str(value) for value in row) + "\n" for row in matrix)
    ).encode("ascii")


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def build_artifacts(
    *,
    result_root: Path,
    pf_labels: Path,
    output_dir: Path,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict[str, bytes], dict[str, object]]:
    gates = analyze_one_sided_gates(result_root)
    if not gates["one_sided_transfer_candidate"]:
        raise ValueError("v152 one-sided transfer gate did not pass")
    observations, snapshot_audit = load_qk_scores(
        result_root / "selector_snapshots.csv.gz"
    )
    discovery_scores = _aggregate_scores(observations, DISCOVERY_REPLICATE)
    validation_scores = _aggregate_scores(observations, VALIDATION_REPLICATE)
    discovery_high = _ranked_map(discovery_scores, high=True)
    discovery_low = _ranked_map(discovery_scores, high=False)
    validation_high = _ranked_map(validation_scores, high=True)
    random_control = _random_map(
        discovery_high, discovery_low, seed=random_seed
    )
    maps = {
        "qk_top4": discovery_high,
        "qk_bottom4_control": discovery_low,
        "random4_control": random_control,
    }
    pf = read_matrix(pf_labels, set(PF_NAMES))
    payloads: dict[str, bytes] = {}
    map_metadata = {}
    for name, selected in maps.items():
        matrix = _binary_matrix(selected)
        payload = _matrix_bytes(matrix)
        filename = MAP_FILENAMES[name]
        payloads[filename] = payload
        map_metadata[name] = {
            "filename": filename,
            "sha256": sha256_bytes(payload),
            "selected_heads_by_layer": {
                str(layer): selected[layer] for layer in range(LAYERS)
            },
            **audit_binary_matrix(
                matrix, pf, expected_label10_per_layer=HEADS_PER_LAYER
            ),
        }

    layer_jaccards = [
        _jaccard(discovery_high[layer], validation_high[layer])
        for layer in range(LAYERS)
    ]
    overlap_count = sum(
        len(set(discovery_high[layer]) & set(validation_high[layer]))
        for layer in range(LAYERS)
    )
    boundary_margins = []
    for layer in range(LAYERS):
        ranked = sorted(
            range(HEADS), key=lambda head: (-discovery_scores[(layer, head)], head)
        )
        boundary_margins.append(
            discovery_scores[(layer, ranked[HEADS_PER_LAYER - 1])]
            - discovery_scores[(layer, ranked[HEADS_PER_LAYER])]
        )
    manifest = {
        "version": 2,
        "experiment": "v152_one_sided_history_critical_reanalysis",
        "classifier": {
            "score": (
                "median over prompts and four denoising contexts of "
                "uniform8 QK compatibility minus recent8 QK compatibility"
            ),
            "discovery_seed_replicate": DISCOVERY_REPLICATE,
            "validation_seed_replicate": VALIDATION_REPLICATE,
            "selection": "top 4 heads independently in each of 30 layers",
            "label10": "history_critical",
            "label11": "default",
            "random_seed": random_seed,
        },
        "source": {
            "result_directory": "docs/results/v152_online_policy_profile/core",
            "files": {
                filename: (
                    sha256_file(result_root / filename)
                    if filename.endswith(".gz")
                    else sha256_normalized_text_file(result_root / filename)
                )
                for filename in (
                    "policy_pair_summary.csv",
                    "random_control_summary.csv",
                    "selector_alignment_summary.csv",
                    "selector_snapshots.csv.gz",
                    "report.json",
                )
            },
            "pf_labels_sha256": sha256_bytes(
                _matrix_bytes(read_matrix(pf_labels, set(PF_NAMES)))
            ),
            "hash_contract": {
                "text": "utf8_bom_removed_and_newlines_normalized_to_lf",
                "gzip": "raw_file_sha256",
                "pf_labels": "parsed_30x12_matrix_with_lf",
            },
        },
        "gate_reanalysis": gates,
        "snapshot_audit": snapshot_audit,
        "discovery_validation_recurrence": {
            "overlap_heads": overlap_count,
            "total_selected_heads": LAYERS * HEADS_PER_LAYER,
            "exact_match_layers": sum(value == 1.0 for value in layer_jaccards),
            "median_layer_jaccard": statistics.median(layer_jaccards),
            "mean_layer_jaccard": statistics.mean(layer_jaccards),
            "layer_jaccards": layer_jaccards,
        },
        "discovery_rank_boundary": {
            "minimum_top4_minus_fifth": min(boundary_margins),
            "median_top4_minus_fifth": statistics.median(boundary_margins),
            "maximum_top4_minus_fifth": max(boundary_margins),
        },
        "maps": map_metadata,
        "generation_contract": {
            "history_critical_route": "sink1 + TemporalPrototype4 + recent4",
            "default_route": "sink1 + recent8",
            "full_frame_equivalents_per_head": 9,
            "purpose": (
                "causal generation transfer screen; no trajectory-level claim "
                "is made by the profiling result alone"
            ),
        },
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    payloads[MANIFEST_FILENAME] = manifest_payload
    return payloads, manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "docs" / "results" / "v152_online_policy_profile" / "core",
    )
    parser.add_argument(
        "--pf-labels",
        type=Path,
        default=(
            root
            / "third_party"
            / "Pyramid-Forcing"
            / "configs"
            / "head_configs"
            / "best_labels.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "configs" / "head_maps",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed artifacts instead of writing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = (
        args.result_root / "policy_pair_summary.csv",
        args.result_root / "random_control_summary.csv",
        args.result_root / "selector_alignment_summary.csv",
        args.result_root / "selector_snapshots.csv.gz",
        args.result_root / "report.json",
        args.pf_labels,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required v152 artifacts: {missing}")
    payloads, manifest = build_artifacts(
        result_root=args.result_root,
        pf_labels=args.pf_labels,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        path = args.output_dir / filename
        if args.check:
            if not path.is_file():
                raise SystemExit(f"frozen artifact mismatch: {path}")
            if filename == MANIFEST_FILENAME:
                matches = json.loads(path.read_text(encoding="utf-8")) == json.loads(
                    payload.decode("utf-8")
                )
            else:
                existing = path.read_text(encoding="utf-8-sig")
                existing = existing.replace("\r\n", "\n").replace("\r", "\n")
                matches = existing.encode("utf-8") == payload
            if not matches:
                raise SystemExit(f"frozen artifact mismatch: {path}")
        else:
            path.write_bytes(payload)
        print(f"[v152-one-sided] {filename} sha256={sha256_bytes(payload)}")
    recurrence = manifest["discovery_validation_recurrence"]
    print(
        "[v152-one-sided] PASS "
        f"overlap={recurrence['overlap_heads']}/"
        f"{recurrence['total_selected_heads']} "
        f"exact_layers={recurrence['exact_match_layers']}/{LAYERS} "
        f"median_jaccard={recurrence['median_layer_jaccard']:.3f}"
    )


if __name__ == "__main__":
    main()
