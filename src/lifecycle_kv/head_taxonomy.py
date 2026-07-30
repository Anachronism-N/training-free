from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


PF_WAVE = -1
PF_ANCHOR = 1
PF_VEIL = 2


def _matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def pf_dominant_period(
    sequence: np.ndarray,
    *,
    max_harmonic: int = 4,
    harmonic_decay: float = 0.5,
) -> float:
    """Reimplement PF's published preprocessing and harmonic-folding outline.

    The paper specifies decaying harmonic weights but not their exact values.
    This implementation is therefore diagnostic and must not be described as
    the authors' official classifier.
    """
    values = np.asarray(sequence, dtype=np.float64).reshape(-1)
    if values.size < 4 or not np.isfinite(values).all():
        return float("inf")
    signal = np.diff(values)
    signal = (signal - signal.mean()) * np.hanning(signal.size)
    amplitude = np.abs(np.fft.rfft(signal))
    if amplitude.size <= 1 or float(amplitude[1:].max()) <= 1e-12:
        return float("inf")
    amplitude[0] = 0.0
    folded = np.zeros_like(amplitude)
    for fundamental in range(1, amplitude.size):
        score = 0.0
        weight = 1.0
        for harmonic in range(1, max_harmonic + 1):
            index = fundamental * harmonic
            if index >= amplitude.size:
                break
            score += weight * float(amplitude[index])
            weight *= harmonic_decay
        folded[fundamental] = score
    peak = int(np.argmax(folded[1:]) + 1)
    return float(signal.size / peak)


def pf_tri_pattern_labels(
    historical_logits: np.ndarray,
    *,
    sign_threshold: float = 0.8,
    period_threshold: float = 6.4,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply the published PF Anchor/Wave/Veil decision outline.

    PF does prompt-level classification followed by majority voting. Callers
    should apply this function to each prompt and vote outside this function.
    The periodicity helper is a formula-level reimplementation, not official
    PF profiling code.
    """
    logits = _matrix(historical_logits, name="historical_logits")
    if not 0.5 < sign_threshold <= 1.0:
        raise ValueError("sign_threshold must be in (0.5, 1]")
    positive_rate = (logits > 0).mean(axis=1)
    negative_rate = 1.0 - positive_rate
    periods = np.asarray(
        [pf_dominant_period(row) for row in logits], dtype=np.float64
    )
    labels = np.zeros(logits.shape[0], dtype=np.int64)
    labels[positive_rate >= sign_threshold] = PF_ANCHOR
    labels[negative_rate >= sign_threshold] = PF_VEIL
    undecided = labels == 0
    labels[undecided & (periods < period_threshold)] = PF_WAVE
    undecided = labels == 0
    means = logits.mean(axis=1)
    labels[undecided & (means > 0)] = PF_ANCHOR
    labels[undecided & (means <= 0)] = PF_VEIL
    diagnostics = {
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "dominant_period": periods,
        "mean_logit": means,
    }
    return labels, diagnostics


def forcing_kv_labels(
    recent_mass: np.ndarray,
    non_sink_mass: np.ndarray,
    *,
    threshold: float = 0.8,
) -> np.ndarray:
    """Reproduce the Forcing-KV static/dynamic threshold rule."""
    recent = np.asarray(recent_mass, dtype=np.float64)
    total = np.asarray(non_sink_mass, dtype=np.float64)
    if recent.shape != total.shape or recent.ndim != 1:
        raise ValueError("recent_mass and non_sink_mass must be equal 1D arrays")
    ratio = recent / np.clip(total, 1e-12, None)
    return np.where(ratio >= threshold, "static", "dynamic")


def head_forcing_labels(
    sink_mass: np.ndarray,
    current_mass: np.ndarray,
    *,
    anchor_fraction: float = 0.25,
    local_fraction: float = 0.20,
) -> np.ndarray:
    """Reimplement Head Forcing's percentile Anchor/Local/Memory assignment."""
    sink = np.asarray(sink_mass, dtype=np.float64)
    current = np.asarray(current_mass, dtype=np.float64)
    if sink.shape != current.shape or sink.ndim != 1:
        raise ValueError("sink_mass and current_mass must be equal 1D arrays")
    count = sink.size
    anchor_count = max(1, int(math.ceil(count * anchor_fraction)))
    labels = np.full(count, "memory", dtype=object)
    anchor = np.argsort(-sink, kind="mergesort")[:anchor_count]
    labels[anchor] = "anchor"
    remaining = np.flatnonzero(labels == "memory")
    local_count = max(1, int(math.ceil(remaining.size * local_fraction)))
    local_order = remaining[
        np.argsort(-current[remaining], kind="mergesort")[:local_count]
    ]
    labels[local_order] = "local"
    return labels.astype(str)


def dummy_forcing_labels(
    current_mass: np.ndarray,
    *,
    dummy_fraction: float = 0.25,
) -> np.ndarray:
    current = np.asarray(current_mass, dtype=np.float64)
    if current.ndim != 1:
        raise ValueError("current_mass must be one-dimensional")
    count = max(1, int(math.ceil(current.size * dummy_fraction)))
    labels = np.full(current.size, "context", dtype=object)
    labels[np.argsort(-current, kind="mergesort")[:count]] = "dummy"
    return labels.astype(str)


def robust_fit_transform(
    discovery: np.ndarray,
    validation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    discovery = _matrix(discovery, name="discovery")
    if validation is not None:
        validation = _matrix(validation, name="validation")
        if validation.shape[1] != discovery.shape[1]:
            raise ValueError("discovery and validation feature counts differ")
    center = np.median(discovery, axis=0)
    q25, q75 = np.quantile(discovery, (0.25, 0.75), axis=0)
    scale = q75 - q25
    fallback = np.std(discovery, axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    transformed = (discovery - center) / scale
    transformed_validation = (
        None if validation is None else (validation - center) / scale
    )
    return transformed, transformed_validation, center, scale


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    iterations: int


def assign_clusters(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = _matrix(values, name="values")
    centers = _matrix(centers, name="centers")
    if values.shape[1] != centers.shape[1]:
        raise ValueError("values and centers feature counts differ")
    distance = ((values[:, None] - centers[None, :]) ** 2).sum(axis=2)
    return distance.argmin(axis=1)


def deterministic_kmeans(
    values: np.ndarray,
    clusters: int,
    *,
    restarts: int = 32,
    max_iterations: int = 200,
    seed: int = 20260730,
) -> KMeansResult:
    values = _matrix(values, name="values")
    if not 2 <= clusters < values.shape[0]:
        raise ValueError("clusters must be in [2, sample_count)")
    best: KMeansResult | None = None
    for restart in range(restarts):
        rng = np.random.default_rng(seed + 104729 * restart)
        first = int(rng.integers(0, values.shape[0]))
        chosen = [first]
        nearest = ((values - values[first]) ** 2).sum(axis=1)
        while len(chosen) < clusters:
            total = float(nearest.sum())
            if total <= 1e-12:
                candidate = next(
                    index
                    for index in range(values.shape[0])
                    if index not in chosen
                )
            else:
                candidate = int(rng.choice(values.shape[0], p=nearest / total))
                if candidate in chosen:
                    candidate = int(np.argmax(nearest))
            chosen.append(candidate)
            candidate_distance = (
                (values - values[candidate]) ** 2
            ).sum(axis=1)
            nearest = np.minimum(nearest, candidate_distance)
        centers = values[chosen].copy()
        labels = np.zeros(values.shape[0], dtype=np.int64)
        for iteration in range(1, max_iterations + 1):
            labels = assign_clusters(values, centers)
            updated = centers.copy()
            for cluster in range(clusters):
                members = values[labels == cluster]
                if members.size:
                    updated[cluster] = members.mean(axis=0)
                else:
                    updated[cluster] = values[int(np.argmax(nearest))]
            if np.allclose(updated, centers, rtol=0, atol=1e-9):
                centers = updated
                break
            centers = updated
        labels = assign_clusters(values, centers)
        inertia = float(
            ((values - centers[labels]) ** 2).sum()
        )
        candidate_result = KMeansResult(
            labels=labels,
            centers=centers,
            inertia=inertia,
            iterations=iteration,
        )
        if best is None or candidate_result.inertia < best.inertia:
            best = candidate_result
    assert best is not None
    return best


def align_labels(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("reference and candidate labels must be equal 1D arrays")
    classes = sorted(set(reference.tolist()) | set(candidate.tolist()))
    if len(classes) > 8:
        raise ValueError("brute-force alignment supports at most eight classes")
    best_mapping = {value: value for value in classes}
    best_matches = -1
    for permutation in itertools.permutations(classes):
        mapping = dict(zip(classes, permutation))
        mapped = np.asarray([mapping[value] for value in candidate])
        matches = int((mapped == reference).sum())
        if matches > best_matches:
            best_matches = matches
            best_mapping = mapping
    return np.asarray([best_mapping[value] for value in candidate])


def adjusted_rand_index(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("ARI labels must be equal one-dimensional arrays")
    left_classes = {value: index for index, value in enumerate(np.unique(left))}
    right_classes = {
        value: index for index, value in enumerate(np.unique(right))
    }
    table = np.zeros((len(left_classes), len(right_classes)), dtype=np.int64)
    for left_value, right_value in zip(left, right):
        table[left_classes[left_value], right_classes[right_value]] += 1

    def choose2(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        return float((values * (values - 1) / 2).sum())

    pairs = choose2(np.asarray([left.size]))
    if pairs == 0:
        return 1.0
    observed = choose2(table)
    left_pairs = choose2(table.sum(axis=1))
    right_pairs = choose2(table.sum(axis=0))
    expected = left_pairs * right_pairs / pairs
    maximum = 0.5 * (left_pairs + right_pairs)
    denominator = maximum - expected
    return 1.0 if abs(denominator) < 1e-12 else (
        observed - expected
    ) / denominator


def normalized_mutual_information(
    left: np.ndarray, right: np.ndarray
) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("NMI labels must be equal one-dimensional arrays")
    _, left_inverse = np.unique(left, return_inverse=True)
    _, right_inverse = np.unique(right, return_inverse=True)
    table = np.zeros(
        (left_inverse.max() + 1, right_inverse.max() + 1),
        dtype=np.float64,
    )
    np.add.at(table, (left_inverse, right_inverse), 1)
    table /= table.sum()
    left_prob = table.sum(axis=1)
    right_prob = table.sum(axis=0)
    expected = left_prob[:, None] * right_prob[None, :]
    active = table > 0
    mutual = float((table[active] * np.log(table[active] / expected[active])).sum())

    def entropy(probability: np.ndarray) -> float:
        probability = probability[probability > 0]
        return float(-(probability * np.log(probability)).sum())

    left_entropy = entropy(left_prob)
    right_entropy = entropy(right_prob)
    denominator = math.sqrt(left_entropy * right_entropy)
    if denominator <= 1e-12:
        return 1.0 if left_entropy <= 1e-12 and right_entropy <= 1e-12 else 0.0
    return mutual / denominator


def silhouette_score(values: np.ndarray, labels: np.ndarray) -> float:
    values = _matrix(values, name="values")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (values.shape[0],):
        raise ValueError("silhouette labels have the wrong shape")
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= values.shape[0]:
        return float("nan")
    distance = np.sqrt(
        np.maximum(
            0.0,
            ((values[:, None] - values[None, :]) ** 2).sum(axis=2),
        )
    )
    scores = []
    for index, label in enumerate(labels):
        own = np.flatnonzero(labels == label)
        own = own[own != index]
        if own.size == 0:
            scores.append(0.0)
            continue
        within = float(distance[index, own].mean())
        between = min(
            float(distance[index, labels == other].mean())
            for other in unique
            if other != label
        )
        scores.append((between - within) / max(within, between, 1e-12))
    return float(np.mean(scores))
