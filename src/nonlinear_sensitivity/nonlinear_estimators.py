"""Frozen nonlinear RRI--SBP coupling estimators for Reviewer 1 Comment 6.

The LP and CE implementations follow the nonuniform-embedding framework in
Porta et al. (2014, PLOS ONE 9:e89463).  ``porta_ratio`` retains that paper's
sign convention (coupling is negative); ``directed_strength`` reverses the
sign so that larger values denote stronger lag-dependent predictability or
conditional information transfer.

The SSC/KNNCP implementation follows Porta et al. (2024, Chaos 34:053115).
It is intentionally kept separate because it does not condition on the
target's own past and therefore answers a different state-space
cross-predictability question.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Literal, Sequence

import numpy as np
from scipy import signal


SignalName = Literal["target", "source"]
MethodName = Literal["LP", "CE", "SSC"]


@dataclass(frozen=True, order=True)
class EmbeddingComponent:
    """One candidate coordinate in a nonuniform embedding."""

    signal: SignalName
    lag: int

    @property
    def label(self) -> str:
        return f"{self.signal}[t-{self.lag}]"


@dataclass
class EstimatorResult:
    """Common machine-readable output from a directional estimator."""

    method: MethodName
    finite: bool
    failure_reason: str
    reduced_score: float
    full_score: float
    directed_strength: float
    porta_ratio: float
    reduced_raw: float
    full_raw: float
    reduced_predictability_r2: float
    full_predictability_r2: float
    target_entropy: float
    tolerance: float
    selected_embedding: str
    selected_dimension: int
    n_input: int
    n_vectors: int
    k: int
    lag_depth: int
    theiler: int
    same_beat_included: bool
    score_curve: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON/CSV-friendly values."""

        return asdict(self)


def detrend_zscore(values: Sequence[float]) -> tuple[np.ndarray, float, float]:
    """Linearly detrend and sample-SD normalise a finite one-dimensional series.

    Returns the normalised series, raw linear slope per beat, and raw sample SD.
    No missing-value interpolation is performed.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 3 or not np.all(np.isfinite(array)):
        return np.full_like(array, np.nan), np.nan, np.nan
    beat_index = np.arange(len(array), dtype=float)
    slope, intercept = np.polyfit(beat_index, array, 1)
    detrended = array - (intercept + slope * beat_index)
    sample_sd = float(np.std(detrended, ddof=1))
    if not np.isfinite(sample_sd) or sample_sd <= 0.0:
        return np.full_like(array, np.nan), float(slope), sample_sd
    normalised = (detrended - float(np.mean(detrended))) / sample_sd
    return normalised, float(slope), float(np.std(array, ddof=1))


def _empty_result(
    method: MethodName,
    reason: str,
    n_input: int,
    k: int,
    lag_depth: int,
    theiler: int,
    same_beat_included: bool,
) -> EstimatorResult:
    return EstimatorResult(
        method=method,
        finite=False,
        failure_reason=reason,
        reduced_score=np.nan,
        full_score=np.nan,
        directed_strength=np.nan,
        porta_ratio=np.nan,
        reduced_raw=np.nan,
        full_raw=np.nan,
        reduced_predictability_r2=np.nan,
        full_predictability_r2=np.nan,
        target_entropy=np.nan,
        tolerance=np.nan,
        selected_embedding="[]",
        selected_dimension=0,
        n_input=n_input,
        n_vectors=0,
        k=k,
        lag_depth=lag_depth,
        theiler=theiler,
        same_beat_included=same_beat_included,
        score_curve="[]",
    )


def _candidate_order(target_lags: Sequence[int], source_lags: Sequence[int]) -> list[EmbeddingComponent]:
    """Return the frozen target-first, increasing-lag candidate order."""

    target_components = [
        EmbeddingComponent("target", int(lag))
        for lag in sorted(set(target_lags))
    ]
    source_components = [
        EmbeddingComponent("source", int(lag))
        for lag in sorted(set(source_lags))
    ]
    return target_components + source_components


def _aligned_values(
    target: np.ndarray,
    source: np.ndarray,
    target_lags: Sequence[int],
    source_lags: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, dict[EmbeddingComponent, np.ndarray]]:
    """Build aligned target values and every frozen candidate coordinate."""

    all_lags = list(target_lags) + list(source_lags)
    maximum_lag = max(all_lags) if all_lags else 0
    reference_indices = np.arange(maximum_lag, len(target), dtype=int)
    aligned_target = target[reference_indices]
    features: dict[EmbeddingComponent, np.ndarray] = {}
    for component in _candidate_order(target_lags, source_lags):
        parent = target if component.signal == "target" else source
        features[component] = parent[reference_indices - component.lag]
    return reference_indices, aligned_target, features


def _component_distance(feature: np.ndarray) -> np.ndarray:
    return np.abs(feature[:, None] - feature[None, :])


def _distance_for_components(
    components: Sequence[EmbeddingComponent],
    component_distances: dict[EmbeddingComponent, np.ndarray],
    n_vectors: int,
) -> np.ndarray | None:
    if not components:
        return None
    distance = np.zeros((n_vectors, n_vectors), dtype=float)
    for component in components:
        distance = np.maximum(distance, component_distances[component])
    return distance


def _eligible_knn(
    distance: np.ndarray,
    reference_indices: np.ndarray,
    k: int,
    theiler: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic k-neighbour indices and original distances."""

    n_vectors = distance.shape[0]
    if distance.shape != (n_vectors, n_vectors):
        raise ValueError("distance matrix must be square")
    exclusion = (
        np.abs(reference_indices[:, None] - reference_indices[None, :])
        <= theiler
    )
    eligible_count = n_vectors - exclusion.sum(axis=1)
    if np.any(eligible_count < k):
        raise ValueError("fewer eligible neighbours than k")

    masked = np.where(exclusion, np.inf, distance)
    finite_scale = float(np.nanmax(np.where(np.isfinite(masked), masked, np.nan)))
    if not np.isfinite(finite_scale):
        finite_scale = 1.0
    tie_step = np.finfo(float).eps * max(1.0, finite_scale)
    ordering_distance = masked + tie_step * np.arange(n_vectors)[None, :]
    indices = np.argpartition(ordering_distance, kth=k - 1, axis=1)[:, :k]
    order_values = np.take_along_axis(ordering_distance, indices, axis=1)
    local_order = np.argsort(order_values, axis=1, kind="stable")
    indices = np.take_along_axis(indices, local_order, axis=1)
    neighbour_distances = np.take_along_axis(masked, indices, axis=1)
    return indices, neighbour_distances


def _inverse_distance_prediction(
    target_values: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_distances: np.ndarray,
) -> np.ndarray:
    neighbour_targets = target_values[neighbour_indices]
    zero_mask = neighbour_distances <= np.finfo(float).eps
    has_zero = zero_mask.any(axis=1)
    weights = np.empty_like(neighbour_distances)
    weights[has_zero] = zero_mask[has_zero].astype(float)
    weights[~has_zero] = 1.0 / neighbour_distances[~has_zero]
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights * neighbour_targets, axis=1)


def _squared_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    if (
        len(observed) < 3
        or not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(predicted))
        or np.std(observed, ddof=1) <= 0.0
        or np.std(predicted, ddof=1) <= 0.0
    ):
        return 0.0
    correlation = float(np.corrcoef(observed, predicted)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation * correlation, 0.0, 1.0))


def _lp_r2(
    distance: np.ndarray | None,
    target_values: np.ndarray,
    reference_indices: np.ndarray,
    k: int,
    theiler: int,
) -> float:
    if distance is None:
        return 0.0
    neighbours, neighbour_distances = _eligible_knn(
        distance, reference_indices, k, theiler
    )
    predicted = _inverse_distance_prediction(
        target_values, neighbours, neighbour_distances
    )
    return _squared_correlation(target_values, predicted)


def _correlation_probability(values: np.ndarray, tolerance: float) -> float:
    """SampEn-style probability for unordered distinct pairs."""

    if len(values) < 2 or not np.isfinite(tolerance) or tolerance <= 0.0:
        return np.nan
    upper = np.triu_indices(len(values), k=1)
    pair_distance = np.abs(values[:, None] - values[None, :])[upper]
    if len(pair_distance) == 0:
        return np.nan
    return float(np.mean(pair_distance < tolerance))


def _conditional_entropy(
    distance: np.ndarray | None,
    target_values: np.ndarray,
    reference_indices: np.ndarray,
    k: int,
    theiler: int,
    tolerance: float,
    target_entropy: float,
) -> float:
    if distance is None:
        return target_entropy
    neighbours, _ = _eligible_knn(distance, reference_indices, k, theiler)
    neighbour_targets = target_values[neighbours]
    left, right = np.triu_indices(k, k=1)
    probabilities = np.mean(
        np.abs(neighbour_targets[:, left] - neighbour_targets[:, right])
        < tolerance,
        axis=1,
    )
    if np.any(probabilities <= 0.0) or not np.all(np.isfinite(probabilities)):
        return np.inf
    return float(np.mean(-np.log(probabilities)))


def _remove_newer_candidates(
    candidates: Sequence[EmbeddingComponent],
    selected: EmbeddingComponent,
) -> list[EmbeddingComponent]:
    """Apply the frozen Porta nonuniform-embedding pruning rule."""

    return [
        candidate
        for candidate in candidates
        if not (
            candidate.signal == selected.signal
            and candidate.lag <= selected.lag
        )
    ]


def _json_components(components: Sequence[EmbeddingComponent]) -> str:
    return json.dumps([component.label for component in components], separators=(",", ":"))


def estimate_porta_lp(
    target: Sequence[float],
    source: Sequence[float],
    *,
    k: int = 30,
    lag_depth: int = 8,
    theiler: int = 8,
    source_lag_zero: bool = False,
) -> EstimatorResult:
    """Estimate Porta-style local-predictability coupling strength."""

    target_array = np.asarray(target, dtype=float)
    source_array = np.asarray(source, dtype=float)
    n_input = min(len(target_array), len(source_array))
    target_array = target_array[:n_input]
    source_array = source_array[:n_input]
    if (
        n_input <= lag_depth + 2 * theiler + k
        or not np.all(np.isfinite(target_array))
        or not np.all(np.isfinite(source_array))
    ):
        return _empty_result(
            "LP", "invalid_or_short_input", n_input, k, lag_depth, theiler, source_lag_zero
        )

    target_lags = list(range(1, lag_depth + 1))
    source_lags = list(range(0 if source_lag_zero else 1, lag_depth + 1))
    reference_indices, target_values, features = _aligned_values(
        target_array, source_array, target_lags, source_lags
    )
    candidates = _candidate_order(target_lags, source_lags)
    component_distances = {
        component: _component_distance(feature)
        for component, feature in features.items()
    }
    current_distance: np.ndarray | None = None
    selected: list[EmbeddingComponent] = []
    curve: list[dict[str, object]] = [{"q": 0, "r2": 0.0, "embedding": []}]

    try:
        while candidates:
            best_component: EmbeddingComponent | None = None
            best_distance: np.ndarray | None = None
            best_r2 = -np.inf
            for candidate in candidates:
                candidate_distance = component_distances[candidate]
                proposed = (
                    candidate_distance
                    if current_distance is None
                    else np.maximum(current_distance, candidate_distance)
                )
                r2 = _lp_r2(
                    proposed, target_values, reference_indices, k, theiler
                )
                if r2 > best_r2 + 1e-15:
                    best_component = candidate
                    best_distance = proposed
                    best_r2 = r2
            if best_component is None or best_distance is None:
                raise RuntimeError("candidate selection failed")
            selected.append(best_component)
            current_distance = best_distance
            curve.append(
                {
                    "q": len(selected),
                    "r2": float(best_r2),
                    "embedding": [item.label for item in selected],
                }
            )
            candidates = _remove_newer_candidates(candidates, best_component)

        r2_values = np.asarray([float(row["r2"]) for row in curve])
        optimal_q = int(np.flatnonzero(r2_values == np.nanmax(r2_values))[0])
        optimal = selected[:optimal_q]
        full_distance = _distance_for_components(
            optimal, component_distances, len(target_values)
        )
        reduced = [item for item in optimal if item.signal == "target"]
        reduced_distance = _distance_for_components(
            reduced, component_distances, len(target_values)
        )
        full_r2 = _lp_r2(
            full_distance, target_values, reference_indices, k, theiler
        )
        reduced_r2 = _lp_r2(
            reduced_distance, target_values, reference_indices, k, theiler
        )
        full_nci = 1.0 - full_r2
        reduced_nci = 1.0 - reduced_r2
        if reduced_nci <= 0.0 or not np.isfinite(full_nci + reduced_nci):
            raise FloatingPointError("non-positive or non-finite reduced NCI")
        porta_ratio = (full_nci - reduced_nci) / reduced_nci
        strength = -porta_ratio
    except (ValueError, FloatingPointError, RuntimeError) as error:
        return _empty_result(
            "LP", str(error), n_input, k, lag_depth, theiler, source_lag_zero
        )

    return EstimatorResult(
        method="LP",
        finite=bool(np.isfinite(strength)),
        failure_reason="NA",
        reduced_score=float(reduced_nci),
        full_score=float(full_nci),
        directed_strength=float(strength),
        porta_ratio=float(porta_ratio),
        reduced_raw=float(reduced_nci),
        full_raw=float(full_nci),
        reduced_predictability_r2=float(reduced_r2),
        full_predictability_r2=float(full_r2),
        target_entropy=np.nan,
        tolerance=np.nan,
        selected_embedding=_json_components(optimal),
        selected_dimension=optimal_q,
        n_input=n_input,
        n_vectors=len(target_values),
        k=k,
        lag_depth=lag_depth,
        theiler=theiler,
        same_beat_included=source_lag_zero,
        score_curve=json.dumps(curve, separators=(",", ":")),
    )


def estimate_porta_ce(
    target: Sequence[float],
    source: Sequence[float],
    *,
    k: int = 30,
    lag_depth: int = 8,
    theiler: int = 8,
    source_lag_zero: bool = False,
) -> EstimatorResult:
    """Estimate Porta-style conditional-entropy coupling strength."""

    target_array = np.asarray(target, dtype=float)
    source_array = np.asarray(source, dtype=float)
    n_input = min(len(target_array), len(source_array))
    target_array = target_array[:n_input]
    source_array = source_array[:n_input]
    if (
        n_input <= lag_depth + 2 * theiler + k
        or not np.all(np.isfinite(target_array))
        or not np.all(np.isfinite(source_array))
    ):
        return _empty_result(
            "CE", "invalid_or_short_input", n_input, k, lag_depth, theiler, source_lag_zero
        )

    target_lags = list(range(1, lag_depth + 1))
    source_lags = list(range(0 if source_lag_zero else 1, lag_depth + 1))
    reference_indices, target_values, features = _aligned_values(
        target_array, source_array, target_lags, source_lags
    )
    tolerance = 0.10 * (
        float(np.percentile(target_values, 84.0))
        - float(np.percentile(target_values, 16.0))
    )
    probability = _correlation_probability(target_values, tolerance)
    if not np.isfinite(probability) or probability <= 0.0:
        return _empty_result(
            "CE", "non-positive target correlation probability", n_input, k,
            lag_depth, theiler, source_lag_zero
        )
    target_entropy = float(-np.log(probability))
    if not np.isfinite(target_entropy) or target_entropy <= 0.0:
        return _empty_result(
            "CE", "non-positive target entropy", n_input, k, lag_depth,
            theiler, source_lag_zero
        )

    candidates = _candidate_order(target_lags, source_lags)
    component_distances = {
        component: _component_distance(feature)
        for component, feature in features.items()
    }
    current_distance: np.ndarray | None = None
    selected: list[EmbeddingComponent] = []
    curve: list[dict[str, object]] = [
        {"q": 0, "ce": target_entropy, "embedding": []}
    ]

    try:
        while candidates:
            best_component: EmbeddingComponent | None = None
            best_distance: np.ndarray | None = None
            best_ce = np.inf
            for candidate in candidates:
                candidate_distance = component_distances[candidate]
                proposed = (
                    candidate_distance
                    if current_distance is None
                    else np.maximum(current_distance, candidate_distance)
                )
                ce = _conditional_entropy(
                    proposed,
                    target_values,
                    reference_indices,
                    k,
                    theiler,
                    tolerance,
                    target_entropy,
                )
                if ce < best_ce - 1e-15:
                    best_component = candidate
                    best_distance = proposed
                    best_ce = ce
            if best_component is None or best_distance is None:
                raise FloatingPointError("all candidate conditional entropies are non-finite")
            selected.append(best_component)
            current_distance = best_distance
            curve.append(
                {
                    "q": len(selected),
                    "ce": float(best_ce),
                    "embedding": [item.label for item in selected],
                }
            )
            candidates = _remove_newer_candidates(candidates, best_component)

        ce_values = np.asarray([float(row["ce"]) for row in curve])
        optimal_q = int(np.flatnonzero(ce_values == np.nanmin(ce_values))[0])
        optimal = selected[:optimal_q]
        full_distance = _distance_for_components(
            optimal, component_distances, len(target_values)
        )
        reduced = [item for item in optimal if item.signal == "target"]
        reduced_distance = _distance_for_components(
            reduced, component_distances, len(target_values)
        )
        full_ce = _conditional_entropy(
            full_distance,
            target_values,
            reference_indices,
            k,
            theiler,
            tolerance,
            target_entropy,
        )
        reduced_ce = _conditional_entropy(
            reduced_distance,
            target_values,
            reference_indices,
            k,
            theiler,
            tolerance,
            target_entropy,
        )
        full_nci = full_ce / target_entropy
        reduced_nci = reduced_ce / target_entropy
        if reduced_nci <= 0.0 or not np.isfinite(full_nci + reduced_nci):
            raise FloatingPointError("non-positive or non-finite reduced CE NCI")
        porta_ratio = (full_nci - reduced_nci) / reduced_nci
        strength = -porta_ratio
    except (ValueError, FloatingPointError, RuntimeError) as error:
        return _empty_result(
            "CE", str(error), n_input, k, lag_depth, theiler, source_lag_zero
        )

    return EstimatorResult(
        method="CE",
        finite=bool(np.isfinite(strength)),
        failure_reason="NA",
        reduced_score=float(reduced_nci),
        full_score=float(full_nci),
        directed_strength=float(strength),
        porta_ratio=float(porta_ratio),
        reduced_raw=float(reduced_ce),
        full_raw=float(full_ce),
        reduced_predictability_r2=np.nan,
        full_predictability_r2=np.nan,
        target_entropy=float(target_entropy),
        tolerance=float(tolerance),
        selected_embedding=_json_components(optimal),
        selected_dimension=optimal_q,
        n_input=n_input,
        n_vectors=len(target_values),
        k=k,
        lag_depth=lag_depth,
        theiler=theiler,
        same_beat_included=source_lag_zero,
        score_curve=json.dumps(curve, separators=(",", ":")),
    )


def _simplex_prediction(
    target_values: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_distances: np.ndarray,
) -> np.ndarray:
    """Sugihara--May exponential simplex weighting used by SSC/KNNCP."""

    neighbour_targets = target_values[neighbour_indices]
    zero_mask = neighbour_distances <= np.finfo(float).eps
    has_zero = zero_mask.any(axis=1)
    weights = np.empty_like(neighbour_distances)
    weights[has_zero] = zero_mask[has_zero].astype(float)
    positive = ~has_zero
    if np.any(positive):
        minimum = neighbour_distances[positive, 0][:, None]
        minimum = np.maximum(minimum, np.finfo(float).eps)
        weights[positive] = np.exp(-neighbour_distances[positive] / minimum)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights * neighbour_targets, axis=1)


def estimate_ssc_knncp(
    target: Sequence[float],
    source: Sequence[float],
    *,
    k: int = 20,
    maximum_m: int = 15,
) -> EstimatorResult:
    """Estimate exact-publication SSC/KNNCP cross-predictability index."""

    target_array = np.asarray(target, dtype=float)
    source_array = np.asarray(source, dtype=float)
    n_input = min(len(target_array), len(source_array))
    target_array = target_array[:n_input]
    source_array = source_array[:n_input]
    if (
        n_input <= maximum_m + k
        or not np.all(np.isfinite(target_array))
        or not np.all(np.isfinite(source_array))
    ):
        return _empty_result(
            "SSC", "invalid_or_short_input", n_input, k, maximum_m, 0, False
        )

    curve: list[dict[str, float | int]] = [{"m": 1, "cpf_r2": 0.0}]
    try:
        for m in range(2, maximum_m + 1):
            lag_count = m - 1
            reference_indices = np.arange(lag_count, n_input, dtype=int)
            embedding = np.column_stack(
                [
                    source_array[reference_indices - lag]
                    for lag in range(1, lag_count + 1)
                ]
            )
            target_values = target_array[reference_indices]
            squared = np.sum(
                (embedding[:, None, :] - embedding[None, :, :]) ** 2,
                axis=2,
            )
            distance = np.sqrt(np.maximum(squared, 0.0))
            np.fill_diagonal(distance, np.inf)
            if distance.shape[0] - 1 < k:
                raise ValueError("fewer SSC neighbours than k")
            finite_scale = float(
                np.nanmax(np.where(np.isfinite(distance), distance, np.nan))
            )
            tie_step = np.finfo(float).eps * max(1.0, finite_scale)
            ordering = distance + tie_step * np.arange(len(distance))[None, :]
            neighbours = np.argpartition(ordering, kth=k - 1, axis=1)[:, :k]
            local_order = np.argsort(
                np.take_along_axis(ordering, neighbours, axis=1),
                axis=1,
                kind="stable",
            )
            neighbours = np.take_along_axis(neighbours, local_order, axis=1)
            neighbour_distances = np.take_along_axis(
                distance, neighbours, axis=1
            )
            predicted = _simplex_prediction(
                target_values, neighbours, neighbour_distances
            )
            curve.append(
                {"m": m, "cpf_r2": _squared_correlation(target_values, predicted)}
            )
        values = np.asarray([float(row["cpf_r2"]) for row in curve])
        best_index = int(np.flatnonzero(values == np.nanmax(values))[0])
        best_m = int(curve[best_index]["m"])
        cpi = float(values[best_index])
    except (ValueError, FloatingPointError, RuntimeError) as error:
        return _empty_result(
            "SSC", str(error), n_input, k, maximum_m, 0, False
        )

    embedding = [f"source[t-{lag}]" for lag in range(1, best_m)]
    return EstimatorResult(
        method="SSC",
        finite=bool(np.isfinite(cpi)),
        failure_reason="NA",
        reduced_score=np.nan,
        full_score=float(1.0 - cpi),
        directed_strength=cpi,
        porta_ratio=np.nan,
        reduced_raw=np.nan,
        full_raw=cpi,
        reduced_predictability_r2=np.nan,
        full_predictability_r2=cpi,
        target_entropy=np.nan,
        tolerance=np.nan,
        selected_embedding=json.dumps(embedding, separators=(",", ":")),
        selected_dimension=max(0, best_m - 1),
        n_input=n_input,
        n_vectors=n_input - max(0, best_m - 1),
        k=k,
        lag_depth=maximum_m,
        theiler=0,
        same_beat_included=False,
        score_curve=json.dumps(curve, separators=(",", ":")),
    )


def estimate_direction(
    method: MethodName,
    target: Sequence[float],
    source: Sequence[float],
    *,
    k: int,
    lag_depth: int,
    theiler: int = 8,
    source_lag_zero: bool = False,
) -> EstimatorResult:
    """Dispatch a frozen directional estimator."""

    if method == "LP":
        return estimate_porta_lp(
            target,
            source,
            k=k,
            lag_depth=lag_depth,
            theiler=theiler,
            source_lag_zero=source_lag_zero,
        )
    if method == "CE":
        return estimate_porta_ce(
            target,
            source,
            k=k,
            lag_depth=lag_depth,
            theiler=theiler,
            source_lag_zero=source_lag_zero,
        )
    if method == "SSC":
        return estimate_ssc_knncp(
            target,
            source,
            k=k,
            maximum_m=lag_depth,
        )
    raise ValueError(f"Unknown method: {method}")


def circular_shift_offsets(
    length: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw frozen admissible circular-shift offsets with replacement."""

    minimum = max(32, int(np.ceil(length / 10.0)))
    maximum = length - minimum
    if maximum < minimum:
        raise ValueError("series too short for frozen circular shifts")
    allowed = np.arange(minimum, maximum + 1, dtype=int)
    return rng.choice(allowed, size=count, replace=True)


def surrogate_p_value(observed: float, surrogates: Sequence[float]) -> float:
    """Finite-sample one-sided Monte Carlo p value for positive strength."""

    surrogate_array = np.asarray(surrogates, dtype=float)
    finite = surrogate_array[np.isfinite(surrogate_array)]
    if not np.isfinite(observed) or len(finite) != len(surrogate_array):
        return np.nan
    return float((1 + np.count_nonzero(finite >= observed)) / (len(finite) + 1))


def surrogate_significant(observed: float, surrogates: Sequence[float]) -> tuple[bool, float, float]:
    """Apply the frozen strict 95th-percentile criterion."""

    surrogate_array = np.asarray(surrogates, dtype=float)
    if not np.isfinite(observed) or not np.all(np.isfinite(surrogate_array)):
        return False, np.nan, np.nan
    threshold = float(np.quantile(surrogate_array, 0.95, method="higher"))
    return bool(observed > threshold), threshold, surrogate_p_value(observed, surrogate_array)


__all__ = [
    "EmbeddingComponent",
    "EstimatorResult",
    "circular_shift_offsets",
    "detrend_zscore",
    "estimate_direction",
    "estimate_porta_ce",
    "estimate_porta_lp",
    "estimate_ssc_knncp",
    "surrogate_p_value",
    "surrogate_significant",
]
