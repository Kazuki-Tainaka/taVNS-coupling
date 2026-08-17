"""Paired statistics, BCa bootstrap, and prevalence helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests


ArrayStatistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class BcaResult:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    n_finite_resamples: int
    seed: int
    bias_correction: float
    acceleration: float


def cohens_dz(diff: np.ndarray) -> float:
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    sd = float(np.std(values, ddof=1))
    if sd == 0.0:
        return np.nan
    return float(np.mean(values) / sd)


def median_stat(diff: np.ndarray) -> float:
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def mean_stat(diff: np.ndarray) -> float:
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def bootstrap_values(
    values: np.ndarray,
    statistic: ArrayStatistic,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Return participant-level bootstrap replicates."""
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return np.full(n_resamples, np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(n_resamples, len(sample)))
    output = np.empty(n_resamples, dtype=float)
    for index, row in enumerate(indices):
        output[index] = statistic(sample[row])
    return output


def bca_interval(
    values: np.ndarray,
    statistic: ArrayStatistic,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20_260_805,
) -> tuple[BcaResult, np.ndarray]:
    """Compute a one-sample participant-level BCa confidence interval."""
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    estimate = statistic(sample)
    replicates = bootstrap_values(sample, statistic, n_resamples, seed)
    finite = replicates[np.isfinite(replicates)]
    if len(sample) < 3 or len(finite) < max(100, n_resamples // 2):
        result = BcaResult(
            estimate=estimate,
            lower=np.nan,
            upper=np.nan,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            n_finite_resamples=len(finite),
            seed=seed,
            bias_correction=np.nan,
            acceleration=np.nan,
        )
        return result, replicates

    n_less = float(np.sum(finite < estimate))
    n_equal = float(np.sum(finite == estimate))
    proportion = (n_less + 0.5 * n_equal) / len(finite)
    epsilon = 0.5 / len(finite)
    proportion = float(np.clip(proportion, epsilon, 1.0 - epsilon))
    z0 = float(stats.norm.ppf(proportion))

    jackknife = np.array(
        [statistic(np.delete(sample, index)) for index in range(len(sample))],
        dtype=float,
    )
    jackknife = jackknife[np.isfinite(jackknife)]
    if len(jackknife) != len(sample):
        acceleration = 0.0
    else:
        jack_mean = float(np.mean(jackknife))
        deviations = jack_mean - jackknife
        denominator = 6.0 * float(np.sum(deviations**2) ** 1.5)
        acceleration = (
            float(np.sum(deviations**3) / denominator)
            if denominator > 0.0
            else 0.0
        )

    alpha = 1.0 - confidence_level
    nominal = np.array([alpha / 2.0, 1.0 - alpha / 2.0], dtype=float)
    z_alpha = stats.norm.ppf(nominal)
    denominator = 1.0 - acceleration * (z0 + z_alpha)
    adjusted_z = z0 + (z0 + z_alpha) / denominator
    adjusted = stats.norm.cdf(adjusted_z)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    lower, upper = np.quantile(finite, adjusted, method="linear")

    result = BcaResult(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n_finite_resamples=len(finite),
        seed=seed,
        bias_correction=z0,
        acceleration=acceleration,
    )
    return result, replicates


def wilcoxon_two_sided(diff: np.ndarray) -> tuple[float, float, str]:
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, "not_estimable"
    if np.all(values == 0.0):
        return 0.0, 1.0, "all_zero"
    has_zero = bool(np.any(values == 0.0))
    absolute = np.abs(values[values != 0.0])
    has_tied_absolute_ranks = len(np.unique(absolute)) != len(absolute)
    method_label = (
        "scipy_auto_exact_eligible"
        if not has_zero and not has_tied_absolute_ranks
        else "scipy_auto_ties_or_zero"
    )
    result = stats.wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )
    return float(result.statistic), float(result.pvalue), method_label


def phase_descriptives(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
        }
    q1, q3 = np.quantile(data, [0.25, 0.75], method="linear")
    return {
        "n": len(data),
        "mean": float(np.mean(data)),
        "sd": float(np.std(data, ddof=1)) if len(data) > 1 else np.nan,
        "median": float(np.median(data)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def paired_summary(
    first: np.ndarray,
    second: np.ndarray,
    first_label: str,
    second_label: str,
    seed: int,
    n_resamples: int = 10_000,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    """Summarize first minus second paired values with separate estimands."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[valid]
    second_array = second_array[valid]
    diff = first_array - second_array
    statistic, p_value, wilcoxon_method = wilcoxon_two_sided(diff)

    mean_bca, mean_boot = bca_interval(
        diff,
        mean_stat,
        n_resamples=n_resamples,
        seed=seed,
    )
    median_bca, median_boot = bca_interval(
        diff,
        median_stat,
        n_resamples=n_resamples,
        seed=seed + 1,
    )
    dz_bca, dz_boot = bca_interval(
        diff,
        cohens_dz,
        n_resamples=n_resamples,
        seed=seed + 2,
    )

    output: dict[str, float | int | str] = {
        "contrast": f"{first_label}-{second_label}",
        "n": len(diff),
        "mean_first": float(np.mean(first_array)) if len(diff) else np.nan,
        "mean_second": float(np.mean(second_array)) if len(diff) else np.nan,
        "mean_difference": mean_bca.estimate,
        "mean_difference_ci_low": mean_bca.lower,
        "mean_difference_ci_high": mean_bca.upper,
        "median_difference": median_bca.estimate,
        "median_difference_ci_low": median_bca.lower,
        "median_difference_ci_high": median_bca.upper,
        "cohens_dz": dz_bca.estimate,
        "cohens_dz_ci_low": dz_bca.lower,
        "cohens_dz_ci_high": dz_bca.upper,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_two_sided": p_value,
        "wilcoxon_method": wilcoxon_method,
        "n_negative": int(np.sum(diff < 0.0)),
        "n_positive": int(np.sum(diff > 0.0)),
        "n_zero": int(np.sum(diff == 0.0)),
        "bootstrap_method": "participant_level_BCa",
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed_mean": seed,
        "bootstrap_seed_median": seed + 1,
        "bootstrap_seed_dz": seed + 2,
        "finite_bootstrap_mean": mean_bca.n_finite_resamples,
        "finite_bootstrap_median": median_bca.n_finite_resamples,
        "finite_bootstrap_dz": dz_bca.n_finite_resamples,
    }
    replicates = {
        "mean_difference": mean_boot,
        "median_difference": median_boot,
        "cohens_dz": dz_boot,
    }
    return output, replicates


def exact_binomial_ci(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence_level
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper


def apply_bh(p_values: Sequence[float]) -> np.ndarray:
    data = np.asarray(p_values, dtype=float)
    output = np.full(len(data), np.nan, dtype=float)
    valid = np.isfinite(data)
    if np.any(valid):
        output[valid] = multipletests(data[valid], method="fdr_bh")[1]
    return output


def mcnemar_exact(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if len(a) != len(b):
        raise ValueError("McNemar vectors must have equal length")
    first_only = int(np.sum(a & ~b))
    second_only = int(np.sum(~a & b))
    discordant = first_only + second_only
    p_value = (
        1.0
        if discordant == 0
        else float(
            stats.binomtest(
                first_only,
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    return {
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "statistic": min(first_only, second_only),
        "p_exact_two_sided": p_value,
    }


def cochran_q_test(matrix: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(matrix, dtype=int)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Cochran Q requires subjects by at least three conditions")
    if np.all(data == data[:, [0]]):
        return {
            "n": data.shape[0],
            "k": data.shape[1],
            "statistic": 0.0,
            "p_value": 1.0,
        }
    result = cochrans_q(data)
    return {
        "n": data.shape[0],
        "k": data.shape[1],
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
