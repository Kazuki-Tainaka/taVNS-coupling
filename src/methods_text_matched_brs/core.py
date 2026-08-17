"""Core data, sequence-BRS, and statistical utilities for the R1 analysis."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import stats


RELEASE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(
    os.environ.get(
        "TAVNS_METHODS_BRS_OUTPUT_ROOT",
        RELEASE_ROOT / "controlled_outputs" / "methods_text_matched_brs",
    )
).resolve()
REVISION_ROOT = Path(
    os.environ.get("TAVNS_DATA_ROOT", RELEASE_ROOT / "controlled_data_not_included")
).resolve()
REVISION_ROOT = REVISION_ROOT / "revision_reference"
PROJECT_ROOT = Path(
    os.environ.get(
        "TAVNS_DATA_ROOT",
        RELEASE_ROOT / "controlled_data_not_included",
    )
).resolve()
PAIRED_DIR = PROJECT_ROOT / "paired"
TABLES_DIR = PACKAGE_ROOT / "outputs" / "tables"
FIGURES_DIR = PACKAGE_ROOT / "outputs" / "figures"
LOGS_DIR = PACKAGE_ROOT / "outputs" / "logs"

SUBJECTS = tuple(range(1, 19))
PHASES = {
    "Pre": (0.0, 300.0),
    "Stim": (300.0, 600.0),
    "Post": (600.0, 900.0),
}
SUBPHASES = {
    "Pre_early": (0.0, 150.0),
    "Pre_late": (150.0, 300.0),
    "Stim_early": (300.0, 450.0),
    "Stim_late": (450.0, 600.0),
    "Post_early": (600.0, 750.0),
    "Post_late": (750.0, 900.0),
}
PHASE_ORDER = tuple(PHASES)
SUBPHASE_ORDER = tuple(SUBPHASES)
PAIRED_COLUMNS = [
    "R_wave_timing_ms",
    "RRI_ms",
    "SBP_peak_timing_ms",
    "SBP_stored_V",
    "PAT_ms",
]

MASTER_SEED = 2_026_080_707
ANTONINO_BCA_BASE = 2_026_080_710
ASSOCIATION_BOOTSTRAP_SEED = 2_026_080_801
ASSOCIATION_PERMUTATION_SEED = 2_026_080_802
THEILSEN_BOOTSTRAP_SEED = 2_026_080_803
PARTIAL_SPEARMAN_SEED = 2_026_080_804
SUBPHASE_BCA_BASE = 2_026_081_000
BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_RESAMPLES = 100_000
IMPLEMENTATION_VERSION = "1.0.0"

Direction = Literal["all", "up", "down"]


@dataclass(frozen=True)
class BranchConfig:
    """Prespecified sequence-method branch."""

    branch: str
    method_family: str
    lag_beats: int
    enumeration: Literal["maximal", "all_contiguous_subramps"]
    sbp_step_threshold: float
    sbp_threshold_strict: bool
    rri_direction_required: bool
    rri_step_threshold: float | None
    rri_threshold_strict: bool
    correlation_threshold: float | None
    correlation_inclusive: bool
    r2_threshold: float | None
    r2_strict: bool
    positive_slope_required: bool
    minimum_length: int = 3

    @property
    def exact_rule_set(self) -> str:
        if self.branch == "REF":
            return (
                "lag1;maximal;SBP_step_abs>=1.0;same_SBP_direction;len>=3;"
                "Pearson_r>=0.80;finite_slope;no_RRI_direction_gate;mean_slopes"
            )
        return (
            f"lag{self.lag_beats};{self.enumeration};SBP_step_abs>1.0;"
            "RRI_step_abs>1.0;same_direction;len>=3;OLS_RRI_on_SBP;"
            "positive_finite_slope;R2>0.85;mean_slopes"
        )


BRANCHES = {
    "REF": BranchConfig(
        branch="REF",
        method_family="CURRENT_SUBMISSION_REFERENCE_LEGACY_CORRELATION_ONLY",
        lag_beats=1,
        enumeration="maximal",
        sbp_step_threshold=1.0,
        sbp_threshold_strict=False,
        rri_direction_required=False,
        rri_step_threshold=None,
        rri_threshold_strict=False,
        correlation_threshold=0.80,
        correlation_inclusive=True,
        r2_threshold=None,
        r2_strict=False,
        positive_slope_required=False,
    ),
    "A0_MAX": BranchConfig(
        branch="A0_MAX",
        method_family="ANTONINO_METHODS_TEXT_MATCHED",
        lag_beats=0,
        enumeration="maximal",
        sbp_step_threshold=1.0,
        sbp_threshold_strict=True,
        rri_direction_required=True,
        rri_step_threshold=1.0,
        rri_threshold_strict=True,
        correlation_threshold=None,
        correlation_inclusive=False,
        r2_threshold=0.85,
        r2_strict=True,
        positive_slope_required=True,
    ),
    "A1_MAX": BranchConfig(
        branch="A1_MAX",
        method_family="ANTONINO_METHODS_TEXT_MATCHED",
        lag_beats=1,
        enumeration="maximal",
        sbp_step_threshold=1.0,
        sbp_threshold_strict=True,
        rri_direction_required=True,
        rri_step_threshold=1.0,
        rri_threshold_strict=True,
        correlation_threshold=None,
        correlation_inclusive=False,
        r2_threshold=0.85,
        r2_strict=True,
        positive_slope_required=True,
    ),
    "A0_OVERLAP": BranchConfig(
        branch="A0_OVERLAP",
        method_family="ANTONINO_OVERLAP_SENSITIVITY",
        lag_beats=0,
        enumeration="all_contiguous_subramps",
        sbp_step_threshold=1.0,
        sbp_threshold_strict=True,
        rri_direction_required=True,
        rri_step_threshold=1.0,
        rri_threshold_strict=True,
        correlation_threshold=None,
        correlation_inclusive=False,
        r2_threshold=0.85,
        r2_strict=True,
        positive_slope_required=True,
    ),
    "A1_OVERLAP": BranchConfig(
        branch="A1_OVERLAP",
        method_family="ANTONINO_OVERLAP_SENSITIVITY",
        lag_beats=1,
        enumeration="all_contiguous_subramps",
        sbp_step_threshold=1.0,
        sbp_threshold_strict=True,
        rri_direction_required=True,
        rri_step_threshold=1.0,
        rri_threshold_strict=True,
        correlation_threshold=None,
        correlation_inclusive=False,
        r2_threshold=0.85,
        r2_strict=True,
        positive_slope_required=True,
    ),
}


def ensure_output_dirs() -> None:
    """Create only package-local output directories."""
    package = PACKAGE_ROOT.resolve()
    for path in (TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        resolved = path.resolve()
        if package not in resolved.parents:
            raise RuntimeError(f"Unsafe output directory: {resolved}")
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base: int, *keys: object) -> int:
    """Return a portable key-derived NumPy seed."""
    key = "|".join(str(value) for value in keys).encode("utf-8")
    return int((base + zlib.crc32(key)) % (2**32 - 1))


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "statsmodels": statsmodels.__version__,
        "implementation_version": IMPLEMENTATION_VERSION,
    }


def load_subject(subject: int) -> pd.DataFrame:
    """Load one retained native paired-beat file without changing it."""
    path = PAIRED_DIR / f"paired_beats_{subject:02d}.csv"
    frame = pd.read_csv(path, header=None, names=PAIRED_COLUMNS)
    frame.insert(0, "source_row", np.arange(len(frame), dtype=int))
    frame.insert(0, "subject", subject)
    frame["R_wave_time_s"] = frame["R_wave_timing_ms"].astype(float) / 1000.0
    frame["SBP_peak_time_s"] = (
        frame["SBP_peak_timing_ms"].astype(float) / 1000.0
    )
    frame["SBP_mmHg"] = frame["SBP_stored_V"].astype(float) * 100.0
    return frame


def clean_and_segment(
    frame: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> pd.DataFrame:
    """Apply current cleaning and a half-open R-wave-time interval."""
    valid = (
        np.isfinite(frame["RRI_ms"].to_numpy(float))
        & np.isfinite(frame["SBP_mmHg"].to_numpy(float))
        & (frame["RRI_ms"].to_numpy(float) > 0.0)
    )
    in_window = (frame["R_wave_time_s"] >= start_s) & (
        frame["R_wave_time_s"] < end_s
    )
    output = frame.loc[valid & in_window].copy()
    return output.sort_values("R_wave_time_s").reset_index(drop=True)


def _step_qualifies(value: float, threshold: float, strict: bool) -> bool:
    return bool(value > threshold if strict else value >= threshold)


def detect_maximal_ramps(
    sbp: np.ndarray,
    direction: Literal["up", "down"],
    minimum_length: int,
    threshold: float,
    strict: bool,
) -> list[tuple[int, int]]:
    """Detect maximal, non-overlapping thresholded SBP ramps."""
    values = np.asarray(sbp, dtype=float)
    diffs = np.diff(values)
    sign = 1.0 if direction == "up" else -1.0
    output: list[tuple[int, int]] = []
    index = 0
    while index < len(diffs):
        signed = sign * diffs[index]
        if not _step_qualifies(signed, threshold, strict):
            index += 1
            continue
        end_diff = index + 1
        while end_diff < len(diffs):
            signed_next = sign * diffs[end_diff]
            if not _step_qualifies(signed_next, threshold, strict):
                break
            end_diff += 1
        end_beat = end_diff
        if end_beat - index + 1 >= minimum_length:
            output.append((index, end_beat))
        index = end_diff
    return output


def enumerate_candidates(
    maximal_ramps: list[tuple[int, int]],
    minimum_length: int,
    enumeration: str,
) -> list[tuple[int, int, int]]:
    """Return candidate start/end/maximal-ramp index triples."""
    output: list[tuple[int, int, int]] = []
    for ramp_id, (start, end) in enumerate(maximal_ramps, start=1):
        if enumeration == "maximal":
            output.append((start, end, ramp_id))
            continue
        for sub_start in range(start, end - minimum_length + 2):
            for sub_end in range(sub_start + minimum_length - 1, end + 1):
                output.append((sub_start, sub_end, ramp_id))
    return output


def _array_text(values: np.ndarray) -> str:
    return "|".join(f"{float(value):.15g}" for value in values)


def evaluate_branch(
    frame: pd.DataFrame,
    subject: int,
    window: str,
    start_s: float,
    end_s: float,
    config: BranchConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate one branch within one already segmented interval."""
    n = len(frame)
    lag = config.lag_beats
    if n <= lag:
        sbp_frame = frame.iloc[0:0].copy().reset_index(drop=True)
        rri_frame = frame.iloc[0:0].copy().reset_index(drop=True)
    elif lag == 0:
        sbp_frame = frame.iloc[:n].reset_index(drop=True)
        rri_frame = frame.iloc[:n].reset_index(drop=True)
    else:
        sbp_frame = frame.iloc[: n - lag].reset_index(drop=True)
        rri_frame = frame.iloc[lag:n].reset_index(drop=True)

    sbp = sbp_frame["SBP_mmHg"].to_numpy(float)
    rri = rri_frame["RRI_ms"].to_numpy(float)
    sequence_rows: list[dict[str, object]] = []
    unique_maximal_counts = {"up": 0, "down": 0}

    if len(sbp) >= config.minimum_length:
        for direction in ("up", "down"):
            maximal = detect_maximal_ramps(
                sbp,
                direction=direction,
                minimum_length=config.minimum_length,
                threshold=config.sbp_step_threshold,
                strict=config.sbp_threshold_strict,
            )
            unique_maximal_counts[direction] = len(maximal)
            candidates = enumerate_candidates(
                maximal,
                config.minimum_length,
                config.enumeration,
            )
            direction_sign = 1.0 if direction == "up" else -1.0
            for candidate_id, (start, end, maximal_id) in enumerate(
                candidates,
                start=1,
            ):
                sbp_values = sbp[start : end + 1]
                rri_values = rri[start : end + 1]
                rri_signed_steps = direction_sign * np.diff(rri_values)
                rri_direction_gate = bool(np.all(rri_signed_steps > 0.0))
                if config.rri_step_threshold is None:
                    rri_threshold_gate = True
                elif config.rri_threshold_strict:
                    rri_threshold_gate = bool(
                        np.all(rri_signed_steps > config.rri_step_threshold)
                    )
                else:
                    rri_threshold_gate = bool(
                        np.all(rri_signed_steps >= config.rri_step_threshold)
                    )

                regression = stats.linregress(sbp_values, rri_values)
                slope = float(regression.slope)
                intercept = float(regression.intercept)
                pearson_r = float(regression.rvalue)
                r_squared = float(pearson_r**2)
                slope_gate = bool(
                    np.isfinite(slope)
                    and (
                        slope > 0.0
                        if config.positive_slope_required
                        else True
                    )
                )
                if config.correlation_threshold is None:
                    correlation_gate = True
                elif config.correlation_inclusive:
                    correlation_gate = bool(
                        np.isfinite(pearson_r)
                        and pearson_r >= config.correlation_threshold
                    )
                else:
                    correlation_gate = bool(
                        np.isfinite(pearson_r)
                        and pearson_r > config.correlation_threshold
                    )
                if config.r2_threshold is None:
                    r2_gate = True
                elif config.r2_strict:
                    r2_gate = bool(
                        np.isfinite(r_squared)
                        and r_squared > config.r2_threshold
                    )
                else:
                    r2_gate = bool(
                        np.isfinite(r_squared)
                        and r_squared >= config.r2_threshold
                    )

                qualifies = bool(
                    slope_gate
                    and correlation_gate
                    and r2_gate
                    and (
                        not config.rri_direction_required
                        or (rri_direction_gate and rri_threshold_gate)
                    )
                )
                sbp_rows = sbp_frame.iloc[start : end + 1]
                rri_rows = rri_frame.iloc[start : end + 1]
                sequence_rows.append(
                    {
                        "subject": subject,
                        "window": window,
                        "window_start_s": start_s,
                        "window_end_s": end_s,
                        "branch": config.branch,
                        "method_family": config.method_family,
                        "direction": direction,
                        "candidate_id": candidate_id,
                        "maximal_ramp_id": maximal_id,
                        "candidate_start_aligned_index": start,
                        "candidate_end_aligned_index": end,
                        "sequence_length_beats": len(sbp_values),
                        "sbp_source_row_start": int(
                            sbp_rows["source_row"].iloc[0]
                        ),
                        "sbp_source_row_end": int(
                            sbp_rows["source_row"].iloc[-1]
                        ),
                        "rri_source_row_start": int(
                            rri_rows["source_row"].iloc[0]
                        ),
                        "rri_source_row_end": int(
                            rri_rows["source_row"].iloc[-1]
                        ),
                        "sbp_rwave_time_start_s": float(
                            sbp_rows["R_wave_time_s"].iloc[0]
                        ),
                        "sbp_rwave_time_end_s": float(
                            sbp_rows["R_wave_time_s"].iloc[-1]
                        ),
                        "rri_rwave_time_start_s": float(
                            rri_rows["R_wave_time_s"].iloc[0]
                        ),
                        "rri_rwave_time_end_s": float(
                            rri_rows["R_wave_time_s"].iloc[-1]
                        ),
                        "sbp_values_mmHg": _array_text(sbp_values),
                        "rri_values_ms": _array_text(rri_values),
                        "sbp_rwave_times_s": _array_text(
                            sbp_rows["R_wave_time_s"].to_numpy(float)
                        ),
                        "rri_rwave_times_s": _array_text(
                            rri_rows["R_wave_time_s"].to_numpy(float)
                        ),
                        "sbp_amplitude_signed_mmHg": float(
                            sbp_values[-1] - sbp_values[0]
                        ),
                        "sbp_amplitude_abs_mmHg": float(
                            abs(sbp_values[-1] - sbp_values[0])
                        ),
                        "rri_response_signed_ms": float(
                            rri_values[-1] - rri_values[0]
                        ),
                        "rri_response_abs_ms": float(
                            abs(rri_values[-1] - rri_values[0])
                        ),
                        "ols_intercept_ms": intercept,
                        "slope_ms_per_mmHg": slope,
                        "pearson_r": pearson_r,
                        "r_squared": r_squared,
                        "rri_direction_gate": rri_direction_gate,
                        "rri_threshold_gate": rri_threshold_gate,
                        "correlation_gate": correlation_gate,
                        "r2_gate": r2_gate,
                        "slope_gate": slope_gate,
                        "qualifying_sequence": qualifies,
                        "lag_beats": config.lag_beats,
                        "enumeration": config.enumeration,
                        "exact_rule_set": config.exact_rule_set,
                        "implementation_version": IMPLEMENTATION_VERSION,
                    }
                )

    sequences = pd.DataFrame(sequence_rows)
    summaries: list[dict[str, object]] = []
    for direction in ("all", "up", "down"):
        if sequences.empty:
            candidates = sequences
        elif direction == "all":
            candidates = sequences
        else:
            candidates = sequences[sequences["direction"] == direction]
        if candidates.empty:
            qualifying = candidates
        else:
            qualifying = candidates[candidates["qualifying_sequence"]]
        n_candidate = len(candidates)
        n_qualifying = len(qualifying)
        if direction == "all":
            n_unique_maximal = sum(unique_maximal_counts.values())
        else:
            n_unique_maximal = unique_maximal_counts[direction]
        if n_candidate == 0:
            reason = "NO_CANDIDATE_SBP_RAMP"
        elif n_qualifying == 0 and config.rri_direction_required:
            rri_ok = int(
                (
                    candidates["rri_direction_gate"]
                    & candidates["rri_threshold_gate"]
                ).sum()
            )
            reason = (
                "NO_RRI_THRESHOLD_CONCORDANT_CANDIDATE"
                if rri_ok == 0
                else "NO_CANDIDATE_PASSED_R2_AND_SLOPE"
            )
        elif n_qualifying == 0:
            reason = "NO_CANDIDATE_PASSED_CORRELATION"
        else:
            reason = "NA"
        summaries.append(
            {
                "subject": subject,
                "window": window,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "branch": config.branch,
                "method_family": config.method_family,
                "direction": direction,
                "n_valid_paired_beats": n,
                "n_aligned_pairs": len(sbp),
                "n_unique_maximal_sbp_ramps": n_unique_maximal,
                "n_candidate_sbp_ramps": n_candidate,
                "n_qualifying_sequences": n_qualifying,
                "gain_ms_per_mmHg": (
                    float(qualifying["slope_ms_per_mmHg"].mean())
                    if n_qualifying
                    else np.nan
                ),
                "BEI": n_qualifying / n_candidate if n_candidate else np.nan,
                "mean_sequence_length_beats": (
                    float(qualifying["sequence_length_beats"].mean())
                    if n_qualifying
                    else np.nan
                ),
                "mean_sequence_r_squared": (
                    float(qualifying["r_squared"].mean())
                    if n_qualifying
                    else np.nan
                ),
                "mean_within_sequence_r": (
                    float(qualifying["pearson_r"].mean())
                    if n_qualifying
                    else np.nan
                ),
                "mean_sbp_ramp_amplitude_abs_mmHg": (
                    float(qualifying["sbp_amplitude_abs_mmHg"].mean())
                    if n_qualifying
                    else np.nan
                ),
                "mean_rri_response_abs_ms": (
                    float(qualifying["rri_response_abs_ms"].mean())
                    if n_qualifying
                    else np.nan
                ),
                "no_valid_sequence_flag": n_qualifying == 0,
                "no_valid_sequence_reason": reason,
                "lag_beats": config.lag_beats,
                "enumeration": config.enumeration,
                "exact_rule_set": config.exact_rule_set,
                "implementation_version": IMPLEMENTATION_VERSION,
            }
        )
    return summaries, sequence_rows


def compute_context(frame: pd.DataFrame) -> dict[str, float | int]:
    """Compute native-beat haemodynamic and HRV context for one interval."""
    rri = frame["RRI_ms"].to_numpy(float)
    sbp = frame["SBP_mmHg"].to_numpy(float)
    if len(rri) == 0:
        return {
            "valid_paired_beat_count": 0,
            "mean_SBP_mmHg": np.nan,
            "mean_RRI_ms": np.nan,
            "mean_HR_bpm": np.nan,
            "RMSSD_ms": np.nan,
        }
    return {
        "valid_paired_beat_count": len(frame),
        "mean_SBP_mmHg": float(np.mean(sbp)),
        "mean_RRI_ms": float(np.mean(rri)),
        "mean_HR_bpm": float(np.mean(60_000.0 / rri)),
        "RMSSD_ms": (
            float(np.sqrt(np.mean(np.diff(rri) ** 2)))
            if len(rri) >= 2
            else np.nan
        ),
    }


def cohens_dz(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return np.nan
    sd = float(np.std(data, ddof=1))
    return float(np.mean(data) / sd) if sd > 0.0 else np.nan


def wilcoxon_two_sided(values: np.ndarray) -> tuple[float, float, str]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.nan, np.nan, "NOT_ESTIMABLE"
    if np.all(data == 0.0):
        return 0.0, 1.0, "ALL_ZERO"
    result = stats.wilcoxon(
        data,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )
    has_zero = bool(np.any(data == 0.0))
    nonzero_abs = np.abs(data[data != 0.0])
    tied = len(nonzero_abs) != len(np.unique(nonzero_abs))
    method = "SCIPY_AUTO_TIES_OR_ZERO" if has_zero or tied else "SCIPY_AUTO_EXACT_ELIGIBLE"
    return float(result.statistic), float(result.pvalue), method


def _bca_limits(
    estimate: float,
    replicates: np.ndarray,
    jackknife: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, int, float, float]:
    finite = np.asarray(replicates, dtype=float)
    finite = finite[np.isfinite(finite)]
    jack = np.asarray(jackknife, dtype=float)
    jack = jack[np.isfinite(jack)]
    if not np.isfinite(estimate) or len(finite) < 100 or len(jack) < 3:
        return np.nan, np.nan, len(finite), np.nan, np.nan
    proportion = (
        np.sum(finite < estimate) + 0.5 * np.sum(finite == estimate)
    ) / len(finite)
    epsilon = 0.5 / len(finite)
    z0 = float(stats.norm.ppf(np.clip(proportion, epsilon, 1.0 - epsilon)))
    jack_mean = float(np.mean(jack))
    deviations = jack_mean - jack
    denominator = 6.0 * float(np.sum(deviations**2) ** 1.5)
    acceleration = (
        float(np.sum(deviations**3) / denominator)
        if denominator > 0.0
        else 0.0
    )
    alpha = 1.0 - confidence_level
    z_alpha = stats.norm.ppf([alpha / 2.0, 1.0 - alpha / 2.0])
    adjusted = stats.norm.cdf(
        z0 + (z0 + z_alpha) / (1.0 - acceleration * (z0 + z_alpha))
    )
    low, high = np.quantile(
        finite,
        np.clip(adjusted, 0.0, 1.0),
        method="linear",
    )
    return float(low), float(high), len(finite), z0, acceleration


def bca_1d(
    values: np.ndarray,
    statistic: Literal["mean", "median", "dz"],
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if statistic == "mean":
        function: Callable[[np.ndarray], float] = lambda x: float(np.mean(x))
    elif statistic == "median":
        function = lambda x: float(np.median(x))
    else:
        function = cohens_dz
    estimate = function(data) if len(data) else np.nan
    if len(data) < 2:
        return {
            "estimate": estimate,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "finite_resamples": 0,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(n_resamples, len(data)))
    samples = data[indices]
    if statistic == "mean":
        replicates = samples.mean(axis=1)
    elif statistic == "median":
        replicates = np.median(samples, axis=1)
    else:
        means = samples.mean(axis=1)
        sds = samples.std(axis=1, ddof=1)
        replicates = np.divide(
            means,
            sds,
            out=np.full(n_resamples, np.nan),
            where=sds > 0.0,
        )
    jackknife = np.array(
        [function(np.delete(data, index)) for index in range(len(data))],
        dtype=float,
    )
    low, high, finite, z0, acceleration = _bca_limits(
        estimate,
        replicates,
        jackknife,
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "finite_resamples": finite,
        "seed": seed,
        "bias_correction": z0,
        "acceleration": acceleration,
    }


def _paired_statistic(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 2:
        return np.nan
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    if kind == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    return float(stats.pearsonr(x, y).statistic)


def bca_paired_correlation(
    x: np.ndarray,
    y: np.ndarray,
    kind: Literal["spearman", "pearson"],
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    x_data = np.asarray(x, dtype=float)
    y_data = np.asarray(y, dtype=float)
    valid = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid]
    y_data = y_data[valid]
    estimate = _paired_statistic(x_data, y_data, kind)
    if len(x_data) < 3:
        return {
            "estimate": estimate,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "finite_resamples": 0,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x_data), size=(n_resamples, len(x_data)))
    replicates = np.array(
        [
            _paired_statistic(x_data[row], y_data[row], kind)
            for row in indices
        ],
        dtype=float,
    )
    jackknife = np.array(
        [
            _paired_statistic(
                np.delete(x_data, index),
                np.delete(y_data, index),
                kind,
            )
            for index in range(len(x_data))
        ],
        dtype=float,
    )
    low, high, finite, z0, acceleration = _bca_limits(
        estimate,
        replicates,
        jackknife,
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "finite_resamples": finite,
        "seed": seed,
        "bias_correction": z0,
        "acceleration": acceleration,
    }


def permutation_spearman(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_permutations: int = PERMUTATION_RESAMPLES,
) -> dict[str, float | int]:
    """Two-sided Monte Carlo permutation test with fixed seed."""
    x_data = np.asarray(x, dtype=float)
    y_data = np.asarray(y, dtype=float)
    valid = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid]
    y_data = y_data[valid]
    observed = _paired_statistic(x_data, y_data, "spearman")
    if len(x_data) < 3 or not np.isfinite(observed):
        return {
            "rho": observed,
            "p_two_sided": np.nan,
            "extreme": 0,
            "n_permutations": n_permutations,
            "seed": seed,
        }
    x_rank = stats.rankdata(x_data)
    y_rank = stats.rankdata(y_data)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    )
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = n_permutations
    chunk_size = 5_000
    while remaining:
        size = min(chunk_size, remaining)
        permutations = np.argsort(rng.random((size, len(x_data))), axis=1)
        permuted = y_centered[permutations]
        correlations = (permuted @ x_centered) / denominator
        extreme += int(np.sum(np.abs(correlations) >= abs(observed) - 1e-15))
        remaining -= size
    return {
        "rho": observed,
        "p_two_sided": (extreme + 1) / (n_permutations + 1),
        "extreme": extreme,
        "n_permutations": n_permutations,
        "seed": seed,
    }


def pearson_fisher_ci(
    x: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    x_data = np.asarray(x, dtype=float)
    y_data = np.asarray(y, dtype=float)
    valid = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid]
    y_data = y_data[valid]
    if len(x_data) < 4 or np.std(x_data) == 0.0 or np.std(y_data) == 0.0:
        return np.nan, np.nan, np.nan
    r_value = float(stats.pearsonr(x_data, y_data).statistic)
    clipped = float(np.clip(r_value, -0.999999999999, 0.999999999999))
    z_value = np.arctanh(clipped)
    z_critical = stats.norm.ppf(0.5 + confidence_level / 2.0)
    se = 1.0 / math.sqrt(len(x_data) - 3)
    low, high = np.tanh([z_value - z_critical * se, z_value + z_critical * se])
    return r_value, float(low), float(high)


def bootstrap_theil_sen(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = THEILSEN_BOOTSTRAP_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    x_data = np.asarray(x, dtype=float)
    y_data = np.asarray(y, dtype=float)
    valid = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid]
    y_data = y_data[valid]
    estimate = (
        float(stats.theilslopes(y_data, x_data).slope)
        if len(x_data) >= 3 and len(np.unique(x_data)) >= 2
        else np.nan
    )
    rng = np.random.default_rng(seed)
    output = np.full(n_resamples, np.nan, dtype=float)
    for index in range(n_resamples):
        row = rng.integers(0, len(x_data), size=len(x_data))
        if len(np.unique(x_data[row])) >= 2:
            output[index] = float(stats.theilslopes(y_data[row], x_data[row]).slope)
    finite = output[np.isfinite(output)]
    if len(finite) >= 100:
        low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    else:
        low, high = np.nan, np.nan
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "finite_resamples": len(finite),
        "seed": seed,
    }


def describe_values(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
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


def summarize_paired_difference(
    diff: np.ndarray,
    seed: int,
) -> dict[str, float | int | str | bool]:
    values = np.asarray(diff, dtype=float)
    values = values[np.isfinite(values)]
    desc = describe_values(values)
    mean_ci = bca_1d(values, "mean", seed)
    dz_ci = bca_1d(values, "dz", seed + 1)
    statistic, p_value, method = wilcoxon_two_sided(values)
    return {
        "paired_n": len(values),
        "mean_difference": desc["mean"],
        "median_difference": desc["median"],
        "sd_difference": desc["sd"],
        "q1_difference": desc["q1"],
        "q3_difference": desc["q3"],
        "iqr_difference": desc["iqr"],
        "mean_difference_ci_low": mean_ci["ci_low"],
        "mean_difference_ci_high": mean_ci["ci_high"],
        "cohens_dz": dz_ci["estimate"],
        "cohens_dz_ci_low": dz_ci["ci_low"],
        "cohens_dz_ci_high": dz_ci["ci_high"],
        "n_negative": int(np.sum(values < 0.0)),
        "n_positive": int(np.sum(values > 0.0)),
        "n_zero": int(np.sum(values == 0.0)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_two_sided": p_value,
        "wilcoxon_method": method,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed_mean": seed,
        "bootstrap_seed_dz": seed + 1,
        "estimable": len(values) >= 3,
        "NA_reason": "NA" if len(values) >= 3 else "FEWER_THAN_3_PAIRS",
    }


def branch_settings_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(config) for config in BRANCHES.values()])
