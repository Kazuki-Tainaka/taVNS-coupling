"""Data, QC, and statistical utilities for the frozen nonlinear analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.tsa.stattools import kpss

from nonlinear_estimators import detrend_zscore


SEED = 20260806
PLAN_SHA256 = "d3ea3d785ff3615eb3370e8fb60daf6fed37440593ae7501e2892be86bd1a0b4"
RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    os.environ.get(
        "TAVNS_NONLINEAR_OUTPUT_ROOT",
        RELEASE_ROOT / "controlled_outputs" / "nonlinear_sensitivity",
    )
).resolve()
PROJECT_ROOT = Path(
    os.environ.get(
        "TAVNS_DATA_ROOT",
        RELEASE_ROOT / "controlled_data_not_included",
    )
).resolve()
PLAN_PATH = Path(
    os.environ.get(
        "TAVNS_NONLINEAR_PLAN_PATH",
        RELEASE_ROOT / "config" / "POST_HOC_NONLINEAR_COUPLING_PLAN.md",
    )
).resolve()
PAIRED_DIR = PROJECT_ROOT / "paired"
ALL_SUBJECTS = tuple(range(1, 19))
EXCLUDED_SUBJECTS: tuple[int, ...] = tuple()
ANALYSIS_SUBJECTS = ALL_SUBJECTS
PHASES = {"Pre": (0.0, 300.0), "Stim": (300.0, 600.0), "Post": (600.0, 900.0)}
PHASE_ORDER = ("Pre", "Stim", "Post")
PAIRED_COLUMNS = (
    "R_wave_timing_ms",
    "RRI_ms",
    "sBP_timing_ms",
    "sBP_mmHg100",
    "PAT_ms",
)


@dataclass(frozen=True)
class SegmentSetting:
    setting_id: str
    segment_kind: str
    segment_length: int | None
    k: int
    lag_depth: int
    same_beat_convention: bool
    primary: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_plan() -> None:
    if not PLAN_PATH.is_file():
        raise RuntimeError(f"Frozen plan missing: {PLAN_PATH}")
    observed = sha256_file(PLAN_PATH)
    if observed != PLAN_SHA256:
        raise RuntimeError(f"Frozen plan hash mismatch: {observed}")


def load_paired_full(subject: int) -> pd.DataFrame:
    """Load an authoritative provider-retained beat table read-only."""

    path = PAIRED_DIR / f"paired_beats_{subject:02d}.csv"
    frame = pd.read_csv(path, header=None, names=PAIRED_COLUMNS)
    frame.insert(0, "original_row", np.arange(1, len(frame) + 1, dtype=int))
    frame["subject_id"] = f"{subject:02d}"
    frame["beat_time_s"] = frame["R_wave_timing_ms"].astype(float) / 1000.0
    frame["SBP_mmHg"] = frame["sBP_mmHg100"].astype(float) * 100.0
    return frame


def load_valid_phase(subject: int, phase: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply frozen phase and finite-pair rules without interpolation."""

    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    start, end = PHASES[phase]
    full = load_paired_full(subject)
    phase_frame = full[
        full["beat_time_s"].ge(start) & full["beat_time_s"].lt(end)
    ].copy()
    valid_mask = (
        np.isfinite(phase_frame["RRI_ms"].to_numpy(float))
        & np.isfinite(phase_frame["SBP_mmHg"].to_numpy(float))
        & (phase_frame["RRI_ms"].to_numpy(float) > 0.0)
    )
    valid = phase_frame.loc[valid_mask].reset_index(drop=True)
    counts = {
        "original_beat_count": len(phase_frame),
        "valid_pair_count": len(valid),
        "missing_pair_count": int((~valid_mask).sum()),
    }
    return valid, counts


def select_centered_segment(
    frame: pd.DataFrame,
    phase: str,
    length: int,
) -> tuple[pd.DataFrame | None, int | None]:
    """Select the deterministic centred window; return 0-based valid-pair start."""

    if len(frame) < length:
        return None, None
    phase_midpoint = float(np.mean(PHASES[phase]))
    times = frame["beat_time_s"].to_numpy(float)
    starts = np.arange(0, len(frame) - length + 1, dtype=int)
    midpoints = (times[starts] + times[starts + length - 1]) / 2.0
    distances = np.abs(midpoints - phase_midpoint)
    start = int(np.flatnonzero(distances == np.min(distances))[0])
    return frame.iloc[start : start + length].copy().reset_index(drop=True), start


def select_segment(
    frame: pd.DataFrame,
    phase: str,
    setting: SegmentSetting,
) -> tuple[pd.DataFrame | None, int | None]:
    if setting.segment_kind == "full":
        return frame.copy().reset_index(drop=True), 0
    if setting.segment_length is None:
        raise ValueError("centred segment requires a length")
    return select_centered_segment(frame, phase, setting.segment_length)


def direction_arrays(
    direction: str,
    rri: Sequence[float],
    sbp: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Map a reported direction to estimator target/source arrays.

    ``A→B`` always means that past A is used as the source to predict current
    B, so the estimator receives B as ``target`` and A as ``source``.
    """

    rri_array = np.asarray(rri, dtype=float)
    sbp_array = np.asarray(sbp, dtype=float)
    if direction == "SBP→RRI":
        return rri_array, sbp_array, "RRI", "SBP"
    if direction == "RRI→SBP":
        return sbp_array, rri_array, "SBP", "RRI"
    raise ValueError(f"Unknown direction: {direction}")


def _kpss_pvalue(values: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_value, _, _ = kpss(values, regression="c", nlags="auto")
    return float(p_value)


def stationarity_flag(rri: np.ndarray, sbp: np.ndarray) -> tuple[str, float, float]:
    try:
        rri_p = _kpss_pvalue(rri)
        sbp_p = _kpss_pvalue(sbp)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return "INDETERMINATE", np.nan, np.nan
    failed = []
    if rri_p < 0.05:
        failed.append("RRI")
    if sbp_p < 0.05:
        failed.append("SBP")
    if not failed:
        return "PASS_BOTH", rri_p, sbp_p
    return "FLAG_" + "_AND_".join(failed), rri_p, sbp_p


def prepare_primary_qc() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]]:
    """Create primary-window QC and normalised in-memory analysis segments."""

    primary = SegmentSetting(
        setting_id="primary",
        segment_kind="centered",
        segment_length=256,
        k=30,
        lag_depth=8,
        same_beat_convention=False,
        primary=True,
    )
    qc_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    segments: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for subject in ALL_SUBJECTS:
        for phase in PHASE_ORDER:
            valid, counts = load_valid_phase(subject, phase)
            selected, selected_start = select_segment(valid, phase, primary)
            included = True
            if selected is None:
                reason = "fewer_than_256_valid_pairs"
                estimable = False
                selected_start_beat = np.nan
                selected_end_beat = np.nan
                selected_start_time = np.nan
                selected_end_time = np.nan
                rri_slope = np.nan
                sbp_slope = np.nan
                rri_sd_raw = np.nan
                sbp_sd_raw = np.nan
                flag = "INDETERMINATE"
                rri_kpss_p = np.nan
                sbp_kpss_p = np.nan
            else:
                rri_raw = selected["RRI_ms"].to_numpy(float)
                sbp_raw = selected["SBP_mmHg"].to_numpy(float)
                rri_normalised, rri_slope, rri_sd_raw = detrend_zscore(rri_raw)
                sbp_normalised, sbp_slope, sbp_sd_raw = detrend_zscore(sbp_raw)
                flag, rri_kpss_p, sbp_kpss_p = stationarity_flag(
                    rri_normalised, sbp_normalised
                )
                selected_start_beat = int(selected_start) + 1
                selected_end_beat = int(selected_start) + len(selected)
                selected_start_time = float(selected["beat_time_s"].iloc[0])
                selected_end_time = float(selected["beat_time_s"].iloc[-1])
                estimable = bool(
                    included
                    and np.all(np.isfinite(rri_normalised))
                    and np.all(np.isfinite(sbp_normalised))
                )
                if not estimable:
                    reason = "detrending_or_scaling_failure"
                else:
                    reason = "NA"
                    segments[(subject, phase)] = (rri_normalised, sbp_normalised)
                for beat_index, row in selected.iterrows():
                    series_rows.append(
                        {
                            "subject_id": f"{subject:02d}",
                            "phase": phase,
                            "included_in_full_cohort_analysis": included,
                            "selected_beat_index": beat_index + 1,
                            "original_paired_row": int(row["original_row"]),
                            "beat_time_s": float(row["beat_time_s"]),
                            "RRI_ms_raw": float(row["RRI_ms"]),
                            "SBP_mmHg_raw": float(row["SBP_mmHg"]),
                            "RRI_detrended_z": float(rri_normalised[beat_index]),
                            "SBP_detrended_z": float(sbp_normalised[beat_index]),
                        }
                    )

            qc_rows.append(
                {
                    "subject_id": f"{subject:02d}",
                    "phase": phase,
                    "included_in_full_cohort_analysis": included,
                    **counts,
                    "selected_start_beat": selected_start_beat,
                    "selected_end_beat": selected_end_beat,
                    "selected_start_time": selected_start_time,
                    "selected_end_time": selected_end_time,
                    "artifact_excluded_count": np.nan,
                    "rri_linear_trend": rri_slope,
                    "sbp_linear_trend": sbp_slope,
                    "rri_sd_raw": rri_sd_raw,
                    "sbp_sd_raw": sbp_sd_raw,
                    "rri_kpss_p": rri_kpss_p,
                    "sbp_kpss_p": sbp_kpss_p,
                    "stationarity_flag": flag,
                    "primary_estimable": estimable,
                    "exclusion_reason": reason,
                }
            )
    return pd.DataFrame(qc_rows), pd.DataFrame(series_rows), segments


def sensitivity_settings() -> list[SegmentSetting]:
    """Return the frozen LP/CE one-factor-at-a-time settings."""

    return [
        SegmentSetting("primary", "centered", 256, 30, 8, False, True),
        SegmentSetting("k20", "centered", 256, 20, 8, False, False),
        SegmentSetting("k40", "centered", 256, 40, 8, False, False),
        SegmentSetting("lag4", "centered", 256, 30, 4, False, False),
        SegmentSetting("lag12", "centered", 256, 30, 12, False, False),
        SegmentSetting(
            "same_beat_convention", "centered", 256, 30, 8, True, False
        ),
        SegmentSetting("centered192", "centered", 192, 30, 8, False, False),
        SegmentSetting("full_phase", "full", None, 30, 8, False, False),
    ]


def wilcoxon_p(differences: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if np.allclose(values, 0.0, atol=0.0, rtol=0.0):
        return 0.0, 1.0
    result = stats.wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def cohens_dz(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    sd = float(np.std(values, ddof=1))
    if sd <= 0.0:
        return 0.0 if np.allclose(values, 0.0) else np.nan
    return float(np.mean(values) / sd)


def bca_mean_ci(
    differences: Sequence[float],
    *,
    resamples: int = 10_000,
    seed_parts: Sequence[int] = (),
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Participant-paired BCa interval for the mean paired difference."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(np.random.SeedSequence([SEED, *seed_parts]))
    indices = rng.integers(0, n, size=(resamples, n))
    bootstrap = values[indices].mean(axis=1)
    observed = float(values.mean())
    # Treat mathematically identical bootstrap means as ties even after a
    # constant translation changes their last floating-point bit.
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(observed))
    centred_bootstrap = bootstrap - observed
    less = np.count_nonzero(centred_bootstrap < -tolerance)
    equal = np.count_nonzero(np.abs(centred_bootstrap) <= tolerance)
    proportion = (less + 0.5 * equal) / resamples
    proportion = np.clip(proportion, 1.0 / (2 * resamples), 1.0 - 1.0 / (2 * resamples))
    z0 = float(stats.norm.ppf(proportion))

    jackknife = np.asarray(
        [np.mean(np.delete(values, index)) for index in range(n)], dtype=float
    )
    jackknife_mean = float(jackknife.mean())
    numerator = np.sum((jackknife_mean - jackknife) ** 3)
    denominator = 6.0 * np.sum((jackknife_mean - jackknife) ** 2) ** 1.5
    acceleration = float(numerator / denominator) if denominator > 0.0 else 0.0
    alpha = (1.0 - confidence) / 2.0
    z_alpha = stats.norm.ppf([alpha, 1.0 - alpha])
    adjusted = stats.norm.cdf(
        z0 + (z0 + z_alpha) / (1.0 - acceleration * (z0 + z_alpha))
    )
    adjusted = np.clip(adjusted, 0.0, 1.0)
    lower, upper = np.quantile(bootstrap, adjusted, method="linear")
    return float(lower), float(upper)


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini--Hochberg adjusted p values with NA preservation."""

    values = np.asarray(p_values, dtype=float)
    output = np.full_like(values, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return output
    finite_values = values[finite_indices]
    order = np.argsort(finite_values, kind="stable")
    ranked = finite_values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[finite_indices] = restored
    return output


def phase_contrast_rows(
    metrics: pd.DataFrame,
    *,
    bootstrap_resamples: int = 10_000,
) -> pd.DataFrame:
    """Summarise all method/direction phase contrasts."""

    rows: list[dict[str, object]] = []
    methods = list(dict.fromkeys(metrics["method"].astype(str)))
    directions = list(dict.fromkeys(metrics["direction"].astype(str)))
    comparisons = (
        ("Stim-Pre", "Stim", "Pre", True),
        ("Post-Pre", "Post", "Pre", False),
        ("Post-Stim", "Post", "Stim", False),
    )
    for method_index, method in enumerate(methods):
        for direction_index, direction in enumerate(directions):
            subset = metrics[
                metrics["method"].eq(method)
                & metrics["direction"].eq(direction)
                & metrics["finite"].eq(True)
            ]
            if subset.empty:
                continue
            pivot = subset.pivot(
                index="subject_id", columns="phase", values="directed_strength"
            )
            descriptive = {}
            for phase in PHASE_ORDER:
                values = pivot[phase].dropna() if phase in pivot else pd.Series(dtype=float)
                descriptive[phase] = {
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "n": len(values),
                }
            for contrast_index, (contrast, first, second, primary) in enumerate(comparisons):
                paired = pivot[[first, second]].dropna()
                differences = (
                    paired[first].to_numpy(float) - paired[second].to_numpy(float)
                )
                statistic, p_value = wilcoxon_p(differences)
                ci_low, ci_high = bca_mean_ci(
                    differences,
                    resamples=bootstrap_resamples,
                    seed_parts=(method_index, direction_index, contrast_index),
                )
                rows.append(
                    {
                        "method": method,
                        "direction": direction,
                        "contrast": contrast,
                        "primary_family": primary,
                        "Pre_mean": descriptive["Pre"]["mean"],
                        "Pre_SD": descriptive["Pre"]["sd"],
                        "Stim_mean": descriptive["Stim"]["mean"],
                        "Stim_SD": descriptive["Stim"]["sd"],
                        "Post_mean": descriptive["Post"]["mean"],
                        "Post_SD": descriptive["Post"]["sd"],
                        "paired_mean_difference": float(np.mean(differences)) if len(differences) else np.nan,
                        "paired_median_difference": float(np.median(differences)) if len(differences) else np.nan,
                        "mean_difference_BCa95_low": ci_low,
                        "mean_difference_BCa95_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_two_sided": p_value,
                        "BH_q_primary_family": np.nan,
                        "cohens_dz": cohens_dz(differences),
                        "negative_difference_n": int(np.count_nonzero(differences < 0.0)),
                        "zero_difference_n": int(np.count_nonzero(differences == 0.0)),
                        "positive_difference_n": int(np.count_nonzero(differences > 0.0)),
                        "evaluable_paired_n": len(differences),
                        "estimability_status": (
                            "ESTIMABLE" if len(differences) >= 15 else "LOW_ESTIMABILITY"
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    primary_mask = output["primary_family"].eq(True)
    output.loc[primary_mask, "BH_q_primary_family"] = bh_adjust(
        output.loc[primary_mask, "wilcoxon_p_two_sided"].to_numpy(float)
    )
    return output


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    lower = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes, total - successes + 1)
    upper = 1.0 if successes == total else stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    return float(lower), float(upper)


def exact_mcnemar(pre: Sequence[bool], stim: Sequence[bool]) -> tuple[int, int, float]:
    pre_array = np.asarray(pre, dtype=bool)
    stim_array = np.asarray(stim, dtype=bool)
    b = int(np.count_nonzero(pre_array & ~stim_array))
    c = int(np.count_nonzero(~pre_array & stim_array))
    discordant = b + c
    if discordant == 0:
        return b, c, 1.0
    p_value = min(1.0, 2.0 * stats.binom.cdf(min(b, c), discordant, 0.5))
    return b, c, float(p_value)


def cochran_q(binary: np.ndarray) -> tuple[float, float]:
    array = np.asarray(binary, dtype=int)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Cochran Q input must be subject x three phases")
    if np.all(array == array[:, [0]]):
        return 0.0, 1.0
    result = cochrans_q(array)
    return float(result.statistic), float(result.pvalue)


__all__ = [
    "ALL_SUBJECTS",
    "ANALYSIS_SUBJECTS",
    "EXCLUDED_SUBJECTS",
    "OUTPUT_ROOT",
    "PAIRED_DIR",
    "PHASE_ORDER",
    "PHASES",
    "PROJECT_ROOT",
    "SegmentSetting",
    "assert_frozen_plan",
    "bca_mean_ci",
    "bh_adjust",
    "clopper_pearson",
    "cochran_q",
    "cohens_dz",
    "direction_arrays",
    "exact_mcnemar",
    "load_paired_full",
    "load_valid_phase",
    "phase_contrast_rows",
    "prepare_primary_qc",
    "select_segment",
    "sensitivity_settings",
    "sha256_file",
    "wilcoxon_p",
]
