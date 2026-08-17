"""Reference and sensitivity analyses for sequence-method BRS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from revision_utils import PHASE_ORDER, SUBJECTS, load_paired_phase
from stats_core import cohens_dz, paired_summary, wilcoxon_two_sided


Direction = Literal["all", "up", "down"]
Aggregation = Literal["mean", "median"]


@dataclass(frozen=True)
class BRSSetting:
    lag_beats: int = 1
    correlation_threshold: float = 0.80
    sbp_increment_threshold: float = 1.0
    minimum_sequence_length: int = 3
    aggregation: Aggregation = "mean"
    require_directional_rri_monotonicity: bool = False

    @property
    def setting_id(self) -> str:
        corr = f"{self.correlation_threshold:.2f}".replace(".", "p")
        sbp = f"{self.sbp_increment_threshold:.1f}".replace(".", "p")
        base = (
            f"lag{self.lag_beats}_r{corr}_sbp{sbp}_"
            f"len{self.minimum_sequence_length}_{self.aggregation}"
        )
        if self.require_directional_rri_monotonicity:
            return base + "_directional_rri_gate"
        return base


CANONICAL_SETTING = BRSSetting(
    require_directional_rri_monotonicity=False,
)


def _detect_ramps(
    sbp: np.ndarray,
    direction: Literal["up", "down"],
    minimum_length: int,
    minimum_increment: float,
) -> list[tuple[int, int]]:
    """Detect maximal non-overlapping thresholded monotonic SBP ramps."""
    differences = np.diff(np.asarray(sbp, dtype=float))
    expected_sign = 1 if direction == "up" else -1
    ramps: list[tuple[int, int]] = []
    index = 0
    while index < len(differences):
        value = differences[index]
        qualifies = (
            np.sign(value) == expected_sign
            and abs(value) >= minimum_increment
        )
        if not qualifies:
            index += 1
            continue
        end_difference = index + 1
        while end_difference < len(differences):
            next_value = differences[end_difference]
            if (
                np.sign(next_value) == expected_sign
                and abs(next_value) >= minimum_increment
            ):
                end_difference += 1
            else:
                break
        n_beats = (end_difference - index) + 1
        if n_beats >= minimum_length:
            ramps.append((index, index + n_beats - 1))
        index = end_difference
    return ramps


def evaluate_brs(
    sbp: np.ndarray,
    rri: np.ndarray,
    setting: BRSSetting,
    subject: int | None = None,
    phase: str | None = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str | bool]]]:
    """Evaluate one BRS setting and retain every candidate ramp."""
    sbp_values = np.asarray(sbp, dtype=float)
    rri_values = np.asarray(rri, dtype=float)
    n = min(len(sbp_values), len(rri_values))
    lag = setting.lag_beats
    if lag < 0:
        raise ValueError("lag_beats must be non-negative")
    if n < lag + setting.minimum_sequence_length:
        return _empty_summary(), []

    if lag == 0:
        aligned_sbp = sbp_values[:n]
        aligned_rri = rri_values[:n]
    else:
        aligned_sbp = sbp_values[: n - lag]
        aligned_rri = rri_values[lag:n]

    sequence_rows: list[dict[str, float | int | str | bool]] = []
    qualifying_slopes: dict[str, list[float]] = {"up": [], "down": []}
    ramp_counts: dict[str, int] = {"up": 0, "down": 0}
    event_counts: dict[str, int] = {"up": 0, "down": 0}

    for direction in ("up", "down"):
        ramps = _detect_ramps(
            aligned_sbp,
            direction=direction,
            minimum_length=setting.minimum_sequence_length,
            minimum_increment=setting.sbp_increment_threshold,
        )
        ramp_counts[direction] = len(ramps)
        expected_sign = 1 if direction == "up" else -1
        for ramp_index, (start, end) in enumerate(ramps, start=1):
            sbp_segment = aligned_sbp[start : end + 1]
            rri_segment = aligned_rri[start : end + 1]
            finite_segment = bool(
                np.all(np.isfinite(sbp_segment))
                and np.all(np.isfinite(rri_segment))
            )
            if finite_segment:
                with np.errstate(invalid="ignore", divide="ignore"):
                    correlation = float(
                        np.corrcoef(sbp_segment, rri_segment)[0, 1]
                    )
                regression = stats.linregress(sbp_segment, rri_segment)
                slope = float(regression.slope)
            else:
                correlation = np.nan
                slope = np.nan
            rri_differences = np.diff(rri_segment)
            rri_direction_monotonic = bool(
                np.all(np.sign(rri_differences) == expected_sign)
            )
            rri_concordance_ge_1ms_descriptive = bool(
                rri_direction_monotonic
                and np.all(np.abs(rri_differences) >= 1.0)
            )
            qualifies = bool(
                (
                    rri_direction_monotonic
                    or not setting.require_directional_rri_monotonicity
                )
                and
                np.isfinite(correlation)
                and correlation >= setting.correlation_threshold
                and np.isfinite(slope)
            )
            if qualifies:
                qualifying_slopes[direction].append(slope)
                event_counts[direction] += 1

            sequence_rows.append(
                {
                    "subject": subject if subject is not None else -1,
                    "phase": phase or "NA",
                    "direction": direction,
                    "ramp_index_within_direction": ramp_index,
                    "start_index_aligned": start,
                    "end_index_aligned": end,
                    "sequence_length_beats": len(sbp_segment),
                    "sbp_start_mmHg": float(sbp_segment[0]),
                    "sbp_end_mmHg": float(sbp_segment[-1]),
                    "sbp_amplitude_signed_mmHg": float(sbp_segment[-1] - sbp_segment[0]),
                    "sbp_amplitude_abs_mmHg": float(abs(sbp_segment[-1] - sbp_segment[0])),
                    "rri_start_ms": float(rri_segment[0]),
                    "rri_end_ms": float(rri_segment[-1]),
                    "rri_response_signed_ms": float(rri_segment[-1] - rri_segment[0]),
                    "rri_response_abs_ms": float(abs(rri_segment[-1] - rri_segment[0])),
                    "pearson_r": correlation,
                    "slope_ms_per_mmHg": slope,
                    "rri_direction_monotonic": rri_direction_monotonic,
                    "rri_concordance_ge_1ms_descriptive": (
                        rri_concordance_ge_1ms_descriptive
                    ),
                    "qualifying_brs_sequence": qualifies,
                    "lag_beats": setting.lag_beats,
                    "correlation_threshold": setting.correlation_threshold,
                    "sbp_increment_threshold": setting.sbp_increment_threshold,
                    "minimum_sequence_length": setting.minimum_sequence_length,
                    "require_directional_rri_monotonicity": (
                        setting.require_directional_rri_monotonicity
                    ),
                }
            )

    def aggregate(values: list[float]) -> float:
        if not values:
            return np.nan
        if setting.aggregation == "median":
            return float(np.median(values))
        return float(np.mean(values))

    all_slopes = qualifying_slopes["up"] + qualifying_slopes["down"]
    total_ramps = ramp_counts["up"] + ramp_counts["down"]
    total_events = event_counts["up"] + event_counts["down"]
    summary: dict[str, float | int] = {
        "n_ramp_up": ramp_counts["up"],
        "n_ramp_down": ramp_counts["down"],
        "n_ramp_all": total_ramps,
        "n_brs_up": event_counts["up"],
        "n_brs_down": event_counts["down"],
        "n_brs_all": total_events,
        "BEI_up": event_counts["up"] / ramp_counts["up"] if ramp_counts["up"] else np.nan,
        "BEI_down": event_counts["down"] / ramp_counts["down"] if ramp_counts["down"] else np.nan,
        "BEI_all": total_events / total_ramps if total_ramps else np.nan,
        "BRS_seq_up": aggregate(qualifying_slopes["up"]),
        "BRS_seq_down": aggregate(qualifying_slopes["down"]),
        "BRS_seq_all": aggregate(all_slopes),
    }
    return summary, sequence_rows


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_ramp_up": 0,
        "n_ramp_down": 0,
        "n_ramp_all": 0,
        "n_brs_up": 0,
        "n_brs_down": 0,
        "n_brs_all": 0,
        "BEI_up": np.nan,
        "BEI_down": np.nan,
        "BEI_all": np.nan,
        "BRS_seq_up": np.nan,
        "BRS_seq_down": np.nan,
        "BRS_seq_all": np.nan,
    }


def compute_canonical_brs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the canonical subject-phase values and sequence diagnostics."""
    summary_rows: list[dict[str, float | int | str]] = []
    sequence_rows: list[dict[str, float | int | str | bool]] = []
    for subject in SUBJECTS:
        for phase in PHASE_ORDER:
            frame = load_paired_phase(subject, phase)
            summary, sequences = evaluate_brs(
                frame["SBP_mmHg"].to_numpy(float),
                frame["RRI_ms"].to_numpy(float),
                CANONICAL_SETTING,
                subject=subject,
                phase=phase,
            )
            summary_rows.append(
                {
                    "subject": subject,
                    "phase": phase,
                    "n_beats": len(frame),
                    **asdict(CANONICAL_SETTING),
                    **summary,
                }
            )
            sequence_rows.extend(sequences)
    summaries = pd.DataFrame(summary_rows)
    sequences = pd.DataFrame(sequence_rows)
    quality = summarize_sequence_quality(sequences, summaries)
    return summaries, sequences, quality


def summarize_sequence_quality(
    sequences: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Create subject-phase-direction quality summaries with explicit NA reasons."""
    rows: list[dict[str, float | int | str | bool]] = []
    for subject in SUBJECTS:
        for phase in PHASE_ORDER:
            subject_phase = sequences[
                (sequences["subject"] == subject) & (sequences["phase"] == phase)
            ]
            canonical_row = canonical[
                (canonical["subject"] == subject) & (canonical["phase"] == phase)
            ].iloc[0]
            for direction in ("all", "up", "down"):
                if direction == "all":
                    ramps = subject_phase
                else:
                    ramps = subject_phase[subject_phase["direction"] == direction]
                qualifying = ramps[ramps["qualifying_brs_sequence"]]
                n_ramps = len(ramps)
                n_sequences = len(qualifying)
                metric_suffix = direction
                no_valid = n_sequences == 0
                if n_ramps == 0:
                    na_reason = "no_sbp_ramps"
                elif no_valid:
                    na_reason = "no_ramps_met_rri_direction_and_correlation_criteria"
                else:
                    na_reason = "NA"
                rows.append(
                    {
                        "subject": subject,
                        "phase": phase,
                        "direction": direction,
                        "n_sbp_ramps": n_ramps,
                        "n_qualifying_brs_sequences": n_sequences,
                        "mean_sequence_length_beats": qualifying["sequence_length_beats"].mean(),
                        "median_sequence_length_beats": qualifying["sequence_length_beats"].median(),
                        "mean_sbp_ramp_amplitude_abs_mmHg": qualifying["sbp_amplitude_abs_mmHg"].mean(),
                        "mean_rri_response_signed_ms": qualifying["rri_response_signed_ms"].mean(),
                        "mean_rri_response_abs_ms": qualifying["rri_response_abs_ms"].mean(),
                        "mean_within_sequence_r": qualifying["pearson_r"].mean(),
                        "mean_sequence_slope_ms_per_mmHg": qualifying["slope_ms_per_mmHg"].mean(),
                        "median_sequence_slope_ms_per_mmHg": qualifying["slope_ms_per_mmHg"].median(),
                        "rri_direction_monotonic_fraction": qualifying[
                            "rri_direction_monotonic"
                        ].mean(),
                        "rri_concordance_ge_1ms_descriptive_fraction": qualifying[
                            "rri_concordance_ge_1ms_descriptive"
                        ].mean(),
                        "BEI": canonical_row[f"BEI_{metric_suffix}"],
                        "BRS_seq": canonical_row[f"BRS_seq_{metric_suffix}"],
                        "no_valid_sequence_indicator": no_valid,
                        "estimability_status": "not_estimable" if no_valid else "estimable",
                        "NA_reason": na_reason,
                    }
                )
    return pd.DataFrame(rows)


def ofat_settings() -> list[tuple[str, BRSSetting]]:
    """Return prespecified one-factor-at-a-time settings including canonical."""
    return [
        ("canonical", CANONICAL_SETTING),
        ("lag_0", BRSSetting(lag_beats=0)),
        ("lag_2", BRSSetting(lag_beats=2)),
        ("correlation_0.70", BRSSetting(correlation_threshold=0.70)),
        ("correlation_0.85", BRSSetting(correlation_threshold=0.85)),
        ("sbp_increment_0.5", BRSSetting(sbp_increment_threshold=0.5)),
        ("minimum_length_4", BRSSetting(minimum_sequence_length=4)),
        ("aggregation_median", BRSSetting(aggregation="median")),
    ]


def full_factorial_settings() -> list[BRSSetting]:
    settings = [
        BRSSetting(
            lag_beats=lag,
            correlation_threshold=correlation,
            sbp_increment_threshold=increment,
            minimum_sequence_length=minimum_length,
            aggregation=aggregation,
        )
        for lag, correlation, increment, minimum_length, aggregation in product(
            (0, 1, 2),
            (0.70, 0.80, 0.85),
            (0.5, 1.0),
            (3, 4),
            ("mean", "median"),
        )
    ]
    return settings


def compute_settings_subject_phase(
    settings: list[BRSSetting],
) -> pd.DataFrame:
    """Compute all/up/down BRS values for every setting and subject-phase."""
    cached_frames = {
        (subject, phase): load_paired_phase(subject, phase)
        for subject in SUBJECTS
        for phase in PHASE_ORDER
    }
    rows: list[dict[str, float | int | str]] = []
    for setting in settings:
        for subject in SUBJECTS:
            for phase in PHASE_ORDER:
                frame = cached_frames[(subject, phase)]
                summary, _ = evaluate_brs(
                    frame["SBP_mmHg"].to_numpy(float),
                    frame["RRI_ms"].to_numpy(float),
                    setting,
                    subject=subject,
                    phase=phase,
                )
                for direction in ("all", "up", "down"):
                    value = summary[f"BRS_seq_{direction}"]
                    rows.append(
                        {
                            "setting_id": setting.setting_id,
                            **asdict(setting),
                            "direction": direction,
                            "subject": subject,
                            "phase": phase,
                            "n_beats": len(frame),
                            "n_ramps": summary[f"n_ramp_{direction}"],
                            "n_qualifying_sequences": summary[f"n_brs_{direction}"],
                            "brs_value_ms_per_mmHg": value,
                            "estimability_status": "estimable" if np.isfinite(value) else "not_estimable",
                            "NA_reason": "NA" if np.isfinite(value) else "no_qualifying_sequences",
                        }
                    )
    return pd.DataFrame(rows)


def summarize_setting_contrasts(
    subject_phase: pd.DataFrame,
    bootstrap_setting_ids: set[str] | None = None,
    retain_bootstrap_keys: set[tuple[str, str]] | None = None,
    seed_base: int = 20_260_805,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize Stim-Pre contrasts for sensitivity settings."""
    bootstrap_setting_ids = bootstrap_setting_ids or set()
    retain_bootstrap_keys = retain_bootstrap_keys or set()
    rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    groups = subject_phase.groupby(
        [
            "setting_id",
            "lag_beats",
            "correlation_threshold",
            "sbp_increment_threshold",
            "minimum_sequence_length",
            "aggregation",
            "require_directional_rri_monotonicity",
            "direction",
        ],
        sort=True,
        dropna=False,
    )
    for group_index, (keys, group) in enumerate(groups):
        (
            setting_id,
            lag,
            correlation,
            increment,
            minimum_length,
            aggregation,
            require_directional_rri_monotonicity,
            direction,
        ) = keys
        pivot = group.pivot(index="subject", columns="phase", values="brs_value_ms_per_mmHg")
        aligned = pivot.reindex(columns=["Pre", "Stim"]).dropna()
        diff = aligned["Stim"].to_numpy(float) - aligned["Pre"].to_numpy(float)
        statistic, p_value, method = wilcoxon_two_sided(diff)
        base = {
            "setting_id": setting_id,
            "lag_beats": lag,
            "correlation_threshold": correlation,
            "sbp_increment_threshold": increment,
            "minimum_sequence_length": minimum_length,
            "aggregation": aggregation,
            "require_directional_rri_monotonicity": (
                require_directional_rri_monotonicity
            ),
            "direction": direction,
            "n_evaluable": len(diff),
            "mean_pre": float(aligned["Pre"].mean()) if len(diff) else np.nan,
            "mean_stim": float(aligned["Stim"].mean()) if len(diff) else np.nan,
            "mean_difference_stim_minus_pre": float(np.mean(diff)) if len(diff) else np.nan,
            "median_difference_stim_minus_pre": float(np.median(diff)) if len(diff) else np.nan,
            "cohens_dz": cohens_dz(diff),
            "wilcoxon_statistic": statistic,
            "wilcoxon_p_two_sided": p_value,
            "wilcoxon_method": method,
            "n_negative": int(np.sum(diff < 0.0)),
            "n_positive": int(np.sum(diff > 0.0)),
            "n_zero": int(np.sum(diff == 0.0)),
            "negative_direction": bool(np.isfinite(np.mean(diff)) and np.mean(diff) < 0.0),
            "sign_reversal_positive": bool(np.isfinite(np.mean(diff)) and np.mean(diff) > 0.0),
            "estimability_status": "estimable" if len(diff) >= 3 else "not_estimable",
            "NA_reason": "NA" if len(diff) >= 3 else "fewer_than_3_paired_subjects",
        }
        if setting_id in bootstrap_setting_ids:
            summary, replicates = paired_summary(
                aligned["Stim"].to_numpy(float),
                aligned["Pre"].to_numpy(float),
                "Stim",
                "Pre",
                seed=seed_base + group_index * 10,
            )
            base.update(
                {
                    "mean_difference_ci_low": summary["mean_difference_ci_low"],
                    "mean_difference_ci_high": summary["mean_difference_ci_high"],
                    "median_difference_ci_low": summary["median_difference_ci_low"],
                    "median_difference_ci_high": summary["median_difference_ci_high"],
                    "cohens_dz_ci_low": summary["cohens_dz_ci_low"],
                    "cohens_dz_ci_high": summary["cohens_dz_ci_high"],
                    "bootstrap_resamples": summary["bootstrap_resamples"],
                    "bootstrap_seed_mean": summary["bootstrap_seed_mean"],
                    "bootstrap_seed_median": summary["bootstrap_seed_median"],
                    "bootstrap_seed_dz": summary["bootstrap_seed_dz"],
                }
            )
            if (setting_id, direction) in retain_bootstrap_keys:
                for estimand, values in replicates.items():
                    for replicate_index, value in enumerate(values, start=1):
                        bootstrap_rows.append(
                            {
                                "setting_id": setting_id,
                                "direction": direction,
                                "estimand": estimand,
                                "replicate": replicate_index,
                                "value": value,
                            }
                        )
        else:
            base.update(
                {
                    "mean_difference_ci_low": np.nan,
                    "mean_difference_ci_high": np.nan,
                    "median_difference_ci_low": np.nan,
                    "median_difference_ci_high": np.nan,
                    "cohens_dz_ci_low": np.nan,
                    "cohens_dz_ci_high": np.nan,
                    "bootstrap_resamples": 0,
                    "bootstrap_seed_mean": np.nan,
                    "bootstrap_seed_median": np.nan,
                    "bootstrap_seed_dz": np.nan,
                }
            )
        rows.append(base)
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)
