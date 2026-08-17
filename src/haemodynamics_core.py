"""Basic haemodynamic and representative HRV context analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from revision_utils import (
    FS_4HZ,
    PHASE_ORDER,
    SUBJECTS,
    load_paired_phase,
    resample_phase_4hz,
)
from stats_core import apply_bh, cohens_dz, wilcoxon_two_sided


def band_power(
    frequencies: np.ndarray,
    power: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    mask = (frequencies >= lower) & (frequencies <= upper)
    if np.sum(mask) < 2:
        return np.nan
    return float(np.trapezoid(power[mask], frequencies[mask]))


def compute_subject_phase_context() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subject in SUBJECTS:
        for phase in PHASE_ORDER:
            paired = load_paired_phase(subject, phase)
            rri = paired["RRI_ms"].to_numpy(float)
            sbp = paired["SBP_mmHg"].to_numpy(float)
            hr = 60_000.0 / rri
            rri_diff = np.diff(rri)
            grid, _, rri_4hz = resample_phase_4hz(paired)
            if len(rri_4hz) >= 256:
                detrended = signal.detrend(rri_4hz)
                frequencies, psd = signal.welch(
                    detrended,
                    fs=FS_4HZ,
                    window="hann",
                    nperseg=256,
                    noverlap=128,
                )
                hf_power = band_power(frequencies, psd, 0.15, 0.40)
            else:
                hf_power = np.nan
            rows.append(
                {
                    "subject": subject,
                    "phase": phase,
                    "n_beats": len(paired),
                    "duration_from_first_to_last_beat_s": float(
                        paired["beat_time_s"].iloc[-1] - paired["beat_time_s"].iloc[0]
                    ),
                    "Mean_RRI_ms": float(np.mean(rri)),
                    "Mean_HR_bpm": float(np.mean(hr)),
                    "SBP_mean_mmHg": float(np.mean(sbp)),
                    "SBP_SD_mmHg": float(np.std(sbp, ddof=1)),
                    "RRI_SD_ms": float(np.std(rri, ddof=1)),
                    "RMSSD_ms": float(np.sqrt(np.mean(rri_diff**2))),
                    "HF_HRV_ms2": hf_power,
                    "estimability_status": "estimable",
                    "NA_reason": "NA",
                }
            )
    return pd.DataFrame(rows)


def summarize_context(subject_phase: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "n_beats",
        "Mean_RRI_ms",
        "Mean_HR_bpm",
        "SBP_mean_mmHg",
        "SBP_SD_mmHg",
        "RRI_SD_ms",
        "RMSSD_ms",
        "HF_HRV_ms2",
    ]
    summary_rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        for phase in PHASE_ORDER:
            values = subject_phase.loc[subject_phase["phase"] == phase, metric].to_numpy(float)
            values = values[np.isfinite(values)]
            q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
            summary_rows.append(
                {
                    "metric": metric,
                    "phase": phase,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                }
            )

    contrast_rows: list[dict[str, float | int | str]] = []
    comparisons = (("Stim", "Pre"), ("Post", "Pre"), ("Post", "Stim"))
    for metric in metrics:
        pivot = subject_phase.pivot(index="subject", columns="phase", values=metric)
        for first, second in comparisons:
            aligned = pivot[[first, second]].dropna()
            diff = aligned[first].to_numpy(float) - aligned[second].to_numpy(float)
            statistic, p_value, method = wilcoxon_two_sided(diff)
            contrast_rows.append(
                {
                    "metric": metric,
                    "contrast": f"{first}-{second}",
                    "n": len(diff),
                    "mean_first": float(aligned[first].mean()),
                    "mean_second": float(aligned[second].mean()),
                    "mean_difference": float(np.mean(diff)),
                    "median_difference": float(np.median(diff)),
                    "cohens_dz": cohens_dz(diff),
                    "wilcoxon_statistic": statistic,
                    "p_two_sided": p_value,
                    "wilcoxon_method": method,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["q_BH_within_contrast_8_context_metrics"] = np.nan
    for contrast in contrasts["contrast"].unique():
        mask = contrasts["contrast"] == contrast
        contrasts.loc[mask, "q_BH_within_contrast_8_context_metrics"] = apply_bh(
            contrasts.loc[mask, "p_two_sided"].to_numpy(float)
        )
    return pd.DataFrame(summary_rows), contrasts
