"""Subject-level coherence and Granger-predictability significance analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.api import VAR

from revision_utils import FS_4HZ, PHASE_ORDER, SUBJECTS, load_paired_phase, resample_phase_4hz
from stats_core import (
    apply_bh,
    cochran_q_test,
    exact_binomial_ci,
    mcnemar_exact,
)


MAYER_BAND = (0.08, 0.12)
COHERENCE_NPERSEG = 512
COHERENCE_NOVERLAP = 256


@dataclass(frozen=True)
class CoherenceEstimatorConfig:
    """Fully specified phase-level Mayer-band coherence estimator."""

    fs_hz: float = FS_4HZ
    window: str = "hann"
    nperseg: int = COHERENCE_NPERSEG
    noverlap: int = COHERENCE_NOVERLAP
    band_low_hz: float = MAYER_BAND[0]
    band_high_hz: float = MAYER_BAND[1]

    def validate(self) -> None:
        if self.fs_hz <= 0.0:
            raise ValueError("fs_hz must be positive")
        if self.nperseg <= 1:
            raise ValueError("nperseg must exceed one sample")
        if not 0 <= self.noverlap < self.nperseg:
            raise ValueError("noverlap must be non-negative and below nperseg")
        if not 0.0 <= self.band_low_hz <= self.band_high_hz <= self.fs_hz / 2.0:
            raise ValueError("coherence band must lie within the Nyquist range")


REFERENCE_COHERENCE_CONFIG = CoherenceEstimatorConfig()
SEGMENT_LENGTH_SENSITIVITY_CONFIG = CoherenceEstimatorConfig(
    nperseg=256,
    noverlap=128,
)


@dataclass(frozen=True)
class SyntheticValidationResult:
    scenario: str
    simulations: int
    detections: int
    detection_rate: float
    ci_low: float
    ci_high: float
    acceptance_rule: str
    passed: bool
    seed: int
    surrogates_per_simulation: int


def preprocess_for_coherence(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    return signal.detrend(data - np.nanmean(data))


def coherence_from_preprocessed(
    sbp_preprocessed: np.ndarray,
    rri_preprocessed: np.ndarray,
    config: CoherenceEstimatorConfig = REFERENCE_COHERENCE_CONFIG,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return mean Mayer-band magnitude-squared coherence and its spectrum."""
    config.validate()
    sbp = np.asarray(sbp_preprocessed, dtype=float)
    rri = np.asarray(rri_preprocessed, dtype=float)
    if len(sbp) < config.nperseg or len(rri) < config.nperseg:
        return np.nan, np.array([], dtype=float), np.array([], dtype=float)
    frequencies, sxx = signal.welch(
        sbp,
        fs=config.fs_hz,
        window=config.window,
        nperseg=config.nperseg,
        noverlap=config.noverlap,
    )
    _, syy = signal.welch(
        rri,
        fs=config.fs_hz,
        window=config.window,
        nperseg=config.nperseg,
        noverlap=config.noverlap,
    )
    _, sxy = signal.csd(
        sbp,
        rri,
        fs=config.fs_hz,
        window=config.window,
        nperseg=config.nperseg,
        noverlap=config.noverlap,
    )
    coherence = np.abs(sxy) ** 2 / (sxx * syy + 1e-20)
    mask = (frequencies >= config.band_low_hz) & (
        frequencies <= config.band_high_hz
    )
    value = float(np.nanmean(coherence[mask])) if np.any(mask) else np.nan
    return value, frequencies, coherence


def coherence_statistic(
    sbp_4hz: np.ndarray,
    rri_4hz: np.ndarray,
    config: CoherenceEstimatorConfig = REFERENCE_COHERENCE_CONFIG,
) -> float:
    sbp = preprocess_for_coherence(sbp_4hz)
    rri = preprocess_for_coherence(rri_4hz)
    value, _, _ = coherence_from_preprocessed(sbp, rri, config=config)
    return value


def effective_welch_segment_count(
    n_samples: int,
    config: CoherenceEstimatorConfig,
) -> int:
    """Return the number of complete overlapping Welch segments."""
    config.validate()
    if n_samples < config.nperseg:
        return 0
    step = config.nperseg - config.noverlap
    return 1 + (n_samples - config.nperseg) // step


def phase_randomize(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserve the rFFT magnitude spectrum while randomizing positive phases."""
    data = np.asarray(values, dtype=float)
    spectrum = np.fft.rfft(data)
    randomized = spectrum.copy()
    if len(data) % 2 == 0:
        random_indices = np.arange(1, len(spectrum) - 1)
    else:
        random_indices = np.arange(1, len(spectrum))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(random_indices))
    randomized[random_indices] = np.abs(spectrum[random_indices]) * np.exp(1j * phases)
    randomized[0] = spectrum[0].real + 0j
    if len(data) % 2 == 0:
        randomized[-1] = spectrum[-1].real + 0j
    return np.fft.irfft(randomized, n=len(data))


def coherence_surrogate_test(
    sbp_4hz: np.ndarray,
    rri_4hz: np.ndarray,
    n_surrogates: int,
    seed: int,
) -> tuple[dict[str, float | int | bool], np.ndarray]:
    sbp = preprocess_for_coherence(sbp_4hz)
    rri = preprocess_for_coherence(rri_4hz)
    observed, frequencies, _ = coherence_from_preprocessed(sbp, rri)
    if not np.isfinite(observed):
        return (
            {
                "observed_coherence": np.nan,
                "null_threshold_95": np.nan,
                "monte_carlo_p": np.nan,
                "exceedances": 0,
                "significant_nominal": False,
                "frequency_bin_count": 0,
            },
            np.full(n_surrogates, np.nan, dtype=float),
        )
    rng = np.random.default_rng(seed)
    null_values = np.empty(n_surrogates, dtype=float)
    for index in range(n_surrogates):
        sbp_null = phase_randomize(sbp, rng)
        rri_null = phase_randomize(rri, rng)
        null_values[index], _, _ = coherence_from_preprocessed(sbp_null, rri_null)
    finite = null_values[np.isfinite(null_values)]
    exceedances = int(np.sum(finite >= observed))
    p_value = (1.0 + exceedances) / (len(finite) + 1.0)
    band_count = int(np.sum((frequencies >= MAYER_BAND[0]) & (frequencies <= MAYER_BAND[1])))
    return (
        {
            "observed_coherence": observed,
            "null_threshold_95": float(np.quantile(finite, 0.95, method="linear")),
            "monte_carlo_p": float(p_value),
            "exceedances": exceedances,
            "significant_nominal": bool(p_value < 0.05),
            "frequency_bin_count": band_count,
        },
        null_values,
    )


def compute_coherence_significance(
    n_surrogates: int = 1_000,
    master_seed: int = 20_260_805,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the frozen surrogate test for all subject-phase records."""
    combinations = [(subject, phase) for subject in SUBJECTS for phase in PHASE_ORDER]
    seed_sequences = np.random.SeedSequence(master_seed).spawn(len(combinations))
    rows: list[dict[str, float | int | str | bool]] = []
    null_rows: list[dict[str, float | int | str]] = []
    for (subject, phase), seed_sequence in zip(combinations, seed_sequences, strict=True):
        seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        frame = load_paired_phase(subject, phase)
        grid, sbp_4hz, rri_4hz = resample_phase_4hz(frame)
        result, null_values = coherence_surrogate_test(
            sbp_4hz,
            rri_4hz,
            n_surrogates=n_surrogates,
            seed=seed,
        )
        rows.append(
            {
                "subject": subject,
                "phase": phase,
                "n_beats": len(frame),
                "n_4hz": len(grid),
                "surrogate_method": "independent_fourier_phase_randomization_both_series",
                "n_surrogates": n_surrogates,
                "seed": seed,
                **result,
                "estimability_status": "estimable" if np.isfinite(result["observed_coherence"]) else "not_estimable",
                "NA_reason": "NA" if np.isfinite(result["observed_coherence"]) else "insufficient_4hz_samples",
            }
        )
        for surrogate_index, value in enumerate(null_values, start=1):
            null_rows.append(
                {
                    "subject": subject,
                    "phase": phase,
                    "surrogate": surrogate_index,
                    "coherence_null": value,
                }
            )

    subject_results = pd.DataFrame(rows)
    subject_results["q_within_phase"] = np.nan
    subject_results["significant_fdr_within_phase"] = False
    for phase in PHASE_ORDER:
        mask = subject_results["phase"] == phase
        q_values = apply_bh(subject_results.loc[mask, "monte_carlo_p"].to_numpy(float))
        subject_results.loc[mask, "q_within_phase"] = q_values
        subject_results.loc[mask, "significant_fdr_within_phase"] = q_values < 0.05
    prevalence = summarize_binary_prevalence(
        subject_results,
        flag_column="significant_nominal",
        analysis="Mayer_band_coherence_surrogate",
    )
    return subject_results, pd.DataFrame(null_rows), prevalence


def summarize_binary_prevalence(
    subject_results: pd.DataFrame,
    flag_column: str,
    analysis: str,
    direction: str = "NA",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    complete_flags: dict[str, pd.Series] = {}
    for phase in PHASE_ORDER:
        phase_data = subject_results[subject_results["phase"] == phase].copy()
        phase_data = phase_data.dropna(subset=[flag_column])
        flags = phase_data.set_index("subject")[flag_column].astype(bool)
        complete_flags[phase] = flags
        successes = int(flags.sum())
        trials = len(flags)
        ci_low, ci_high = exact_binomial_ci(successes, trials)
        rows.append(
            {
                "analysis": analysis,
                "direction": direction,
                "summary_type": "phase_prevalence",
                "phase_or_contrast": phase,
                "successes": successes,
                "trials": trials,
                "prevalence": successes / trials if trials else np.nan,
                "ci_low_exact": ci_low,
                "ci_high_exact": ci_high,
                "test_statistic": np.nan,
                "p_value": np.nan,
                "test": "exact_Clopper_Pearson_CI",
            }
        )

    pre_stim_subjects = complete_flags["Pre"].index.intersection(complete_flags["Stim"].index)
    mcnemar = mcnemar_exact(
        complete_flags["Pre"].loc[pre_stim_subjects].to_numpy(bool),
        complete_flags["Stim"].loc[pre_stim_subjects].to_numpy(bool),
    )
    rows.append(
        {
            "analysis": analysis,
            "direction": direction,
            "summary_type": "prevalence_comparison",
            "phase_or_contrast": "Stim-vs-Pre",
            "successes": np.nan,
            "trials": len(pre_stim_subjects),
            "prevalence": np.nan,
            "ci_low_exact": np.nan,
            "ci_high_exact": np.nan,
            "test_statistic": mcnemar["statistic"],
            "p_value": mcnemar["p_exact_two_sided"],
            "test": "exact_McNemar",
        }
    )

    all_subjects = complete_flags["Pre"].index
    for phase in ("Stim", "Post"):
        all_subjects = all_subjects.intersection(complete_flags[phase].index)
    matrix = np.column_stack(
        [complete_flags[phase].loc[all_subjects].to_numpy(bool) for phase in PHASE_ORDER]
    )
    q_result = cochran_q_test(matrix)
    rows.append(
        {
            "analysis": analysis,
            "direction": direction,
            "summary_type": "prevalence_comparison",
            "phase_or_contrast": "Pre-Stim-Post",
            "successes": np.nan,
            "trials": len(all_subjects),
            "prevalence": np.nan,
            "ci_low_exact": np.nan,
            "ci_high_exact": np.nan,
            "test_statistic": q_result["statistic"],
            "p_value": q_result["p_value"],
            "test": "Cochran_Q",
        }
    )
    return pd.DataFrame(rows)


def _ar1_noise(length: int, coefficient: float, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(size=length)
    return signal.lfilter([1.0], [1.0, -coefficient], white)


def validate_coherence_surrogates(
    simulations: int = 200,
    n_surrogates: int = 199,
    seed: int = 20_260_806,
) -> pd.DataFrame:
    """Estimate synthetic false-positive rate and shared-signal sensitivity."""
    rng = np.random.default_rng(seed)
    length = 1_200
    time = np.arange(length) / FS_4HZ
    records: list[SyntheticValidationResult] = []
    for scenario in ("uncoupled_colored_noise", "coupled_shared_0p1Hz"):
        detections = 0
        for simulation_index in range(simulations):
            first = _ar1_noise(length, 0.85, rng)
            second = _ar1_noise(length, 0.80, rng)
            if scenario == "coupled_shared_0p1Hz":
                # A stochastic Mayer-band component exercises coherence across
                # multiple Fourier bins. A single deterministic sinusoid is an
                # invalid sensitivity fixture for magnitude-squared coherence:
                # independent whole-record phase shifts leave its coherence
                # magnitude unchanged.
                sos = signal.butter(
                    4,
                    [MAYER_BAND[0], MAYER_BAND[1]],
                    btype="bandpass",
                    fs=FS_4HZ,
                    output="sos",
                )
                shared = signal.sosfiltfilt(sos, rng.normal(size=length))
                shared = 2.5 * (shared - np.mean(shared)) / np.std(shared)
                first = first + shared
                second = second + 0.9 * shared
            test_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            result, _ = coherence_surrogate_test(
                first,
                second,
                n_surrogates=n_surrogates,
                seed=test_seed,
            )
            detections += int(result["significant_nominal"])
        rate = detections / simulations
        ci_low, ci_high = exact_binomial_ci(detections, simulations)
        if scenario == "uncoupled_colored_noise":
            rule = "exact_binomial_95CI_contains_0.05"
            passed = bool(ci_low <= 0.05 <= ci_high)
        else:
            rule = "detection_rate_at_least_0.80"
            passed = bool(rate >= 0.80)
        records.append(
            SyntheticValidationResult(
                scenario=scenario,
                simulations=simulations,
                detections=detections,
                detection_rate=rate,
                ci_low=ci_low,
                ci_high=ci_high,
                acceptance_rule=rule,
                passed=passed,
                seed=seed,
                surrogates_per_simulation=n_surrogates,
            )
        )
    return pd.DataFrame([record.__dict__ for record in records])


def compute_gc_significance(max_order: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit bivariate VARs and summarize subject-level directional predictability."""
    rows: list[dict[str, float | int | str | bool]] = []
    for subject in SUBJECTS:
        for phase in PHASE_ORDER:
            frame = load_paired_phase(subject, phase)
            grid, sbp_4hz, rri_4hz = resample_phase_4hz(frame)
            base = {
                "subject": subject,
                "phase": phase,
                "n_beats": len(frame),
                "n_4hz": len(grid),
                "max_order": max_order,
            }
            try:
                sbp = signal.detrend(sbp_4hz)
                rri = signal.detrend(rri_4hz)
                sbp = (sbp - np.nanmean(sbp)) / np.nanstd(sbp)
                rri = (rri - np.nanmean(rri)) / np.nanstd(rri)
                data = pd.DataFrame({"SBP": sbp, "RRI": rri})
                model = VAR(data)
                fitted = model.fit(maxlags=max_order, ic="aic", trend="c")
                order = int(fitted.k_ar)
                if order < 1:
                    raise ValueError("AIC selected order zero; causality not estimable")
                stable = bool(fitted.is_stable(verbose=False))
                roots = np.asarray(fitted.roots)
                minimum_root_modulus = float(np.min(np.abs(roots))) if len(roots) else np.nan
                whiteness_lags = max(20, order + 5)
                try:
                    whiteness = fitted.test_whiteness(whiteness_lags, adjusted=True)
                    whiteness_statistic = float(whiteness.test_statistic)
                    whiteness_p = float(whiteness.pvalue)
                except Exception:
                    whiteness_statistic = np.nan
                    whiteness_p = np.nan
                try:
                    normality = fitted.test_normality()
                    normality_statistic = float(normality.test_statistic)
                    normality_p = float(normality.pvalue)
                except Exception:
                    normality_statistic = np.nan
                    normality_p = np.nan

                directions = (
                    ("SBP_to_RRI", "RRI", ["SBP"]),
                    ("RRI_to_SBP", "SBP", ["RRI"]),
                )
                for direction, caused, causing in directions:
                    test = fitted.test_causality(caused=caused, causing=causing, kind="f")
                    eligible = stable and np.isfinite(test.pvalue)
                    rows.append(
                        {
                            **base,
                            "direction": direction,
                            "selected_order_aic": order,
                            "model_stable": stable,
                            "minimum_root_modulus": minimum_root_modulus,
                            "residual_whiteness_lags": whiteness_lags,
                            "residual_whiteness_statistic": whiteness_statistic,
                            "residual_whiteness_p": whiteness_p,
                            "residual_normality_statistic": normality_statistic,
                            "residual_normality_p": normality_p,
                            "f_statistic": float(np.atleast_1d(test.test_statistic)[0]),
                            "p_value": float(test.pvalue),
                            "significant_nominal": bool(test.pvalue < 0.05) if eligible else np.nan,
                            "model_status": "stable_estimable" if eligible else "unstable_not_estimable",
                            "estimability_status": "estimable" if eligible else "not_estimable",
                            "NA_reason": "NA" if eligible else "unstable_var_model",
                        }
                    )
            except Exception as error:
                for direction in ("SBP_to_RRI", "RRI_to_SBP"):
                    rows.append(
                        {
                            **base,
                            "direction": direction,
                            "selected_order_aic": np.nan,
                            "model_stable": False,
                            "minimum_root_modulus": np.nan,
                            "residual_whiteness_lags": np.nan,
                            "residual_whiteness_statistic": np.nan,
                            "residual_whiteness_p": np.nan,
                            "residual_normality_statistic": np.nan,
                            "residual_normality_p": np.nan,
                            "f_statistic": np.nan,
                            "p_value": np.nan,
                            "significant_nominal": np.nan,
                            "model_status": "failed",
                            "estimability_status": "not_estimable",
                            "NA_reason": f"VAR_failure:{type(error).__name__}:{error}",
                        }
                    )

    results = pd.DataFrame(rows)
    results["q_within_phase_direction"] = np.nan
    results["significant_fdr_within_phase_direction"] = False
    for phase in PHASE_ORDER:
        for direction in ("SBP_to_RRI", "RRI_to_SBP"):
            mask = (results["phase"] == phase) & (results["direction"] == direction)
            q_values = apply_bh(results.loc[mask, "p_value"].to_numpy(float))
            results.loc[mask, "q_within_phase_direction"] = q_values
            results.loc[mask, "significant_fdr_within_phase_direction"] = q_values < 0.05

    summaries = []
    for direction in ("SBP_to_RRI", "RRI_to_SBP"):
        direction_data = results[results["direction"] == direction]
        summaries.append(
            summarize_binary_prevalence(
                direction_data,
                flag_column="significant_nominal",
                analysis="bivariate_VAR_lag_dependent_predictability",
                direction=direction,
            )
        )
    return results, pd.concat(summaries, ignore_index=True)
