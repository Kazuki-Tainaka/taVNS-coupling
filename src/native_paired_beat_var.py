"""Native paired-beat bivariate VAR analysis for the major revision.

This module is deliberately separate from the legacy 4-Hz significance
pipeline.  It fits one bivariate model per participant and phase to the same
non-interpolated paired-beat SBP/RRI records used by the sequence-BRS analysis.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import signal, stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR

from revision_utils import (
    PAIRED_DIR,
    PHASE_ORDER,
    PHASES,
    PROJECT_ROOT,
    SUBJECTS,
    load_paired_full,
)
from stats_core import cochran_q_test


ANALYSIS_DOMAIN: Final[str] = "native_paired_beats"
CANDIDATE_ORDER_MIN: Final[int] = 1
CANDIDATE_ORDER_MAX: Final[int] = 10
ALPHA: Final[float] = 0.05
WHITENESS_MIN_LAGS: Final[int] = 20
WHITENESS_ORDER_OFFSET: Final[int] = 5
WHITENESS_ADJUSTED: Final[bool] = True
DIRECTIONS: Final[tuple[tuple[str, str, list[str]], ...]] = (
    ("past_SBP_to_RRI", "RRI", ["SBP"]),
    ("past_RRI_to_SBP", "SBP", ["RRI"]),
)


@dataclass(frozen=True)
class FitInput:
    """Inputs and source traceability for one participant-phase model."""

    subject_id: str
    phase: str
    sbp: np.ndarray
    rri: np.ndarray
    source_file: str
    source_row_or_record: str


def _finite_float(value: Any) -> float:
    array = np.atleast_1d(value).astype(float)
    return float(array.flat[0])


def _explicit_na_row(base: dict[str, Any], direction: str, reason: str) -> dict[str, Any]:
    """Return a schema-complete failed-model row."""
    return {
        **base,
        "direction": direction,
        "n_effective_model_observations": "NA",
        "selected_order": "NA",
        "candidate_order_min": CANDIDATE_ORDER_MIN,
        "candidate_order_max": CANDIDATE_ORDER_MAX,
        "aic_selected_model": "NA",
        "f_statistic": "NA",
        "p_value": "NA",
        "q_within_phase_direction": "NA",
        "nominal_significant": "NA",
        "fdr_significant": "NA",
        "model_fit_status": "failed",
        "stability_status": "NA",
        "stability_metric_name": "maximum_companion_eigenvalue_modulus",
        "stability_metric_value": "NA",
        "statsmodels_inverse_root_minimum_modulus": "NA",
        "residual_whiteness_test": "adjusted_Portmanteau",
        "residual_whiteness_nlags": "NA",
        "residual_whiteness_df": "NA",
        "residual_whiteness_statistic": "NA",
        "residual_whiteness_p": "NA",
        "residual_normality_test": "Jarque_Bera_style_omnibus_chi_square",
        "residual_normality_df": "NA",
        "residual_normality_statistic": "NA",
        "residual_normality_p": "NA",
        "any_residual_diagnostic_failed": "NA",
        "full_model_definition": (
            "bivariate_VAR_p_on_detrended_z_scored_SBP_and_RRI_with_constant"
        ),
        "reduced_model_definition": (
            "same_caused_equation_and_order_with_all_lags_of_causing_variable_"
            "jointly_restricted_to_zero"
        ),
        "causality_test": "statsmodels_VARResults_test_causality_F_Wald",
        "model_constant": "included",
        "within_phase_detrending": "linear_scipy_signal_detrend",
        "within_phase_standardization": "z_score_population_sd_ddof_0",
        "NA_reason": reason,
    }


def prepare_participant_phase(subject: int, phase: str) -> FitInput:
    """Load one native participant-phase series with original row traceability."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    full = load_paired_full(subject).copy()
    full["source_row_1based"] = np.arange(1, len(full) + 1, dtype=int)
    t0, t1 = PHASES[phase]
    phase_mask = (full["beat_time_s"] >= t0) & (full["beat_time_s"] < t1)
    phase_frame = full.loc[phase_mask].copy()
    valid = (
        np.isfinite(phase_frame["RRI_ms"].to_numpy(float))
        & np.isfinite(phase_frame["SBP_mmHg"].to_numpy(float))
        & (phase_frame["RRI_ms"].to_numpy(float) > 0.0)
    )
    phase_frame = phase_frame.loc[valid].copy()
    if phase_frame.empty:
        source_record = "no_retained_rows"
    else:
        first = int(phase_frame["source_row_1based"].iloc[0])
        last = int(phase_frame["source_row_1based"].iloc[-1])
        source_record = (
            f"1-based_data_rows_{first}-{last};retained_rows={len(phase_frame)}"
        )
    source_path = PAIRED_DIR / f"paired_beats_{subject:02d}.csv"
    return FitInput(
        subject_id=f"S{subject:02d}",
        phase=phase,
        sbp=phase_frame["SBP_mmHg"].to_numpy(float),
        rri=phase_frame["RRI_ms"].to_numpy(float),
        source_file=source_path.relative_to(PROJECT_ROOT).as_posix(),
        source_row_or_record=source_record,
    )


def select_order_aic_1_to_10(model: VAR) -> tuple[int, list[dict[str, Any]]]:
    """Select AIC strictly among orders 1-10 using a common estimation sample."""
    selection = model.select_order(maxlags=CANDIDATE_ORDER_MAX, trend="c")
    aic_values = np.asarray(selection.ics["aic"], dtype=float)
    if len(aic_values) <= CANDIDATE_ORDER_MAX:
        raise ValueError(
            "statsmodels returned fewer AIC candidates than orders 0-10"
        )
    candidates = aic_values[CANDIDATE_ORDER_MIN : CANDIDATE_ORDER_MAX + 1]
    if not np.isfinite(candidates).any():
        raise ValueError("No finite AIC value among candidate orders 1-10")
    selected = int(np.nanargmin(candidates) + CANDIDATE_ORDER_MIN)
    rows = [
        {
            "candidate_order": order,
            "aic": float(aic_values[order]) if np.isfinite(aic_values[order]) else "NA",
            "selected": order == selected,
        }
        for order in range(CANDIDATE_ORDER_MIN, CANDIDATE_ORDER_MAX + 1)
    ]
    return selected, rows


def fit_native_var(fit_input: FitInput) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit one participant-phase model and return its two directional tests."""
    base: dict[str, Any] = {
        "subject_id": fit_input.subject_id,
        "phase": fit_input.phase,
        "analysis_domain": ANALYSIS_DOMAIN,
        "n_observations": len(fit_input.sbp),
        "source_file": fit_input.source_file,
        "source_row_or_record": fit_input.source_row_or_record,
    }
    try:
        if len(fit_input.sbp) != len(fit_input.rri):
            raise ValueError("SBP and RRI lengths differ")
        if len(fit_input.sbp) <= 5 * CANDIDATE_ORDER_MAX:
            raise ValueError("Too few observations for candidate order 10")
        if not np.isfinite(fit_input.sbp).all() or not np.isfinite(fit_input.rri).all():
            raise ValueError("Non-finite values remain after source QC")
        if np.any(fit_input.rri <= 0.0):
            raise ValueError("Non-positive RRI remains after source QC")

        sbp_detrended = signal.detrend(fit_input.sbp, type="linear")
        rri_detrended = signal.detrend(fit_input.rri, type="linear")
        sbp_sd = float(np.std(sbp_detrended, ddof=0))
        rri_sd = float(np.std(rri_detrended, ddof=0))
        if sbp_sd <= 0.0 or rri_sd <= 0.0:
            raise ValueError("Zero within-phase standard deviation")
        sbp_z = (sbp_detrended - np.mean(sbp_detrended)) / sbp_sd
        rri_z = (rri_detrended - np.mean(rri_detrended)) / rri_sd
        data = pd.DataFrame({"SBP": sbp_z, "RRI": rri_z})
        model = VAR(data)
        selected_order, order_rows = select_order_aic_1_to_10(model)
        fitted = model.fit(maxlags=selected_order, ic=None, trend="c")

        stable = bool(fitted.is_stable(verbose=False))
        inverse_roots = np.asarray(fitted.roots, dtype=complex)
        inverse_root_min = (
            float(np.min(np.abs(inverse_roots))) if len(inverse_roots) else np.nan
        )
        maximum_companion_modulus = (
            float(1.0 / inverse_root_min)
            if np.isfinite(inverse_root_min) and inverse_root_min > 0.0
            else np.nan
        )
        # Statsmodels returns inverse roots: stability requires every returned
        # root outside the unit circle.  The reciprocal metric stored here is
        # therefore a companion-eigenvalue modulus and must be below one.
        root_rule_stable = bool(
            np.isfinite(maximum_companion_modulus)
            and maximum_companion_modulus < 1.0
        )
        if stable != root_rule_stable:
            raise RuntimeError("Statsmodels stability result and root rule disagree")

        whiteness_nlags = max(
            WHITENESS_MIN_LAGS,
            selected_order + WHITENESS_ORDER_OFFSET,
        )
        if whiteness_nlags >= int(fitted.nobs):
            raise ValueError("Insufficient effective observations for whiteness lag")
        whiteness = fitted.test_whiteness(
            nlags=whiteness_nlags,
            adjusted=WHITENESS_ADJUSTED,
        )
        normality = fitted.test_normality(signif=ALPHA)
        diagnostic_failed = bool(
            float(whiteness.pvalue) < ALPHA
            or float(normality.pvalue) < ALPHA
        )

        common = {
            **base,
            "n_effective_model_observations": int(fitted.nobs),
            "selected_order": selected_order,
            "candidate_order_min": CANDIDATE_ORDER_MIN,
            "candidate_order_max": CANDIDATE_ORDER_MAX,
            "aic_selected_model": float(fitted.aic),
            "model_fit_status": "fit_succeeded",
            "stability_status": "stable" if stable else "unstable",
            "stability_metric_name": "maximum_companion_eigenvalue_modulus",
            "stability_metric_value": maximum_companion_modulus,
            "statsmodels_inverse_root_minimum_modulus": inverse_root_min,
            "residual_whiteness_test": "adjusted_Portmanteau",
            "residual_whiteness_nlags": whiteness_nlags,
            "residual_whiteness_df": int(whiteness.df),
            "residual_whiteness_statistic": float(whiteness.test_statistic),
            "residual_whiteness_p": float(whiteness.pvalue),
            "residual_normality_test": "Jarque_Bera_style_omnibus_chi_square",
            "residual_normality_df": int(normality.df),
            "residual_normality_statistic": float(normality.test_statistic),
            "residual_normality_p": float(normality.pvalue),
            "any_residual_diagnostic_failed": diagnostic_failed,
            "full_model_definition": (
                "bivariate_VAR_p_on_detrended_z_scored_SBP_and_RRI_with_constant"
            ),
            "reduced_model_definition": (
                "same_caused_equation_and_order_with_all_lags_of_causing_"
                "variable_jointly_restricted_to_zero"
            ),
            "causality_test": "statsmodels_VARResults_test_causality_F_Wald",
            "model_constant": "included",
            "within_phase_detrending": "linear_scipy_signal_detrend",
            "within_phase_standardization": "z_score_population_sd_ddof_0",
            "NA_reason": "NA",
        }
        directional_rows: list[dict[str, Any]] = []
        for direction, caused, causing in DIRECTIONS:
            test = fitted.test_causality(
                caused=caused,
                causing=causing,
                kind="f",
                signif=ALPHA,
            )
            p_value = float(test.pvalue)
            directional_rows.append(
                {
                    **common,
                    "direction": direction,
                    "f_statistic": _finite_float(test.test_statistic),
                    "p_value": p_value,
                    "q_within_phase_direction": "NA",
                    "nominal_significant": p_value < ALPHA,
                    "fdr_significant": "NA",
                }
            )
        order_trace = [
            {
                "subject_id": fit_input.subject_id,
                "phase": fit_input.phase,
                "analysis_domain": ANALYSIS_DOMAIN,
                **row,
            }
            for row in order_rows
        ]
        return directional_rows, order_trace
    except Exception as exc:  # A failed row must remain machine-auditable.
        reason = f"{type(exc).__name__}:{exc}"
        rows = [
            _explicit_na_row(base, direction=direction, reason=reason)
            for direction, _, _ in DIRECTIONS
        ]
        return rows, []


def apply_within_phase_direction_bh(results: pd.DataFrame) -> pd.DataFrame:
    """Apply BH separately to the 18 tests in each phase-direction family."""
    output = results.copy()
    for phase in PHASE_ORDER:
        for direction, _, _ in DIRECTIONS:
            mask = (output["phase"] == phase) & (output["direction"] == direction)
            numeric_p = pd.to_numeric(output.loc[mask, "p_value"], errors="coerce")
            valid = numeric_p.notna()
            q_values = pd.Series(np.nan, index=numeric_p.index, dtype=float)
            if valid.any():
                q_values.loc[valid] = multipletests(
                    numeric_p.loc[valid].to_numpy(float), method="fdr_bh"
                )[1]
            output.loc[mask, "q_within_phase_direction"] = q_values
            output.loc[mask, "fdr_significant"] = q_values < ALPHA
            failed = output.loc[mask, "model_fit_status"].ne("fit_succeeded")
            output.loc[mask & failed.reindex(output.index, fill_value=False), "fdr_significant"] = "NA"
    return output


def exact_clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    """Return the two-sided exact 95% Clopper-Pearson interval."""
    if trials <= 0:
        return np.nan, np.nan
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(ALPHA / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(
            stats.beta.ppf(
                1.0 - ALPHA / 2.0,
                successes + 1,
                trials - successes,
            )
        )
    )
    return lower, upper


def _paired_boolean_table(
    results: pd.DataFrame,
    direction: str,
    flag: str,
) -> pd.DataFrame:
    data = results[results["direction"] == direction].copy()
    data[flag] = data[flag].map({True: 1.0, False: 0.0})
    return data.pivot(index="subject_id", columns="phase", values=flag)


def summarize_prevalence(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize nominal and BH-significant prevalence with paired tests."""
    rows: list[dict[str, Any]] = []
    for direction, _, _ in DIRECTIONS:
        for flag in ("nominal_significant", "fdr_significant"):
            pivot = _paired_boolean_table(results, direction, flag)
            for phase in PHASE_ORDER:
                values = pivot[phase].dropna().astype(int)
                successes = int(values.sum())
                trials = int(len(values))
                ci_low, ci_high = exact_clopper_pearson(successes, trials)
                rows.append(
                    {
                        "analysis_domain": ANALYSIS_DOMAIN,
                        "direction": direction,
                        "significance_criterion": flag,
                        "summary_type": "phase_prevalence",
                        "phase_or_contrast": phase,
                        "successes": successes,
                        "trials": trials,
                        "prevalence": successes / trials if trials else "NA",
                        "exact_ci_95_low": ci_low,
                        "exact_ci_95_high": ci_high,
                        "test": "Clopper_Pearson_exact_binomial_interval",
                        "exact_or_asymptotic": "exact",
                        "test_statistic": "NA",
                        "p_value": "NA",
                        "discordant_pre_only": "NA",
                        "discordant_stim_only": "NA",
                        "status": "estimable" if trials else "not_estimable",
                        "NA_reason": "NA" if trials else "no_evaluable_participants",
                    }
                )

            paired = pivot[["Pre", "Stim"]].dropna().astype(bool)
            pre_only = int((paired["Pre"] & ~paired["Stim"]).sum())
            stim_only = int((~paired["Pre"] & paired["Stim"]).sum())
            discordant = pre_only + stim_only
            p_mcnemar = (
                1.0
                if discordant == 0
                else float(
                    stats.binomtest(
                        pre_only,
                        discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )
            )
            rows.append(
                {
                    "analysis_domain": ANALYSIS_DOMAIN,
                    "direction": direction,
                    "significance_criterion": flag,
                    "summary_type": "paired_prevalence_comparison",
                    "phase_or_contrast": "Stim-vs-Pre",
                    "successes": "NA",
                    "trials": len(paired),
                    "prevalence": "NA",
                    "exact_ci_95_low": "NA",
                    "exact_ci_95_high": "NA",
                    "test": "McNemar_exact_conditional_binomial",
                    "exact_or_asymptotic": "exact",
                    "test_statistic": min(pre_only, stim_only),
                    "p_value": p_mcnemar,
                    "discordant_pre_only": pre_only,
                    "discordant_stim_only": stim_only,
                    "status": "estimable",
                    "NA_reason": "NA",
                }
            )

            three_phase = pivot[list(PHASE_ORDER)].dropna().astype(int)
            q_result = cochran_q_test(three_phase.to_numpy())
            rows.append(
                {
                    "analysis_domain": ANALYSIS_DOMAIN,
                    "direction": direction,
                    "significance_criterion": flag,
                    "summary_type": "paired_prevalence_comparison",
                    "phase_or_contrast": "Pre-Stim-Post",
                    "successes": "NA",
                    "trials": len(three_phase),
                    "prevalence": "NA",
                    "exact_ci_95_low": "NA",
                    "exact_ci_95_high": "NA",
                    "test": "Cochran_Q_chi_square",
                    "exact_or_asymptotic": q_result["exact_or_asymptotic"],
                    "test_statistic": q_result["statistic"],
                    "p_value": q_result["p_value"],
                    "discordant_pre_only": "NA",
                    "discordant_stim_only": "NA",
                    "status": q_result["status"],
                    "NA_reason": q_result["NA_reason"],
                }
            )
    return pd.DataFrame(rows)


def summarize_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize the 54 unique fits rather than double-counting directions."""
    fit_columns = [
        "subject_id",
        "phase",
        "n_observations",
        "selected_order",
        "model_fit_status",
        "stability_status",
        "residual_whiteness_p",
        "residual_normality_p",
        "any_residual_diagnostic_failed",
    ]
    fits = results[fit_columns].drop_duplicates(["subject_id", "phase"])
    rows: list[dict[str, Any]] = []
    for phase in (*PHASE_ORDER, "All"):
        phase_fits = fits if phase == "All" else fits[fits["phase"] == phase]
        fit_ok = phase_fits["model_fit_status"].eq("fit_succeeded")
        selected = pd.to_numeric(phase_fits["selected_order"], errors="coerce")
        white_p = pd.to_numeric(phase_fits["residual_whiteness_p"], errors="coerce")
        normal_p = pd.to_numeric(phase_fits["residual_normality_p"], errors="coerce")
        n_total = int(len(phase_fits))
        n_fit = int(fit_ok.sum())
        rows.append(
            {
                "phase": phase,
                "unique_fit_count": n_total,
                "fit_succeeded_count": n_fit,
                "model_failure_count": n_total - n_fit,
                "model_failure_rate": (n_total - n_fit) / n_total if n_total else "NA",
                "order_10_count": int((selected == CANDIDATE_ORDER_MAX).sum()),
                "order_10_saturation_rate": (
                    float((selected == CANDIDATE_ORDER_MAX).sum() / n_fit)
                    if n_fit
                    else "NA"
                ),
                "selected_order_median": float(selected.median()) if n_fit else "NA",
                "selected_order_min": int(selected.min()) if n_fit else "NA",
                "selected_order_max": int(selected.max()) if n_fit else "NA",
                "whiteness_pass_count": int((white_p >= ALPHA).sum()),
                "whiteness_pass_rate": (
                    float((white_p >= ALPHA).sum() / white_p.notna().sum())
                    if white_p.notna().sum()
                    else "NA"
                ),
                "normality_pass_count": int((normal_p >= ALPHA).sum()),
                "normality_pass_rate": (
                    float((normal_p >= ALPHA).sum() / normal_p.notna().sum())
                    if normal_p.notna().sum()
                    else "NA"
                ),
                "both_residual_diagnostics_pass_count": int(
                    ((white_p >= ALPHA) & (normal_p >= ALPHA)).sum()
                ),
                "both_residual_diagnostics_pass_rate": (
                    float(
                        ((white_p >= ALPHA) & (normal_p >= ALPHA)).sum()
                        / max(white_p.notna().sum(), normal_p.notna().sum())
                    )
                    if max(white_p.notna().sum(), normal_p.notna().sum())
                    else "NA"
                ),
            }
        )
    return pd.DataFrame(rows)


def order_distribution(results: pd.DataFrame) -> pd.DataFrame:
    """Return selected-order counts for the 54 unique participant-phase fits."""
    fits = results[
        ["subject_id", "phase", "selected_order", "model_fit_status"]
    ].drop_duplicates(["subject_id", "phase"])
    selected = pd.to_numeric(fits["selected_order"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for phase in (*PHASE_ORDER, "All"):
        mask = pd.Series(True, index=fits.index) if phase == "All" else fits["phase"].eq(phase)
        values = selected.loc[mask].dropna().astype(int)
        for order in range(CANDIDATE_ORDER_MIN, CANDIDATE_ORDER_MAX + 1):
            rows.append(
                {
                    "phase": phase,
                    "selected_order": order,
                    "count": int((values == order).sum()),
                    "denominator": int(len(values)),
                    "proportion": float((values == order).mean()) if len(values) else "NA",
                }
            )
    return pd.DataFrame(rows)


def validate_no_empty_cells(frame: pd.DataFrame, name: str) -> None:
    """Fail if any exported cell would be blank rather than explicit NA."""
    rendered = frame.astype(object).where(pd.notna(frame), "NA")
    empty = rendered.map(lambda value: str(value).strip() == "").to_numpy().sum()
    if empty:
        raise ValueError(f"{name} contains {int(empty)} empty cells")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write UTF-8 CSV with explicit NA strings and no blank cells."""
    rendered = frame.astype(object).where(pd.notna(frame), "NA")
    validate_no_empty_cells(rendered, path.name)
    rendered.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def run_analysis(output_dir: Path) -> dict[str, Path]:
    """Run all 54 fits, FDR, prevalence, and diagnostic summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_rows: list[dict[str, Any]] = []
    order_trace_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for phase in PHASE_ORDER:
            fit_input = prepare_participant_phase(subject, phase)
            directional, order_trace = fit_native_var(fit_input)
            participant_rows.extend(directional)
            order_trace_rows.extend(order_trace)

    participant = apply_within_phase_direction_bh(pd.DataFrame(participant_rows))
    prevalence = summarize_prevalence(participant)
    diagnostics = summarize_diagnostics(participant)
    distribution = order_distribution(participant)
    order_trace = pd.DataFrame(order_trace_rows)

    paths = {
        "participant": output_dir
        / "native_paired_beat_var_participant_results.csv",
        "prevalence": output_dir
        / "native_paired_beat_var_prevalence.csv",
        "diagnostics": output_dir
        / "native_paired_beat_var_diagnostic_summary.csv",
        "distribution": output_dir
        / "native_paired_beat_var_order_distribution.csv",
        "order_trace": output_dir
        / "native_paired_beat_var_aic_trace.csv",
        "metadata": output_dir
        / "native_paired_beat_var_run_metadata.json",
    }
    write_csv(participant, paths["participant"])
    write_csv(prevalence, paths["prevalence"])
    write_csv(diagnostics, paths["diagnostics"])
    write_csv(distribution, paths["distribution"])
    write_csv(order_trace, paths["order_trace"])

    metadata = {
        "analysis_domain": ANALYSIS_DOMAIN,
        "variables": ["SBP_mmHg", "RRI_ms"],
        "sampling_domain": "beat_index",
        "interpolation": "none",
        "bandpass_filtering": "none",
        "phase_boundary_crossing": False,
        "detrending": "scipy.signal.detrend(type='linear') within phase",
        "standardization": "within-phase z-score using population SD (ddof=0)",
        "constant": "included",
        "candidate_orders": list(range(CANDIDATE_ORDER_MIN, CANDIDATE_ORDER_MAX + 1)),
        "selection": "minimum AIC among orders 1-10 from statsmodels common-sample select_order",
        "whiteness": {
            "test": "statsmodels adjusted Portmanteau",
            "nlags_rule": "max(20, selected_order + 5)",
            "null": "residual autocorrelation through nlags is zero",
        },
        "normality": {
            "test": "statsmodels Jarque-Bera-style omnibus chi-square",
            "degrees_of_freedom": "stored from test result; equals 2 times number of variables (4)",
            "null": "Gaussian-distributed residual process",
        },
        "stability": {
            "statsmodels_roots": "inverse roots; all moduli must exceed 1",
            "reported_metric": "maximum companion eigenvalue modulus = reciprocal of minimum inverse-root modulus",
            "reported_rule": "stable when reported metric is below 1",
        },
        "full_model": "bivariate VAR(p) on detrended z-scored SBP/RRI with constant",
        "reduced_model": "same caused equation and p with every lag of the causing variable jointly restricted to zero",
        "directional_test": "statsmodels VARResults.test_causality(kind='f') joint F/Wald test",
        "multiple_testing": "BH within each phase-direction family of 18 participants",
        "prevalence_interval": "two-sided exact 95% Clopper-Pearson",
        "pre_vs_stim": "two-sided exact conditional McNemar binomial test",
        "three_phase": (
            "Cochran Q with asymptotic chi-square reference when estimable; "
            "not estimable when all participants have identical binary status "
            "across phases"
        ),
        "unique_models": 54,
        "directional_rows": 108,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "outputs": {key: str(path) for key, path in paths.items() if key != "metadata"},
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for versioned analysis outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
