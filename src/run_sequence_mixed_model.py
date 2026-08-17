"""Exploratory sequence-level mixed model fixed in the pre-analysis plan.

The model is attempted only after the eligibility gate has confirmed that every
participant contributes at least two qualifying sequences in both Pre and Stim.
It is a sensitivity analysis and does not replace the participant-level paired
canonical analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from revision_utils import REVISION_ROOT, write_csv


RESULTS_DIR = REVISION_ROOT / "02_analysis" / "results"
REPORTS_DIR = REVISION_ROOT / "02_analysis" / "reports"


def main() -> None:
    source = RESULTS_DIR / "canonical_brs_sequence_level.csv"
    frame = pd.read_csv(source)
    model_frame = frame.loc[
        frame["phase"].isin(["Pre", "Stim"])
        & frame["qualifying_brs_sequence"].astype(bool),
        ["subject", "phase", "direction", "slope_ms_per_mmHg"],
    ].dropna()
    model_frame["subject"] = model_frame["subject"].astype(str)
    model_frame["phase"] = pd.Categorical(
        model_frame["phase"], categories=["Pre", "Stim"], ordered=True
    )

    counts = (
        model_frame.groupby(["subject", "phase"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    write_csv(counts, RESULTS_DIR / "sequence_mixed_model_eligibility_counts.csv")
    eligible = bool((counts[["Pre", "Stim"]] >= 2).all(axis=None))

    diagnostic: dict[str, object] = {
        "source_file": str(source.relative_to(REVISION_ROOT)),
        "model": (
            "slope_ms_per_mmHg ~ C(phase, Treatment(reference='Pre')); "
            "participant random intercept; qualifying sequences; Pre and Stim"
        ),
        "eligibility_rule": (
            "every participant has at least two qualifying sequences in Pre and Stim"
        ),
        "eligible": eligible,
        "n_participants": int(model_frame["subject"].nunique()),
        "n_sequences": int(len(model_frame)),
        "minimum_sequences_per_subject_phase": int(
            counts[["Pre", "Stim"]].min(axis=None)
        ),
        "estimability_status": "not_estimable",
        "NA_reason": "eligibility_gate_failed" if not eligible else "NA",
    }

    if not eligible:
        (RESULTS_DIR / "sequence_mixed_model_diagnostics.json").write_text(
            json.dumps(diagnostic, indent=2), encoding="utf-8"
        )
        return

    model = smf.mixedlm(
        "slope_ms_per_mmHg ~ C(phase, Treatment(reference='Pre'))",
        data=model_frame,
        groups=model_frame["subject"],
        re_formula="1",
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        fit = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)

    warning_messages = [str(item.message) for item in caught_warnings]
    random_variance = float(np.asarray(fit.cov_re)[0, 0])
    residual_variance = float(fit.scale)
    variance_ratio = random_variance / residual_variance if residual_variance > 0 else np.nan
    singular = bool(
        (not np.isfinite(random_variance))
        or random_variance <= 1e-8
        or (np.isfinite(variance_ratio) and variance_ratio <= 1e-8)
        or any("singular" in message.lower() for message in warning_messages)
    )

    coefficients = pd.DataFrame(
        {
            "term": fit.params.index,
            "estimate": fit.params.to_numpy(float),
            "standard_error": fit.bse.to_numpy(float),
            "z_or_wald_statistic": fit.tvalues.to_numpy(float),
            "p_two_sided": fit.pvalues.to_numpy(float),
            "ci95_lower": fit.conf_int()[0].to_numpy(float),
            "ci95_upper": fit.conf_int()[1].to_numpy(float),
            "estimability_status": (
                "estimable" if bool(fit.converged) and not singular else "not_estimable"
            ),
            "NA_reason": (
                "NA"
                if bool(fit.converged) and not singular
                else "nonconvergence_or_singular_random_intercept"
            ),
        }
    )
    write_csv(coefficients, RESULTS_DIR / "sequence_mixed_model_coefficients.csv")

    condition_number = float(np.linalg.cond(np.asarray(model.exog, dtype=float)))
    residual_summary: dict[str, float | str | None] = {
        "residual_mean": None,
        "residual_sd": None,
        "residual_skewness": None,
        "residual_excess_kurtosis": None,
        "residual_shapiro_w": None,
        "residual_shapiro_p": None,
        "abs_residual_vs_fitted_spearman_rho": None,
        "abs_residual_vs_fitted_spearman_p": None,
        "residual_diagnostics_NA_reason": (
            "singular_random_effects_prevent_prediction" if singular else "NA"
        ),
    }
    if not singular:
        residuals = np.asarray(fit.resid, dtype=float)
        fitted = np.asarray(fit.fittedvalues, dtype=float)
        shapiro_values = residuals
        if len(shapiro_values) > 5000:
            rng = np.random.default_rng(2026080506)
            shapiro_values = rng.choice(shapiro_values, size=5000, replace=False)
        shapiro = stats.shapiro(shapiro_values)
        heteroscedasticity = stats.spearmanr(np.abs(residuals), fitted)
        residual_summary = {
            "residual_mean": float(np.mean(residuals)),
            "residual_sd": float(np.std(residuals, ddof=1)),
            "residual_skewness": float(stats.skew(residuals, bias=False)),
            "residual_excess_kurtosis": float(stats.kurtosis(residuals, bias=False)),
            "residual_shapiro_w": float(shapiro.statistic),
            "residual_shapiro_p": float(shapiro.pvalue),
            "abs_residual_vs_fitted_spearman_rho": float(
                heteroscedasticity.statistic
            ),
            "abs_residual_vs_fitted_spearman_p": float(heteroscedasticity.pvalue),
            "residual_diagnostics_NA_reason": "NA",
        }

    diagnostic.update(
        {
            "converged": bool(fit.converged),
            "optimizer": "lbfgs",
            "reml": True,
            "log_likelihood": float(fit.llf) if np.isfinite(fit.llf) else None,
            "random_intercept_variance": random_variance,
            "residual_variance": residual_variance,
            "random_to_residual_variance_ratio": variance_ratio,
            "singular_by_variance_threshold_1e-8": singular,
            "fixed_effect_design_condition_number": condition_number,
            "warnings": warning_messages,
            **residual_summary,
            "estimability_status": (
                "estimable" if bool(fit.converged) and not singular else "not_estimable"
            ),
            "NA_reason": (
                "NA"
                if bool(fit.converged) and not singular
                else "nonconvergence_or_singular_random_intercept"
            ),
            "interpretation_rule": (
                "The participant-level paired canonical analysis remains primary. "
                "Marked residual non-normality or variance-pattern evidence is recorded "
                "as a limitation and precludes treating this sensitivity model as decisive."
            ),
        }
    )
    (RESULTS_DIR / "sequence_mixed_model_diagnostics.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )

    report = [
        "# Sequence-level mixed-effects sensitivity model",
        "",
        "This exploratory model was attempted because every participant contributed at least two qualifying sequences in both Pre and Stim. It does not replace the participant-level paired canonical analysis.",
        "",
        f"- Eligible: `{eligible}`",
        f"- Participants: `{diagnostic['n_participants']}`",
        f"- Qualifying sequences: `{diagnostic['n_sequences']}`",
        f"- Minimum sequences per participant-phase: `{diagnostic['minimum_sequences_per_subject_phase']}`",
        f"- Converged: `{diagnostic['converged']}`",
        f"- Singular random-intercept fit: `{diagnostic['singular_by_variance_threshold_1e-8']}`",
        f"- Residual Shapiro-Wilk p: `{diagnostic['residual_shapiro_p']}`",
        f"- |residual| versus fitted Spearman p: `{diagnostic['abs_residual_vs_fitted_spearman_p']}`",
        "",
    ]
    if singular or not bool(fit.converged):
        report.extend(
            [
                "",
                "The random-intercept covariance collapsed to zero and the Hessian was not positive definite. In accordance with the pre-analysis plan, this model is classified as not estimable and is omitted from scientific interpretation. The failed-fit coefficient file is retained only as a diagnostic record.",
            ]
        )
    else:
        report.extend(
            [
                "",
                "## Coefficients",
                "",
                coefficients.to_markdown(index=False),
                "",
                "The coefficient for Stim is the sequence-level mean slope difference relative to Pre under a participant random-intercept model. Sequence observations are not an independent-subject sample, and the unequal number of sequences per participant is an additional limitation.",
            ]
        )
    (REPORTS_DIR / "sequence_mixed_model_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
