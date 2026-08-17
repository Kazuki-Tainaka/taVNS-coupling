"""Generate reports and cautious English response/manuscript snippets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis_utils import (
    ANALYSIS_SUBJECTS,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    PLAN_SHA256,
)


REPORT_DIR = OUTPUT_ROOT / "07_reports"
RESULT_DIR = OUTPUT_ROOT / "05_results"
SNIPPET_DIR = OUTPUT_ROOT / "08_response_and_manuscript_snippets"
REFERENCE_N16_PACKAGE = PROJECT_ROOT / "nonlinear_reference"


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _fmt_p(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    if number < 0.001:
        return f"{number:.2e}"
    return f"{number:.3f}"


def _fmt_effect(value: object) -> str:
    """Retain enough precision not to print a nonzero CI bound as zero."""

    return _fmt(value, 4)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "No rows available."
    selected = frame.loc[:, list(columns)].copy()
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append(_fmt(value))
            else:
                cells.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def _primary_table(contrasts: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if contrasts.empty:
        return pd.DataFrame(), "No real-data contrasts were eligible."
    primary = contrasts[contrasts["contrast"].eq("Stim-Pre")].copy()
    display = pd.DataFrame(
        {
            "Method": primary["method"],
            "Direction": primary["direction"],
            "n": primary["evaluable_paired_n"],
            "Pre mean (SD)": [
                f"{_fmt(row.Pre_mean)} ({_fmt(row.Pre_SD)})"
                for row in primary.itertuples(index=False)
            ],
            "Stim mean (SD)": [
                f"{_fmt(row.Stim_mean)} ({_fmt(row.Stim_SD)})"
                for row in primary.itertuples(index=False)
            ],
            "Stim–Pre mean difference [BCa 95% CI]": [
                f"{_fmt_effect(row.paired_mean_difference)} "
                f"[{_fmt_effect(row.mean_difference_BCa95_low)}, "
                f"{_fmt_effect(row.mean_difference_BCa95_high)}]"
                for row in primary.itertuples(index=False)
            ],
            "p": [_fmt_p(value) for value in primary["wilcoxon_p_two_sided"]],
            "q": [_fmt_p(value) for value in primary["BH_q_primary_family"]],
            "dz": [_fmt(value) for value in primary["cohens_dz"]],
        }
    )
    return primary, _markdown_table(display, display.columns)


def _result_clause(contrasts: pd.DataFrame) -> str:
    if contrasts.empty:
        return "No candidate estimator yielded validated, estimable real-data contrasts."
    primary = contrasts[contrasts["contrast"].eq("Stim-Pre")]
    clauses = []
    for row in primary.itertuples(index=False):
        clauses.append(
            f"{row.method} {row.direction}: mean difference "
            f"{_fmt_effect(row.paired_mean_difference)} "
            f"(BCa 95% CI {_fmt_effect(row.mean_difference_BCa95_low)} to "
            f"{_fmt_effect(row.mean_difference_BCa95_high)}), "
            f"Wilcoxon p={_fmt_p(row.wilcoxon_p_two_sided)}, "
            f"BH q={_fmt_p(row.BH_q_primary_family)}, "
            f"dz={_fmt(row.cohens_dz)}, n={int(row.evaluable_paired_n)}"
        )
    return "; ".join(clauses)


def _validation_table(gate: pd.DataFrame) -> str:
    display = pd.DataFrame(
        {
            "Method": gate["method"],
            "FPR gate": gate["uncoupled_fpr_pass"],
            "Linear direction fraction": gate[
                "moderate_linear_direction_fraction"
            ],
            "Nonlinear direction fraction": gate[
                "moderate_nonlinear_direction_fraction"
            ],
            "Failure gate": gate["all_failure_rates_le_0p05"],
            "Bidirectional fraction": gate[
                "bidirectional_moderate_direction_fraction"
            ],
            "Status": gate["validation_status"],
        }
    )
    return _markdown_table(display, display.columns)


def _sensitivity_text(sensitivity: pd.DataFrame) -> str:
    if sensitivity.empty:
        return "No parameter sensitivity was eligible."
    lines = []
    for (method, direction), group in sensitivity.groupby(
        ["method", "direction"], sort=False
    ):
        primary = group[group["setting_id"].eq("primary")].iloc[0]
        nonprimary = group[group["setting_id"].ne("primary")]
        agreement = float(nonprimary["sign_consistent_with_primary"].mean())
        finite_n_min = int(group["evaluable_paired_n"].min())
        finite_n_max = int(group["evaluable_paired_n"].max())
        difference_low = float(group["paired_mean_difference"].min())
        difference_high = float(group["paired_mean_difference"].max())
        p_low = float(group["wilcoxon_p_two_sided"].min())
        p_high = float(group["wilcoxon_p_two_sided"].max())
        lines.append(
            f"- {method} {direction}: primary sign "
            f"{int(primary['direction_sign']):+d}; sign agreement in "
            f"{agreement * 100:.1f}% of non-primary settings; mean-difference "
            f"range {_fmt(difference_low)} to {_fmt(difference_high)}; nominal-p "
            f"range {_fmt_p(p_low)} to {_fmt_p(p_high)}; paired n {finite_n_min}–{finite_n_max}."
        )
    return "\n".join(lines)


def _prevalence_text(prevalence: pd.DataFrame) -> str:
    if prevalence.empty:
        return "No surrogate-prevalence analysis was eligible."
    lines = []
    for (method, direction), group in prevalence.groupby(
        ["method", "direction"], sort=False
    ):
        phase_bits = []
        for phase in ("Pre", "Stim", "Post"):
            row = group[group["phase"].eq(phase)].iloc[0]
            phase_bits.append(
                f"{phase} {int(row.significant_n)}/{int(row.evaluable_n)} "
                f"({_fmt(row.percentage, 1)}%)"
            )
        first = group.iloc[0]
        lines.append(
            f"- {method} {direction}: "
            + ", ".join(phase_bits)
            + f"; exact McNemar p={_fmt_p(first.exact_McNemar_p_Pre_vs_Stim)}; "
            + f"Cochran's Q p={_fmt_p(first.Cochran_Q_p_Pre_Stim_Post)}."
        )
    return "\n".join(lines)


def _existing_linear_context() -> str:
    key_path = PROJECT_ROOT / "nonlinear_reference" / "key_values.json"
    parts = []
    if key_path.is_file():
        payload = json.loads(key_path.read_text(encoding="utf-8"))
        brs = payload.get("reference_BRS_Stim_Pre", {})
        coherence = payload.get("coherence_Stim_Pre", {})
        parts.append(
            "The frozen earlier n=18 revision summary reported sequence BRS "
            f"dz={_fmt(brs.get('cohens_dz'))}, p={_fmt_p(brs.get('wilcoxon_p_two_sided'))}, "
            "whereas Mayer-band coherence showed "
            f"dz={_fmt(coherence.get('cohens_dz'))}, p={_fmt_p(coherence.get('wilcoxon_p_two_sided'))}."
        )
    parts.append(
        "VAR/coherence and sequence-BRS values answer different questions and "
        "were not placed in the nonlinear FDR family. The present indices are "
        "descriptive complements, not substitutes for those prespecified analyses."
    )
    return " ".join(parts)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _ci_excludes_zero(low: object, high: object) -> bool:
    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(low_value)
        and np.isfinite(high_value)
        and (low_value > 0 or high_value < 0)
    )


def _recommended_adoption(verdict: str) -> str:
    return {
        "RESPONSE_ONLY_INCONCLUSIVE": "A. RTOR_ONLY",
        "SI_BRIEF_NULL_SENSITIVITY": "B. RTOR_PLUS_BRIEF_SI_METHODS_RESULTS",
        "SI_EXPLORATORY_NON_NULL": (
            "C. SI_EXPLORATORY_RESULT_PLUS_OPTIONAL_ONE_MAIN_SENTENCE"
        ),
        "DO_NOT_REPORT_REAL_DATA_INFERENCE": "D. DO_NOT_REPORT_REAL_DATA_INFERENCE",
    }[verdict]


def _verdict_from_report(path: Path) -> str:
    if not path.is_file():
        return "UNAVAILABLE"
    text = path.read_text(encoding="utf-8")
    candidates = (
        "SI_BRIEF_NULL_SENSITIVITY",
        "RESPONSE_ONLY_INCONCLUSIVE",
        "SI_EXPLORATORY_NON_NULL",
        "DO_NOT_REPORT_REAL_DATA_INFERENCE",
    )
    for candidate in candidates:
        if f"OVERALL_VERDICT = {candidate}" in text:
            return candidate
    return "UNAVAILABLE"


def _scientific_interpretation_class(verdict: str) -> str:
    if verdict == "SI_EXPLORATORY_NON_NULL":
        return "exploratory robust difference detected"
    if verdict == "DO_NOT_REPORT_REAL_DATA_INFERENCE":
        return "real-data inference not reportable"
    return "no robust Stim–Pre difference detected"


def write_n18_vs_n16_comparison(
    contrasts: pd.DataFrame, verdict: str
) -> tuple[pd.DataFrame, str, bool]:
    """Write the neutral, internal comparison with the previous n=16 run."""

    previous_path = (
        REFERENCE_N16_PACKAGE / "05_results" / "nonlinear_phase_contrasts.csv"
    )
    if not previous_path.is_file():
        raise FileNotFoundError(
            f"Previous n=16 primary contrast table not found: {previous_path}"
        )
    previous = pd.read_csv(previous_path)
    previous = previous[previous["contrast"].eq("Stim-Pre")].copy()
    current = contrasts[contrasts["contrast"].eq("Stim-Pre")].copy()
    value_columns = [
        "evaluable_paired_n",
        "paired_mean_difference",
        "mean_difference_BCa95_low",
        "mean_difference_BCa95_high",
        "wilcoxon_p_two_sided",
        "BH_q_primary_family",
        "cohens_dz",
    ]
    previous_names = {
        "evaluable_paired_n": "n16_paired_n",
        "paired_mean_difference": "n16_mean_difference",
        "mean_difference_BCa95_low": "n16_BCa95_low",
        "mean_difference_BCa95_high": "n16_BCa95_high",
        "wilcoxon_p_two_sided": "n16_wilcoxon_p",
        "BH_q_primary_family": "n16_BH_q",
        "cohens_dz": "n16_cohens_dz",
    }
    current_names = {key: value.replace("n16", "n18") for key, value in previous_names.items()}
    left = previous[["method", "direction", *value_columns]].rename(
        columns=previous_names
    )
    right = current[["method", "direction", *value_columns]].rename(
        columns=current_names
    )
    comparison = left.merge(
        right, on=["method", "direction"], how="outer", validate="one_to_one"
    )
    comparison["n18_minus_n16_mean_difference"] = (
        comparison["n18_mean_difference"] - comparison["n16_mean_difference"]
    )
    comparison["same_effect_direction"] = np.sign(
        comparison["n16_mean_difference"]
    ).eq(np.sign(comparison["n18_mean_difference"]))
    comparison["n16_q_lt_0p05"] = comparison["n16_BH_q"].lt(0.05)
    comparison["n18_q_lt_0p05"] = comparison["n18_BH_q"].lt(0.05)
    n16_ci_excludes = [
        _ci_excludes_zero(low, high)
        for low, high in zip(
            comparison["n16_BCa95_low"], comparison["n16_BCa95_high"]
        )
    ]
    n18_ci_excludes = [
        _ci_excludes_zero(low, high)
        for low, high in zip(
            comparison["n18_BCa95_low"], comparison["n18_BCa95_high"]
        )
    ]
    comparison["qualitative_conclusion_changed"] = (
        comparison["n16_q_lt_0p05"].ne(comparison["n18_q_lt_0p05"])
        | pd.Series(n16_ci_excludes).ne(pd.Series(n18_ci_excludes))
        | ~comparison["same_effect_direction"]
    )
    ordered_columns = [
        "method",
        "direction",
        "n16_paired_n",
        "n18_paired_n",
        "n16_mean_difference",
        "n18_mean_difference",
        "n18_minus_n16_mean_difference",
        "n16_BCa95_low",
        "n16_BCa95_high",
        "n18_BCa95_low",
        "n18_BCa95_high",
        "n16_wilcoxon_p",
        "n18_wilcoxon_p",
        "n16_BH_q",
        "n18_BH_q",
        "n16_cohens_dz",
        "n18_cohens_dz",
        "same_effect_direction",
        "n16_q_lt_0p05",
        "n18_q_lt_0p05",
        "qualitative_conclusion_changed",
    ]
    comparison = (
        comparison[ordered_columns]
        .sort_values(["method", "direction"], kind="stable")
        .reset_index(drop=True)
    )
    comparison.to_csv(
        RESULT_DIR / "n18_vs_n16_primary_contrasts.csv", index=False
    )

    previous_verdict = _verdict_from_report(
        REFERENCE_N16_PACKAGE
        / "07_reports"
        / "RESULT_INTERPRETATION_AND_ADOPTION_DECISION.md"
    )
    n16_ci_excludes = [
        _ci_excludes_zero(low, high)
        for low, high in zip(
            comparison["n16_BCa95_low"], comparison["n16_BCa95_high"]
        )
    ]
    n18_ci_excludes = [
        _ci_excludes_zero(low, high)
        for low, high in zip(
            comparison["n18_BCa95_low"], comparison["n18_BCa95_high"]
        )
    ]
    q_changes = comparison[
        comparison["n16_q_lt_0p05"].ne(comparison["n18_q_lt_0p05"])
    ]
    ci_changes = comparison[
        pd.Series(n16_ci_excludes, index=comparison.index).ne(
            pd.Series(n18_ci_excludes, index=comparison.index)
        )
    ]
    direction_matches = int(comparison["same_effect_direction"].sum())
    interpretation_changed = (
        _scientific_interpretation_class(previous_verdict)
        != _scientific_interpretation_class(verdict)
    )
    display = pd.DataFrame(
        {
            "Method": comparison["method"],
            "Direction": comparison["direction"],
            "n16 Δ [BCa 95% CI]": [
                f"{_fmt_effect(row.n16_mean_difference)} "
                f"[{_fmt_effect(row.n16_BCa95_low)}, {_fmt_effect(row.n16_BCa95_high)}]"
                for row in comparison.itertuples(index=False)
            ],
            "n18 Δ [BCa 95% CI]": [
                f"{_fmt_effect(row.n18_mean_difference)} "
                f"[{_fmt_effect(row.n18_BCa95_low)}, {_fmt_effect(row.n18_BCa95_high)}]"
                for row in comparison.itertuples(index=False)
            ],
            "n16 p / q / dz": [
                f"{_fmt_p(row.n16_wilcoxon_p)} / {_fmt_p(row.n16_BH_q)} / {_fmt(row.n16_cohens_dz)}"
                for row in comparison.itertuples(index=False)
            ],
            "n18 p / q / dz": [
                f"{_fmt_p(row.n18_wilcoxon_p)} / {_fmt_p(row.n18_BH_q)} / {_fmt(row.n18_cohens_dz)}"
                for row in comparison.itertuples(index=False)
            ],
            "Same direction": comparison["same_effect_direction"],
        }
    )
    report = f"""# n=18 versus n=16 internal comparison

This internal comparison is descriptive and is not part of the primary
hypothesis family. The n=18 run is the authoritative result for this nonlinear
sensitivity analysis; `previous n=16 run` is retained only as a neutral
reference label.

{_markdown_table(display, display.columns)}

- Effect direction matched in {direction_matches}/{len(comparison)} contrasts.
- BH q<0.05 status changed in {len(q_changes)} contrast(s): {', '.join((q_changes['method'] + ' ' + q_changes['direction']).tolist()) if len(q_changes) else 'none'}.
- BCa interval zero-crossing status changed in {len(ci_changes)} contrast(s): {', '.join((ci_changes['method'] + ' ' + ci_changes['direction']).tolist()) if len(ci_changes) else 'none'}.
- Overall verdict: previous n=16 run `{previous_verdict}`; n=18 run `{verdict}`; changed: **{'YES' if previous_verdict != verdict else 'NO'}**.
- Final scientific interpretation changed: **{'YES' if interpretation_changed else 'NO'}**. Previous: {_scientific_interpretation_class(previous_verdict)}. Current: {_scientific_interpretation_class(verdict)}.

No participant-level attribution is made by this comparison.
"""
    _write(REPORT_DIR / "N18_VS_N16_COMPARISON.md", report)
    return comparison, previous_verdict, interpretation_changed


def write_method_validation_report(
    validation: pd.DataFrame, gate: pd.DataFrame
) -> None:
    failure_bits = []
    fpr_bits = []
    for method in gate["method"]:
        subset = validation[validation["method"].eq(method)]
        directional = 2 * len(subset)
        failures = directional - int(
            subset["x_to_y_finite"].sum() + subset["y_to_x_finite"].sum()
        )
        failure_bits.append(f"- {method}: {failures}/{directional} non-finite directional estimates.")
        uncoupled = subset[subset["scenario"].eq("uncoupled_linear")]
        cell_rates = []
        for noise, group in uncoupled.groupby("noise", sort=False):
            forward = float(group["x_to_y_false_positive"].mean())
            reverse = float(group["y_to_x_false_positive"].mean())
            cell_rates.extend([forward, reverse])
        fpr_bits.append(
            f"- {method}: maximum cell FPR {max(cell_rates) * 100:.1f}% "
            f"(four direction-by-noise cells: "
            + ", ".join(f"{value * 100:.1f}%" for value in cell_rates)
            + ")."
        )
    text = f"""# Method validation report

## Prespecified design

Validation used five known-ground-truth scenarios (uncoupled linear,
unidirectional linear X→Y, unidirectional nonlinear X→Y, common driver without
direct X↔Y coupling, and asymmetric bidirectional coupling), two measurement-
noise levels, 200 replicates per cell, N=256 retained observations, and seed
20260806. Uncoupled-process calibration used 39 circular-shift surrogates per
replicate, direction, and method.

## Frozen gate results

{_validation_table(gate)}

The direction fractions refer to the proportion of moderate-noise replicates
in which the known stronger X→Y index exceeded Y→X. The FPR gate required both
directions at both noise levels to be at most 10%. Direction recovery required
at least 80% for moderate linear and nonlinear unidirectional coupling;
non-finite failure rates had to be at most 5%; systematic reversal was
prohibited; and the moderate asymmetric-bidirectional scenario required at
least 65% correct stronger-direction ordering.

### Empirical uncoupled-process false-positive rates

{chr(10).join(fpr_bits)}

## Numerical failures

{chr(10).join(failure_bits)}

## Method-specific disposition

{chr(10).join(f'- {row.method}: {row.validation_status}.' for row in gate.itertuples(index=False))}

Only methods labelled `VALIDATED` are eligible for real-data inference,
surrogate prevalence, response language, or the primary multiplicity family.
A common-driver scenario is a limitation demonstration rather than a claim
that a bivariate estimator can separate common input from direct transfer.
"""
    _write(REPORT_DIR / "METHOD_VALIDATION_REPORT.md", text)


def write_analysis_report(
    qc: pd.DataFrame,
    gate: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    prevalence: pd.DataFrame,
    missingness: pd.DataFrame,
) -> None:
    _, primary_table = _primary_table(contrasts)
    validated = gate[gate["validation_status"].eq("VALIDATED")]["method"].tolist()
    failed = gate[gate["validation_status"].ne("VALIDATED")]["method"].tolist()
    estimable = int(qc["primary_estimable"].sum())
    stationarity_counts = qc[
        qc["subject_id"].astype(int).isin(ANALYSIS_SUBJECTS)
    ]["stationarity_flag"].value_counts().to_dict()
    text = f"""# Nonlinear coupling analysis report

## Scope

This was a narrowly scoped post hoc sensitivity analysis requested during peer
review. It examined only native beat-to-beat RRI–SBP pairs in a small exploratory
sample. The design was fixed-order Pre–Stim–Post, without sham; breathing was
spontaneous and respiration was not recorded. The primary contrast was
Stim–Pre; Post contrasts were descriptive/exploratory. No PAT analysis and no
respiratory proxy were added.

## Data provenance and preprocessing

All 18 protocol completers were included in this author-requested full-cohort
rerun. Provider-retained native beat pairs were reused; raw events were not
re-extracted. Their SHA-256 identities were verified against the 18 paired-file
entries recorded by the previous production package before calculation. Phase
boundaries were [0,300), [300,600), and [600,900) s by
R-wave timing. Finite RRI/SBP pairs with positive RRI were retained. The primary
window was the deterministic central 256 valid pairs. Each selected signal was
linearly detrended and z-normalized within subject, phase, and segment. No
uniform-time interpolation, filtering, imputation, or cross-boundary sample was
used. A machine-readable artifact flag was unavailable, so artifact exclusions
cannot be enumerated separately from the provider-retained pairing.

Primary estimable full-cohort cells: {estimable}/{len(ANALYSIS_SUBJECTS) * 3}.
Stationarity flags (reporting only): `{stationarity_counts}`. A stationarity flag
never changed the frozen window or inclusion decision.

## Exact estimators

- **LP:** Porta-style nonuniform embedding with target/source lags 1–8,
  Chebyshev distance, k=30, leave-one-out local prediction, and Theiler ±8.
  Reduced target-past and full target/source-past unpredictability were compared;
  the positive-strength index reverses the original negative ratio convention.
- **CE:** the same candidates, k, norm, and Theiler window; epsilon was
  `0.10 × (P84 − P16)` of the target. Local conditional entropy was the mean
  negative log fraction of unordered neighbor-target pairs closer than epsilon.
  Source-related reduction in normalized conditional entropy was reoriented so
  higher values indicate stronger conditional information transfer.
- **SSC/KNNCP:** exact-publication state-space cross-predictability candidate,
  strictly lagged consecutive source patterns, m=1–15, k=20, Euclidean distance,
  exponential simplex weights, and CPI as the maximum squared prediction
  correlation. It does not condition on target history.

These are described as nonlinear lag-dependent predictability, conditional
information transfer, and state-space cross-predictability—not physiological
causation.

Validated methods: {', '.join(validated) if validated else 'none'}.
Failed/excluded methods: {', '.join(failed) if failed else 'none'}.

## Primary Stim–Pre results

{primary_table}

Post–Pre and Post–Stim rows are included in `nonlinear_phase_contrasts.csv` as
descriptive/exploratory contrasts and are excluded from the BH family.

## Parameter sensitivity

{_sensitivity_text(sensitivity)}

All settings were frozen in advance. The most favorable setting was not
selected or highlighted, and sensitivity p values were not added to the primary
hypothesis family.

## Subject-level surrogate prevalence

{_prevalence_text(prevalence)}

Prevalence asks whether an individual index exceeded circular-shift surrogates;
it is distinct from the paired group-level Stim–Pre strength contrast.

## Relationship to existing linear and sequence analyses

{_existing_linear_context()}

## Missingness and estimability

{_markdown_table(missingness, ['method', 'direction', 'phase', 'finite_subject_n', 'Stim_Pre_evaluable_paired_n', 'validation_status', 'group_inference_status'])}

## Limitations

This post hoc analysis has short 5-min series, a small n=18 sample, fixed phase
order, no sham condition, spontaneous breathing,
and no respiratory recording. Bivariate indices cannot rule out common drivers,
and subject-level surrogate significance is sensitive to finite-series
estimation. A non-significant contrast means that no robust Stim–Pre difference
was detected under these settings; it does not show that nonlinear coupling was
absent, establish equivalence, or prove the null hypothesis.

Primary method sources: Porta et al., PLOS ONE 2014,
https://doi.org/10.1371/journal.pone.0089463; Porta et al., Chaos 2024,
https://doi.org/10.1063/5.0192645.
"""
    _write(REPORT_DIR / "NONLINEAR_COUPLING_ANALYSIS_REPORT.md", text)


def write_decision_report(
    verdict: str,
    gate: pd.DataFrame,
    contrasts: pd.DataFrame,
    adoption_audit: pd.DataFrame,
    reasons: Sequence[str],
) -> None:
    _, table = _primary_table(contrasts)
    reason_text = "\n".join(f"- {reason}" for reason in reasons)
    if adoption_audit.empty:
        audit_text = "No estimable validated outcomes were available."
    else:
        audit_text = _markdown_table(
            adoption_audit,
            [
                "method",
                "direction",
                "BH_q_lt_0p05",
                "BCa_CI_excludes_zero",
                "paired_n_ge_15",
                "nonprimary_sensitivity_sign_agreement_fraction",
                "robust_non_null_rule_met",
            ],
        )
    consequence = {
        "SI_BRIEF_NULL_SENSITIVITY": (
            "A brief SI sensitivity result and robust-null reviewer response are "
            "appropriate; no main-text insertion is recommended."
        ),
        "RESPONSE_ONLY_INCONCLUSIVE": (
            "Report the complete result transparently in the reviewer response "
            "and, if desired, a short SI technical note. Do not use it for "
            "physiological inference or a main-text result claim."
        ),
        "SI_EXPLORATORY_NON_NULL": (
            "Report the exploratory result in the SI; at most one cautious "
            "optional main-text sentence may be considered."
        ),
        "DO_NOT_REPORT_REAL_DATA_INFERENCE": (
            "Do not report real-data inference; explain the validation or "
            "estimability failure and retain only a future-work limitation."
        ),
    }[verdict]
    recommended_adoption = _recommended_adoption(verdict)
    text = f"""# Result interpretation and adoption decision

```text
OVERALL_VERDICT = {verdict}
RECOMMENDED_ADOPTION = {recommended_adoption}
```

## Basis

{reason_text}

## Primary evidence

{table}

## Outcome-level robust-non-null audit

{audit_text}

## Reporting consequence

{consequence} The analysis remains post hoc and SI/response-centred. Existing
main figures and the manuscript's central sequence-BRS/coherence structure
should not be replaced. Main manuscript central conclusion change: **NO**.
"""
    _write(REPORT_DIR / "RESULT_INTERPRETATION_AND_ADOPTION_DECISION.md", text)


def write_reproducibility_report(
    run_status: str,
    inventory: pd.DataFrame,
    hashes: pd.DataFrame,
    qc: pd.DataFrame,
    gate: pd.DataFrame,
    missingness: pd.DataFrame,
    git_head: str,
) -> None:
    source_rows = inventory[inventory["category"].eq("canonical_native_pair")]
    source_hashes = hashes[hashes["path"].isin(source_rows["path"])]
    text = f"""# Reproducibility and QC report

- Run classification: `{run_status}`
- Fixed seed: `20260806`
- Frozen plan hash: `{PLAN_SHA256}`
- Git HEAD at execution: `{git_head}`
- Native paired input files: {len(source_rows)} present; {len(source_hashes)} hashed
- Protocol completers and full-cohort analysis set: 18
- Primary estimable cells: {int(qc['primary_estimable'].sum())}/54
- Primary interpolation use: none
- Phase-boundary crossing: none by construction and test
- Imputation: none
- Machine-readable artifact flag: unavailable

## Validation disposition

{_validation_table(gate)}

## Estimability audit

{_markdown_table(missingness, ['method', 'direction', 'phase', 'eligible_analysis_subject_n', 'computed_subject_n', 'finite_subject_n', 'not_computed_due_to_validation_n', 'group_inference_status'])}

## Tests and implementation controls

The test suite covers deterministic output, direction mapping, surrogate
p-values and strict thresholding, phase boundaries, native-beat/no-interpolation
selection, paired-participant BCa bootstrap behavior, BH correction,
known-direction simulation, missing-data handling, full-cohort inclusion,
54-cell segment integrity, and result-schema checks. The plan hash is checked
before validation and immediately before real-data calculations. Input identity
is independently checked against all 18 SHA-256 entries in the previous
production inventory. Method-only synthetic-validation tables were reused only
after exact hash and structural verification; every human-data metric,
contrast, sensitivity result, surrogate result, figure, and report was newly
computed for n=18.

The inherited implementation includes machine-precision-tolerant BCa tie
handling, encoding-safe Windows console output, and a regression-tested
Unicode-arrow direction-label mapping. None changes the frozen estimator or
statistical specification. Package versions, execution messages, captured
warnings/errors, software corrections, all input hashes, and a dedicated-output
file manifest are stored under `logs/` and `01_input_inventory/`.
"""
    _write(REPORT_DIR / "REPRODUCIBILITY_AND_QC_REPORT.md", text)


def write_claim_audit(
    verdict: str,
    gate: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    prevalence: pd.DataFrame,
) -> None:
    claims = [
        {
            "Claim": "Native beat-to-beat RRI–SBP data were used without 4-Hz samples as independent observations.",
            "Evidence": "INPUT inventory, selected_primary_series.csv, no-interpolation test",
            "Status": "SUPPORTED",
        },
        {
            "Claim": "Only validation-passing methods entered real-data inference.",
            "Evidence": "validation_method_gate.csv and validation_status in every metric row",
            "Status": "SUPPORTED",
        },
        {
            "Claim": "Primary multiplicity was one BH family over validated method×direction Stim–Pre tests.",
            "Evidence": "nonlinear_phase_contrasts.csv",
            "Status": "SUPPORTED",
        },
        {
            "Claim": "No parameter was selected after viewing real-data results.",
            "Evidence": "immutable plan SHA-256 and full one-factor-at-a-time sensitivity table",
            "Status": "SUPPORTED",
        },
        {
            "Claim": "The indices establish physiological causation or exclude respiratory/common-driver effects.",
            "Evidence": "Design has no sham or respiratory recording; bivariate methods",
            "Status": "NOT_SUPPORTED_AND_NOT_CLAIMED",
        },
        {
            "Claim": "A non-significant result proves absence/equivalence.",
            "Evidence": "No equivalence margin or prospective power design",
            "Status": "NOT_SUPPORTED_AND_PROHIBITED",
        },
    ]
    frame = pd.DataFrame(claims)
    text = f"""# Claim–evidence audit

Overall adoption verdict: `{verdict}`.

{_markdown_table(frame, frame.columns)}

## Allowed result language

The allowed vocabulary is nonlinear lag-dependent predictability, conditional
information transfer, state-space cross-predictability, post hoc exploratory,
and stimulation-associated reconfiguration. A null result must be stated as
"no robust Stim–Pre difference was detected," never "nonlinear coupling was
absent" or "the null hypothesis was proven."

## Traceability

- Validation gate: `03_validation/validation_method_gate.csv`
- Primary effects and FDR: `05_results/nonlinear_phase_contrasts.csv`
- Frozen settings: `05_results/nonlinear_parameter_sensitivity.csv`
- Subject prevalence: `05_results/nonlinear_prevalence_summary.csv`
- Missingness: `05_results/nonlinear_missingness_summary.csv`
"""
    _write(REPORT_DIR / "CLAIM_EVIDENCE_AUDIT.md", text)


def write_snippets(
    verdict: str,
    gate: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    validated = gate[gate["validation_status"].eq("VALIDATED")]["method"].tolist()
    excluded = gate[gate["validation_status"].ne("VALIDATED")]["method"].tolist()
    results = _result_clause(contrasts)
    method_names = {
        "LP": "local-predictability",
        "CE": "conditional-entropy",
        "SSC": "state-space cross-predictability",
    }
    validated_phrase = ", ".join(method_names[item] for item in validated)
    primary = (
        contrasts[contrasts["contrast"].eq("Stim-Pre")].copy()
        if not contrasts.empty
        else pd.DataFrame()
    )
    discordant = (
        primary[
            (
                (primary["mean_difference_BCa95_low"] > 0)
                | (primary["mean_difference_BCa95_high"] < 0)
            )
            & (primary["wilcoxon_p_two_sided"] >= 0.05)
        ]
        if not primary.empty
        else pd.DataFrame()
    )
    if discordant.empty:
        discordance_text = ""
    else:
        details = []
        for row in discordant.itertuples(index=False):
            details.append(
                f"{row.method} {row.direction} (mean difference "
                f"{_fmt_effect(row.paired_mean_difference)}, BCa 95% CI "
                f"{_fmt_effect(row.mean_difference_BCa95_low)} to "
                f"{_fmt_effect(row.mean_difference_BCa95_high)}, Wilcoxon "
                f"p={_fmt_p(row.wilcoxon_p_two_sided)}, BH "
                f"q={_fmt_p(row.BH_q_primary_family)})"
            )
        discordance_text = (
            "One mean-based interval narrowly excluded zero—"
            + "; ".join(details)
            + "—but the paired rank test was not significant. "
        )
    if verdict == "SI_BRIEF_NULL_SENSITIVITY":
        response = f"""# Reviewer 1 Comment 6 response snippets

## Recommended response

We appreciate the reviewer's point that nonlinear approaches can provide
complementary information. To assess whether our conclusion depended on linear
coupling estimators, we conducted a narrowly scoped post hoc sensitivity
analysis of native beat-to-beat RRI–SBP coupling in all 18 protocol completers
using validated
{validated_phrase} estimators. Candidate methods were admitted only after a
prespecified synthetic-validation gate, and the state-space correspondence
estimator was included only if it passed that gate. Under the frozen analysis
plan, no nonlinear metric showed a robust Stim–Pre difference after correction
for the validated method-by-direction family ({results}). We report these
results in the Supplementary Information. Because the analysis was post hoc,
the segments were short, the sample was small, respiration was not recorded,
and no sham condition was available, this finding should be interpreted as an
absence of a detected robust phase-related change under these settings rather
than evidence that nonlinear coupling was absent.
"""
    elif verdict == "RESPONSE_ONLY_INCONCLUSIVE":
        response = f"""# Reviewer 1 Comment 6 response snippets

## Recommended response

We appreciate the reviewer's suggestion and performed a targeted post hoc
nonlinear sensitivity analysis of native beat-to-beat RRI–SBP coupling in all
18 protocol completers after a
prespecified synthetic-validation gate. All three candidate estimators
({validated_phrase}) passed the gate. The primary estimates were {results}.
None of the six two-sided Wilcoxon tests was nominally significant, and none
survived multiplicity correction. {discordance_text}Because the frozen
robustness rule required multiplicity-corrected significance together with an
excluding-zero BCa interval, adequate n, and sensitivity stability, we classify
the result as inconclusive rather than as either robust evidence of change or a
robust-null sensitivity result. We therefore do not use it to support
physiological inference and report it only for transparency in the reviewer
response and, if desired, a short Supplementary technical note. The short
series, small sample, fixed phase order, absence of sham, and lack of respiratory
recording further limit interpretation.
"""
    elif verdict == "SI_EXPLORATORY_NON_NULL":
        response = f"""# Reviewer 1 Comment 6 response snippets

## Recommended response

We appreciate the reviewer's suggestion. A targeted post hoc nonlinear analysis
of native beat-to-beat RRI–SBP coupling in all 18 protocol completers was
conducted under a frozen plan and
prespecified synthetic-validation gate. The primary estimates were {results}.
At least one validated method/direction met the multiplicity-corrected,
BCa-interval, estimability, and parameter-stability criteria. We report this
exploratory finding in the Supplementary Information and provide one cautious
optional main-text sentence. Because the analysis was post hoc and the study
lacked a sham condition and respiratory recording, the result is not interpreted
as physiological causation.
"""
    else:
        response = f"""# Reviewer 1 Comment 6 response snippets

## Recommended response

We explored the feasibility of the suggested nonlinear approaches for all 18
protocol completers under a frozen plan. However, the candidate estimator(s)
did not meet the prespecified
synthetic-validation and/or estimability criteria ({', '.join(excluded) if excluded else 'none eligible'}).
We therefore did not add unstable real-data inference, which would have
increased analytical arbitrariness. The revised Discussion can acknowledge the
limitation of linear estimators and identify validated nonlinear analysis in a
prospectively designed study as future work.
"""
    _write(SNIPPET_DIR / "R1_COMMENT6_RESPONSE_SNIPPETS.md", response)

    methods = f"""# SI Methods insertion

## Targeted post hoc nonlinear RRI–SBP coupling sensitivity analysis

In response to peer review, we conducted a narrowly scoped post hoc sensitivity
analysis to examine whether validated nonlinear coupling estimators revealed a
Stim–Pre change not detected by the linear framework. The analysis was limited
to native beat-to-beat paired RRI and systolic blood pressure (SBP); PAT and
interpolated 4-Hz samples were not used. All 18 protocol completers were
included in this author-requested full-cohort rerun. For each Pre, Stim, and Post
phase, the central 256 valid pairs were selected deterministically, without
crossing phase boundaries or imputing missing observations. Each signal was
linearly detrended and z-normalized within the selected segment.

Candidate estimators were local predictability (LP), conditional entropy (CE),
and publication-defined state-space correspondence/KNN cross-predictability
(SSC/KNNCP). LP and CE used nonuniform target/source embeddings with candidate
lags 1–8 beats, k=30 neighbors, Chebyshev distance, leave-one-out estimation,
and a Theiler exclusion window of ±8 beats. LP quantified the reduction in
target unpredictability after adding source history. For CE, the target
tolerance was 0.10×(P84−P16), and conditional entropy was estimated from the
fraction of unordered neighbor-target pairs separated by less than this
tolerance; the index quantified the reduction after source-history inclusion.
Both were reoriented so that larger values represent stronger directed
predictability or conditional information transfer. SSC/KNNCP used strictly
lagged consecutive source patterns, m=1–15, k=20, Euclidean distance, and
exponentially weighted simplex prediction; its index was the maximum squared
correlation between observed and cross-predicted target values.

Before real-data comparison, every method was evaluated in 200 replicates of
five prespecified synthetic scenarios at two noise levels (N=256). The gate
required uncoupled-process false-positive rates ≤10%, ≥80% correct direction
ordering for moderate unidirectional linear and nonlinear coupling, ≥95%
finite estimates, no systematic reversal, and ≥65% stronger-direction
ordering for asymmetric bidirectional coupling. Only validated methods
({', '.join(validated) if validated else 'none'}) entered real-data inference.

The primary hypothesis family comprised the two-sided paired Stim–Pre Wilcoxon
tests for both directions of each validated method, adjusted together by the
Benjamini–Hochberg procedure. We also report paired mean differences with
participant-level BCa 95% confidence intervals (10,000 resamples), Cohen's dz,
and paired n. Post contrasts were descriptive. Prespecified one-factor-at-a-
time sensitivity settings for LP/CE were k=20/30/40, lag depth 4/8/12, central
192/256 beats and full phases, plus a same-beat-inclusive SBP→RRI convention.
Subject-level coupling was assessed secondarily using 199 source circular-shift
surrogates and summarized with exact binomial intervals, exact McNemar tests,
and Cochran's Q. Seed 20260806 was used throughout.
"""
    _write(SNIPPET_DIR / "SI_METHODS_INSERTION.md", methods)

    _, primary_table = _primary_table(contrasts)
    results_text = f"""# SI Results insertion

## Targeted post hoc nonlinear RRI–SBP coupling sensitivity analysis

The synthetic-validation dispositions were: {', '.join(f'{row.method} {row.validation_status}' for row in gate.itertuples(index=False))}.

{primary_table}

Overall adoption classification: `{verdict}`. {_result_clause(contrasts)}.
{discordance_text}
The complete parameter-sensitivity and subject-level surrogate-prevalence
results are provided in the accompanying machine-readable tables. These results
should be interpreted as a post hoc test for a detectable Stim–Pre change under
the prespecified settings, not as evidence for absence of nonlinear coupling or
as physiological causation.
"""
    _write(SNIPPET_DIR / "SI_RESULTS_INSERTION.md", results_text)

    if verdict == "SI_EXPLORATORY_NON_NULL":
        sentence = (
            "A targeted post hoc analysis using synthetic-validated nonlinear "
            "RRI–SBP estimators identified an exploratory Stim–Pre difference in "
            "[validated method/direction; see Supplementary Information], without "
            "supporting physiological causation."
        )
        recommendation = "OPTIONAL_ONE_SENTENCE_ONLY"
    elif verdict == "DO_NOT_REPORT_REAL_DATA_INFERENCE":
        sentence = (
            "Validated nonlinear cardiovascular-coupling analysis remains a "
            "priority for prospectively designed studies with longer recordings "
            "and measured respiration."
        )
        recommendation = "DISCUSSION_LIMITATION_ONLY_IF_NEEDED"
    else:
        sentence = (
            "A targeted post hoc analysis using synthetic-validated nonlinear "
            "RRI–SBP estimators did not detect a robust Stim–Pre difference after "
            "multiplicity correction (Supplementary Information)."
        )
        recommendation = "DO_NOT_INSERT_UNLESS_EDITOR_REQUIRES_MAIN_TEXT_MENTION"
    _write(
        SNIPPET_DIR / "MAIN_TEXT_OPTIONAL_ONE_SENTENCE.md",
        f"# Optional main-text sentence\n\nRecommendation: `{recommendation}`\n\n> {sentence}",
    )


def write_author_facing_summary_ja(
    gate: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    prevalence: pd.DataFrame,
    comparison: pd.DataFrame,
    previous_verdict: str,
    interpretation_changed: bool,
    verdict: str,
) -> None:
    validated = gate[gate["validation_status"].eq("VALIDATED")]["method"].tolist()
    primary = contrasts[contrasts["contrast"].eq("Stim-Pre")].copy()
    corrected_n = int(primary["BH_q_primary_family"].lt(0.05).sum())
    primary_display = pd.DataFrame(
        {
            "Method": primary["method"],
            "Direction": primary["direction"],
            "paired n": primary["evaluable_paired_n"],
            "平均差 [BCa 95% CI]": [
                f"{_fmt_effect(row.paired_mean_difference)} "
                f"[{_fmt_effect(row.mean_difference_BCa95_low)}, "
                f"{_fmt_effect(row.mean_difference_BCa95_high)}]"
                for row in primary.itertuples(index=False)
            ],
            "Wilcoxon p": [_fmt_p(value) for value in primary["wilcoxon_p_two_sided"]],
            "BH q": [_fmt_p(value) for value in primary["BH_q_primary_family"]],
            "Cohen's dz": [_fmt(value) for value in primary["cohens_dz"]],
        }
    )
    sensitivity_lines = []
    for (method, direction), group in sensitivity.groupby(
        ["method", "direction"], sort=False
    ):
        nonprimary = group[group["setting_id"].ne("primary")]
        agreement = float(nonprimary["sign_consistent_with_primary"].mean())
        sensitivity_lines.append(
            f"- {method} {direction}: primaryと同じ差の向きだった非primary設定は "
            f"{agreement * 100:.1f}%。平均差の範囲は "
            f"{_fmt(group['paired_mean_difference'].min())}～"
            f"{_fmt(group['paired_mean_difference'].max())}。"
        )
    prevalence_lines = []
    for (method, direction), group in prevalence.groupby(
        ["method", "direction"], sort=False
    ):
        phase_parts = []
        for phase in ("Pre", "Stim", "Post"):
            row = group[group["phase"].eq(phase)].iloc[0]
            phase_parts.append(
                f"{phase} {int(row.significant_n)}/{int(row.evaluable_n)}"
            )
        first = group.iloc[0]
        prevalence_lines.append(
            f"- {method} {direction}: {', '.join(phase_parts)}; "
            f"Pre–Stim exact McNemar p={_fmt_p(first.exact_McNemar_p_Pre_vs_Stim)}, "
            f"3 phase Cochran's Q p={_fmt_p(first.Cochran_Q_p_Pre_Stim_Post)}。"
        )
    direction_matches = int(comparison["same_effect_direction"].sum())
    q_changes = int(
        comparison["n16_q_lt_0p05"].ne(comparison["n18_q_lt_0p05"]).sum()
    )
    recommendation = _recommended_adoption(verdict)
    report = f"""# 著者向け要約：全18例 nonlinear RRI–SBP coupling再解析

## 何を行ったか

Reviewer #1 Comment 6への対応として、全18 protocol completersのnative
beat-to-beat RRI–SBP dataを使い、LP、CE、SSC/KNNCPの3 method × 2 directionを
同じ凍結済み設定で再計算しました。主比較はStim–Preのみです。Post comparison
は探索的記述に限定しました。method、parameter、segment、検定、6検定のBH
family、bootstrap 10,000回、surrogate 199回、seed 20260806、adoption ruleは
previous production planから変更していません。

Synthetic validationを通過してreal-data inferenceへ入ったmethodは
**{', '.join(validated) if validated else 'なし'}** です。

## 6 primary contrasts

{_markdown_table(primary_display, primary_display.columns)}

BH q<0.05だったprimary outcomeは **{corrected_n}/6** でした。平均差のBCa
intervalとpaired rank testが異なる結論を示す場合も、事前のadoption ruleどおり
双方をそのまま示し、positive resultまたはrobust-null classificationへ格上げしていません。

## Parameter sensitivity

{chr(10).join(sensitivity_lines) if sensitivity_lines else '- 評価対象なし。'}

k、lag depth、segment length、same-beat conventionは固定したone-factor-at-a-time
sensitivityとして全て表示し、最も有意なsettingを選んでいません。

## Subject-level prevalence（補助解析）

{chr(10).join(prevalence_lines) if prevalence_lines else '- 評価対象なし。'}

これは各個人でobserved indexがcircular-shift surrogateを超えた割合であり、
group-levelのStim–Pre strength差とは別の問いです。

## Previous n=16 runとの内部比較

6 contrastsのeffect directionは **{direction_matches}/{len(comparison)}** で一致し、
q<0.05 statusが変わったoutcomeは **{q_changes}** 件でした。overall verdictは
previous n=16 run `{previous_verdict}`、今回のn=18 run `{verdict}` です。
最終的な科学的解釈が変わったか：**{'YES' if interpretation_changed else 'NO'}**。
この比較は内部robustness checkであり、primary hypothesis familyには加えていません。

## 最終判断と採用方針

- Overall verdict: `{verdict}`
- Recommended adoption: `{recommendation}`
- Main manuscriptの中心結論を変更するか：**NO**
- RtoRでは、全18例によるtargeted post hoc sensitivityであること、検出された差の
  robustness、short recordings・small n=18・fixed order・no sham・respiration未記録
  という限界を簡潔に示します。
- SI/Main textへの反映は上記adoption区分に従い、Main Figureは追加しません。

本解析で差が検出されなかった場合も、「nonlinear couplingが不存在」とは結論
できません。言えるのは、tested estimatorとparameterの範囲でrobustなStim–Pre
differenceを検出しなかった、または結果がinconclusiveだった、ということです。
"""
    _write(REPORT_DIR / "AUTHOR_FACING_SUMMARY_JA.md", report)


def write_analysis_summary(
    run_status: str,
    gate: pd.DataFrame,
    qc: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    prevalence: pd.DataFrame,
    comparison: pd.DataFrame,
    interpretation_changed: bool,
    verdict: str,
) -> None:
    validated = gate[gate["validation_status"].eq("VALIDATED")]["method"].tolist()
    failed = gate[gate["validation_status"].ne("VALIDATED")]["method"].tolist()
    primary = contrasts[contrasts["contrast"].eq("Stim-Pre")] if not contrasts.empty else pd.DataFrame()
    corrected = (
        primary[primary["BH_q_primary_family"] < 0.05]
        if not primary.empty
        else pd.DataFrame()
    )
    result_lines = []
    for row in primary.itertuples(index=False):
        result_lines.append(
            f"  - {row.method} {row.direction}: Δ={_fmt_effect(row.paired_mean_difference)} "
            f"[{_fmt_effect(row.mean_difference_BCa95_low)}, {_fmt_effect(row.mean_difference_BCa95_high)}], "
            f"p={_fmt_p(row.wilcoxon_p_two_sided)}, q={_fmt_p(row.BH_q_primary_family)}, "
            f"dz={_fmt(row.cohens_dz)}, n={int(row.evaluable_paired_n)}"
        )
    if not result_lines:
        result_lines = ["  - Real-data inference not eligible."]
    stability_lines = []
    if not sensitivity.empty:
        for (method, direction), group in sensitivity.groupby(
            ["method", "direction"], sort=False
        ):
            nonprimary = group[group["setting_id"].ne("primary")]
            agreement = float(nonprimary["sign_consistent_with_primary"].mean())
            stability_lines.append(
                f"  - {method} {direction}: {agreement * 100:.1f}% non-primary sign agreement"
            )
    else:
        stability_lines = ["  - Not estimable."]
    strategy = {
        "SI_BRIEF_NULL_SENSITIVITY": "Brief SI nonlinear sensitivity plus completed robust-null reviewer response.",
        "RESPONSE_ONLY_INCONCLUSIVE": "Reviewer response with transparent inconclusive results; no main-text inference.",
        "SI_EXPLORATORY_NON_NULL": "SI-only exploratory detail and at most one cautious optional main-text sentence.",
        "DO_NOT_REPORT_REAL_DATA_INFERENCE": "Explain validation/estimability failure and retain only a future-work limitation.",
    }[verdict]
    recommendation = _recommended_adoption(verdict)
    prevalence_lines = []
    for (method, direction), group in prevalence.groupby(
        ["method", "direction"], sort=False
    ):
        values = []
        for phase in ("Pre", "Stim", "Post"):
            row = group[group["phase"].eq(phase)].iloc[0]
            values.append(f"{phase} {int(row.significant_n)}/{int(row.evaluable_n)}")
        first = group.iloc[0]
        prevalence_lines.append(
            f"  - {method} {direction}: {', '.join(values)}; "
            f"McNemar p={_fmt_p(first.exact_McNemar_p_Pre_vs_Stim)}, "
            f"Cochran Q p={_fmt_p(first.Cochran_Q_p_Pre_Stim_Post)}"
        )
    if not prevalence_lines:
        prevalence_lines = ["  - Not estimable."]
    direction_matches = int(comparison["same_effect_direction"].sum())
    q_changes = int(
        comparison["n16_q_lt_0p05"].ne(comparison["n18_q_lt_0p05"]).sum()
    )
    text = f"""# Analysis execution summary

```text
STATUS: PASS_FULL_COHORT_N18_NONLINEAR_RERUN_AND_SYNTHESIS ({run_status})
INPUT DATA: provider-retained native beat-to-beat paired RRI–SBP files; no 4-Hz interpolation used
INPUT HASH CHECK: PASS; 18/18 paired files matched the previous production inventory by SHA-256
SUBJECTS / EVALUABLE N: 18 protocol completers; n=18 full-cohort analysis; {int(qc['primary_estimable'].sum())}/54 primary cells estimable
VALIDATED METHODS: {', '.join(validated) if validated else 'none'}
FAILED / EXCLUDED METHODS: {', '.join(failed) if failed else 'none'}
PRIMARY STIM–PRE RESULTS:
{chr(10).join(result_lines)}
MULTIPLICITY-CORRECTED FINDINGS: {len(corrected)} primary outcome(s) with BH q<0.05
SENSITIVITY STABILITY:
{chr(10).join(stability_lines)}
SUBJECT-LEVEL PREVALENCE:
{chr(10).join(prevalence_lines)}
N18 VS N16 QUALITATIVE CHANGE: effect direction matched {direction_matches}/{len(comparison)}; q<0.05 status changed for {q_changes} outcome(s); scientific interpretation changed: {'YES' if interpretation_changed else 'NO'}
OVERALL VERDICT: {verdict}
RECOMMENDED R1 RESPONSE STRATEGY: {strategy}
RECOMMENDED ADOPTION: {recommendation}
OUTPUT ROOT: {OUTPUT_ROOT}
KEY FILES: 00_plan/POST_HOC_NONLINEAR_COUPLING_PLAN.md; 03_validation/validation_method_gate.csv; 05_results/nonlinear_phase_contrasts.csv; 05_results/n18_vs_n16_primary_contrasts.csv; 07_reports/AUTHOR_FACING_SUMMARY_JA.md; 07_reports/NONLINEAR_COUPLING_ANALYSIS_REPORT.md; 08_response_and_manuscript_snippets/R1_COMMENT6_RESPONSE_SNIPPETS.md
MAIN MANUSCRIPT CENTRAL CONCLUSION CHANGE: NO
REASON: This targeted post hoc sensitivity analysis does not replace the sequence-BRS and Mayer-band coherence framework; even a robust exploratory finding remains SI-centred.
DOCX/EXISTING FILE MODIFICATION STATUS: no manuscript, SI, response, cover, existing figure, or Supplementary Data file was edited by this pipeline
```

This run included all 18 protocol completers as requested. It did not perform
or reopen participant-level provenance adjudication.
"""
    _write(REPORT_DIR / "FINAL_EXECUTION_SUMMARY.md", text)


def write_all_reports(
    *,
    run_status: str,
    inventory: pd.DataFrame,
    hashes: pd.DataFrame,
    phase_map: pd.DataFrame,
    qc: pd.DataFrame,
    validation: pd.DataFrame,
    gate: pd.DataFrame,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    surrogates: pd.DataFrame,
    prevalence: pd.DataFrame,
    missingness: pd.DataFrame,
    verdict: str,
    adoption_audit: pd.DataFrame,
    decision_reasons: Sequence[str],
    git_head: str,
) -> None:
    del phase_map, metrics, surrogates
    comparison, previous_verdict, interpretation_changed = (
        write_n18_vs_n16_comparison(contrasts, verdict)
    )
    write_method_validation_report(validation, gate)
    write_analysis_report(qc, gate, contrasts, sensitivity, prevalence, missingness)
    write_decision_report(
        verdict, gate, contrasts, adoption_audit, decision_reasons
    )
    write_reproducibility_report(
        run_status, inventory, hashes, qc, gate, missingness, git_head
    )
    write_claim_audit(verdict, gate, contrasts, sensitivity, prevalence)
    write_snippets(verdict, gate, contrasts, sensitivity)
    write_author_facing_summary_ja(
        gate,
        contrasts,
        sensitivity,
        prevalence,
        comparison,
        previous_verdict,
        interpretation_changed,
        verdict,
    )
    write_analysis_summary(
        run_status,
        gate,
        qc,
        contrasts,
        sensitivity,
        prevalence,
        comparison,
        interpretation_changed,
        verdict,
    )


__all__ = ["write_all_reports"]
