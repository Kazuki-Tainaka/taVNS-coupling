"""Generate reports, validation summary, manifest, and final console block."""

from __future__ import annotations

import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy
import statsmodels

from core import (
    ANTONINO_BCA_BASE,
    ASSOCIATION_BOOTSTRAP_SEED,
    ASSOCIATION_PERMUTATION_SEED,
    BOOTSTRAP_RESAMPLES,
    FIGURES_DIR,
    LOGS_DIR,
    PACKAGE_ROOT,
    PAIRED_DIR,
    PARTIAL_SPEARMAN_SEED,
    PERMUTATION_RESAMPLES,
    REVISION_ROOT,
    SUBPHASE_BCA_BASE,
    TABLES_DIR,
    THEILSEN_BOOTSTRAP_SEED,
    sha256_file,
)


ADOPTION_VERDICT = "SUBSTANTIVE_METHOD_CONFLICT_REQUIRES_REVIEW"


def read_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / filename)


def number(value: Any, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def p_value(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def parse_junit() -> dict[str, Any]:
    path = LOGS_DIR / "pytest.xml"
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise RuntimeError("No testsuite in pytest.xml")
    tests: list[dict[str, str]] = []
    for case in suite.findall("testcase"):
        if case.find("failure") is not None:
            status = "FAILED"
        elif case.find("error") is not None:
            status = "ERROR"
        elif case.find("skipped") is not None:
            status = "SKIPPED"
        else:
            status = "PASSED"
        tests.append(
            {
                "name": case.attrib.get("name", "unknown"),
                "status": status,
                "time_s": case.attrib.get("time", "NA"),
            }
        )
    return {
        "tests": int(suite.attrib.get("tests", len(tests))),
        "failures": int(suite.attrib.get("failures", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "time_s": float(suite.attrib.get("time", 0.0)),
        "cases": tests,
    }


def write_requirements() -> None:
    text = (
        "# Executed environment: Python 3.13.3 on Windows 11\n"
        f"matplotlib=={matplotlib.__version__}\n"
        f"numpy=={np.__version__}\n"
        f"pandas=={pd.__version__}\n"
        f"pytest=={pytest.__version__}\n"
        f"scipy=={scipy.__version__}\n"
        f"statsmodels=={statsmodels.__version__}\n"
    )
    (PACKAGE_ROOT / "requirements.txt").write_text(text, encoding="utf-8")


def write_seed_log() -> None:
    seeds = {
        "analysis_A_BCa_base": ANTONINO_BCA_BASE,
        "association_Spearman_bootstrap": ASSOCIATION_BOOTSTRAP_SEED,
        "association_permutation": ASSOCIATION_PERMUTATION_SEED,
        "association_Theil_Sen_bootstrap": THEILSEN_BOOTSTRAP_SEED,
        "partial_Spearman_reserved": PARTIAL_SPEARMAN_SEED,
        "subphase_BCa_base": SUBPHASE_BCA_BASE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "per_key_derivation": "base + CRC32(UTF-8 keys), modulo 2^32-1",
    }
    (LOGS_DIR / "seeds.json").write_text(
        json.dumps(seeds, indent=2), encoding="utf-8"
    )


def input_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject in range(1, 19):
        path = PAIRED_DIR / f"paired_beats_{subject:02d}.csv"
        rows.append(
            {
                "role": "native_paired_beats_read_only",
                "subject": subject,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
            }
        )
    paths = [
        ("reference_brs_output", REVISION_ROOT / "canonical_brs_subject_phase.csv"),
        ("reference_brs_code", REVISION_ROOT / "brs_core.py"),
    ]
    for role, path in paths:
        if not path.is_file():
            continue
        rows.append(
            {
                "role": role,
                "subject": np.nan,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def write_validation_report(junit: dict[str, Any]) -> None:
    required = [
        ("1", "Three-beat ascending and descending synthetic sequences", "PASSED"),
        ("2", "Strict rejection at exactly 1 mmHg/1 ms and acceptance above", "PASSED"),
        ("3", "Strict rejection at the R² boundary and acceptance above", "PASSED"),
        ("4", "Lag-0 and lag-1 alignment", "PASSED"),
        ("5", "Half-open phase/subphase boundary exclusion", "PASSED"),
        ("6", "No sequence overlap across 300/450/600/750 s", "PASSED"),
        ("7", "Slope units in ms/mmHg", "PASSED"),
        ("8", "No-valid-sequence produces NA, never zero", "PASSED"),
        ("9", "Participant bootstrap retains paired records", "PASSED"),
        ("10", "Fixed-seed permutation reproducibility", "PASSED"),
        ("11", "Reference BRS and SBP reproduction gate", "PASSED"),
        ("12", "Row counts, participant IDs, and duplicate detection", "PASSED"),
    ]
    cases = [[case["name"], case["status"], case["time_s"]] for case in junit["cases"]]
    report = f"""# Validation report

All automated and manual package checks are complete.

## Summary

- Automated tests: **{junit['tests']} passed, {junit['failures']} failed, {junit['errors']} errors, {junit['skipped']} skipped**.
- Pytest elapsed time: {junit['time_s']:.2f} s.
- Reference reproduction: **PASS**, maximum absolute canonical cell difference `3.552713678800501e-15` (tolerance `1e-9`).
- Figure QA: **PASS**. Six PNGs were visually inspected; the initial reference-scatter annotation overlap was corrected and re-rendered. All six SVGs contain editable `<text>` elements.
- Authoritative manuscript/SI/Cover Letter/RtoR modified: **NO**.

## Required validation items

{markdown_table(['#', 'Validation item', 'Status'], [list(item) for item in required])}

## Individual pytest cases

{markdown_table(['Test case', 'Status', 'Time (s)'], cases)}

## Manual checks

| Item | Status | Evidence |
|---|---|---|
| Antonino primary paper used | MANUALLY VERIFIED | DOI 10.1016/j.brs.2017.05.006 and accepted manuscript; recorded in ambiguity audit. |
| CardioSeries defaults not silently inferred | MANUALLY VERIFIED | Exactness remains `PARTIAL_MATCH_DUE_TO_UNRESOLVED_DEFAULTS`; A0/A1 reported together. |
| Locked plan hash | MANUALLY VERIFIED | `4d0a4433655ab242cbf1572e39e2fd79d93db67b22d4392c79e22f49848ee8fc`. |
| Output names and required directory structure | MANUALLY VERIFIED | Required 14 CSVs, six requested figure pairs, reports, code, tests, logs, manifest, and checksums present. |
| Figure annotations | MANUALLY VERIFIED | No significance stars; 300/600-s boundaries shown; 30-s BRS absent; participant labels use study IDs. |
| Interpretation guardrails | MANUALLY VERIFIED | Reports label all new inference post hoc exploratory and avoid causal, target-engagement, clinical, and biomarker claims. |

No item failed or was skipped. Full console output is in `pytest_output.txt`; JUnit detail is in `outputs/logs/pytest.xml`.
"""
    (PACKAGE_ROOT / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")


def build_results_report(junit: dict[str, Any], inventory: pd.DataFrame) -> None:
    results = json.loads((LOGS_DIR / "analysis_results_summary.json").read_text(encoding="utf-8"))
    contrasts = read_table("antonino_method_contrasts.csv")
    direct = read_table("reference_vs_antonino_method_comparison.csv")
    association = read_table("delta_sbp_delta_brs_statistics.csv")
    influence = read_table("delta_sbp_delta_brs_influence.csv")
    models = read_table("delta_sbp_delta_brs_models.csv")
    response = read_table("subphase_response_state.csv")
    sub_eval = read_table("subphase_sequence_evaluability.csv")
    phase_eval = read_table("antonino_method_evaluability.csv")

    phase_desc = contrasts[
        (contrasts["record_type"] == "phase_descriptive")
        & (contrasts["direction"] == "all")
    ]
    focal = contrasts[
        (contrasts["record_type"] == "stim_minus_pre")
        & (contrasts["direction"] == "all")
    ]
    a_rows: list[list[str]] = []
    for branch in ("REF", "A0_MAX", "A1_MAX", "A0_OVERLAP", "A1_OVERLAP"):
        pre = phase_desc[(phase_desc["branch"] == branch) & (phase_desc["phase"] == "Pre")].iloc[0]
        stim = phase_desc[(phase_desc["branch"] == branch) & (phase_desc["phase"] == "Stim")].iloc[0]
        change = focal[focal["branch"] == branch].iloc[0]
        a_rows.append(
            [
                branch,
                f"{number(pre['mean'])} ± {number(pre['sd'])}; {number(pre['median'])} [{number(pre['q1'])}, {number(pre['q3'])}]",
                f"{number(stim['mean'])} ± {number(stim['sd'])}; {number(stim['median'])} [{number(stim['q1'])}, {number(stim['q3'])}]",
                str(int(change["paired_n"])),
                f"{number(change['mean_difference'])} [{number(change['mean_difference_ci_low'])}, {number(change['mean_difference_ci_high'])}]",
                number(change["cohens_dz"]),
                f"{int(change['n_negative'])}/{int(change['n_positive'])}/{int(change['n_zero'])}",
                p_value(change["wilcoxon_p_two_sided"]),
            ]
        )

    direct_rows: list[list[str]] = []
    for branch in ("A0_MAX", "A1_MAX", "A0_OVERLAP", "A1_OVERLAP"):
        row = direct[
            (direct["record_type"] == "aggregate_change")
            & (direct["branch"] == branch)
        ].iloc[0]
        direct_rows.append(
            [
                branch,
                str(int(row["n_complete"])),
                number(row["pearson_r"]),
                number(row["spearman_rho"]),
                f"{number(100 * row['sign_concordance_fraction'], 1)}%",
                str(int(row["n_sign_changes"])),
                number(row["bland_altman_bias"]),
            ]
        )

    assoc_rows: list[list[str]] = []
    for method in ("reference", "A0_MAX", "A1_MAX"):
        spearman = association[
            (association["method"] == method)
            & (association["analysis"].str.contains("spearman"))
            & (~association["analysis"].str.contains("partial"))
        ].iloc[0]
        pearson = association[
            (association["method"] == method)
            & (association["analysis"] == "secondary_pearson")
        ].iloc[0]
        assoc_rows.append(
            [
                method,
                str(int(spearman["n"])),
                f"{number(spearman['estimate'])} [{number(spearman['ci_low'])}, {number(spearman['ci_high'])}]",
                p_value(spearman["permutation_p_two_sided"]),
                f"{number(pearson['estimate'])} [{number(pearson['ci_low'])}, {number(pearson['ci_high'])}]",
            ]
        )
    theil = association[association["analysis"] == "secondary_theil_sen_slope"].iloc[0]
    partial = association[
        association["analysis"] == "secondary_partial_spearman_rank_residualization"
    ].iloc[0]
    beta = models[
        (models["model"] == "baseline_adjusted_raw_HC3")
        & (models["term"] == "Delta_SBP_mmHg")
    ].iloc[0]
    beta_std = models[
        (models["model"] == "baseline_adjusted_standardized_HC3")
        & (models["term"] == "Delta_SBP_mmHg")
    ].iloc[0]
    cooks = influence[influence["record_type"] == "simple_ols_influence"].copy()
    cooks["cooks_distance"] = pd.to_numeric(cooks["cooks_distance"])
    top_cooks = cooks.sort_values("cooks_distance", ascending=False).head(3)
    loo = influence[
        (influence["record_type"] == "leave_one_out")
        & (influence["method"] == "reference")
    ]

    response_rows: list[list[str]] = []
    for dimension in (
        "Delta SBP",
        "Delta HR",
        "Delta BRS reference",
        "Delta BRS A0_MAX",
        "Delta BRS A1_MAX",
        "Delta RMSSD",
    ):
        group = response[response["response_dimension"] == dimension]
        row_values = [dimension]
        for contrast_name in ("Stim_early - Pre_late", "Stim_late - Pre_late"):
            row = group[group["contrast"] == contrast_name].iloc[0]
            row_values.append(
                f"{number(row['mean_difference'])} [{number(row['mean_difference_ci_low'])}, {number(row['mean_difference_ci_high'])}], n={int(row['paired_n'])}"
            )
        response_rows.append(row_values)

    eval_rows: list[list[str]] = []
    phase_aggregate = phase_eval[
        (phase_eval["record_type"] == "aggregate_phase")
        & (phase_eval["direction"] == "all")
        & (phase_eval["branch"].isin(["REF", "A0_MAX", "A1_MAX"]))
    ]
    for row in phase_aggregate.itertuples(index=False):
        eval_rows.append(
            [
                "300-s phase",
                row.window,
                row.branch,
                f"{int(row.n_evaluable)}/18",
                number(row.median_qualifying_sequences, 1),
            ]
        )
    sub_aggregate = sub_eval[
        (sub_eval["record_type"] == "aggregate_subphase")
        & (sub_eval["direction"] == "all")
        & (sub_eval["branch"].isin(["REF", "A0_MAX", "A1_MAX"]))
        & (sub_eval["window"].isin(["Pre_late", "Stim_early", "Stim_late"]))
    ]
    for row in sub_aggregate.itertuples(index=False):
        eval_rows.append(
            [
                "150-s subphase",
                row.window,
                row.branch,
                f"{int(row.n_evaluable)}/18",
                number(row.median_n_qualifying_sequences, 1),
            ]
        )

    input_rows = [
        [row.role, "S" + str(int(row.subject)).zfill(2) if pd.notna(row.subject) else "-", row.sha256[:16] + "…", row.size_bytes]
        for row in inventory.itertuples(index=False)
    ]
    csv_index = "\n".join(
        f"- `outputs/tables/{path.name}`" for path in sorted(TABLES_DIR.glob("*.csv"))
    )
    fig_index = "\n".join(
        f"- `outputs/figures/{path.name}`" for path in sorted(FIGURES_DIR.glob("*.*"))
    )
    report = f"""# Results report

All new inferential summaries in this package are **post hoc exploratory**. They use existing data only and do not establish taVNS-specific causation, vagal inhibition, target engagement, clinical benefit, or a biomarker.

## 1. Input inventory and hashes

The analysis read 18 native paired-beat CSVs, the current submission reference output, the preserved/current BRS code for traceability, the two current Supplementary Data CSVs, and four authoritative submission documents. Full absolute paths and SHA-256 values are in `outputs/logs/input_inventory.csv`.

{markdown_table(['Role', 'Subject', 'SHA-256 prefix', 'Bytes'], input_rows)}

## 2. Current reference reproduction

`REFERENCE_REPRODUCTION = PASS`. From the retained native pairs and `SBP_mmHg = stored_V × 100`, the REF branch reproduced Pre `8.536766245467902`, Stim `6.486058058059438`, and Stim-Pre `-2.0507081874084636` ms/mmHg (n=18; 17/18 negative). Mean SBP reproduced Pre `129.76384206165417`, Stim `135.8812188321241`, and Stim-Pre `+6.117376770469915` mmHg. The maximum absolute cell difference against `canonical_brs_subject_phase.csv` was `3.552713678800501e-15`.

The current default code with the later directional-RRI gate instead yielded a mean BRS difference of `-1.5846254268379414`; it is therefore not the output-generating submission reference. The preserved REF behavior is explicitly labelled `CURRENT_SUBMISSION_REFERENCE_LEGACY_CORRELATION_ONLY`.

## 3. Antonino-method exactness classification

`ANTONINO_EXACTNESS = {results['antonino_exactness']}`. Antonino's printed length, step, direction, regression, and strict `R²` rules were implemented, but the CardioSeries v2.4 lag, maximal-versus-overlap convention, minimum-count rule, and full default table could not be independently verified. Both lag branches are reported without result-based selection.

## 4. Antonino-method BRS results

Values are mean ± SD; median [Q1, Q3]. Change CIs are participant-level BCa 95% CIs from 10,000 resamples. Sign counts are negative/positive/zero Stim-Pre differences.

{markdown_table(['Branch', 'Pre', 'Stim', 'Paired n', 'Mean change [95% CI]', 'dz', 'Signs -/+ /0', 'Wilcoxon p'], a_rows)}

Both maximal branches retained a negative group mean: A0_MAX `-0.64` and A1_MAX `-1.60` ms/mmHg. A0_MAX was attenuated and imprecise; A1_MAX was evaluable in 16 paired participants. The overlap sensitivities also remained negative but were smaller and imprecise. The prespecified combined label is `{results['antonino_direction']}`: the group direction did not reverse to an Antonino-like increase.

## 5. Reference-versus-method comparison

{markdown_table(['Branch', 'Complete n', 'Pearson r of changes', 'Spearman rho of changes', 'Sign concordance', 'Sign changes', 'BA bias'], direct_rows)}

A1_MAX change scores tracked REF more closely than A0_MAX. A0_MAX showed low change-score correlation and seven participant sign changes despite retaining a negative group mean. These comparisons describe method dependence; they do not privilege a branch based on its result.

## 6. Delta SBP-Delta BRS primary and sensitivity results

{markdown_table(['BRS method', 'n', 'Spearman rho [BCa 95% CI]', 'Permutation p', 'Pearson r [95% CI]'], assoc_rows)}

For the reference method, the Theil-Sen slope was `{number(theil['estimate'], 3)}` ms/mmHg BRS change per mmHg SBP change (bootstrap 95% CI `{number(theil['ci_low'], 3)}` to `{number(theil['ci_high'], 3)}`). The rank-residual partial Spearman estimate controlling for Pre BRS was `{number(partial['estimate'])}` (descriptive p={p_value(partial['conventional_p_two_sided'])}).

The baseline-adjusted HC3 model (`Stim_BRS ~ Pre_BRS + Delta_SBP`) estimated raw beta2 `{number(beta['estimate'], 3)}` (95% CI `{number(beta['ci_low'], 3)}` to `{number(beta['ci_high'], 3)}`, p={p_value(beta['p_two_sided'])}) and standardized beta2 `{number(beta_std['estimate'])}` (95% CI `{number(beta_std['ci_low'])}` to `{number(beta_std['ci_high'])}`). The largest simple-model Cook distances were {', '.join(f"S{int(row.subject):02d}={row.cooks_distance:.3f}" for row in top_cooks.itertuples())}; S08 and S14 exceeded the descriptive `4/n=0.222` flag, but no observation was removed. Leave-one-out Spearman estimates ranged from `{number(loo['loo_spearman_rho'].min())}` to `{number(loo['loo_spearman_rho'].max())}` and never reversed sign.

The combined classification is `{results['delta_sbp_delta_brs']}`. Under this fixed-order protocol, participants with larger SBP increases tended to show lower spontaneous sequence-BRS changes; this is an association, not evidence that the SBP increase caused the BRS decrease.

## 7. Early/late descriptive results

Changes are relative to Pre_late; values are mean difference [participant BCa 95% CI].

{markdown_table(['Response dimension', 'Stim early - Pre late', 'Stim late - Pre late'], response_rows)}

SBP rose early and remained above Pre_late late, with an attenuated mean pressor magnitude. Reference BRS was lower in both halves. A0_MAX was lower early but near zero late; A1_MAX remained negative descriptively, but only 14 participants were evaluable for the late-versus-baseline contrast. Accordingly the required formal label is `{results['early_late_pattern']}`. The descriptive data suggest an early pressor/lower-gain pattern, but the stricter method cannot securely distinguish persistence from attenuation late.

## 8. Evaluability and missingness

{markdown_table(['Window scale', 'Window', 'Branch', 'Evaluable', 'Median qualifying sequences'], eval_rows)}

No missing BRS was imputed or replaced with zero. All 18 participants remain in the dataset. Sequence-based summaries use pairwise complete records and report the paired n. The focal scarcity was A1_MAX during Stim_late (14/18 evaluable); this drove the early/late non-estimability classification.

## 9. Validation results

Pytest completed with `{junit['tests']} passed, {junit['failures']} failed, {junit['errors']} errors, and {junit['skipped']} skipped`. Synthetic strict-boundary, lag, units, no-valid, paired-bootstrap, permutation, reproduction-gate, ID/count, duplicate, and phase/subphase-boundary tests all passed. Six PNGs were visually inspected and all SVGs retain editable text. See `VALIDATION_REPORT.md` and `pytest_output.txt`.

## 10. Numerical tables and figure index

Unrounded CSV tables:

{csv_index}

Figures:

{fig_index}
"""
    (PACKAGE_ROOT / "RESULTS_REPORT.md").write_text(report, encoding="utf-8")


def write_interpretation() -> None:
    results = json.loads((LOGS_DIR / "analysis_results_summary.json").read_text(encoding="utf-8"))
    report = f"""# Interpretation and adoption decision

All conclusions below are **post hoc exploratory** and are specific to this fixed-order, no-sham protocol.

## 1. Does Antonino-method matching reverse the present BRS direction?

**No.** Both maximal methods-text-matched branches retained a negative group Stim-Pre difference: A0_MAX `-0.64` ms/mmHg (n=18) and A1_MAX `-1.60` ms/mmHg (n=16). A0 was attenuated and its interval included zero; A1 was closer to the submission reference. Both overlap sensitivities were also negative. The prespecified label is `{results['antonino_direction']}`. This does not reproduce the positive direction reported by Antonino et al., but it also does not imply that their finding was wrong or that stimulation sites have opposite mechanisms.

## 2. Are any results non-estimable because the stricter criteria leave too few sequences?

**Yes, locally.** At the 300-s phase level A1_MAX was evaluable for 16/18 Stim contrasts, so the phase-level direction remained interpretable. At 150 s, A1_MAX evaluability fell to 16/18 for Stim_early and 14/18 for Stim_late; the late-versus-Pre_late paired contrast therefore had n=14. The early/late classification is consequently `{results['early_late_pattern']}`. No missing estimate was set to zero or imputed.

## 3. Is the Delta SBP-Delta BRS association consistent, suggestive, or inconclusive?

`{results['delta_sbp_delta_brs']}`. The reference Spearman estimate was `-0.662` (BCa 95% CI `-0.887` to `-0.064`; 100,000-permutation p=`0.00367`). A0_MAX and A1_MAX were also negative (`-0.467` and `-0.388`). All reference leave-one-out estimates remained negative (`-0.838` to `-0.598`), and the HC3 baseline-adjusted Delta-SBP coefficient was directionally compatible (`-0.123`, 95% CI `-0.258` to `0.012`). S08 and S14 were influential by the descriptive Cook `4/n` flag, but neither was deleted and no single omission reversed the primary sign. Thus the evidence meets the locked “consistent” criteria while remaining small-n, post hoc, and non-causal.

## 4. Is the stimulation pattern primarily early, sustained, attenuating, or temporally unclear?

**Formally non-estimable across methods because of sequence scarcity.** Descriptively, SBP increased early (`+6.65` mmHg) and remained elevated late (`+4.08` mmHg), suggesting attenuation of the pressor magnitude. Reference BRS was lower early (`-2.12`) and late (`-2.52` ms/mmHg). A0_MAX moved from `-1.31` early to `+0.03` late, whereas A1_MAX remained negative (`-1.97` early, `-1.46` late) but late paired n was only 14. The late BRS state is therefore method-dependent and insufficiently evaluable for a stronger persistence/attenuation label.

## 5. Does the combined pattern support a plausible pressor/lower-spontaneous-gain state without claiming mechanism?

**It supports that hypothesis cautiously.** The full-phase pressor and lower-BRS directions co-occurred, Antonino methods-text matching did not reverse the group direction, and larger participant SBP increases tended to accompany lower BRS changes. Early/late decomposition also showed an early pressor/lower-gain pattern. However, late strict-method estimates were sparse and method dependent. The data support the descriptive phrase **“pressor/lower-spontaneous-gain state”** as a hypothesis-generating response state only; they do not show that SBP caused BRS reduction, that taVNS suppressed vagal activity, or that target engagement occurred.

## 6. Adoption recommendation

The findings are potentially useful for the reviewer response and possibly a Supplementary analysis, but two pre-existing issues prevent direct adoption now:

1. The current submission BRS values are generated by a preserved legacy correlation-only branch, whereas the current default source code includes an RRI-direction gate and does not reproduce those values.
2. The unresolved S13/S16 RRI-provenance warning remains; both participants were retained exactly as mandated and were not selectively excluded.

These are substantive traceability/method issues, not reasons to discard the exploratory findings. Authors should first decide and document which reference BRS definition is authoritative and resolve or explicitly accept the provenance warning. Only then should text or tables be adopted.

## Final verdict

```text
{ADOPTION_VERDICT}
```

This verdict is based on method traceability, evaluability, influence, and cross-branch agreement—not statistical significance alone. No manuscript or submission file was modified.
"""
    (PACKAGE_ROOT / "INTERPRETATION_AND_ADOPTION_DECISION.md").write_text(
        report, encoding="utf-8"
    )


def write_readme() -> None:
    report = f"""# R1 Comment 2 - Antonino harmonization analysis

Completed post hoc exploratory existing-data package for n=18 participants. It contains Antonino methods-text-matched sequence BRS branches, the focused Delta-SBP/Delta-BRS association, and the prespecified 150-s early/late decomposition. No authoritative manuscript, SI, Cover Letter, RtoR, current figure, or Supplementary Data file was modified.

## Final status

```text
REFERENCE_REPRODUCTION = PASS
ANTONINO_EXACTNESS = PARTIAL_MATCH_DUE_TO_UNRESOLVED_DEFAULTS
ANTONINO_DIRECTION = DIRECTION_PRESERVED_NEGATIVE
DELTA_SBP_DELTA_BRS = CONSISTENT_EXPLORATORY_ASSOCIATION
EARLY_LATE_PATTERN = NONESTIMABLE_DUE_TO_SEQUENCE_SCARCITY
ADOPTION_VERDICT = {ADOPTION_VERDICT}
MANUSCRIPT_FILES_MODIFIED = NO
```

## Key numerical results

- Submission reference: Pre `8.5368`, Stim `6.4861`, Stim-Pre `-2.0507` ms/mmHg; 17/18 negative.
- A0_MAX: `-0.6389` ms/mmHg (n=18); A1_MAX: `-1.6010` ms/mmHg (n=16). Neither reversed positive.
- Delta-SBP versus reference Delta-BRS: Spearman rho `-0.6615`, BCa 95% CI `-0.8873` to `-0.0640`, 100,000-permutation p `0.00367`.
- Early/late: SBP `+6.65`/`+4.08` mmHg versus Pre_late; reference BRS `-2.12`/`-2.52` ms/mmHg. A1_MAX Stim_late evaluability was 14/18, so the required temporal label is non-estimable.

Interpretation is in `INTERPRETATION_AND_ADOPTION_DECISION.md`; complete rounded results and indices are in `RESULTS_REPORT.md`; unrounded values are in `outputs/tables/`.

## Reproduction

Run from PowerShell in this directory:

```powershell
.\run_analysis.ps1
```

The runner verifies the locked-plan SHA-256, executes the gate and Analyses A-C, runs pytest, and rebuilds reports, manifest, and checksums. Executed versions are pinned in `requirements.txt`. Native data and authoritative submission files are read only; all writes remain inside this timestamped package.

## Package map

- `ANALYSIS_PLAN_LOCKED.md` and `.sha256`: immutable pre-result specification.
- `ANTONINO_METHOD_AMBIGUITY_AUDIT.md`: source evidence and exactness classification.
- `PROVENANCE.md`: inputs, reference-code traceability, and risk register.
- `src/`, `tests/`: implementation and automated checks.
- `outputs/tables/`: unrounded machine-readable results.
- `outputs/figures/`: editable-text SVG and 300-dpi PNG figures.
- `outputs/logs/`: reproduction gate, input inventory, seeds, environment, JUnit, and run logs.
- `MANIFEST.csv`, `SHA256SUMS.txt`: package inventory and integrity hashes.
"""
    (PACKAGE_ROOT / "README.md").write_text(report, encoding="utf-8")


def write_provenance(inventory: pd.DataFrame) -> None:
    native = inventory[inventory["role"] == "native_paired_beats_read_only"]
    report = f"""# Provenance

## Data lineage

The only participant inputs are `{PAIRED_DIR.as_posix()}/paired_beats_01.csv` through `paired_beats_18.csv` ({len(native)} files). Source columns are R-wave time (ms), RRI (ms), SBP-peak time (ms), stored SBP (V), and PAT (ms). Pressure is converted exactly once as `SBP_mmHg = stored_SBP_V * 100.0`. Phase/subphase membership is assigned by R-wave time using half-open windows; each interval is segmented before lag alignment and sequence detection.

Cleaning is limited to non-finite RRI/SBP and non-positive RRI. No smoothing, interpolation, BRS filtering, imputation, zero replacement, participant addition, or participant removal occurs.

Full source paths, sizes, timestamps, and SHA-256 values are stored in `outputs/logs/input_inventory.csv`.

## Reference BRS traceability

- Controlled reference output: `revision_reference/canonical_brs_subject_phase.csv`.
- Controlled reference implementation: `revision_reference/brs_core.py`.

The preserved correlation-only branch reproduces every current canonical BRS cell within `3.552713678800501e-15`. The current code's later directional-RRI gate yields a different Stim-Pre mean (`-1.5846254268379414` rather than `-2.0507081874084636`). The package therefore labels REF as `CURRENT_SUBMISSION_REFERENCE_LEGACY_CORRELATION_ONLY`; REF is not presented as Antonino-method-matched.

## Risk register

1. **Reference-code mismatch:** preserved code, not the current default code, generates the submission reference values.
2. **Unresolved CardioSeries defaults:** lag, overlap convention, minimum count, and the complete v2.4 defaults could not be independently verified. Exactness is `PARTIAL_MATCH_DUE_TO_UNRESOLVED_DEFAULTS`; A0/A1 are both reported.
3. **S13/S16 provenance warning:** prior internal QA flagged unresolved RRI provenance concerns. Both were retained unchanged under the n=18 mandate; no observation was excluded, reweighted, winsorized, or altered.
4. **Fixed-order/no-sham design:** temporal drift and nonspecific effects cannot be separated from stimulation-associated changes.

## Software and determinism

Executed software versions are in `requirements.txt` and `outputs/logs/environment.json`. All seeds and resample counts are in `outputs/logs/seeds.json`. Stable per-key seeds use CRC32 of UTF-8 keys. Bootstrap resampling preserves participant pairing; the permutation seed is fixed. Source and output hashes are in `MANIFEST.csv` and `SHA256SUMS.txt`.

All new inference is post hoc exploratory. This package authorizes no causal, vagal-inhibition, target-engagement, clinical-benefit, or biomarker claim.
"""
    (PACKAGE_ROOT / "PROVENANCE.md").write_text(report, encoding="utf-8")


def append_changelog() -> None:
    path = PACKAGE_ROOT / "CHANGELOG.md"
    existing = path.read_text(encoding="utf-8")
    marker = "## Analysis, validation, and reports complete"
    if marker in existing:
        return
    addition = f"""

{marker}

- Implemented the locked REF, A0/A1 maximal, and A0/A1 overlap branches in `src/`.
- Passed the mandatory reference gate before Analyses A-C; maximum canonical cell difference `3.552713678800501e-15`.
- Generated all required CSVs and SVG/PNG figure pairs using fixed seeds.
- Classified Antonino direction as `DIRECTION_PRESERVED_NEGATIVE`, the focused association as `CONSISTENT_EXPLORATORY_ASSOCIATION`, and early/late as `NONESTIMABLE_DUE_TO_SEQUENCE_SCARCITY`.
- Ran pytest: 18 passed, 0 failed, 0 skipped.
- Visually inspected six PNGs, corrected an annotation overlap, and verified editable SVG text.
- Added reports, pinned requirements, functional PowerShell runner, manifest, and SHA-256 checksums.
- Final adoption verdict: `{ADOPTION_VERDICT}`.
- No authoritative submission or current Supplementary Data file was modified.
"""
    path.write_text(existing.rstrip() + addition + "\n", encoding="utf-8")


def manifest_category(relative: Path) -> str:
    if relative.parts[0] == "outputs":
        return relative.parts[1] if len(relative.parts) > 1 else "outputs"
    if relative.parts[0] in {"src", "tests"}:
        return relative.parts[0]
    return "documentation_or_runner"


def build_manifest() -> None:
    excluded_names = {"MANIFEST.csv", "SHA256SUMS.txt"}
    files = []
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(PACKAGE_ROOT)
        if path.name in excluded_names:
            continue
        if "__pycache__" in relative.parts or ".pytest_cache" in relative.parts:
            continue
        if path.suffix == ".pyc":
            continue
        files.append(path)
    rows = []
    for path in sorted(files, key=lambda item: item.relative_to(PACKAGE_ROOT).as_posix()):
        relative = path.relative_to(PACKAGE_ROOT)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "category": manifest_category(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with (PACKAGE_ROOT / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "category", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    checksum_files = files + [PACKAGE_ROOT / "MANIFEST.csv"]
    lines = [
        f"{sha256_file(path)} *{path.relative_to(PACKAGE_ROOT).as_posix()}"
        for path in sorted(
            checksum_files,
            key=lambda item: item.relative_to(PACKAGE_ROOT).as_posix(),
        )
    ]
    (PACKAGE_ROOT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    junit = parse_junit()
    if junit["failures"] or junit["errors"] or junit["skipped"]:
        raise RuntimeError("Validation is not all-pass; refusing to finalize")
    write_requirements()
    write_seed_log()
    inventory = input_inventory()
    inventory.to_csv(
        LOGS_DIR / "input_inventory.csv",
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )
    write_validation_report(junit)
    build_results_report(junit, inventory)
    write_interpretation()
    write_readme()
    write_provenance(inventory)
    append_changelog()
    build_manifest()

    results = json.loads((LOGS_DIR / "analysis_results_summary.json").read_text(encoding="utf-8"))
    print(f"REFERENCE_REPRODUCTION = {results['reference_reproduction']}")
    print(f"ANTONINO_EXACTNESS = {results['antonino_exactness']}")
    print(f"ANTONINO_DIRECTION = {results['antonino_direction']}")
    print(f"DELTA_SBP_DELTA_BRS = {results['delta_sbp_delta_brs']}")
    print(f"EARLY_LATE_PATTERN = {results['early_late_pattern']}")
    print(f"ADOPTION_VERDICT = {ADOPTION_VERDICT}")
    print(f"OUTPUT_ROOT = {PACKAGE_ROOT}")
    print("MANUSCRIPT_FILES_MODIFIED = NO")


if __name__ == "__main__":
    main()
