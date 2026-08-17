"""Reproduce submitted anchor values before reviewer-requested analyses."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import signal, stats
from statsmodels.tsa.api import VAR

from brs_core import compute_canonical_brs
from coupling_core import coherence_statistic
from haemodynamics_core import compute_subject_phase_context
from revision_utils import (
    PHASE_ORDER,
    PROJECT_ROOT,
    REPORTS_DIR,
    RESULTS_DIR,
    SUBJECTS,
    assert_boundaries,
    ensure_output_dirs,
    load_paired_phase,
    resample_phase_4hz,
    write_csv,
)
from stats_core import cohens_dz, wilcoxon_two_sided


def legacy_gc(sbp_4hz: np.ndarray, rri_4hz: np.ndarray) -> dict[str, float | int]:
    sbp = signal.detrend(sbp_4hz)
    rri = signal.detrend(rri_4hz)
    sbp = (sbp - np.nanmean(sbp)) / np.nanstd(sbp)
    rri = (rri - np.nanmean(rri)) / np.nanstd(rri)
    data = pd.DataFrame({"SBP_mmHg": sbp, "RRI_ms": rri})
    result = VAR(data).fit(ic="aic", maxlags=12, trend="c")
    sbp_to_rri = result.test_causality("RRI_ms", ["SBP_mmHg"], kind="f")
    rri_to_sbp = result.test_causality("SBP_mmHg", ["RRI_ms"], kind="f")
    return {
        "GC_F_BP_to_RRI": float(np.atleast_1d(sbp_to_rri.test_statistic)[0]),
        "GC_p_BP_to_RRI": float(sbp_to_rri.pvalue),
        "GC_F_RRI_to_SBP": float(np.atleast_1d(rri_to_sbp.test_statistic)[0]),
        "GC_p_RRI_to_SBP": float(rri_to_sbp.pvalue),
        "GC_order_p": int(result.k_ar),
    }


def compute_observed_coupling() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subject in SUBJECTS:
        for phase in PHASE_ORDER:
            frame = load_paired_phase(subject, phase)
            grid, sbp_4hz, rri_4hz = resample_phase_4hz(frame)
            rows.append(
                {
                    "subject": subject,
                    "phase": phase,
                    "n_beats": len(frame),
                    "n_4hz": len(grid),
                    "Coh_mean": coherence_statistic(sbp_4hz, rri_4hz),
                    **legacy_gc(sbp_4hz, rri_4hz),
                }
            )
    return pd.DataFrame(rows)


def compare_cells(
    reproduced: pd.DataFrame,
    reference: pd.DataFrame,
    metrics: list[str],
    label: str,
) -> pd.DataFrame:
    merged = reproduced.merge(reference, on=["subject", "phase"], suffixes=("_reproduced", "_reference"))
    rows: list[dict[str, float | int | str | bool]] = []
    for _, row in merged.iterrows():
        for metric in metrics:
            reproduced_value = float(row[f"{metric}_reproduced"])
            reference_value = float(row[f"{metric}_reference"])
            both_missing = bool(
                not np.isfinite(reproduced_value) and not np.isfinite(reference_value)
            )
            one_missing = bool(
                np.isfinite(reproduced_value) != np.isfinite(reference_value)
            )
            absolute_delta = (
                0.0 if both_missing else abs(reproduced_value - reference_value)
            )
            rows.append(
                {
                    "analysis": label,
                    "subject": int(row["subject"]),
                    "phase": row["phase"],
                    "metric": metric,
                    "reproduced": reproduced_value,
                    "reference": reference_value,
                    "absolute_delta": absolute_delta,
                    "both_values_missing": both_missing,
                    "one_value_missing": one_missing,
                    "within_numeric_tolerance": bool(
                        both_missing or (not one_missing and absolute_delta <= 1e-9)
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    assert_boundaries()
    ensure_output_dirs()
    print("Phase 1 baseline reproduction")

    canonical, _, _ = compute_canonical_brs()
    write_csv(canonical, RESULTS_DIR / "baseline_canonical_brs_subject_phase.csv")

    legacy_brs = pd.read_csv(PROJECT_ROOT / "phase5_step1" / "outputs" / "phase5_step1_raw.csv")
    legacy_brs = legacy_brs.rename(columns={"Subject": "subject", "Phase": "phase"})
    brs_metrics = ["BRS_seq_all", "BRS_seq_up", "BRS_seq_down", "BEI_all"]
    brs_comparison = compare_cells(canonical, legacy_brs, brs_metrics, "canonical_BRS")
    write_csv(brs_comparison, RESULTS_DIR / "baseline_brs_cell_comparison.csv")

    coupling = compute_observed_coupling()
    write_csv(coupling, RESULTS_DIR / "baseline_coupling_subject_phase.csv")
    legacy_coupling = pd.read_csv(
        PROJECT_ROOT / "_archive" / "phase0" / "outputs" / "phase0_n18_raw.csv"
    )
    legacy_coupling = legacy_coupling[legacy_coupling["Mode"] == "ungated"].copy()
    legacy_coupling = legacy_coupling.rename(columns={"Subject": "subject", "Phase": "phase"})
    coupling_metrics = [
        "Coh_mean",
        "GC_F_BP_to_RRI",
        "GC_p_BP_to_RRI",
        "GC_F_RRI_to_SBP",
        "GC_p_RRI_to_SBP",
        "GC_order_p",
    ]
    coupling_comparison = compare_cells(
        coupling,
        legacy_coupling,
        coupling_metrics,
        "coherence_and_bivariate_GC",
    )
    write_csv(coupling_comparison, RESULTS_DIR / "baseline_coupling_cell_comparison.csv")

    context = compute_subject_phase_context()
    write_csv(context, RESULTS_DIR / "baseline_context_subject_phase.csv")
    legacy_hrv = pd.read_csv(PROJECT_ROOT / "hrv_panel" / "outputs" / "hrv_panel_raw.csv")
    legacy_hrv["subject"] = legacy_hrv["Subject"].str.replace("S", "", regex=False).astype(int)
    legacy_hrv = legacy_hrv.rename(columns={"Phase": "phase"})
    hrv_mapping = {
        "Mean_RRI_ms": "Mean_RRI",
        "Mean_HR_bpm": "Mean_HR",
        "RRI_SD_ms": "SDNN",
        "RMSSD_ms": "RMSSD",
        "HF_HRV_ms2": "HF_power",
    }
    hrv_rows = []
    hrv_merged = context.merge(legacy_hrv, on=["subject", "phase"])
    for _, row in hrv_merged.iterrows():
        for reproduced_metric, reference_metric in hrv_mapping.items():
            reproduced_value = float(row[reproduced_metric])
            reference_value = float(row[reference_metric])
            delta = abs(reproduced_value - reference_value)
            hrv_rows.append(
                {
                    "analysis": "representative_HRV",
                    "subject": int(row["subject"]),
                    "phase": row["phase"],
                    "metric": reproduced_metric,
                    "reference_metric": reference_metric,
                    "reproduced": reproduced_value,
                    "reference": reference_value,
                    "absolute_delta": delta,
                    "within_numeric_tolerance": delta <= 1e-8,
                }
            )
    hrv_comparison = pd.DataFrame(hrv_rows)
    write_csv(hrv_comparison, RESULTS_DIR / "baseline_hrv_cell_comparison.csv")

    brs_pivot = canonical.pivot(index="subject", columns="phase", values="BRS_seq_all")
    diff = brs_pivot["Stim"].to_numpy(float) - brs_pivot["Pre"].to_numpy(float)
    wilcoxon_statistic, wilcoxon_p, wilcoxon_method = wilcoxon_two_sided(diff)
    targets = {
        "n": len(diff),
        "pre_mean": float(brs_pivot["Pre"].mean()),
        "stim_mean": float(brs_pivot["Stim"].mean()),
        "mean_difference": float(np.mean(diff)),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "wilcoxon_method": wilcoxon_method,
        "cohens_dz": cohens_dz(diff),
        "n_stim_below_pre": int(np.sum(diff < 0.0)),
    }
    target_checks = {
        "n_18": targets["n"] == 18,
        "pre_mean_tolerance": abs(targets["pre_mean"] - 8.54) <= 0.02,
        "stim_mean_tolerance": abs(targets["stim_mean"] - 6.49) <= 0.02,
        "difference_tolerance": abs(targets["mean_difference"] - (-2.05)) <= 0.02,
        "dz_tolerance": abs(targets["cohens_dz"] - (-0.74)) <= 0.03,
        "sign_count_17_of_18": targets["n_stim_below_pre"] == 17,
        "p_same_exact_rank_result": math.isclose(
            targets["wilcoxon_p_two_sided"],
            2.288818359375e-05,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "subject_cell_BRS_match": bool(brs_comparison["within_numeric_tolerance"].all()),
        "subject_cell_coherence_GC_match": bool(coupling_comparison["within_numeric_tolerance"].all()),
        "representative_HRV_match": bool(hrv_comparison["within_numeric_tolerance"].all()),
    }

    coherence_pivot = coupling.pivot(index="subject", columns="phase", values="Coh_mean")
    coherence_diff = coherence_pivot["Stim"].to_numpy(float) - coherence_pivot["Pre"].to_numpy(float)
    coherence_dz = cohens_dz(coherence_diff)
    t_raw = float(stats.ttest_1samp(coherence_diff, 0.0).statistic)
    t_from_dz = coherence_dz * math.sqrt(len(coherence_diff))
    prior = math.sqrt(2.0) / 2.0
    bf10_raw = float(pg.bayesfactor_ttest(t_raw, len(coherence_diff), paired=True, r=prior))
    bf10_reconstructed = float(
        pg.bayesfactor_ttest(t_from_dz, len(coherence_diff), paired=True, r=prior)
    )
    bayes_audit = {
        "metric": "Coh_mean",
        "n": len(coherence_diff),
        "dz_raw_paired": coherence_dz,
        "t_raw_paired_differences": t_raw,
        "t_reconstructed_dz_sqrt_n": t_from_dz,
        "t_absolute_delta": abs(t_raw - t_from_dz),
        "BF01_raw_paired": 1.0 / bf10_raw,
        "BF01_reconstructed": 1.0 / bf10_reconstructed,
        "BF01_absolute_delta": abs((1.0 / bf10_raw) - (1.0 / bf10_reconstructed)),
        "submitted_BF01": 3.91,
        "submitted_rounding_consistent": abs((1.0 / bf10_raw) - 3.91) <= 0.02,
        "test_model": "paired-difference JZS Bayesian t test",
        "prior_r": prior,
    }
    (RESULTS_DIR / "baseline_bayesian_audit.json").write_text(
        json.dumps(bayes_audit, indent=2), encoding="utf-8"
    )

    passed = all(target_checks.values())
    report_lines = [
        "# Baseline reproduction report",
        "",
        f"Verdict: `{'PASS' if passed else 'STOP-C FAIL'}`",
        "",
        "## Canonical BRSseq,all Stim-Pre",
        "",
        f"- n: {targets['n']}",
        f"- Pre mean: {targets['pre_mean']:.12f} ms/mmHg",
        f"- Stim mean: {targets['stim_mean']:.12f} ms/mmHg",
        f"- mean paired difference: {targets['mean_difference']:.12f} ms/mmHg",
        f"- Wilcoxon statistic: {targets['wilcoxon_statistic']:.12g}",
        f"- Wilcoxon two-sided p: {targets['wilcoxon_p_two_sided']:.15g}",
        f"- Cohen's dz: {targets['cohens_dz']:.12f}",
        f"- Stim < Pre: {targets['n_stim_below_pre']}/18",
        "",
        "## Gate checks",
        "",
    ]
    report_lines.extend(
        f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in target_checks.items()
    )
    report_lines.extend(
        [
            "",
            "## Cell-level reproduction",
            "",
            f"- Maximum BRS absolute delta: {brs_comparison['absolute_delta'].max():.3g}",
            f"- Maximum coherence/GC absolute delta: {coupling_comparison['absolute_delta'].max():.3g}",
            f"- Maximum representative-HRV absolute delta: {hrv_comparison['absolute_delta'].max():.3g}",
            "",
            "## Bayesian audit",
            "",
            f"- Raw paired t: {t_raw:.12f}",
            f"- dz x sqrt(n) t: {t_from_dz:.12f}",
            f"- Raw-paired BF01: {bayes_audit['BF01_raw_paired']:.6f}",
            f"- Reconstructed BF01: {bayes_audit['BF01_reconstructed']:.6f}",
            "- The reconstruction is algebraically identical when dz is computed from the same paired difference vector.",
            "- Wilcoxon remains the frequentist paired test; the JZS t model is retained only as a parametric Bayesian sensitivity analysis and may be moved to SI.",
        ]
    )
    (REPORTS_DIR / "baseline_reproduction_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    gate = {"passed": passed, "targets": targets, "checks": target_checks, "bayes_audit": bayes_audit}
    (RESULTS_DIR / "baseline_reproduction_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )

    print(json.dumps(targets, indent=2))
    print(f"baseline_gate={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
