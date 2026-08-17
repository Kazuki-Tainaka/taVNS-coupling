"""Run all prespecified reviewer-requested revision analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from brs_core import (
    CANONICAL_SETTING,
    compute_canonical_brs,
    compute_settings_subject_phase,
    full_factorial_settings,
    ofat_settings,
    summarize_setting_contrasts,
)
from coupling_core import (
    compute_coherence_significance,
    compute_gc_significance,
    phase_randomize,
    validate_coherence_surrogates,
)
from haemodynamics_core import compute_subject_phase_context, summarize_context
from revision_utils import (
    PHASE_ORDER,
    PROJECT_ROOT,
    REPORTS_DIR,
    RESULTS_DIR,
    SUBJECTS,
    assert_boundaries,
    ensure_output_dirs,
    write_csv,
)
from stats_core import (
    bca_interval,
    cohens_dz,
    mean_stat,
    paired_summary,
    wilcoxon_two_sided,
)


MASTER_SEED = 20_260_805


def json_dump(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value)}")


def validate_bca_implementation() -> dict[str, float | int | bool | str]:
    """Compare the local BCa mean CI with SciPy on a smooth synthetic sample."""
    rng = np.random.default_rng(20_260_807)
    sample = rng.normal(loc=-2.0, scale=2.5, size=18)
    local, _ = bca_interval(
        sample,
        mean_stat,
        n_resamples=10_000,
        seed=20_260_808,
    )
    reference = stats.bootstrap(
        (sample,),
        np.mean,
        confidence_level=0.95,
        n_resamples=10_000,
        method="BCa",
        rng=np.random.default_rng(20_260_808),
    )
    low_delta = abs(local.lower - float(reference.confidence_interval.low))
    high_delta = abs(local.upper - float(reference.confidence_interval.high))
    passed = bool(low_delta <= 0.02 and high_delta <= 0.02)
    return {
        "test": "BCa_mean_against_scipy_stats_bootstrap",
        "sample_n": len(sample),
        "local_low": local.lower,
        "local_high": local.upper,
        "reference_low": float(reference.confidence_interval.low),
        "reference_high": float(reference.confidence_interval.high),
        "absolute_delta_low": low_delta,
        "absolute_delta_high": high_delta,
        "tolerance": 0.02,
        "passed": passed,
    }


def validate_phase_randomization() -> dict[str, float | int | bool | str]:
    rng = np.random.default_rng(20_260_809)
    source = rng.normal(size=1_200)
    randomized = phase_randomize(source, rng)
    source_magnitude = np.abs(np.fft.rfft(source))
    randomized_magnitude = np.abs(np.fft.rfft(randomized))
    maximum_absolute_delta = float(np.max(np.abs(source_magnitude - randomized_magnitude)))
    tolerance = 1e-10
    return {
        "test": "Fourier_phase_randomization_preserves_rFFT_magnitude",
        "n": len(source),
        "maximum_absolute_delta": maximum_absolute_delta,
        "tolerance": tolerance,
        "passed": bool(maximum_absolute_delta <= tolerance),
    }


def canonical_descriptives_and_contrasts(
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    descriptive_rows: list[dict[str, float | int | str]] = []
    contrast_rows: list[dict[str, float | int | str]] = []
    replicate_rows: list[dict[str, float | int | str]] = []
    metrics = ("BRS_seq_all", "BRS_seq_up", "BRS_seq_down")
    comparisons = (("Stim", "Pre"), ("Post", "Pre"), ("Post", "Stim"))
    for metric_index, metric in enumerate(metrics):
        for phase in PHASE_ORDER:
            values = canonical.loc[canonical["phase"] == phase, metric].dropna().to_numpy(float)
            q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
            descriptive_rows.append(
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
        pivot = canonical.pivot(index="subject", columns="phase", values=metric)
        for comparison_index, (first, second) in enumerate(comparisons):
            aligned = pivot[[first, second]].dropna()
            seed = MASTER_SEED + metric_index * 1_000 + comparison_index * 10
            summary, replicates = paired_summary(
                aligned[first].to_numpy(float),
                aligned[second].to_numpy(float),
                first,
                second,
                seed=seed,
            )
            contrast_rows.append({"metric": metric, **summary})
            if metric == "BRS_seq_all" and first == "Stim" and second == "Pre":
                for estimand, values in replicates.items():
                    for replicate, value in enumerate(values, start=1):
                        replicate_rows.append(
                            {
                                "metric": metric,
                                "contrast": "Stim-Pre",
                                "estimand": estimand,
                                "replicate": replicate,
                                "value": value,
                            }
                        )
    return (
        pd.DataFrame(descriptive_rows),
        pd.DataFrame(contrast_rows),
        pd.DataFrame(replicate_rows),
    )


def canonical_loo(canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    pivot = canonical.pivot(index="subject", columns="phase", values="BRS_seq_all")
    rows: list[dict[str, float | int | str | bool]] = []
    for omitted in pivot.index:
        retained = pivot.drop(index=omitted).dropna(subset=["Pre", "Stim"])
        diff = retained["Stim"].to_numpy(float) - retained["Pre"].to_numpy(float)
        statistic, p_value, method = wilcoxon_two_sided(diff)
        rows.append(
            {
                "omitted_subject": int(omitted),
                "n": len(diff),
                "mean_difference_stim_minus_pre": float(np.mean(diff)),
                "median_difference_stim_minus_pre": float(np.median(diff)),
                "cohens_dz": cohens_dz(diff),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_two_sided": p_value,
                "wilcoxon_method": method,
                "negative_direction": bool(np.mean(diff) < 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "iterations": len(frame),
        "mean_difference_min": float(frame["mean_difference_stim_minus_pre"].min()),
        "mean_difference_max": float(frame["mean_difference_stim_minus_pre"].max()),
        "dz_min": float(frame["cohens_dz"].min()),
        "dz_max": float(frame["cohens_dz"].max()),
        "p_min": float(frame["wilcoxon_p_two_sided"].min()),
        "p_max": float(frame["wilcoxon_p_two_sided"].max()),
        "all_negative_direction": bool(frame["negative_direction"].all()),
        "all_p_below_0.05": bool((frame["wilcoxon_p_two_sided"] < 0.05).all()),
    }
    return frame, summary


def sequence_quality_contrasts(quality: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_sbp_ramps",
        "n_qualifying_brs_sequences",
        "mean_sequence_length_beats",
        "mean_sbp_ramp_amplitude_abs_mmHg",
        "mean_rri_response_signed_ms",
        "mean_rri_response_abs_ms",
        "mean_within_sequence_r",
        "BEI",
        "BRS_seq",
    ]
    comparisons = (("Stim", "Pre"), ("Post", "Pre"), ("Post", "Stim"))
    rows: list[dict[str, float | int | str]] = []
    for direction in ("all", "up", "down"):
        direction_data = quality[quality["direction"] == direction]
        for metric in metrics:
            pivot = direction_data.pivot(index="subject", columns="phase", values=metric)
            for first, second in comparisons:
                aligned = pivot[[first, second]].dropna()
                diff = aligned[first].to_numpy(float) - aligned[second].to_numpy(float)
                statistic, p_value, method = wilcoxon_two_sided(diff)
                rows.append(
                    {
                        "direction": direction,
                        "metric": metric,
                        "contrast": f"{first}-{second}",
                        "n": len(diff),
                        "mean_first": float(aligned[first].mean()) if len(diff) else np.nan,
                        "mean_second": float(aligned[second].mean()) if len(diff) else np.nan,
                        "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                        "median_difference": float(np.median(diff)) if len(diff) else np.nan,
                        "cohens_dz": cohens_dz(diff),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_two_sided": p_value,
                        "wilcoxon_method": method,
                        "estimability_status": "estimable" if len(diff) >= 3 else "not_estimable",
                        "NA_reason": "NA" if len(diff) >= 3 else "fewer_than_3_pairs",
                    }
                )
    return pd.DataFrame(rows)


def mixed_model_eligibility(quality: pd.DataFrame) -> dict[str, object]:
    all_direction = quality[quality["direction"] == "all"]
    pre_stim = all_direction[all_direction["phase"].isin(["Pre", "Stim"])]
    counts = pre_stim.pivot(
        index="subject", columns="phase", values="n_qualifying_brs_sequences"
    )
    eligible_by_subject = (counts["Pre"] >= 2) & (counts["Stim"] >= 2)
    eligible = bool(eligible_by_subject.all())
    return {
        "criterion": "every subject has at least two qualifying sequences in both Pre and Stim",
        "eligible": eligible,
        "eligible_subjects": [int(value) for value in eligible_by_subject[eligible_by_subject].index],
        "ineligible_subjects": [int(value) for value in eligible_by_subject[~eligible_by_subject].index],
        "decision": "run_mixed_model" if eligible else "omit_mixed_model",
        "reason": "criterion_met" if eligible else "prespecified_sequence_count_criterion_failed",
    }


def optional_brs_eligibility() -> pd.DataFrame:
    legacy = pd.read_csv(
        PROJECT_ROOT / "_archive" / "phase0" / "outputs" / "phase0_n18_raw.csv"
    )
    legacy = legacy[legacy["Mode"] == "ungated"]
    rows = []
    for phase in PHASE_ORDER:
        phase_data = legacy[legacy["Phase"] == phase]
        valid = pd.to_numeric(phase_data["BRS_TF_mean"], errors="coerce").notna()
        rows.append(
            {
                "phase": phase,
                "valid_participants": int(valid.sum()),
                "required_participants": 15,
                "participant_availability_pass": bool(valid.sum() >= 15),
                "estimator": "coherence-gated_Mayer_transfer_gain",
            }
        )
    frame = pd.DataFrame(rows)
    frame["overall_eligible_for_SI"] = bool(frame["participant_availability_pass"].all())
    frame["decision"] = np.where(
        frame["overall_eligible_for_SI"],
        "eligible_pending_frequency_bin_audit",
        "do_not_add_optional_estimator",
    )
    frame["NA_reason"] = np.where(
        frame["participant_availability_pass"],
        "NA",
        "fewer_than_15_valid_participants",
    )
    return frame


def main() -> int:
    assert_boundaries()
    ensure_output_dirs()
    baseline_gate = json.loads(
        (RESULTS_DIR / "baseline_reproduction_gate.json").read_text(encoding="utf-8")
    )
    if not baseline_gate.get("passed", False):
        raise RuntimeError("STOP-C baseline gate has not passed")

    print("[1/9] Implementation validation")
    bca_validation = validate_bca_implementation()
    phase_validation = validate_phase_randomization()
    json_dump(RESULTS_DIR / "implementation_validation.json", {
        "bca": bca_validation,
        "phase_randomization": phase_validation,
    })

    print("[2/9] Canonical BRS, bootstrap, LOO, and sequence quality")
    canonical, sequences, quality = compute_canonical_brs()
    write_csv(canonical, RESULTS_DIR / "canonical_brs_subject_phase.csv")
    write_csv(sequences, RESULTS_DIR / "canonical_brs_sequence_level.csv")
    write_csv(quality, RESULTS_DIR / "canonical_brs_sequence_quality_subject_phase.csv")
    descriptives, contrasts, canonical_bootstrap = canonical_descriptives_and_contrasts(canonical)
    write_csv(descriptives, RESULTS_DIR / "canonical_brs_phase_descriptives.csv")
    write_csv(contrasts, RESULTS_DIR / "canonical_brs_contrasts.csv")
    write_csv(canonical_bootstrap, RESULTS_DIR / "canonical_brs_bootstrap_replicates.csv")
    loo, loo_summary = canonical_loo(canonical)
    write_csv(loo, RESULTS_DIR / "canonical_brs_leave_one_out.csv")
    json_dump(RESULTS_DIR / "canonical_brs_leave_one_out_summary.json", loo_summary)
    quality_contrasts = sequence_quality_contrasts(quality)
    write_csv(quality_contrasts, RESULTS_DIR / "canonical_brs_sequence_quality_contrasts.csv")
    mixed_eligibility = mixed_model_eligibility(quality)
    json_dump(RESULTS_DIR / "sequence_level_mixed_model_eligibility.json", mixed_eligibility)

    print("[3/9] Full-factorial and OFAT BRS sensitivity")
    settings = full_factorial_settings()
    sensitivity_subject_phase = compute_settings_subject_phase(settings)
    write_csv(sensitivity_subject_phase, RESULTS_DIR / "brs_full_factorial_subject_phase.csv")
    ofat_pairs = ofat_settings()
    ofat_label_by_id = {setting.setting_id: label for label, setting in ofat_pairs}
    ofat_ids = set(ofat_label_by_id)
    full_summary, retained_bootstrap = summarize_setting_contrasts(
        sensitivity_subject_phase,
        bootstrap_setting_ids=ofat_ids,
        retain_bootstrap_keys={(CANONICAL_SETTING.setting_id, "all")},
        seed_base=MASTER_SEED,
    )
    write_csv(full_summary, RESULTS_DIR / "brs_full_factorial_contrasts.csv")
    write_csv(retained_bootstrap, RESULTS_DIR / "brs_sensitivity_retained_bootstrap_replicates.csv")
    ofat = full_summary[full_summary["setting_id"].isin(ofat_ids)].copy()
    ofat.insert(0, "ofat_change", ofat["setting_id"].map(ofat_label_by_id))
    write_csv(ofat, RESULTS_DIR / "brs_ofat_sensitivity.csv")
    sensitivity_summary = {
        "factorial_base_settings": len(settings),
        "factorial_setting_direction_rows": len(full_summary),
        "evaluable_rows": int((full_summary["estimability_status"] == "estimable").sum()),
        "negative_direction_rows": int(full_summary["negative_direction"].sum()),
        "negative_direction_fraction": float(full_summary["negative_direction"].mean()),
        "positive_sign_reversal_rows": int(full_summary["sign_reversal_positive"].sum()),
        "dz_min": float(full_summary["cohens_dz"].min()),
        "dz_median": float(full_summary["cohens_dz"].median()),
        "dz_max": float(full_summary["cohens_dz"].max()),
        "mean_difference_min": float(full_summary["mean_difference_stim_minus_pre"].min()),
        "mean_difference_median": float(full_summary["mean_difference_stim_minus_pre"].median()),
        "mean_difference_max": float(full_summary["mean_difference_stim_minus_pre"].max()),
    }
    json_dump(RESULTS_DIR / "brs_full_factorial_summary.json", sensitivity_summary)

    print("[4/9] Basic haemodynamics and representative HRV")
    context = compute_subject_phase_context()
    context_summary, context_contrasts = summarize_context(context)
    write_csv(context, RESULTS_DIR / "haemodynamics_hrv_subject_phase.csv")
    write_csv(context_summary, RESULTS_DIR / "haemodynamics_hrv_phase_summary.csv")
    write_csv(context_contrasts, RESULTS_DIR / "haemodynamics_hrv_contrasts.csv")

    print("[5/9] Synthetic coherence-surrogate validation")
    synthetic = validate_coherence_surrogates()
    write_csv(synthetic, RESULTS_DIR / "coherence_surrogate_synthetic_validation.csv")

    print("[6/9] Subject-level coherence significance")
    coherence, coherence_null, coherence_prevalence = compute_coherence_significance()
    write_csv(coherence, RESULTS_DIR / "coherence_subject_significance.csv")
    write_csv(coherence_null, RESULTS_DIR / "coherence_surrogate_null_values.csv")
    write_csv(coherence_prevalence, RESULTS_DIR / "coherence_prevalence_summary.csv")

    print("[7/9] Subject-level bivariate GC significance and diagnostics")
    gc_results, gc_prevalence = compute_gc_significance()
    write_csv(gc_results, RESULTS_DIR / "gc_subject_significance_and_diagnostics.csv")
    write_csv(gc_prevalence, RESULTS_DIR / "gc_prevalence_summary.csv")

    print("[8/9] Optional BRS estimator eligibility")
    optional_brs = optional_brs_eligibility()
    write_csv(optional_brs, RESULTS_DIR / "optional_brs_estimator_eligibility.csv")

    print("[9/9] Analysis completion summary")
    summary = {
        "baseline_reproduction_passed": bool(baseline_gate["passed"]),
        "bca_validation_passed": bool(bca_validation["passed"]),
        "phase_randomization_validation_passed": bool(phase_validation["passed"]),
        "synthetic_coherence_validation_passed": bool(synthetic["passed"].all()),
        "canonical_loo": loo_summary,
        "mixed_model": mixed_eligibility,
        "sensitivity": sensitivity_summary,
        "coherence_model_failures": int((coherence["estimability_status"] != "estimable").sum()),
        "gc_model_failures": int((gc_results["estimability_status"] != "estimable").sum()),
        "optional_brs_eligible": bool(optional_brs["overall_eligible_for_SI"].all()),
        "new_participant_experiment_performed": False,
    }
    json_dump(RESULTS_DIR / "revision_analysis_completion_summary.json", summary)
    report = [
        "# Revision analysis completion report",
        "",
        f"- Baseline reproduction: {'PASS' if summary['baseline_reproduction_passed'] else 'FAIL'}",
        f"- BCa validation: {'PASS' if summary['bca_validation_passed'] else 'FAIL'}",
        f"- Fourier magnitude preservation: {'PASS' if summary['phase_randomization_validation_passed'] else 'FAIL'}",
        f"- Synthetic coherence validation: {'PASS' if summary['synthetic_coherence_validation_passed'] else 'FAIL'}",
        f"- Full-factorial negative direction: {sensitivity_summary['negative_direction_rows']}/{sensitivity_summary['factorial_setting_direction_rows']} ({100*sensitivity_summary['negative_direction_fraction']:.1f}%)",
        f"- Full-factorial positive sign reversals: {sensitivity_summary['positive_sign_reversal_rows']}",
        f"- LOO mean-difference range: {loo_summary['mean_difference_min']:.4f} to {loo_summary['mean_difference_max']:.4f} ms/mmHg",
        f"- LOO dz range: {loo_summary['dz_min']:.4f} to {loo_summary['dz_max']:.4f}",
        f"- Coherence non-estimable records: {summary['coherence_model_failures']}",
        f"- GC non-estimable records: {summary['gc_model_failures']}",
        f"- Optional alternative BRS eligible: {summary['optional_brs_eligible']}",
        "- No new participant experiment was performed.",
    ]
    (REPORTS_DIR / "revision_analysis_completion_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
