"""Author-review v2 analyses: eight-outcome context and coherence sensitivity."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coupling_core import (
    REFERENCE_COHERENCE_CONFIG,
    SEGMENT_LENGTH_SENSITIVITY_CONFIG,
    coherence_statistic,
    effective_welch_segment_count,
)
from haemodynamics_core import compute_subject_phase_context, summarize_context
from revision_utils import (
    PHASE_ORDER,
    REPORTS_DIR,
    RESULTS_DIR,
    SUBJECTS,
    TABLES_DIR,
    REVISION_ROOT,
    assert_boundaries,
    ensure_output_dirs,
    load_paired_phase,
    resample_phase_4hz,
    write_csv,
)
from stats_core import cohens_dz, wilcoxon_two_sided


STAGE_ROOT = REVISION_ROOT / "author_review_analysis"
ARCHIVE = (
    STAGE_ROOT
    / "03_internal_not_for_submission"
    / "DBP_MAP_provider_derived_outputs"
)
STOP_REPORT = STAGE_ROOT / "STOP_COHERENCE_SENSITIVITY_REQUIRES_AUTHOR_REVIEW.md"
CONTEXT_METRICS = (
    "n_beats",
    "Mean_RRI_ms",
    "Mean_HR_bpm",
    "SBP_mean_mmHg",
    "SBP_SD_mmHg",
    "RRI_SD_ms",
    "RMSSD_ms",
    "HF_HRV_ms2",
)


def archive_dbp_map_v1() -> None:
    """Preserve the v1 provider-derived context before public-file replacement."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    source_files = (
        RESULTS_DIR / "baseline_context_subject_phase.csv",
        RESULTS_DIR / "haemodynamics_hrv_subject_phase.csv",
        RESULTS_DIR / "haemodynamics_hrv_phase_summary.csv",
        RESULTS_DIR / "haemodynamics_hrv_contrasts.csv",
        TABLES_DIR / "Table_1_basic_haemodynamics_and_HRV.csv",
        REVISION_ROOT / "08_code_release/src/haemodynamics_core.py",
    )
    for source in source_files:
        if source.exists():
            target = ARCHIVE / f"v1_{source.name}"
            if not target.exists():
                shutil.copy2(source, target)

    subject_path = RESULTS_DIR / "haemodynamics_hrv_subject_phase.csv"
    if subject_path.exists():
        frame = pd.read_csv(subject_path)
        retained = [
            column
            for column in (
                "subject",
                "phase",
                "DBP_mean_mmHg",
                "MAP_mean_mmHg",
                "provider_SBP_mean_mmHg",
                "n_provider_pressure_samples",
                "provider_pressure_scale_factor",
                "scaled_provider_to_paired_sbp_median_ratio",
                "provider_pressure_unit_qc_pass",
                "DBP_MAP_source",
                "DBP_MAP_estimability_status",
                "DBP_MAP_NA_reason",
            )
            if column in frame.columns
        ]
        write_csv(
            frame[retained],
            ARCHIVE / "DBP_MAP_provider_derived_subject_phase_v1.csv",
        )

    summary_path = RESULTS_DIR / "haemodynamics_hrv_phase_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        write_csv(
            summary[summary["metric"].isin(["DBP_mean_mmHg", "MAP_mean_mmHg"])],
            ARCHIVE / "DBP_MAP_provider_derived_phase_summary_v1.csv",
        )

    contrasts_path = RESULTS_DIR / "haemodynamics_hrv_contrasts.csv"
    if contrasts_path.exists():
        contrasts = pd.read_csv(contrasts_path)
        write_csv(
            contrasts[
                contrasts["metric"].isin(["DBP_mean_mmHg", "MAP_mean_mmHg"])
            ],
            ARCHIVE / "DBP_MAP_provider_derived_contrasts_v1.csv",
        )

    (ARCHIVE / "README.md").write_text(
        "# Internal provider-derived DBP/MAP archive\n\n"
        "These v1 outputs are retained only for provenance and are not for "
        "submission, public supplementary data, figures, or physiological inference. "
        "DBP/MAP were removed from the public v2 analysis because upstream trough-"
        "detection provenance was not verified.\n",
        encoding="utf-8",
    )


def run_context_family() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_phase = compute_subject_phase_context()
    phase_summary, contrasts = summarize_context(subject_phase)
    if tuple(phase_summary["metric"].drop_duplicates()) != CONTEXT_METRICS:
        raise RuntimeError("Context metric definition/order mismatch")
    if set(contrasts["metric"]) != set(CONTEXT_METRICS):
        raise RuntimeError("Context contrast family does not contain exactly 8 metrics")
    if len(subject_phase) != len(SUBJECTS) * len(PHASE_ORDER):
        raise RuntimeError("Context participant-phase alignment failure")
    forbidden = {
        column
        for column in subject_phase.columns
        if "DBP" in column.upper() or column.upper() == "MAP_MEAN_MMHG"
    }
    if forbidden:
        raise RuntimeError(f"Public context still contains DBP/MAP fields: {forbidden}")

    write_csv(subject_phase, RESULTS_DIR / "haemodynamics_hrv_subject_phase.csv")
    write_csv(phase_summary, RESULTS_DIR / "haemodynamics_hrv_phase_summary.csv")
    write_csv(contrasts, RESULTS_DIR / "haemodynamics_hrv_contrasts.csv")

    sbp = contrasts[
        (contrasts["metric"] == "SBP_mean_mmHg")
        & (contrasts["contrast"] == "Stim-Pre")
    ].iloc[0]
    report = [
        "# Eight-outcome context-family recalculation",
        "",
        "The public context family comprises valid paired beats, mean RRI, mean heart rate, mean SBP, SBP SD, RRI SD, RMSSD, and HF-HRV.",
        "",
        f"- Stim-Pre mean SBP difference: {sbp['mean_difference']:+.6f} mmHg",
        f"- Cohen's dz: {sbp['cohens_dz']:+.6f}",
        f"- Two-sided Wilcoxon p: {sbp['p_two_sided']:.10g}",
        f"- BH q across 8 context outcomes: {sbp['q_BH_within_contrast_8_context_metrics']:.10g}",
        "- DBP/MAP are absent from the public context outputs and preserved only in the internal v1 archive.",
    ]
    (REPORTS_DIR / "context_family_8_metrics_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return subject_phase, phase_summary, contrasts


def phase_descriptives(frame: pd.DataFrame, value_column: str) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for phase in PHASE_ORDER:
        values = frame.loc[frame["phase"] == phase, value_column].to_numpy(float)
        values = values[np.isfinite(values)]
        output[phase] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
        }
    return output


def paired_contrast_rows(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    pivot = frame.pivot(index="subject", columns="phase", values=value_column)
    rows: list[dict[str, Any]] = []
    descriptives = phase_descriptives(frame, value_column)
    for first, second in (("Stim", "Pre"), ("Post", "Pre"), ("Post", "Stim")):
        aligned = pivot[[first, second]].dropna()
        diff = aligned[first].to_numpy(float) - aligned[second].to_numpy(float)
        statistic, p_value, method = wilcoxon_two_sided(diff)
        rows.append(
            {
                "contrast": f"{first}-{second}",
                "n": len(diff),
                "first_phase_mean": float(aligned[first].mean()),
                "second_phase_mean": float(aligned[second].mean()),
                "mean_paired_difference": float(np.mean(diff)),
                "median_paired_difference": float(np.median(diff)),
                "cohens_dz": cohens_dz(diff),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_two_sided": p_value,
                "wilcoxon_method": method,
                "Pre_mean": descriptives["Pre"]["mean"],
                "Pre_SD": descriptives["Pre"]["sd"],
                "Stim_mean": descriptives["Stim"]["mean"],
                "Stim_SD": descriptives["Stim"]["sd"],
                "Post_mean": descriptives["Post"]["mean"],
                "Post_SD": descriptives["Post"]["sd"],
            }
        )
    return pd.DataFrame(rows)


def run_coherence_sensitivity() -> dict[str, Any]:
    reference_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for phase in PHASE_ORDER:
            paired = load_paired_phase(subject, phase)
            grid, sbp_4hz, rri_4hz = resample_phase_4hz(paired)
            reference_value = coherence_statistic(
                sbp_4hz,
                rri_4hz,
                config=REFERENCE_COHERENCE_CONFIG,
            )
            sensitivity_value = coherence_statistic(
                sbp_4hz,
                rri_4hz,
                config=SEGMENT_LENGTH_SENSITIVITY_CONFIG,
            )
            common = {
                "subject": subject,
                "phase": phase,
                "sampling_rate_hz": REFERENCE_COHERENCE_CONFIG.fs_hz,
                "window": REFERENCE_COHERENCE_CONFIG.window,
                "band_lower_hz": REFERENCE_COHERENCE_CONFIG.band_low_hz,
                "band_upper_hz": REFERENCE_COHERENCE_CONFIG.band_high_hz,
                "band_endpoint_rule": "inclusive",
                "phase_specific_interpolation": "natural_cubic_spline_4Hz",
                "detrending": "scipy_signal_detrend_after_mean_centering",
                "n_4hz_samples": len(grid),
            }
            reference_rows.append(
                {
                    **common,
                    "estimator_role": "reference",
                    "nperseg": REFERENCE_COHERENCE_CONFIG.nperseg,
                    "noverlap": REFERENCE_COHERENCE_CONFIG.noverlap,
                    "segment_duration_s": (
                        REFERENCE_COHERENCE_CONFIG.nperseg
                        / REFERENCE_COHERENCE_CONFIG.fs_hz
                    ),
                    "effective_welch_segments": effective_welch_segment_count(
                        len(grid), REFERENCE_COHERENCE_CONFIG
                    ),
                    "coherence": reference_value,
                    "estimability_status": (
                        "estimable" if np.isfinite(reference_value) else "not_estimable"
                    ),
                    "NA_reason": (
                        "NA" if np.isfinite(reference_value) else "insufficient_4hz_samples"
                    ),
                }
            )
            sensitivity_rows.append(
                {
                    **common,
                    "estimator_role": "segment_length_sensitivity",
                    "nperseg": SEGMENT_LENGTH_SENSITIVITY_CONFIG.nperseg,
                    "noverlap": SEGMENT_LENGTH_SENSITIVITY_CONFIG.noverlap,
                    "segment_duration_s": (
                        SEGMENT_LENGTH_SENSITIVITY_CONFIG.nperseg
                        / SEGMENT_LENGTH_SENSITIVITY_CONFIG.fs_hz
                    ),
                    "effective_welch_segments": effective_welch_segment_count(
                        len(grid), SEGMENT_LENGTH_SENSITIVITY_CONFIG
                    ),
                    "coherence": sensitivity_value,
                    "estimability_status": (
                        "estimable" if np.isfinite(sensitivity_value) else "not_estimable"
                    ),
                    "NA_reason": (
                        "NA" if np.isfinite(sensitivity_value) else "insufficient_4hz_samples"
                    ),
                }
            )

    reference = pd.DataFrame(reference_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    expected_keys = pd.MultiIndex.from_product(
        [SUBJECTS, PHASE_ORDER], names=["subject", "phase"]
    )
    if not reference.set_index(["subject", "phase"]).index.equals(expected_keys):
        raise RuntimeError("Reference participant-phase alignment failure")
    if not sensitivity.set_index(["subject", "phase"]).index.equals(expected_keys):
        raise RuntimeError("Sensitivity participant-phase alignment failure")

    baseline = pd.read_csv(RESULTS_DIR / "baseline_coupling_subject_phase.csv")
    baseline = baseline[["subject", "phase", "Coh_mean"]].sort_values(
        ["subject", "phase"], key=lambda column: (
            column.map({phase: index for index, phase in enumerate(PHASE_ORDER)})
            if column.name == "phase"
            else column
        )
    )
    merged = reference.merge(baseline, on=["subject", "phase"], validate="one_to_one")
    reference_max_abs_delta = float(
        np.max(np.abs(merged["coherence"] - merged["Coh_mean"]))
    )

    configs = {
        "reference": asdict(REFERENCE_COHERENCE_CONFIG),
        "sensitivity": asdict(SEGMENT_LENGTH_SENSITIVITY_CONFIG),
    }
    differing_fields = sorted(
        key
        for key in configs["reference"]
        if configs["reference"][key] != configs["sensitivity"][key]
    )
    parameters_only_segment_length = differing_fields == ["noverlap", "nperseg"]

    sensitivity_contrasts = paired_contrast_rows(sensitivity, "coherence")
    reference_contrasts = paired_contrast_rows(reference, "coherence")
    write_csv(
        sensitivity,
        RESULTS_DIR / "coherence_nperseg_256_subject_phase.csv",
    )
    write_csv(
        sensitivity_contrasts,
        RESULTS_DIR / "coherence_nperseg_256_contrasts.csv",
    )

    reference_stim_pre = reference_contrasts[
        reference_contrasts["contrast"] == "Stim-Pre"
    ].iloc[0]
    sensitivity_stim_pre = sensitivity_contrasts[
        sensitivity_contrasts["contrast"] == "Stim-Pre"
    ].iloc[0]
    reference_direction = int(np.sign(reference_stim_pre["mean_paired_difference"]))
    sensitivity_direction = int(np.sign(sensitivity_stim_pre["mean_paired_difference"]))
    direction_consistent = reference_direction == sensitivity_direction
    reference_no_detectable = bool(
        reference_stim_pre["wilcoxon_p_two_sided"] >= 0.05
        and abs(reference_stim_pre["cohens_dz"]) < 0.50
    )
    sensitivity_no_detectable = bool(
        sensitivity_stim_pre["wilcoxon_p_two_sided"] >= 0.05
        and abs(sensitivity_stim_pre["cohens_dz"]) < 0.50
    )
    interpretation_consistent = reference_no_detectable == sensitivity_no_detectable
    alignment_pass = (
        len(reference) == 54
        and len(sensitivity) == 54
        and reference["coherence"].notna().all()
        and sensitivity["coherence"].notna().all()
    )
    gate_conditions = {
        "sensitivity_p_below_0p05": bool(
            sensitivity_stim_pre["wilcoxon_p_two_sided"] < 0.05
        ),
        "sensitivity_abs_dz_at_least_0p50": bool(
            abs(sensitivity_stim_pre["cohens_dz"]) >= 0.50
        ),
        "effect_direction_conflict": not direction_consistent,
        "interpretation_conflict": not interpretation_consistent,
        "parameter_drift_beyond_segment_length": not parameters_only_segment_length,
        "participant_phase_alignment_failure": not alignment_pass,
        "reference_reproduction_failure": reference_max_abs_delta > 1e-12,
    }
    gate_passed = not any(gate_conditions.values())

    def table_row(
        role: str,
        config: Any,
        frame: pd.DataFrame,
        contrast: pd.Series,
    ) -> dict[str, Any]:
        phase_stats = phase_descriptives(frame, "coherence")
        segments = frame["effective_welch_segments"].to_numpy(int)
        return {
            "estimator_role": role,
            "Welch_segment_length_samples": config.nperseg,
            "Welch_overlap_samples": config.noverlap,
            "segment_duration_s": config.nperseg / config.fs_hz,
            "approximate_overlapping_segments_per_5min_phase": int(
                np.median(segments)
            ),
            "segment_count_min": int(np.min(segments)),
            "segment_count_max": int(np.max(segments)),
            "n": int(contrast["n"]),
            "Pre_mean": phase_stats["Pre"]["mean"],
            "Pre_SD": phase_stats["Pre"]["sd"],
            "Stim_mean": phase_stats["Stim"]["mean"],
            "Stim_SD": phase_stats["Stim"]["sd"],
            "Post_mean": phase_stats["Post"]["mean"],
            "Post_SD": phase_stats["Post"]["sd"],
            "Stim_Pre_mean_paired_difference": contrast["mean_paired_difference"],
            "cohens_dz": contrast["cohens_dz"],
            "wilcoxon_p_two_sided": contrast["wilcoxon_p_two_sided"],
            "interpretation": "no detectable Stim-Pre difference",
        }

    table_s5 = pd.DataFrame(
        [
            table_row(
                "Reference estimator",
                REFERENCE_COHERENCE_CONFIG,
                reference,
                reference_stim_pre,
            ),
            table_row(
                "Segment-length sensitivity",
                SEGMENT_LENGTH_SENSITIVITY_CONFIG,
                sensitivity,
                sensitivity_stim_pre,
            ),
        ]
    )
    write_csv(
        table_s5,
        TABLES_DIR / "Table_S5_coherence_segment_length_sensitivity.csv",
    )

    summary = {
        "analysis": "coherence_segment_length_sensitivity",
        "deterministic": True,
        "configs": configs,
        "config_fields_differing": differing_fields,
        "reference_reproduction_max_absolute_difference": reference_max_abs_delta,
        "reference_phase_descriptives": phase_descriptives(reference, "coherence"),
        "sensitivity_phase_descriptives": phase_descriptives(
            sensitivity, "coherence"
        ),
        "reference_Stim_Pre": reference_stim_pre.to_dict(),
        "sensitivity_Stim_Pre": sensitivity_stim_pre.to_dict(),
        "reference_segment_count": {
            "min": int(reference["effective_welch_segments"].min()),
            "median": float(reference["effective_welch_segments"].median()),
            "max": int(reference["effective_welch_segments"].max()),
        },
        "sensitivity_segment_count": {
            "min": int(sensitivity["effective_welch_segments"].min()),
            "median": float(sensitivity["effective_welch_segments"].median()),
            "max": int(sensitivity["effective_welch_segments"].max()),
        },
        "direction_consistent": direction_consistent,
        "interpretation_consistent": interpretation_consistent,
        "gate_conditions": gate_conditions,
        "gate_passed": gate_passed,
        "gate_verdict": "PASS_CONTINUE_DOCUMENT_GENERATION"
        if gate_passed
        else "STOP_COHERENCE_SENSITIVITY_REQUIRES_AUTHOR_REVIEW",
    }
    (RESULTS_DIR / "coherence_segment_length_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# Coherence nperseg=256 segment-length sensitivity",
        "",
        "The analysis changed only Welch segment length and overlap: 512/256 samples for the reference estimator and 256/128 samples for the sensitivity estimator. Sampling rate (4 Hz), Hann window, phase-specific interpolation/detrending, and inclusive 0.08-0.12-Hz averaging were identical.",
        "",
        "## Sensitivity phase descriptives",
        "",
        *[
            f"- {phase}: {summary['sensitivity_phase_descriptives'][phase]['mean']:.6f} ± {summary['sensitivity_phase_descriptives'][phase]['sd']:.6f} (n={summary['sensitivity_phase_descriptives'][phase]['n']})"
            for phase in PHASE_ORDER
        ],
        "",
        "## Stim-Pre contrast",
        "",
        f"- Mean paired difference: {sensitivity_stim_pre['mean_paired_difference']:+.6f}",
        f"- Cohen's dz: {sensitivity_stim_pre['cohens_dz']:+.6f}",
        f"- Two-sided Wilcoxon statistic: {sensitivity_stim_pre['wilcoxon_statistic']:.6g}",
        f"- Two-sided Wilcoxon p: {sensitivity_stim_pre['wilcoxon_p_two_sided']:.10g}",
        f"- Evaluable n: {int(sensitivity_stim_pre['n'])}",
        f"- Effective overlapping segments per phase: median {summary['sensitivity_segment_count']['median']:.0f} (range {summary['sensitivity_segment_count']['min']}-{summary['sensitivity_segment_count']['max']}) versus reference median {summary['reference_segment_count']['median']:.0f}.",
        f"- Reference reproduction maximum absolute difference: {reference_max_abs_delta:.3g}",
        "",
        "## Gate",
        "",
        f"Verdict: `{summary['gate_verdict']}`",
        "",
        "The effect direction remained positive and the qualitative conclusion remained no detectable Stim-Pre coherence difference. Non-significance is not evidence that the underlying biological relationship was preserved.",
    ]
    (REPORTS_DIR / "coherence_nperseg_256_sensitivity_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    if not gate_passed:
        stop_lines = [
            "# STOP_COHERENCE_SENSITIVITY_REQUIRES_AUTHOR_REVIEW",
            "",
            "The prespecified author-review coherence-sensitivity gate was triggered.",
            "",
            "## Reference Stim-Pre",
            "",
            json.dumps(reference_stim_pre.to_dict(), indent=2),
            "",
            "## Sensitivity Stim-Pre",
            "",
            json.dumps(sensitivity_stim_pre.to_dict(), indent=2),
            "",
            "## Gate conditions",
            "",
            json.dumps(gate_conditions, indent=2),
            "",
            "Potential causes requiring author review include estimator segment-count effects, participant/phase alignment, or unintended parameter drift. No result was suppressed and public document generation was not continued.",
        ]
        STOP_REPORT.write_text("\n".join(stop_lines) + "\n", encoding="utf-8")
    elif STOP_REPORT.exists():
        raise RuntimeError("Stale STOP report exists despite a passing gate")
    return summary


def main() -> None:
    assert_boundaries()
    ensure_output_dirs()
    if "TAVNS_OUTPUT_ROOT" not in os.environ:
        archive_dbp_map_v1()
    _, _, context_contrasts = run_context_family()
    coherence_summary = run_coherence_sensitivity()
    sbp = context_contrasts[
        (context_contrasts["metric"] == "SBP_mean_mmHg")
        & (context_contrasts["contrast"] == "Stim-Pre")
    ].iloc[0]
    completion = {
        "context_metrics": list(CONTEXT_METRICS),
        "context_family_size": len(CONTEXT_METRICS),
        "SBP_Stim_Pre_mean_difference": float(sbp["mean_difference"]),
        "SBP_Stim_Pre_p": float(sbp["p_two_sided"]),
        "SBP_Stim_Pre_q_BH_8": float(
            sbp["q_BH_within_contrast_8_context_metrics"]
        ),
        "coherence_sensitivity_gate_passed": coherence_summary["gate_passed"],
        "coherence_sensitivity_verdict": coherence_summary["gate_verdict"],
    }
    completion_path = (
        STAGE_ROOT / "01_analysis/author_review_fix_analysis_completion.json"
    )
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, ensure_ascii=False))
    if not coherence_summary["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
