"""Run the locked R1 Antonino harmonization analyses A-C."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from core import (
    ANTONINO_BCA_BASE,
    ASSOCIATION_BOOTSTRAP_SEED,
    ASSOCIATION_PERMUTATION_SEED,
    BOOTSTRAP_RESAMPLES,
    BRANCHES,
    FIGURES_DIR,
    IMPLEMENTATION_VERSION,
    LOGS_DIR,
    PACKAGE_ROOT,
    PERMUTATION_RESAMPLES,
    PHASES,
    PHASE_ORDER,
    REVISION_ROOT,
    SUBJECTS,
    SUBPHASES,
    SUBPHASE_BCA_BASE,
    SUBPHASE_ORDER,
    TABLES_DIR,
    THEILSEN_BOOTSTRAP_SEED,
    bca_paired_correlation,
    bootstrap_theil_sen,
    clean_and_segment,
    cohens_dz,
    compute_context,
    describe_values,
    ensure_output_dirs,
    environment_versions,
    evaluate_branch,
    load_subject,
    pearson_fisher_ci,
    permutation_spearman,
    sha256_file,
    stable_seed,
    summarize_paired_difference,
    wilcoxon_two_sided,
)


EXACTNESS = "PARTIAL_MATCH_DUE_TO_UNRESOLVED_DEFAULTS"
PALETTE = {
    "Pre_late": "#777777",
    "Stim_early": "#0072B2",
    "Stim_late": "#D55E00",
    "REF": "#222222",
    "A0_MAX": "#0072B2",
    "A1_MAX": "#D55E00",
    "early": "#0072B2",
    "late": "#D55E00",
}


def save_csv(frame: pd.DataFrame, filename: str) -> Path:
    """Write an unrounded machine-readable CSV to the package."""
    path = TABLES_DIR / filename
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
    return path


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
) -> tuple[int, float, float]:
    """Return complete-pair n, estimate, and conventional p value."""
    x_data = np.asarray(x, dtype=float)
    y_data = np.asarray(y, dtype=float)
    valid = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid]
    y_data = y_data[valid]
    if len(x_data) < 3 or np.std(x_data) == 0 or np.std(y_data) == 0:
        return len(x_data), np.nan, np.nan
    result = (
        stats.spearmanr(x_data, y_data)
        if kind == "spearman"
        else stats.pearsonr(x_data, y_data)
    )
    return len(x_data), float(result.statistic), float(result.pvalue)


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save editable-text SVG and 300-dpi PNG."""
    fig.savefig(
        FIGURES_DIR / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Creator": f"R1 analysis {IMPLEMENTATION_VERSION}"},
    )
    fig.savefig(
        FIGURES_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def load_all_subjects() -> dict[int, pd.DataFrame]:
    return {subject: load_subject(subject) for subject in SUBJECTS}


def compute_windows(
    all_subjects: dict[int, pd.DataFrame],
    windows: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute context, branch summaries, and candidate sequences."""
    context_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        native = all_subjects[subject]
        for window, (start_s, end_s) in windows.items():
            frame = clean_and_segment(native, start_s, end_s)
            context_rows.append(
                {
                    "subject": subject,
                    "window": window,
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    **compute_context(frame),
                }
            )
            for branch, config in BRANCHES.items():
                summaries, sequences = evaluate_branch(
                    frame,
                    subject,
                    window,
                    start_s,
                    end_s,
                    config,
                )
                summary_rows.extend(summaries)
                sequence_rows.extend(sequences)
    context = pd.DataFrame(context_rows)
    summaries = pd.DataFrame(summary_rows).merge(
        context,
        on=["subject", "window", "window_start_s", "window_end_s"],
        how="left",
        validate="many_to_one",
    )
    return context, summaries, pd.DataFrame(sequence_rows)


def reference_reproduction_gate(
    all_subjects: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    """Run the mandatory computational gate before Analyses A-C."""
    canonical_path = (
        REVISION_ROOT
        / "02_analysis"
        / "results"
        / "canonical_brs_subject_phase.csv"
    )
    canonical = pd.read_csv(canonical_path)
    rows: list[dict[str, Any]] = []
    sbp_rows: list[dict[str, Any]] = []
    current_default_rows: list[dict[str, Any]] = []
    current_default = replace(
        BRANCHES["REF"],
        branch="CURRENT_DEFAULT_RRI_MONOTONIC",
        method_family="CURRENT_CODE_DEFAULT_DIAGNOSTIC_ONLY",
        rri_direction_required=True,
        rri_step_threshold=0.0,
        rri_threshold_strict=True,
    )
    for subject in SUBJECTS:
        for phase, (start_s, end_s) in PHASES.items():
            frame = clean_and_segment(all_subjects[subject], start_s, end_s)
            summaries, _ = evaluate_branch(
                frame,
                subject,
                phase,
                start_s,
                end_s,
                BRANCHES["REF"],
            )
            rows.extend(summaries)
            current, _ = evaluate_branch(
                frame,
                subject,
                phase,
                start_s,
                end_s,
                current_default,
            )
            current_default_rows.extend(current)
            sbp_rows.append(
                {
                    "subject": subject,
                    "phase": phase,
                    "mean_SBP_mmHg": float(frame["SBP_mmHg"].mean()),
                    "n_beats": len(frame),
                }
            )

    calculated = pd.DataFrame(rows)
    calculated = calculated[calculated["direction"].isin(["all", "up", "down"])]
    metric_map = {"all": "BRS_seq_all", "up": "BRS_seq_up", "down": "BRS_seq_down"}
    comparisons: list[dict[str, Any]] = []
    for row in calculated.itertuples(index=False):
        target_row = canonical[
            (canonical["subject"] == row.subject)
            & (canonical["phase"] == row.window)
        ].iloc[0]
        target = float(target_row[metric_map[row.direction]])
        difference = float(row.gain_ms_per_mmHg - target)
        comparisons.append(
            {
                "subject": row.subject,
                "phase": row.window,
                "direction": row.direction,
                "calculated": row.gain_ms_per_mmHg,
                "canonical": target,
                "absolute_difference": abs(difference),
            }
        )
    comparison = pd.DataFrame(comparisons)
    ref_all = calculated[calculated["direction"] == "all"].pivot(
        index="subject", columns="window", values="gain_ms_per_mmHg"
    )
    sbp = pd.DataFrame(sbp_rows).pivot(
        index="subject", columns="phase", values="mean_SBP_mmHg"
    )
    current_all = pd.DataFrame(current_default_rows)
    current_all = current_all[current_all["direction"] == "all"].pivot(
        index="subject", columns="window", values="gain_ms_per_mmHg"
    )
    max_cell_difference = float(comparison["absolute_difference"].max())
    targets = {
        "reference_pre_mean": 8.536766245467902,
        "reference_stim_mean": 6.486058058059438,
        "reference_difference_mean": -2.0507081874084636,
        "reference_negative_count": 17,
        "sbp_pre_mean": 129.76384206165417,
        "sbp_stim_mean": 135.8812188321241,
        "sbp_difference_mean": 6.117376770469915,
    }
    observed = {
        "reference_pre_mean": float(ref_all["Pre"].mean()),
        "reference_stim_mean": float(ref_all["Stim"].mean()),
        "reference_difference_mean": float((ref_all["Stim"] - ref_all["Pre"]).mean()),
        "reference_negative_count": int(((ref_all["Stim"] - ref_all["Pre"]) < 0).sum()),
        "sbp_pre_mean": float(sbp["Pre"].mean()),
        "sbp_stim_mean": float(sbp["Stim"].mean()),
        "sbp_difference_mean": float((sbp["Stim"] - sbp["Pre"]).mean()),
    }
    numeric_pass = all(
        abs(observed[key] - target) <= 1e-9
        for key, target in targets.items()
        if key != "reference_negative_count"
    )
    count_pass = observed["reference_negative_count"] == targets["reference_negative_count"]
    ids_pass = set(ref_all.index) == set(SUBJECTS) and len(ref_all) == 18
    canonical_pass = max_cell_difference <= 1e-9
    current_diff = float((current_all["Stim"] - current_all["Pre"]).mean())
    current_default_fails = not math.isclose(
        current_diff,
        targets["reference_difference_mean"],
        abs_tol=1e-6,
    )
    passed = bool(
        numeric_pass
        and count_pass
        and ids_pass
        and canonical_pass
        and current_default_fails
    )
    gate = {
        "status": "PASS" if passed else "FAIL",
        "canonical_path": str(canonical_path),
        "canonical_sha256": sha256_file(canonical_path),
        "max_absolute_cell_difference": max_cell_difference,
        "tolerance": 1e-9,
        "participant_ids": list(map(int, ref_all.index)),
        "participant_phase_rows": int(len(ref_all) * 3),
        "observed": observed,
        "targets": targets,
        "current_default_rri_monotonic_difference_mean": current_diff,
        "current_default_demonstrably_not_submission_reference": current_default_fails,
        "phase_assignment_time": "R_wave_time_s",
        "phase_intervals": {key: list(value) for key, value in PHASES.items()},
        "pressure_conversion": "SBP_mmHg = SBP_stored_V * 100.0",
    }
    comparison.to_csv(
        LOGS_DIR / "reference_reproduction_cell_comparison.csv",
        index=False,
        float_format="%.17g",
    )
    (LOGS_DIR / "reference_reproduction_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not passed:
        failed = comparison.sort_values("absolute_difference", ascending=False).iloc[0]
        report = (
            "# BLOCKER: reference reproduction failed\n\n"
            f"First/worst divergent participant: {int(failed['subject'])}\n\n"
            f"Phase: {failed['phase']}\n\n"
            f"Direction: {failed['direction']}\n\n"
            f"Absolute difference: {failed['absolute_difference']:.17g}\n\n"
            "Downstream Analyses A-C were not run.\n"
        )
        (PACKAGE_ROOT / "BLOCKER_REPORT_REFERENCE_REPRODUCTION_FAILED.md").write_text(
            report,
            encoding="utf-8",
        )
        raise RuntimeError("REFERENCE_REPRODUCTION_FAILED")
    return gate


def analysis_a_contrasts(subject_phase: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        for direction in ("all", "up", "down"):
            group = subject_phase[
                (subject_phase["branch"] == branch)
                & (subject_phase["direction"] == direction)
            ]
            pivot = group.pivot(
                index="subject", columns="window", values="gain_ms_per_mmHg"
            )
            for phase in PHASE_ORDER:
                desc = describe_values(pivot[phase].to_numpy(float))
                rows.append(
                    {
                        "record_type": "phase_descriptive",
                        "branch": branch,
                        "direction": direction,
                        "phase": phase,
                        "omitted_subject": np.nan,
                        **desc,
                    }
                )
            complete = pivot.reindex(columns=["Pre", "Stim"]).dropna()
            diff = (complete["Stim"] - complete["Pre"]).to_numpy(float)
            seed = stable_seed(ANTONINO_BCA_BASE, branch, direction)
            summary = summarize_paired_difference(diff, seed)
            rows.append(
                {
                    "record_type": "stim_minus_pre",
                    "branch": branch,
                    "direction": direction,
                    "phase": "Stim-Pre",
                    "omitted_subject": np.nan,
                    "pre_mean": float(complete["Pre"].mean()) if len(complete) else np.nan,
                    "stim_mean": float(complete["Stim"].mean()) if len(complete) else np.nan,
                    **summary,
                }
            )
            for omitted in complete.index:
                loo = complete.drop(index=omitted)
                loo_diff = (loo["Stim"] - loo["Pre"]).to_numpy(float)
                statistic, p_value, method = wilcoxon_two_sided(loo_diff)
                mean_diff = float(np.mean(loo_diff)) if len(loo_diff) else np.nan
                rows.append(
                    {
                        "record_type": "leave_one_out",
                        "branch": branch,
                        "direction": direction,
                        "phase": "Stim-Pre",
                        "omitted_subject": int(omitted),
                        "paired_n": len(loo_diff),
                        "mean_difference": mean_diff,
                        "median_difference": float(np.median(loo_diff)),
                        "cohens_dz": cohens_dz(loo_diff),
                        "estimate_sign": (
                            "negative" if mean_diff < 0 else "positive" if mean_diff > 0 else "zero"
                        ),
                        "n_negative": int(np.sum(loo_diff < 0)),
                        "n_positive": int(np.sum(loo_diff > 0)),
                        "n_zero": int(np.sum(loo_diff == 0)),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_two_sided": p_value,
                        "wilcoxon_method": method,
                    }
                )
    return pd.DataFrame(rows)


def analysis_a_evaluability(subject_phase: pd.DataFrame) -> pd.DataFrame:
    subject_rows = subject_phase.copy()
    subject_rows.insert(0, "record_type", "subject_phase")
    aggregate_rows: list[dict[str, Any]] = []
    for (branch, direction, phase), group in subject_phase.groupby(
        ["branch", "direction", "window"], sort=False
    ):
        aggregate_rows.append(
            {
                "record_type": "aggregate_phase",
                "branch": branch,
                "direction": direction,
                "window": phase,
                "n_subjects": len(group),
                "n_evaluable": int(group["gain_ms_per_mmHg"].notna().sum()),
                "n_no_valid_sequence": int(group["no_valid_sequence_flag"].sum()),
                "evaluable_fraction": float(group["gain_ms_per_mmHg"].notna().mean()),
                "median_qualifying_sequences": float(group["n_qualifying_sequences"].median()),
                "minimum_qualifying_sequences": int(group["n_qualifying_sequences"].min()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    return pd.concat([subject_rows, aggregate], ignore_index=True, sort=False)


def direct_method_comparison(
    subject_phase: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_gain = subject_phase[subject_phase["direction"] == "all"]
    ref = all_gain[all_gain["branch"] == "REF"].pivot(
        index="subject", columns="window", values="gain_ms_per_mmHg"
    )
    rows: list[dict[str, Any]] = []
    ba_subject_rows: list[dict[str, Any]] = []
    for branch in ("A0_MAX", "A1_MAX", "A0_OVERLAP", "A1_OVERLAP"):
        method = all_gain[all_gain["branch"] == branch].pivot(
            index="subject", columns="window", values="gain_ms_per_mmHg"
        )
        for phase in PHASE_ORDER:
            joined = pd.concat(
                [ref[phase].rename("reference"), method[phase].rename("method")],
                axis=1,
            )
            for subject, values in joined.iterrows():
                rows.append(
                    {
                        "record_type": "subject_phase",
                        "branch": branch,
                        "subject": subject,
                        "phase": phase,
                        "reference_value": values["reference"],
                        "method_value": values["method"],
                        "absolute_difference": abs(values["method"] - values["reference"])
                        if np.isfinite(values["method"])
                        else np.nan,
                        "method_to_reference_ratio": values["method"] / values["reference"]
                        if np.isfinite(values["method"]) and values["reference"] != 0
                        else np.nan,
                    }
                )
            n_p, pearson, pearson_p = safe_correlation(
                joined["reference"].to_numpy(), joined["method"].to_numpy(), "pearson"
            )
            n_s, spearman, spearman_p = safe_correlation(
                joined["reference"].to_numpy(), joined["method"].to_numpy(), "spearman"
            )
            rows.append(
                {
                    "record_type": "aggregate_phase",
                    "branch": branch,
                    "phase": phase,
                    "n_complete": n_p,
                    "pearson_r": pearson,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman,
                    "spearman_p": spearman_p,
                    "n_reference_available": int(joined["reference"].notna().sum()),
                    "n_method_available": int(joined["method"].notna().sum()),
                }
            )

        changes = pd.concat(
            [
                (ref["Stim"] - ref["Pre"]).rename("reference_delta"),
                (method["Stim"] - method["Pre"]).rename("method_delta"),
            ],
            axis=1,
        )
        for subject, values in changes.iterrows():
            ref_delta = values["reference_delta"]
            method_delta = values["method_delta"]
            sign_same = (
                bool(np.sign(ref_delta) == np.sign(method_delta))
                if np.isfinite(method_delta)
                else np.nan
            )
            rows.append(
                {
                    "record_type": "subject_change",
                    "branch": branch,
                    "subject": subject,
                    "phase": "Stim-Pre",
                    "reference_delta": ref_delta,
                    "method_delta": method_delta,
                    "absolute_difference": abs(method_delta - ref_delta)
                    if np.isfinite(method_delta)
                    else np.nan,
                    "method_to_reference_ratio": method_delta / ref_delta
                    if np.isfinite(method_delta) and ref_delta != 0
                    else np.nan,
                    "sign_concordant": sign_same,
                    "sign_changed": (not sign_same) if isinstance(sign_same, bool) else np.nan,
                }
            )
            if np.isfinite(method_delta):
                ba_subject_rows.append(
                    {
                        "branch": branch,
                        "subject": subject,
                        "mean_of_delta_methods": (ref_delta + method_delta) / 2.0,
                        "method_minus_reference_delta": method_delta - ref_delta,
                    }
                )
        complete = changes.dropna()
        n_p, pearson, pearson_p = safe_correlation(
            complete["reference_delta"].to_numpy(),
            complete["method_delta"].to_numpy(),
            "pearson",
        )
        n_s, spearman, spearman_p = safe_correlation(
            complete["reference_delta"].to_numpy(),
            complete["method_delta"].to_numpy(),
            "spearman",
        )
        ba_diff = complete["method_delta"] - complete["reference_delta"]
        ba_mean = float(ba_diff.mean()) if len(ba_diff) else np.nan
        ba_sd = float(ba_diff.std(ddof=1)) if len(ba_diff) > 1 else np.nan
        rows.append(
            {
                "record_type": "aggregate_change",
                "branch": branch,
                "phase": "Stim-Pre",
                "n_complete": n_p,
                "pearson_r": pearson,
                "pearson_p": pearson_p,
                "spearman_rho": spearman,
                "spearman_p": spearman_p,
                "reference_mean_delta": float(complete["reference_delta"].mean())
                if len(complete)
                else np.nan,
                "method_mean_delta": float(complete["method_delta"].mean())
                if len(complete)
                else np.nan,
                "sign_concordance_fraction": float(
                    (np.sign(complete["reference_delta"]) == np.sign(complete["method_delta"])).mean()
                )
                if len(complete)
                else np.nan,
                "n_sign_changes": int(
                    (np.sign(complete["reference_delta"]) != np.sign(complete["method_delta"])).sum()
                ),
                "n_reference_available": int(changes["reference_delta"].notna().sum()),
                "n_method_available": int(changes["method_delta"].notna().sum()),
                "bland_altman_bias": ba_mean,
                "bland_altman_lower_loa": ba_mean - 1.96 * ba_sd,
                "bland_altman_upper_loa": ba_mean + 1.96 * ba_sd,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(ba_subject_rows)


def classify_antonino_direction(contrasts: pd.DataFrame) -> str:
    focal = contrasts[
        (contrasts["record_type"] == "stim_minus_pre")
        & (contrasts["direction"] == "all")
        & (contrasts["branch"].isin(["A0_MAX", "A1_MAX"]))
    ].set_index("branch")
    if len(focal) != 2 or bool((focal["paired_n"] < 15).any()):
        return "DIRECTION_INCONCLUSIVE_DUE_TO_EVALUABILITY"
    means = focal["mean_difference"]
    if means.loc["A0_MAX"] * means.loc["A1_MAX"] < 0:
        return "METHOD_BRANCHES_DISCORDANT"
    if bool((means > 0).all()):
        return "DIRECTION_REVERSED_POSITIVE"
    if bool((means < 0).all()):
        reference = contrasts[
            (contrasts["record_type"] == "stim_minus_pre")
            & (contrasts["direction"] == "all")
            & (contrasts["branch"] == "REF")
        ]["mean_difference"].iloc[0]
        ratio = float(abs(means.mean()) / abs(reference))
        if ratio < 0.50:
            return "DIRECTION_ATTENUATED_BUT_NEGATIVE"
        return "DIRECTION_PRESERVED_NEGATIVE"
    return "DIRECTION_INCONCLUSIVE_DUE_TO_EVALUABILITY"


def build_delta_subject_table(
    phase_context: pd.DataFrame,
    subject_phase: pd.DataFrame,
) -> pd.DataFrame:
    sbp = phase_context.pivot(index="subject", columns="window", values="mean_SBP_mmHg")
    rows = pd.DataFrame(index=SUBJECTS)
    rows.index.name = "subject"
    rows["Pre_SBP_mmHg"] = sbp["Pre"]
    rows["Stim_SBP_mmHg"] = sbp["Stim"]
    rows["Delta_SBP_mmHg"] = sbp["Stim"] - sbp["Pre"]
    all_gain = subject_phase[subject_phase["direction"] == "all"]
    for branch, label in (
        ("REF", "reference"),
        ("A0_MAX", "A0_MAX"),
        ("A1_MAX", "A1_MAX"),
    ):
        values = all_gain[all_gain["branch"] == branch].pivot(
            index="subject", columns="window", values="gain_ms_per_mmHg"
        )
        rows[f"Pre_BRS_{label}_ms_per_mmHg"] = values["Pre"]
        rows[f"Stim_BRS_{label}_ms_per_mmHg"] = values["Stim"]
        rows[f"Delta_BRS_{label}_ms_per_mmHg"] = values["Stim"] - values["Pre"]
        counts = all_gain[all_gain["branch"] == branch].pivot(
            index="subject", columns="window", values="n_qualifying_sequences"
        )
        rows[f"Pre_Nseq_{label}"] = counts["Pre"]
        rows[f"Stim_Nseq_{label}"] = counts["Stim"]
    return rows.reset_index()


def association_analysis(
    subject: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    x_name = "Delta_SBP_mmHg"
    method_columns = {
        "reference": "Delta_BRS_reference_ms_per_mmHg",
        "A0_MAX": "Delta_BRS_A0_MAX_ms_per_mmHg",
        "A1_MAX": "Delta_BRS_A1_MAX_ms_per_mmHg",
    }
    statistics_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    branch_rhos: dict[str, float] = {}
    branch_ns: dict[str, int] = {}

    for method, y_name in method_columns.items():
        complete = subject[["subject", x_name, y_name]].dropna()
        x = complete[x_name].to_numpy(float)
        y = complete[y_name].to_numpy(float)
        n, rho, rho_p = safe_correlation(x, y, "spearman")
        branch_rhos[method] = rho
        branch_ns[method] = n
        if method == "reference":
            rho_ci = bca_paired_correlation(
                x,
                y,
                "spearman",
                ASSOCIATION_BOOTSTRAP_SEED,
                BOOTSTRAP_RESAMPLES,
            )
            permutation = permutation_spearman(
                x,
                y,
                ASSOCIATION_PERMUTATION_SEED,
                PERMUTATION_RESAMPLES,
            )
        else:
            rho_ci = bca_paired_correlation(
                x,
                y,
                "spearman",
                stable_seed(ASSOCIATION_BOOTSTRAP_SEED, method),
                BOOTSTRAP_RESAMPLES,
            )
            permutation = {
                "p_two_sided": np.nan,
                "n_permutations": 0,
                "seed": np.nan,
            }
        statistics_rows.append(
            {
                "method": method,
                "analysis": "primary_spearman" if method == "reference" else "method_dependence_spearman",
                "n": n,
                "estimate": rho,
                "ci_low": rho_ci["ci_low"],
                "ci_high": rho_ci["ci_high"],
                "conventional_p_two_sided": rho_p,
                "permutation_p_two_sided": permutation["p_two_sided"],
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": rho_ci["seed"],
                "permutation_resamples": permutation["n_permutations"],
                "permutation_seed": permutation["seed"],
                "estimand_units": "rank correlation",
            }
        )
        r_value, r_low, r_high = pearson_fisher_ci(x, y)
        _, _, r_p = safe_correlation(x, y, "pearson")
        statistics_rows.append(
            {
                "method": method,
                "analysis": "secondary_pearson",
                "n": n,
                "estimate": r_value,
                "ci_low": r_low,
                "ci_high": r_high,
                "conventional_p_two_sided": r_p,
                "estimand_units": "linear correlation",
            }
        )

        for omitted in complete["subject"]:
            loo = complete[complete["subject"] != omitted]
            _, loo_rho, loo_rho_p = safe_correlation(
                loo[x_name].to_numpy(), loo[y_name].to_numpy(), "spearman"
            )
            _, loo_r, loo_r_p = safe_correlation(
                loo[x_name].to_numpy(), loo[y_name].to_numpy(), "pearson"
            )
            influence_rows.append(
                {
                    "record_type": "leave_one_out",
                    "method": method,
                    "subject": int(omitted),
                    "loo_n": len(loo),
                    "loo_spearman_rho": loo_rho,
                    "loo_spearman_p": loo_rho_p,
                    "loo_pearson_r": loo_r,
                    "loo_pearson_p": loo_r_p,
                }
            )

    ref_complete = subject[
        [
            "subject",
            x_name,
            "Delta_BRS_reference_ms_per_mmHg",
            "Pre_BRS_reference_ms_per_mmHg",
            "Stim_BRS_reference_ms_per_mmHg",
        ]
    ].dropna()
    x = ref_complete[x_name].to_numpy(float)
    y_change = ref_complete["Delta_BRS_reference_ms_per_mmHg"].to_numpy(float)
    theil = bootstrap_theil_sen(
        x,
        y_change,
        THEILSEN_BOOTSTRAP_SEED,
        BOOTSTRAP_RESAMPLES,
    )
    statistics_rows.append(
        {
            "method": "reference",
            "analysis": "secondary_theil_sen_slope",
            "n": len(ref_complete),
            "estimate": theil["estimate"],
            "ci_low": theil["ci_low"],
            "ci_high": theil["ci_high"],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": THEILSEN_BOOTSTRAP_SEED,
            "estimand_units": "(ms/mmHg BRS change) per mmHg SBP change",
        }
    )

    simple_x = sm.add_constant(ref_complete[[x_name]], has_constant="add")
    simple_fit = sm.OLS(y_change, simple_x).fit()
    simple_influence = simple_fit.get_influence()
    simple_dfbetas = simple_influence.dfbetas
    for index, row in ref_complete.reset_index(drop=True).iterrows():
        influence_rows.append(
            {
                "record_type": "simple_ols_influence",
                "method": "reference",
                "subject": int(row["subject"]),
                "leverage": float(simple_influence.hat_matrix_diag[index]),
                "cooks_distance": float(simple_influence.cooks_distance[0][index]),
                "dfbeta_intercept": float(simple_dfbetas[index, 0]),
                "dfbeta_delta_sbp": float(simple_dfbetas[index, 1]),
                "cooks_threshold_4_over_n": 4.0 / len(ref_complete),
            }
        )

    raw_predictors = ref_complete[
        ["Pre_BRS_reference_ms_per_mmHg", x_name]
    ]
    raw_x = sm.add_constant(raw_predictors, has_constant="add")
    raw_y = ref_complete["Stim_BRS_reference_ms_per_mmHg"]
    raw_base = sm.OLS(raw_y, raw_x).fit()
    raw_fit = sm.OLS(raw_y, raw_x).fit(cov_type="HC3")
    raw_ci = raw_fit.conf_int(alpha=0.05)
    for term in raw_x.columns:
        model_rows.append(
            {
                "record_type": "coefficient",
                "model": "baseline_adjusted_raw_HC3",
                "term": term,
                "n": int(raw_fit.nobs),
                "estimate": float(raw_fit.params[term]),
                "HC3_standard_error": float(raw_fit.bse[term]),
                "ci_low": float(raw_ci.loc[term, 0]),
                "ci_high": float(raw_ci.loc[term, 1]),
                "p_two_sided": float(raw_fit.pvalues[term]),
                "r_squared": float(raw_base.rsquared),
                "adjusted_r_squared": float(raw_base.rsquared_adj),
            }
        )

    standardized = ref_complete[
        [
            "Stim_BRS_reference_ms_per_mmHg",
            "Pre_BRS_reference_ms_per_mmHg",
            x_name,
        ]
    ].apply(lambda value: (value - value.mean()) / value.std(ddof=1))
    std_y = standardized["Stim_BRS_reference_ms_per_mmHg"]
    std_x = sm.add_constant(
        standardized[["Pre_BRS_reference_ms_per_mmHg", x_name]],
        has_constant="add",
    )
    std_base = sm.OLS(std_y, std_x).fit()
    std_fit = sm.OLS(std_y, std_x).fit(cov_type="HC3")
    std_ci = std_fit.conf_int(alpha=0.05)
    for term in std_x.columns:
        model_rows.append(
            {
                "record_type": "coefficient",
                "model": "baseline_adjusted_standardized_HC3",
                "term": term,
                "n": int(std_fit.nobs),
                "estimate": float(std_fit.params[term]),
                "HC3_standard_error": float(std_fit.bse[term]),
                "ci_low": float(std_ci.loc[term, 0]),
                "ci_high": float(std_ci.loc[term, 1]),
                "p_two_sided": float(std_fit.pvalues[term]),
                "r_squared": float(std_base.rsquared),
                "adjusted_r_squared": float(std_base.rsquared_adj),
            }
        )

    raw_influence = raw_base.get_influence()
    raw_dfbetas = raw_influence.dfbetas
    for index, row in ref_complete.reset_index(drop=True).iterrows():
        influence_rows.append(
            {
                "record_type": "baseline_adjusted_ols_influence",
                "method": "reference",
                "subject": int(row["subject"]),
                "leverage": float(raw_influence.hat_matrix_diag[index]),
                "cooks_distance": float(raw_influence.cooks_distance[0][index]),
                "dfbeta_intercept": float(raw_dfbetas[index, 0]),
                "dfbeta_pre_brs": float(raw_dfbetas[index, 1]),
                "dfbeta_delta_sbp": float(raw_dfbetas[index, 2]),
                "cooks_threshold_4_over_n": 4.0 / len(ref_complete),
            }
        )

    residuals = raw_base.resid.to_numpy(float)
    shapiro = stats.shapiro(residuals)
    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, raw_x)
    diagnostic_values = {
        "shapiro_w": (float(shapiro.statistic), float(shapiro.pvalue)),
        "jarque_bera": (float(jb_stat), float(jb_p)),
        "residual_skew": (float(skew), np.nan),
        "residual_kurtosis": (float(kurtosis), np.nan),
        "breusch_pagan_lm": (float(bp_stat), float(bp_p)),
        "breusch_pagan_f": (float(bp_f), float(bp_f_p)),
        "durbin_watson_descriptive": (float(durbin_watson(residuals)), np.nan),
        "max_leverage": (float(np.max(raw_influence.hat_matrix_diag)), np.nan),
        "max_cooks_distance": (float(np.max(raw_influence.cooks_distance[0])), np.nan),
    }
    for term, (estimate, p_value) in diagnostic_values.items():
        model_rows.append(
            {
                "record_type": "diagnostic",
                "model": "baseline_adjusted_raw_HC3",
                "term": term,
                "n": len(ref_complete),
                "estimate": estimate,
                "p_two_sided": p_value,
            }
        )

    rank_frame = ref_complete[
        [x_name, "Stim_BRS_reference_ms_per_mmHg", "Pre_BRS_reference_ms_per_mmHg"]
    ].rank(method="average")
    control = sm.add_constant(rank_frame[["Pre_BRS_reference_ms_per_mmHg"]])
    x_resid = sm.OLS(rank_frame[x_name], control).fit().resid
    y_resid = sm.OLS(rank_frame["Stim_BRS_reference_ms_per_mmHg"], control).fit().resid
    partial = stats.spearmanr(x_resid, y_resid)
    statistics_rows.append(
        {
            "method": "reference",
            "analysis": "secondary_partial_spearman_rank_residualization",
            "n": len(ref_complete),
            "estimate": float(partial.statistic),
            "conventional_p_two_sided": float(partial.pvalue),
            "estimand_units": "rank-residual correlation controlling Pre BRS",
        }
    )

    influence = pd.DataFrame(influence_rows)
    loo_ref = influence[
        (influence["record_type"] == "leave_one_out")
        & (influence["method"] == "reference")
    ]
    no_loo_reversal = bool((loo_ref["loo_spearman_rho"] <= 0).all())
    adequate_branches = [
        method
        for method in ("A0_MAX", "A1_MAX")
        if branch_ns[method] >= 15
    ]
    branch_compatible = all(branch_rhos[method] <= 0 for method in adequate_branches)
    beta2 = float(raw_fit.params[x_name])
    primary_negative = bool(branch_rhos["reference"] < 0)
    if primary_negative and no_loo_reversal and branch_compatible and beta2 <= 0:
        classification = "CONSISTENT_EXPLORATORY_ASSOCIATION"
    elif primary_negative:
        classification = "SUGGESTIVE_BUT_INFLUENCE_SENSITIVE"
    elif np.isfinite(branch_rhos["reference"]) and branch_rhos["reference"] >= 0:
        classification = "NO_EVIDENCE_OF_MONOTONIC_ASSOCIATION"
    else:
        classification = "INCONCLUSIVE"
    return (
        pd.DataFrame(statistics_rows),
        influence,
        pd.DataFrame(model_rows),
        classification,
    )


def build_subphase_wide(
    context: pd.DataFrame,
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        for window in SUBPHASE_ORDER:
            base = context[
                (context["subject"] == subject) & (context["window"] == window)
            ].iloc[0].to_dict()
            group = summaries[
                (summaries["subject"] == subject)
                & (summaries["window"] == window)
            ]
            for branch in BRANCHES:
                for direction in ("all", "up", "down"):
                    item = group[
                        (group["branch"] == branch)
                        & (group["direction"] == direction)
                    ].iloc[0]
                    prefix = f"{branch}_{direction}"
                    base[f"gain_{prefix}_ms_per_mmHg"] = item["gain_ms_per_mmHg"]
                    base[f"n_candidate_{prefix}"] = item["n_candidate_sbp_ramps"]
                    base[f"n_qualifying_{prefix}"] = item["n_qualifying_sequences"]
                    base[f"no_valid_{prefix}"] = int(item["no_valid_sequence_flag"])
                    if direction == "all":
                        base[f"n_unique_maximal_ramps_{branch}"] = item[
                            "n_unique_maximal_sbp_ramps"
                        ]
                        base[f"BEI_{branch}"] = item["BEI"]
                        base[f"mean_sequence_length_{branch}"] = item[
                            "mean_sequence_length_beats"
                        ]
                        base[f"mean_sequence_r_squared_{branch}"] = item[
                            "mean_sequence_r_squared"
                        ]
                        base[f"mean_within_sequence_r_{branch}"] = item[
                            "mean_within_sequence_r"
                        ]
                        base[f"mean_sbp_ramp_amplitude_abs_{branch}_mmHg"] = item[
                            "mean_sbp_ramp_amplitude_abs_mmHg"
                        ]
                        base[f"mean_rri_response_abs_{branch}_ms"] = item[
                            "mean_rri_response_abs_ms"
                        ]
            rows.append(base)
    return pd.DataFrame(rows)


def subphase_evaluability(summaries: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "subject",
        "window",
        "branch",
        "direction",
        "n_valid_paired_beats",
        "n_aligned_pairs",
        "n_unique_maximal_sbp_ramps",
        "n_candidate_sbp_ramps",
        "n_qualifying_sequences",
        "gain_ms_per_mmHg",
        "no_valid_sequence_flag",
        "no_valid_sequence_reason",
        "lag_beats",
        "enumeration",
    ]
    subject_rows = summaries[columns].copy()
    subject_rows.insert(0, "record_type", "subject_subphase")
    aggregates: list[dict[str, Any]] = []
    for (window, branch, direction), group in summaries.groupby(
        ["window", "branch", "direction"], sort=False
    ):
        aggregates.append(
            {
                "record_type": "aggregate_subphase",
                "window": window,
                "branch": branch,
                "direction": direction,
                "n_subjects": len(group),
                "n_evaluable": int(group["gain_ms_per_mmHg"].notna().sum()),
                "evaluable_fraction": float(group["gain_ms_per_mmHg"].notna().mean()),
                "n_no_valid_sequence": int(group["no_valid_sequence_flag"].sum()),
                "median_n_qualifying_sequences": float(
                    group["n_qualifying_sequences"].median()
                ),
            }
        )
    return pd.concat(
        [subject_rows, pd.DataFrame(aggregates)], ignore_index=True, sort=False
    )


def subphase_contrast_analysis(
    wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [
        ("Pre_late-Pre_early", "Pre_late", "Pre_early"),
        ("Stim_early-Pre_late", "Stim_early", "Pre_late"),
        ("Stim_late-Stim_early", "Stim_late", "Stim_early"),
        ("Stim_late-Pre_late", "Stim_late", "Pre_late"),
        ("Post_early-Stim_late", "Post_early", "Stim_late"),
        ("Post_late-Post_early", "Post_late", "Post_early"),
    ]
    central = [
        "mean_SBP_mmHg",
        "mean_HR_bpm",
        "mean_RRI_ms",
        "RMSSD_ms",
        "valid_paired_beat_count",
    ]
    for branch in ("REF", "A0_MAX", "A1_MAX"):
        central.extend(
            [
                f"gain_{branch}_all_ms_per_mmHg",
                f"gain_{branch}_up_ms_per_mmHg",
                f"gain_{branch}_down_ms_per_mmHg",
                f"n_unique_maximal_ramps_{branch}",
                f"n_candidate_{branch}_all",
                f"n_qualifying_{branch}_all",
                f"BEI_{branch}",
                f"mean_sequence_length_{branch}",
                f"mean_sequence_r_squared_{branch}",
                f"mean_within_sequence_r_{branch}",
                f"mean_sbp_ramp_amplitude_abs_{branch}_mmHg",
                f"mean_rri_response_abs_{branch}_ms",
                f"no_valid_{branch}_all",
            ]
        )
    rows: list[dict[str, Any]] = []
    p_rows: list[dict[str, Any]] = []
    for contrast, lhs, rhs in pairs:
        left = wide[wide["window"] == lhs].set_index("subject")
        right = wide[wide["window"] == rhs].set_index("subject")
        for outcome in central:
            joined = pd.concat(
                [left[outcome].rename("left"), right[outcome].rename("right")],
                axis=1,
            ).dropna()
            diff = (joined["left"] - joined["right"]).to_numpy(float)
            seed = stable_seed(SUBPHASE_BCA_BASE, contrast, outcome)
            summary = summarize_paired_difference(diff, seed)
            p_rows.append(
                {
                    "contrast": contrast,
                    "left_window": lhs,
                    "right_window": rhs,
                    "outcome": outcome,
                    "paired_n": summary["paired_n"],
                    "wilcoxon_statistic": summary["wilcoxon_statistic"],
                    "wilcoxon_p_two_sided_unadjusted_post_hoc_descriptor": summary[
                        "wilcoxon_p_two_sided"
                    ],
                    "wilcoxon_method": summary["wilcoxon_method"],
                    "interpretive_role": "machine-readable descriptor only; not used for classification",
                }
            )
            for key in (
                "wilcoxon_statistic",
                "wilcoxon_p_two_sided",
                "wilcoxon_method",
            ):
                summary.pop(key, None)
            rows.append(
                {
                    "contrast": contrast,
                    "left_window": lhs,
                    "right_window": rhs,
                    "outcome": outcome,
                    **summary,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(p_rows)


def build_response_state(contrasts: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Delta SBP": "mean_SBP_mmHg",
        "Delta HR": "mean_HR_bpm",
        "Delta BRS reference": "gain_REF_all_ms_per_mmHg",
        "Delta BRS A0_MAX": "gain_A0_MAX_all_ms_per_mmHg",
        "Delta BRS A1_MAX": "gain_A1_MAX_all_ms_per_mmHg",
        "Delta RMSSD": "RMSSD_ms",
    }
    contrast_labels = {
        "Stim_early-Pre_late": "Stim_early - Pre_late",
        "Stim_late-Pre_late": "Stim_late - Pre_late",
    }
    rows: list[dict[str, Any]] = []
    for label, outcome in mapping.items():
        for key, display in contrast_labels.items():
            row = contrasts[
                (contrasts["contrast"] == key)
                & (contrasts["outcome"] == outcome)
            ].iloc[0]
            rows.append(
                {
                    "response_dimension": label,
                    "source_outcome": outcome,
                    "contrast": display,
                    "paired_n": row["paired_n"],
                    "mean_difference": row["mean_difference"],
                    "median_difference": row["median_difference"],
                    "mean_difference_ci_low": row["mean_difference_ci_low"],
                    "mean_difference_ci_high": row["mean_difference_ci_high"],
                    "cohens_dz": row["cohens_dz"],
                    "n_negative": row["n_negative"],
                    "n_positive": row["n_positive"],
                    "n_zero": row["n_zero"],
                }
            )
    return pd.DataFrame(rows)


def classify_temporal_pattern(response: pd.DataFrame) -> str:
    pivot = response.pivot(
        index="response_dimension", columns="contrast", values="mean_difference"
    )
    n_pivot = response.pivot(
        index="response_dimension", columns="contrast", values="paired_n"
    )
    brs_methods = ["Delta BRS reference", "Delta BRS A0_MAX", "Delta BRS A1_MAX"]
    if bool((n_pivot.loc[brs_methods].to_numpy() < 15).any()):
        return "NONESTIMABLE_DUE_TO_SEQUENCE_SCARCITY"
    early_col = "Stim_early - Pre_late"
    late_col = "Stim_late - Pre_late"
    early_brs = pivot.loc[brs_methods, early_col]
    late_brs = pivot.loc[brs_methods, late_col]
    if len(set(np.sign(early_brs))) > 1 or len(set(np.sign(late_brs))) > 1:
        return "METHOD_DEPENDENT_TEMPORAL_PATTERN"
    early_pressor = pivot.loc["Delta SBP", early_col] > 0
    late_pressor = pivot.loc["Delta SBP", late_col] > 0
    early_lower = bool((early_brs < 0).all())
    late_lower = bool((late_brs < 0).all())
    if early_pressor and early_lower and late_pressor and late_lower:
        early_mag = float(np.mean(np.abs(early_brs)))
        late_mag = float(np.mean(np.abs(late_brs)))
        if late_mag >= 0.67 * early_mag:
            return "EARLY_PRESSOR_LOWER_BRS_PERSISTS_LATE"
        return "EARLY_PRESSOR_LOWER_BRS_ATTENUATES_LATE"
    if bool((late_brs > 0).all()) and pivot.loc["Delta HR", late_col] < 0:
        return "LATE_SHIFT_TOWARD_HIGHER_BRS_LOWER_HR"
    return "NO_CLEAR_EARLY_LATE_PATTERN"


def compute_30s_trajectory(
    all_subjects: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    subject_rows: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        native = all_subjects[subject]
        for start_s in np.arange(0.0, 900.0, 30.0):
            end_s = start_s + 30.0
            frame = clean_and_segment(native, start_s, end_s)
            context = compute_context(frame)
            subject_rows.append(
                {
                    "record_type": "subject_bin",
                    "subject": subject,
                    "bin_start_s": start_s,
                    "bin_end_s": end_s,
                    "bin_center_s": start_s + 15.0,
                    "mean_SBP_mmHg": context["mean_SBP_mmHg"],
                    "mean_HR_bpm": context["mean_HR_bpm"],
                    "n_beats": context["valid_paired_beat_count"],
                }
            )
    subject_frame = pd.DataFrame(subject_rows)
    group_rows: list[dict[str, Any]] = []
    for (start_s, end_s, center_s), group in subject_frame.groupby(
        ["bin_start_s", "bin_end_s", "bin_center_s"], sort=True
    ):
        row: dict[str, Any] = {
            "record_type": "group_bin",
            "bin_start_s": start_s,
            "bin_end_s": end_s,
            "bin_center_s": center_s,
        }
        for outcome in ("mean_SBP_mmHg", "mean_HR_bpm"):
            values = group[outcome].dropna().to_numpy(float)
            mean_value = float(np.mean(values))
            se = float(stats.sem(values))
            critical = float(stats.t.ppf(0.975, len(values) - 1))
            row[f"{outcome}_n"] = len(values)
            row[outcome] = mean_value
            row[f"{outcome}_ci_low"] = mean_value - critical * se
            row[f"{outcome}_ci_high"] = mean_value + critical * se
        group_rows.append(row)
    return pd.concat(
        [subject_frame, pd.DataFrame(group_rows)], ignore_index=True, sort=False
    )


def create_figures(
    delta: pd.DataFrame,
    association_stats: pd.DataFrame,
    ba_subject: pd.DataFrame,
    subphase_wide: pd.DataFrame,
    response: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    x = delta["Delta_SBP_mmHg"].to_numpy(float)
    y = delta["Delta_BRS_reference_ms_per_mmHg"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.3, 4.3))
    ax.axhline(0, color="#BBBBBB", lw=0.8)
    ax.axvline(0, color="#BBBBBB", lw=0.8)
    ax.scatter(x, y, color=PALETTE["REF"], s=32, zorder=3)
    slope = stats.theilslopes(y, x).slope
    intercept = float(np.median(y - slope * x))
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(grid, intercept + slope * grid, color="#666666", lw=1.4, label="Theil-Sen guide")
    label_offsets = {
        2: (3, -11),
        5: (3, -11),
        6: (3, 6),
        10: (3, 6),
        11: (3, 6),
        13: (3, -11),
    }
    for row in delta.itertuples(index=False):
        offset = label_offsets.get(int(row.subject), (3, 3))
        ax.annotate(
            f"S{int(row.subject):02d}",
            (row.Delta_SBP_mmHg, row.Delta_BRS_reference_ms_per_mmHg),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
        )
    rho_row = association_stats[
        (association_stats["method"] == "reference")
        & (association_stats["analysis"] == "primary_spearman")
    ].iloc[0]
    ax.set_title(
        "Post hoc exploratory association\n"
        f"Spearman rho = {rho_row['estimate']:.2f}; "
        f"95% BCa CI [{rho_row['ci_low']:.2f}, {rho_row['ci_high']:.2f}]",
        pad=10,
    )
    ax.set_xlabel("Stim-Pre mean SBP (mmHg)")
    ax.set_ylabel("Stim-Pre reference BRS (ms/mmHg)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "Fig_delta_SBP_vs_delta_BRS_reference")

    method_info = [
        ("reference", "Delta_BRS_reference_ms_per_mmHg", "Reference"),
        ("A0_MAX", "Delta_BRS_A0_MAX_ms_per_mmHg", "Antonino A0_MAX"),
        ("A1_MAX", "Delta_BRS_A1_MAX_ms_per_mmHg", "Antonino A1_MAX"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharex=True, sharey=True)
    for ax, (method, column, title) in zip(axes, method_info):
        complete = delta[["subject", "Delta_SBP_mmHg", column]].dropna()
        ax.axhline(0, color="#BBBBBB", lw=0.8)
        ax.axvline(0, color="#BBBBBB", lw=0.8)
        ax.scatter(
            complete["Delta_SBP_mmHg"],
            complete[column],
            color=PALETTE.get(method, PALETTE.get(method.upper(), "#555555")),
            s=28,
        )
        _, rho, _ = safe_correlation(
            complete["Delta_SBP_mmHg"].to_numpy(),
            complete[column].to_numpy(),
            "spearman",
        )
        for row in complete.itertuples(index=False):
            ax.annotate(
                f"S{int(row.subject):02d}",
                (row.Delta_SBP_mmHg, getattr(row, column)),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=6,
            )
        ax.set_title(f"{title}\nn={len(complete)}, rho={rho:.2f}")
        ax.set_xlabel("Delta SBP (mmHg)")
    axes[0].set_ylabel("Delta BRS (ms/mmHg)")
    fig.suptitle("Post hoc method-dependence comparison", y=1.02)
    fig.tight_layout()
    save_figure(fig, "Fig_delta_SBP_vs_delta_BRS_method_comparison")

    display_windows = ["Pre_late", "Stim_early", "Stim_late"]
    display_outcomes = [
        ("gain_REF_all_ms_per_mmHg", "Reference BRS", "ms/mmHg"),
        ("mean_SBP_mmHg", "Mean SBP", "mmHg"),
        ("mean_HR_bpm", "Mean HR", "beats/min"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (outcome, title, unit) in zip(axes, display_outcomes):
        pivot = subphase_wide.pivot(index="subject", columns="window", values=outcome)
        for subject in pivot.index:
            ax.plot(
                range(3),
                pivot.loc[subject, display_windows],
                color="#BBBBBB",
                lw=0.7,
                alpha=0.55,
            )
        means = pivot[display_windows].mean(axis=0)
        ci = []
        for window in display_windows:
            values = pivot[window].dropna().to_numpy(float)
            critical = stats.t.ppf(0.975, len(values) - 1)
            ci.append(critical * stats.sem(values))
        ax.errorbar(
            range(3),
            means,
            yerr=ci,
            marker="o",
            color="#111111",
            capsize=3,
            lw=1.7,
            zorder=4,
        )
        ax.set_xticks(range(3), ["Pre late", "Stim early", "Stim late"])
        ax.set_title(title)
        ax.set_ylabel(unit)
    fig.suptitle("Prespecified 150-s temporal decomposition", y=1.02)
    fig.tight_layout()
    save_figure(fig, "Fig_stim_early_late_BRS_SBP_HR")

    dimensions = list(response["response_dimension"].drop_duplicates())
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2))
    for ax, dimension in zip(axes.flat, dimensions):
        group = response[response["response_dimension"] == dimension]
        x_pos = np.arange(2)
        values = group["mean_difference"].to_numpy(float)
        low = values - group["mean_difference_ci_low"].to_numpy(float)
        high = group["mean_difference_ci_high"].to_numpy(float) - values
        ax.axhline(0, color="#999999", lw=0.8)
        ax.errorbar(
            x_pos,
            values,
            yerr=np.vstack([low, high]),
            fmt="o",
            color="#222222",
            ecolor="#555555",
            capsize=3,
        )
        ax.scatter(x_pos, values, c=[PALETTE["early"], PALETTE["late"]], zorder=4)
        ax.set_xticks(x_pos, ["Early", "Late"])
        ax.set_title(dimension)
        ax.set_ylabel("Change from Pre late")
    fig.suptitle("Response-state summary (mean and participant BCa 95% CI)", y=1.01)
    fig.tight_layout()
    save_figure(fig, "Fig_subphase_response_state")

    subject_traj = trajectory[trajectory["record_type"] == "subject_bin"]
    group_traj = trajectory[trajectory["record_type"] == "group_bin"]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True)
    for ax, outcome, ylabel in (
        (axes[0], "mean_SBP_mmHg", "Mean SBP (mmHg)"),
        (axes[1], "mean_HR_bpm", "Mean HR (beats/min)"),
    ):
        for _, group in subject_traj.groupby("subject"):
            ax.plot(group["bin_center_s"], group[outcome], color="#BBBBBB", alpha=0.25, lw=0.55)
        ax.plot(group_traj["bin_center_s"], group_traj[outcome], color="#111111", lw=1.7)
        ax.fill_between(
            group_traj["bin_center_s"].to_numpy(float),
            group_traj[f"{outcome}_ci_low"].to_numpy(float),
            group_traj[f"{outcome}_ci_high"].to_numpy(float),
            color="#777777",
            alpha=0.22,
        )
        for boundary in (300.0, 600.0):
            ax.axvline(boundary, color="#D55E00", ls="--", lw=1.0)
        ax.set_ylabel(ylabel)
    axes[1].set_xlabel("Protocol time (s)")
    axes[0].set_title("Non-overlapping 30-s context (no 30-s BRS)")
    fig.tight_layout()
    save_figure(fig, "Fig_30s_SBP_HR_trajectory")

    max_ba = ba_subject[ba_subject["branch"].isin(["A0_MAX", "A1_MAX"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, branch in zip(axes, ("A0_MAX", "A1_MAX")):
        group = max_ba[max_ba["branch"] == branch]
        bias = group["method_minus_reference_delta"].mean()
        sd = group["method_minus_reference_delta"].std(ddof=1)
        ax.scatter(
            group["mean_of_delta_methods"],
            group["method_minus_reference_delta"],
            color=PALETTE[branch],
            s=28,
        )
        for level, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
            ax.axhline(level, color="#555555", ls=style, lw=1.0)
        ax.set_title(f"{branch} (n={len(group)})")
        ax.set_xlabel("Mean of Stim-Pre BRS differences")
    axes[0].set_ylabel("Antonino-method minus reference difference")
    fig.suptitle("Bland-Altman description of participant change scores", y=1.02)
    fig.tight_layout()
    save_figure(fig, "Fig_reference_vs_antonino_bland_altman")


def main() -> None:
    ensure_output_dirs()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    all_subjects = load_all_subjects()
    gate = reference_reproduction_gate(all_subjects)

    phase_context, subject_phase, sequence_level = compute_windows(all_subjects, PHASES)
    save_csv(subject_phase, "antonino_method_subject_phase.csv")
    save_csv(sequence_level, "antonino_method_sequence_level.csv")
    contrasts_a = analysis_a_contrasts(subject_phase)
    save_csv(contrasts_a, "antonino_method_contrasts.csv")
    evaluability_a = analysis_a_evaluability(subject_phase)
    save_csv(evaluability_a, "antonino_method_evaluability.csv")
    direct_comparison, ba_subject = direct_method_comparison(subject_phase)
    save_csv(direct_comparison, "reference_vs_antonino_method_comparison.csv")
    antonino_direction = classify_antonino_direction(contrasts_a)

    delta_subject = build_delta_subject_table(phase_context, subject_phase)
    save_csv(delta_subject, "delta_sbp_delta_brs_subject_level.csv")
    association_stats, influence, models, association_class = association_analysis(delta_subject)
    save_csv(association_stats, "delta_sbp_delta_brs_statistics.csv")
    save_csv(influence, "delta_sbp_delta_brs_influence.csv")
    save_csv(models, "delta_sbp_delta_brs_models.csv")

    sub_context, sub_summaries, _ = compute_windows(all_subjects, SUBPHASES)
    sub_wide = build_subphase_wide(sub_context, sub_summaries)
    save_csv(sub_wide, "subphase_subject_level_values.csv")
    sub_eval = subphase_evaluability(sub_summaries)
    save_csv(sub_eval, "subphase_sequence_evaluability.csv")
    sub_contrasts, sub_wilcoxon = subphase_contrast_analysis(sub_wide)
    save_csv(sub_contrasts, "subphase_contrasts.csv")
    save_csv(sub_wilcoxon, "subphase_optional_wilcoxon_descriptive.csv")
    response = build_response_state(sub_contrasts)
    save_csv(response, "subphase_response_state.csv")
    trajectory = compute_30s_trajectory(all_subjects)
    save_csv(trajectory, "trajectory_30s_SBP_HR.csv")
    temporal_class = classify_temporal_pattern(response)

    create_figures(
        delta_subject,
        association_stats,
        ba_subject,
        sub_wide,
        response,
        trajectory,
    )

    focal_a = contrasts_a[
        (contrasts_a["record_type"] == "stim_minus_pre")
        & (contrasts_a["direction"] == "all")
        & (contrasts_a["branch"].isin(["REF", "A0_MAX", "A1_MAX"]))
    ][
        [
            "branch",
            "paired_n",
            "pre_mean",
            "stim_mean",
            "mean_difference",
            "mean_difference_ci_low",
            "mean_difference_ci_high",
            "cohens_dz",
            "wilcoxon_p_two_sided",
            "n_negative",
            "n_positive",
        ]
    ]
    focal_association = association_stats[
        association_stats["analysis"].isin(
            ["primary_spearman", "method_dependence_spearman"]
        )
    ][
        [
            "method",
            "n",
            "estimate",
            "ci_low",
            "ci_high",
            "permutation_p_two_sided",
        ]
    ]
    focal_response = response[
        response["response_dimension"].isin(
            ["Delta SBP", "Delta HR", "Delta BRS reference", "Delta BRS A0_MAX", "Delta BRS A1_MAX"]
        )
    ]
    results = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reference_reproduction": gate["status"],
        "antonino_exactness": EXACTNESS,
        "antonino_direction": antonino_direction,
        "delta_sbp_delta_brs": association_class,
        "early_late_pattern": temporal_class,
        "focal_antonino_results": focal_a.to_dict(orient="records"),
        "focal_association_results": focal_association.to_dict(orient="records"),
        "focal_response_state": focal_response.to_dict(orient="records"),
        "software": environment_versions(),
        "output_root": str(PACKAGE_ROOT),
        "manuscript_files_modified": "NO",
        "palette": PALETTE,
    }
    (LOGS_DIR / "analysis_results_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    (LOGS_DIR / "environment.json").write_text(
        json.dumps(environment_versions(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("REFERENCE_REPRODUCTION = PASS")
    print(f"ANTONINO_EXACTNESS = {EXACTNESS}")
    print(f"ANTONINO_DIRECTION = {antonino_direction}")
    print(f"DELTA_SBP_DELTA_BRS = {association_class}")
    print(f"EARLY_LATE_PATTERN = {temporal_class}")
    print(f"OUTPUT_ROOT = {PACKAGE_ROOT}")
    print("MANUSCRIPT_FILES_MODIFIED = NO")


if __name__ == "__main__":
    main()
