"""Generate the Codex-updated main Figure 2 and Supplementary Figure S5.

The figures are rebuilt from machine-readable source data.  Figure 2c and the
coherence-prevalence panel are read from revised Supplementary Data 3; Figure
S5 is read from its native paired-beat VAR records in the same file.  Existing
figure files are never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASES = ["Pre", "Stim", "Post"]
PHASE_X = np.arange(3, dtype=float)
PRE = "#3B83BD"
STIM = "#E69F00"
POST = "#009E73"
PHASE_COLORS = {"Pre": PRE, "Stim": STIM, "Post": POST}
NAVY = "#1F4E79"
BLUE = "#2C7FB8"
TEAL = "#1B9E77"
AMBER = "#D89000"
DARK = "#222222"
MID = "#6F6F6F"
LIGHT = "#D5D5D5"
GRID = "#E8E8E8"
PALE = "#F6F6F6"
NEGATIVE = "#2166AC"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.85,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "svg.hashsalt": "tavns-scirep-analysis-figures",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def clean_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label.lower(),
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_all(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {suffix: output_dir / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png", "tif", "jpg")}
    metadata = {
        "Title": stem,
        "Creator": "generate_summary_figures.py",
        "Description": "Rebuilt from revised machine-readable source data",
    }
    fig.savefig(paths["svg"], format="svg", facecolor="white", metadata=metadata)
    fig.savefig(paths["pdf"], format="pdf", facecolor="white", metadata={"Title": stem, "Creator": metadata["Creator"]})
    fig.savefig(paths["png"], format="png", facecolor="white", dpi=300)
    fig.savefig(paths["tif"], format="tiff", facecolor="white", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["jpg"], format="jpg", facecolor="white", dpi=600, pil_kwargs={"quality": 95, "subsampling": 0})
    return paths


def require_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def load_sequence_quality(data3: pd.DataFrame) -> pd.DataFrame:
    subset = data3[
        data3["analysis_family"].eq("sequence_quality")
        & data3["record_type"].eq("contrast_summary")
        & data3["direction"].eq("all")
        & data3["contrast"].eq("Stim-Pre")
        & data3["metric"].eq("cohens_dz")
    ][["outcome", "value", "p_value"]].copy()
    wanted = [
        "mean_within_sequence_r",
        "mean_rri_response_abs_ms",
        "mean_sbp_ramp_amplitude_abs_mmHg",
        "n_qualifying_brs_sequences",
        "n_sbp_ramps",
        "BEI",
    ]
    subset = subset.set_index("outcome").loc[wanted].reset_index()
    subset = subset.rename(columns={"value": "cohens_dz", "p_value": "wilcoxon_p_two_sided"})
    subset = require_numeric(subset, ["cohens_dz", "wilcoxon_p_two_sided"])
    if len(subset) != 6:
        raise RuntimeError("Figure 2c must contain exactly six sequence-quality outcomes")
    return subset


def load_coherence_prevalence(data3: pd.DataFrame) -> pd.DataFrame:
    subset = data3[
        data3["analysis_family"].eq("coherence_significance")
        & data3["record_type"].eq("phase_prevalence")
    ][["phase", "successes", "trials", "prevalence", "ci_low", "ci_high"]].copy()
    subset = subset.set_index("phase").loc[PHASES].reset_index()
    return require_numeric(subset, ["successes", "trials", "prevalence", "ci_low", "ci_high"])


def make_figure_2(
    *,
    data3_path: Path,
    differences_path: Path,
    ofat_path: Path,
    output_dir: Path,
    stem: str,
) -> tuple[dict[str, Path], pd.DataFrame]:
    data3 = pd.read_csv(data3_path, dtype=str, keep_default_na=False)
    differences = pd.read_csv(differences_path)
    ofat = pd.read_csv(ofat_path)
    quality = load_sequence_quality(data3)
    prevalence = load_coherence_prevalence(data3)
    if int((differences["Stim_minus_Pre_BRS_ms_per_mmHg"] < 0).sum()) != 17:
        raise RuntimeError("The 17/18 central direction anchor was not reproduced")

    fig = plt.figure(figsize=(7.2, 6.45))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.13,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.58,
        hspace=0.44,
        width_ratios=[1.0, 1.12],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    values = differences["Stim_minus_Pre_BRS_ms_per_mmHg"].to_numpy(float)
    x_values = np.arange(1, len(values) + 1)
    colors = [NEGATIVE if value < 0 else STIM for value in values]
    ax_a.bar(x_values, values, width=0.75, color=colors, edgecolor="white", linewidth=0.35, zorder=2)
    ax_a.axhline(0, color=DARK, linewidth=0.9, zorder=3)
    ax_a.set_xlim(0.25, len(values) + 0.75)
    ax_a.set_ylim(-11.2, 1.15)
    ax_a.set_xticks([1, 6, 12, 18])
    ax_a.set_xlabel("Participants, sorted by Stim − Pre difference")
    ax_a.set_ylabel("Stim − Pre BRSseq,all (ms/mmHg)")
    ax_a.set_title("Individual consistency", loc="left", pad=5.5)
    ax_a.text(
        0.98,
        0.97,
        "17/18 lower",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=NEGATIVE,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
    )
    clean_axis(ax_a)
    panel_label(ax_a, "a")

    order = [
        "Reference setting",
        "SBP increment ≥0.5 mmHg/beat",
        "Minimum length = 4 beats",
        "Lag = 0 beats",
        "Lag = 2 beats",
        "Correlation threshold r ≥0.70",
        "Correlation threshold r ≥0.85",
        "Median aggregation",
    ]
    public_labels = {
        "Reference setting": "Reference setting",
        "SBP increment ≥0.5 mmHg/beat": "SBP increment ≥0.5 mmHg/beat",
        "Minimum length = 4 beats": "Minimum length = 4 beats",
        "Lag = 0 beats": "Lag = 0 beats",
        "Lag = 2 beats": "Lag = 2 beats",
        "Correlation threshold r ≥0.70": "Correlation r ≥0.70",
        "Correlation threshold r ≥0.85": "Correlation r ≥0.85",
        "Median aggregation": "Median aggregation",
    }
    ofat = ofat.set_index("display_setting").loc[order].reset_index()
    y_values = np.arange(len(ofat))
    for yi in y_values:
        if yi % 2 == 0:
            ax_b.axhspan(yi - 0.5, yi + 0.5, color=PALE, zorder=0)
    for yi, item in enumerate(ofat.itertuples(index=False)):
        is_reference = item.display_setting == "Reference setting"
        color = NAVY if is_reference else BLUE
        ax_b.plot(
            [item.mean_difference_ci_low, item.mean_difference_ci_high],
            [yi, yi],
            color=color,
            linewidth=1.8 if is_reference else 1.2,
            zorder=2,
        )
        ax_b.scatter(
            item.mean_difference_stim_minus_pre,
            yi,
            s=40 if is_reference else 28,
            facecolor=color if is_reference else "white",
            edgecolor=color,
            linewidth=1.35,
            zorder=3,
        )
        if int(item.n_evaluable) < 18:
            ax_b.text(0.68, yi, f"n={int(item.n_evaluable)}", ha="right", va="center", fontsize=6.2, color=MID)
    ax_b.axvline(0, color=DARK, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax_b.set_yticks(y_values, [public_labels[item] for item in ofat["display_setting"]])
    ax_b.invert_yaxis()
    ax_b.set_xlim(-4.15, 0.8)
    ax_b.set_xlabel("Mean Stim − Pre BRS difference\n(ms/mmHg; BCa 95% CI)")
    ax_b.set_title("One-factor-at-a-time sensitivity", loc="left", pad=5.5)
    ax_b.tick_params(axis="y", labelsize=6.7)
    clean_axis(ax_b, grid=False)
    panel_label(ax_b, "b", x=-0.30)

    metric_labels = {
        "mean_within_sequence_r": "Within-sequence r",
        "mean_rri_response_abs_ms": "Mean absolute\nRRI response",
        "mean_sbp_ramp_amplitude_abs_mmHg": "SBP-ramp\namplitude",
        "n_qualifying_brs_sequences": "Qualifying\nsequences",
        "n_sbp_ramps": "SBP ramps",
        "BEI": "BEI",
    }
    quality["display"] = quality["outcome"].map(metric_labels)
    y_quality = np.arange(len(quality))
    for yi, item in enumerate(quality.itertuples(index=False)):
        ax_c.plot([0, item.cohens_dz], [yi, yi], color=MID, linewidth=1.5, zorder=2)
        ax_c.scatter(item.cohens_dz, yi, s=34, facecolor="white", edgecolor=MID, linewidth=1.4, zorder=3)
        ax_c.text(
            0.82,
            yi,
            f"p={item.wilcoxon_p_two_sided:.3g}",
            ha="right",
            va="center",
            fontsize=6.5,
            color=DARK,
        )
    ax_c.axvline(0, color=DARK, linewidth=0.85, zorder=1)
    ax_c.set_yticks(y_quality, quality["display"])
    ax_c.invert_yaxis()
    ax_c.set_xlim(-0.82, 0.88)
    ax_c.set_xlabel("Paired effect size, Cohen’s dz (Stim − Pre)")
    ax_c.set_title("Sequence-quality context", loc="left", pad=5.5)
    ax_c.tick_params(axis="y", labelsize=6.9)
    clean_axis(ax_c, grid=False)
    panel_label(ax_c, "c", x=-0.22)

    for idx, item in prevalence.iterrows():
        phase = item["phase"]
        rate = 100 * item["prevalence"]
        ax_d.errorbar(
            idx,
            rate,
            yerr=[[100 * (item["prevalence"] - item["ci_low"])], [100 * (item["ci_high"] - item["prevalence"])]],
            fmt="o",
            markersize=6.6,
            markerfacecolor="white",
            markeredgecolor=PHASE_COLORS[phase],
            markeredgewidth=1.8,
            ecolor=PHASE_COLORS[phase],
            elinewidth=1.6,
            capsize=3,
            zorder=3,
        )
        ax_d.text(
            idx,
            min(101.5, 100 * item["ci_high"] + 3.0),
            f"{int(item['successes'])}/{int(item['trials'])}",
            ha="center",
            va="bottom",
            fontsize=7.1,
            fontweight="bold",
            color=PHASE_COLORS[phase],
        )
    ax_d.plot(PHASE_X, 100 * prevalence["prevalence"].to_numpy(), color=LIGHT, linewidth=1.2, zorder=1)
    ax_d.set_xticks(PHASE_X, PHASES)
    ax_d.set_xlim(-0.35, 2.35)
    ax_d.set_ylim(0, 106)
    ax_d.set_ylabel("Participants exceeding surrogate threshold (%)")
    ax_d.set_title("Significant Mayer-band coherence\nprevalence", loc="left", pad=5.5)
    ax_d.text(
        0.02,
        0.035,
        "Pre vs Stim: exact McNemar p = 1.00\nThree phases: Cochran Q p = 0.846",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": LIGHT, "linewidth": 0.7},
    )
    clean_axis(ax_d)
    panel_label(ax_d, "d", x=-0.20)

    paths = save_all(fig, output_dir, stem)
    plt.close(fig)
    return paths, quality


def make_supplementary_figure_s5(
    *, data3_path: Path, output_dir: Path, stem: str
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    data3 = pd.read_csv(data3_path, dtype=str, keep_default_na=False)
    records = data3[
        data3["analysis_family"].eq("GC_significance_native_paired_beats")
        & data3["record_type"].eq("participant_phase_direction")
    ].copy()
    if len(records) != 108:
        raise RuntimeError(f"Expected 108 direction rows, found {len(records)}")
    model_rows = records.drop_duplicates(["subject_id", "phase"]).copy()
    if len(model_rows) != 54:
        raise RuntimeError(f"Expected 54 unique participant-phase fits, found {len(model_rows)}")
    numeric = ["selected_order", "candidate_order_max", "residual_whiteness_p", "residual_normality_p"]
    model_rows = require_numeric(model_rows, numeric)
    model_rows["whiteness_pass"] = model_rows["residual_whiteness_p"] >= 0.05
    model_rows["normality_pass"] = model_rows["residual_normality_p"] >= 0.05
    model_rows["both_residual_pass"] = model_rows["whiteness_pass"] & model_rows["normality_pass"]
    model_rows["stability_pass"] = model_rows["stability_status"].eq("stable")

    order_counts = (
        model_rows.groupby(["phase", "selected_order"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    order_grid = pd.MultiIndex.from_product([PHASES, range(1, 11)], names=["phase", "selected_order"]).to_frame(index=False)
    order_counts = order_grid.merge(order_counts, how="left", on=["phase", "selected_order"])
    order_counts["count"] = order_counts["count"].fillna(0).astype(int)

    diagnostic_rows = []
    for phase in [*PHASES, "All"]:
        subset = model_rows if phase == "All" else model_rows[model_rows["phase"].eq(phase)]
        for diagnostic, column in (
            ("Stable", "stability_pass"),
            ("Whiteness", "whiteness_pass"),
            ("Normality", "normality_pass"),
            ("Both residual", "both_residual_pass"),
        ):
            passed = int(subset[column].sum())
            diagnostic_rows.append(
                {
                    "phase": phase,
                    "diagnostic": diagnostic,
                    "passed": passed,
                    "total": len(subset),
                    "pass_rate": passed / len(subset),
                }
            )
    diagnostic_summary = pd.DataFrame(diagnostic_rows)

    fig = plt.figure(figsize=(7.2, 3.75))
    grid = fig.add_gridspec(1, 2, left=0.09, right=0.985, bottom=0.23, top=0.91, wspace=0.34, width_ratios=[1.05, 1.15])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    orders = np.arange(1, 11)
    width = 0.24
    offsets = {"Pre": -width, "Stim": 0.0, "Post": width}
    for phase in PHASES:
        subset = order_counts[order_counts["phase"].eq(phase)].set_index("selected_order").loc[orders]
        ax_a.bar(
            orders + offsets[phase],
            subset["count"],
            width=width,
            color=PHASE_COLORS[phase],
            label=phase,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
    ax_a.axvline(10.5, color=DARK, linewidth=0.0)
    ax_a.set_xticks(orders)
    ax_a.set_xlim(0.5, 10.5)
    ax_a.set_ylim(0, 7)
    ax_a.set_xlabel("AIC-selected VAR order (beats)")
    ax_a.set_ylabel("Unique participant-phase fits")
    ax_a.set_title("Native paired-beat model order", loc="left", pad=5.5)
    ax_a.text(
        0.02,
        0.035,
        "Candidate orders 1–10; order 10 in 1/54 fits",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": LIGHT, "linewidth": 0.7},
    )
    ax_a.legend(frameon=False, ncol=3, loc="upper right")
    clean_axis(ax_a)
    panel_label(ax_a, "a", x=-0.18)

    diagnostic_order = ["Stable", "Whiteness", "Normality", "Both residual"]
    x_diag = np.arange(len(diagnostic_order), dtype=float)
    for phase in PHASES:
        subset = diagnostic_summary[diagnostic_summary["phase"].eq(phase)].set_index("diagnostic").loc[diagnostic_order]
        x_positions = x_diag + offsets[phase]
        rates = 100 * subset["pass_rate"].to_numpy(float)
        ax_b.bar(
            x_positions,
            rates,
            width=width,
            color=PHASE_COLORS[phase],
            label=phase,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        for xpos, rate, passed, total in zip(x_positions, rates, subset["passed"], subset["total"]):
            high_bar = rate >= 45
            ax_b.text(
                xpos,
                rate - 4.0 if high_bar else rate + 3.0,
                f"{int(passed)}/{int(total)}",
                ha="center",
                va="top" if high_bar else "bottom",
                fontsize=5.7,
                rotation=90 if high_bar else 0,
                color="white" if high_bar else DARK,
                fontweight="bold" if high_bar else "normal",
            )
    ax_b.set_xticks(x_diag, ["Stable", "Whiteness", "Normality", "Both residual\ndiagnostics"])
    ax_b.set_xlim(-0.55, 3.55)
    ax_b.set_ylim(0, 106)
    ax_b.set_ylabel("Unique fits passing diagnostic (%)")
    ax_b.set_title("Model and residual diagnostics", loc="left", pad=5.5)
    clean_axis(ax_b)
    panel_label(ax_b, "b", x=-0.17)

    fig.text(
        0.76,
        0.055,
        "Overall: whiteness 42/54; normality 7/54; both residual diagnostics 4/54.",
        ha="center",
        va="center",
        fontsize=6.5,
        color=DARK,
    )

    paths = save_all(fig, output_dir, stem)
    plt.close(fig)
    return paths, order_counts, diagnostic_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data3", required=True, type=Path)
    parser.add_argument("--differences", required=True, type=Path)
    parser.add_argument("--ofat", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure2-stem", default="figure_2_analysis_summary")
    parser.add_argument(
        "--s5-stem", default="supplementary_figure_s5_native_var_diagnostics"
    )
    args = parser.parse_args()
    configure_style()
    figure2_paths, quality = make_figure_2(
        data3_path=args.data3,
        differences_path=args.differences,
        ofat_path=args.ofat,
        output_dir=args.output_dir,
        stem=args.figure2_stem,
    )
    s5_paths, order_counts, diagnostic_summary = make_supplementary_figure_s5(
        data3_path=args.data3,
        output_dir=args.output_dir,
        stem=args.s5_stem,
    )
    quality.to_csv(
        args.output_dir / "figure_2c_sequence_quality_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    order_counts.to_csv(
        args.output_dir / "supplementary_figure_s5_order_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostic_summary.to_csv(
        args.output_dir / "supplementary_figure_s5_diagnostic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for path in [*figure2_paths.values(), *s5_paths.values()]:
        print(path)


if __name__ == "__main__":
    main()
