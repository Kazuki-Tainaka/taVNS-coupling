"""Generate editable SVG figures from frozen nonlinear-analysis CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    os.environ.get(
        "TAVNS_NONLINEAR_OUTPUT_ROOT",
        RELEASE_ROOT / "controlled_outputs" / "nonlinear_sensitivity",
    )
).resolve()
RESULTS_DIR = OUTPUT_ROOT / "05_results"
FIGURE_DIR = OUTPUT_ROOT / "06_figures"
PHASES = ("Pre", "Stim", "Post")
PHASE_X = np.arange(3, dtype=float)
METHOD_COLORS = {"LP": "#0072B2", "CE": "#D55E00", "SSC": "#009E73"}
DIRECTION_MARKERS = {"SBP→RRI": "o", "RRI→SBP": "s"}


def configure() -> None:
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "Arial",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "savefig.transparent": False,
        }
    )


def _label(method: str, direction: str) -> str:
    if direction not in DIRECTION_MARKERS:
        raise ValueError(f"Unknown direction label: {direction}")
    return f"{method} {direction}"


def plot_forest(contrasts: pd.DataFrame, path: Path) -> None:
    primary = contrasts[contrasts["contrast"].eq("Stim-Pre")].copy()
    if primary.empty:
        return
    primary = primary.sort_values(["method", "direction"]).reset_index(drop=True)
    paired_counts = sorted(primary["evaluable_paired_n"].dropna().astype(int).unique())
    n_label = str(paired_counts[0]) if len(paired_counts) == 1 else "/".join(
        str(value) for value in paired_counts
    )
    y = np.arange(len(primary))[::-1]
    figure, axis = plt.subplots(figsize=(7.4, max(2.8, 0.58 * len(primary) + 1.2)))
    for index, row in primary.iterrows():
        color = METHOD_COLORS.get(str(row["method"]), "0.25")
        marker = DIRECTION_MARKERS.get(str(row["direction"]), "o")
        value = float(row["paired_mean_difference"])
        low = float(row["mean_difference_BCa95_low"])
        high = float(row["mean_difference_BCa95_high"])
        axis.errorbar(
            value,
            y[index],
            xerr=[[value - low], [high - value]],
            fmt=marker,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.2,
            capsize=3,
        )
        q_value = row["BH_q_primary_family"]
        note = (
            f"q={float(q_value):.3f}"
            if pd.notna(q_value) and float(q_value) < 0.05
            else "not robust after FDR"
        )
        axis.annotate(
            note,
            xy=(high, y[index]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="0.35",
        )
    axis.axvline(0.0, color="0.35", linewidth=0.9)
    axis.set_yticks(
        y,
        [_label(str(row.method), str(row.direction)) for row in primary.itertuples(index=False)],
    )
    axis.set_xlabel("Stim − Pre mean directed-strength difference (BCa 95% CI)")
    axis.set_title(
        f"Targeted nonlinear RRI–SBP sensitivity analysis (paired n={n_label})",
        loc="left",
    )
    axis.grid(axis="x", color="0.88", linewidth=0.6)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def plot_paired_values(metrics: pd.DataFrame, path: Path) -> None:
    outcomes = metrics[["method", "direction"]].drop_duplicates().sort_values(
        ["method", "direction"]
    )
    if outcomes.empty or len(outcomes) > 6:
        return
    n_panels = len(outcomes)
    subject_n = int(metrics.loc[metrics["finite"].eq(True), "subject_id"].nunique())
    n_columns = 2
    n_rows = int(np.ceil(n_panels / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(8.6, 2.8 * n_rows),
        squeeze=False,
    )
    for panel_index, outcome in enumerate(outcomes.itertuples(index=False)):
        axis = axes.flat[panel_index]
        subset = metrics[
            metrics["method"].eq(outcome.method)
            & metrics["direction"].eq(outcome.direction)
            & metrics["finite"].eq(True)
        ]
        pivot = subset.pivot(
            index="subject_id", columns="phase", values="directed_strength"
        ).reindex(columns=PHASES)
        color = METHOD_COLORS.get(str(outcome.method), "0.25")
        for _, values in pivot.iterrows():
            if values.notna().all():
                axis.plot(
                    PHASE_X,
                    values.to_numpy(float),
                    color="0.72",
                    linewidth=0.7,
                    marker="o",
                    markersize=2.5,
                    zorder=1,
                )
        means = pivot.mean(axis=0).to_numpy(float)
        sem = pivot.sem(axis=0).to_numpy(float)
        axis.errorbar(
            PHASE_X,
            means,
            yerr=sem,
            color=color,
            marker=DIRECTION_MARKERS.get(str(outcome.direction), "o"),
            markerfacecolor="white",
            markeredgewidth=1.2,
            capsize=3,
            linewidth=1.6,
            zorder=2,
        )
        axis.set_xticks(PHASE_X, PHASES)
        axis.set_ylabel("Directed strength")
        axis.set_title(_label(str(outcome.method), str(outcome.direction)), loc="left")
        axis.grid(axis="y", color="0.90", linewidth=0.5)
    for unused in range(n_panels, n_rows * n_columns):
        axes.flat[unused].axis("off")
    figure.suptitle(
        f"Primary setting: n={subject_n} individual values and mean ± SEM",
        x=0.06,
        ha="left",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def plot_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    if sensitivity.empty:
        return
    outcomes = sensitivity[["method", "direction"]].drop_duplicates().sort_values(
        ["method", "direction"]
    )
    n_panels = len(outcomes)
    paired_counts = sorted(
        sensitivity["evaluable_paired_n"].dropna().astype(int).unique()
    )
    n_label = str(paired_counts[0]) if len(paired_counts) == 1 else "/".join(
        str(value) for value in paired_counts
    )
    figure, axes = plt.subplots(
        n_panels,
        1,
        figsize=(8.4, max(3.0, 2.5 * n_panels)),
        squeeze=False,
    )
    setting_order = [
        "primary",
        "k20",
        "k40",
        "lag4",
        "lag12",
        "same_beat_convention",
        "centered192",
        "full_phase",
    ]
    for panel_index, outcome in enumerate(outcomes.itertuples(index=False)):
        axis = axes.flat[panel_index]
        subset = sensitivity[
            sensitivity["method"].eq(outcome.method)
            & sensitivity["direction"].eq(outcome.direction)
        ].copy()
        subset["setting_id"] = pd.Categorical(
            subset["setting_id"], categories=setting_order, ordered=True
        )
        subset = subset.sort_values("setting_id")
        y = np.arange(len(subset))[::-1]
        color = METHOD_COLORS.get(str(outcome.method), "0.25")
        for index, row in enumerate(subset.itertuples(index=False)):
            face = color if row.setting_id == "primary" else "white"
            axis.plot(
                row.paired_mean_difference,
                y[index],
                marker=DIRECTION_MARKERS.get(str(outcome.direction), "o"),
                markerfacecolor=face,
                markeredgecolor=color,
                linestyle="none",
            )
            axis.annotate(
                f"p={row.wilcoxon_p_two_sided:.3f}, n={int(row.evaluable_paired_n)}",
                (row.paired_mean_difference, y[index]),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color="0.38",
            )
        axis.axvline(0.0, color="0.35", linewidth=0.8)
        axis.set_yticks(y, subset["setting_id"].astype(str))
        axis.set_xlabel("Stim − Pre mean directed-strength difference")
        axis.set_title(_label(str(outcome.method), str(outcome.direction)), loc="left")
        axis.grid(axis="x", color="0.90", linewidth=0.5)
    figure.suptitle(
        f"Prespecified one-factor-at-a-time parameter sensitivity (paired n={n_label})",
        x=0.08,
        ha="left",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def make_all_figures() -> list[Path]:
    configure()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    contrast_path = RESULTS_DIR / "nonlinear_phase_contrasts.csv"
    metric_path = RESULTS_DIR / "nonlinear_subject_phase_metrics.csv"
    sensitivity_path = RESULTS_DIR / "nonlinear_parameter_sensitivity.csv"
    if contrast_path.is_file():
        target = FIGURE_DIR / "nonlinear_stim_pre_forest.svg"
        plot_forest(pd.read_csv(contrast_path), target)
        if target.is_file():
            created.append(target)
    if metric_path.is_file():
        target = FIGURE_DIR / "nonlinear_paired_values.svg"
        plot_paired_values(pd.read_csv(metric_path), target)
        if target.is_file():
            created.append(target)
    if sensitivity_path.is_file():
        target = FIGURE_DIR / "nonlinear_parameter_sensitivity.svg"
        plot_sensitivity(pd.read_csv(sensitivity_path), target)
        if target.is_file():
            created.append(target)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    for path in make_all_figures():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
