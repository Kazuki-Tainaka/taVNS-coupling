"""Generate Supplementary Figure S7 from bundled delayed-effect data.

Data dependencies:
  - figures/data/coupling_raw.csv
  - figures/data/supplementary_data_1.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "figures" / "data"
SUP_DATA = ROOT / "figures" / "data"
OUT_DIR = ROOT / "figures" / "outputs"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
}
plt.rcParams.update(STYLE)

W_DOUBLE = 170 / 25.4

TYPE_C_METRICS = [
    "SGC_SBP_to_RRI_LF",
    "SGC_SBP_to_RRI_Mayer",
    "rho_u",
    "innov_RRI_own_frac_Mayer",
    "innov_RRI_from_SBP_frac_Mayer",
]

TYPE_C_LABELS = [
    r"SGC SBP$\to$RRI (LF)",
    r"SGC SBP$\to$RRI (Mayer)",
    r"$\rho_u$",
    "Innov RRI self frac",
    "Innov SBP-driven frac",
]

OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_GREY = "#999999"


def ci_for_dz(dz: float, n: float) -> tuple[float, float]:
    if not np.isfinite(dz) or not np.isfinite(n) or n <= 1:
        return np.nan, np.nan
    se = np.sqrt(1.0 / n + dz**2 / (2.0 * n))
    return float(dz - 1.96 * se), float(dz + 1.96 * se)


def sig_text(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main():
    print("Loading data...")
    raw = pd.read_csv(DATA_DIR / "coupling_raw.csv")
    sup = pd.read_csv(SUP_DATA / "supplementary_data_1.csv")
    sup_idx = sup.set_index("Metric")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3,
        figsize=(W_DOUBLE * 1.35, W_DOUBLE * 0.52),
        gridspec_kw={"width_ratios": [1.75, 1.05, 1.05]},
    )

    # Panel A: Type C forest plot.
    y_pos = np.arange(len(TYPE_C_METRICS))
    dz_sp, dz_pp, p_sp, p_pp, n_vals = [], [], [], [], []
    for metric in TYPE_C_METRICS:
        if metric in sup_idx.index:
            row = sup_idx.loc[metric]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            dz_sp.append(row.get("dz_Stim_Pre", np.nan))
            dz_pp.append(row.get("dz_Post_Pre", np.nan))
            p_sp.append(row.get("p_Stim_Pre", np.nan))
            p_pp.append(row.get("p_Post_Pre", np.nan))
            n_vals.append(row.get("n", 16))
        else:
            dz_sp.append(np.nan)
            dz_pp.append(np.nan)
            p_sp.append(np.nan)
            p_pp.append(np.nan)
            n_vals.append(16)

    dz_sp = np.array(dz_sp, dtype=float)
    dz_pp = np.array(dz_pp, dtype=float)
    p_sp = np.array(p_sp, dtype=float)
    p_pp = np.array(p_pp, dtype=float)
    n_vals = np.array(n_vals, dtype=float)

    offset = 0.18
    for i, metric in enumerate(TYPE_C_METRICS):
        y = y_pos[i]
        if np.isfinite(dz_sp[i]):
            lo, hi = ci_for_dz(dz_sp[i], n_vals[i])
            ax_a.plot([lo, hi], [y - offset, y - offset],
                      color=OI_GREY, lw=1.5, alpha=0.6,
                      solid_capstyle="butt")
            ax_a.plot(dz_sp[i], y - offset, marker="o", markersize=5.5,
                      color=OI_GREY, alpha=0.75, zorder=5)

        if np.isfinite(dz_pp[i]):
            lo, hi = ci_for_dz(dz_pp[i], n_vals[i])
            color_pp = OI_BLUE if dz_pp[i] > 0 else OI_ORANGE
            ax_a.plot([lo, hi], [y + offset, y + offset],
                      color=color_pp, lw=2.0, solid_capstyle="butt")
            ax_a.plot(dz_pp[i], y + offset, marker="o", markersize=6.5,
                      color=color_pp, zorder=6)
            marker = sig_text(p_pp[i])
            if marker:
                text_y = y + offset + (0.14 if dz_pp[i] < 0 else -0.14)
                ax_a.text(dz_pp[i], text_y, marker, ha="center",
                          va="center", fontsize=10, fontweight="bold",
                          color=color_pp)

    ax_a.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(TYPE_C_LABELS, fontsize=7.2)
    ax_a.set_xlabel(r"Effect size (Cohen's $d_z$)")
    ax_a.set_xlim(-1.2, 1.2)
    ax_a.set_title("Type C (Delayed) Metrics:\n"
                   r"Stim$-$Pre vs Post$-$Pre",
                   fontsize=9, fontweight="bold")
    ax_a.grid(axis="x", alpha=0.2)
    ax_a.invert_yaxis()
    legend_elements = [
        Line2D([0], [0], color=OI_GREY, lw=2, marker="o", alpha=0.6,
               label=r"Stim$-$Pre (n.s.)"),
        Line2D([0], [0], color=OI_BLUE, lw=2, marker="o",
               label=r"Post$-$Pre (positive)"),
        Line2D([0], [0], color=OI_ORANGE, lw=2, marker="o",
               label=r"Post$-$Pre (negative)"),
    ]
    ax_a.legend(handles=legend_elements, fontsize=5.5, loc="upper left",
                framealpha=0.75, handlelength=1.7, borderpad=0.25)
    ax_a.text(-0.18, 1.02, "A", transform=ax_a.transAxes,
              fontsize=12, fontweight="bold", va="top")

    # Panel B: Innovation variance redistribution.
    phases = ["Pre", "Stim", "Post"]
    x = np.arange(3)
    own_subjects = raw.pivot_table(
        index="Subject", columns="Phase", values="innov_RRI_own_frac_Mayer")
    sbp_subjects = raw.pivot_table(
        index="Subject", columns="Phase", values="innov_RRI_from_SBP_frac_Mayer")

    for _, row in own_subjects.iterrows():
        vals = [row.get(phase, np.nan) for phase in phases]
        if np.all(np.isfinite(vals)):
            ax_b.plot(x, vals, color=OI_BLUE, alpha=0.15, linewidth=0.5)
    for _, row in sbp_subjects.iterrows():
        vals = [row.get(phase, np.nan) for phase in phases]
        if np.all(np.isfinite(vals)):
            ax_b.plot(x, vals, color=OI_ORANGE, alpha=0.15, linewidth=0.5)

    own_means = [raw[raw["Phase"] == phase]["innov_RRI_own_frac_Mayer"].mean()
                 for phase in phases]
    sbp_means = [raw[raw["Phase"] == phase]["innov_RRI_from_SBP_frac_Mayer"].mean()
                 for phase in phases]
    ax_b.plot(x, own_means, color=OI_BLUE, linewidth=2.3, marker="s",
              markersize=8, label="RRI self fraction", zorder=10)
    ax_b.plot(x, sbp_means, color=OI_ORANGE, linewidth=2.3, marker="^",
              markersize=8, label="SBP-driven fraction", zorder=10)
    ax_b.text(2.05, max(own_means[2], sbp_means[2]) + 0.035, "*",
              fontsize=12, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(phases)
    ax_b.set_ylabel("Innovation variance fraction")
    ax_b.set_title("Innovation Variance Redistribution\n(Mayer band)",
                   fontsize=9, fontweight="bold")
    ax_b.set_ylim(0, 1.0)
    ax_b.grid(axis="y", alpha=0.2)
    ax_b.legend(fontsize=5.6, loc="lower center", framealpha=0.75)
    ax_b.text(-0.28, 1.02, "B", transform=ax_b.transAxes,
              fontsize=12, fontweight="bold", va="top")

    # Panel C: rho_u box plots with paired lines.
    rho_pivot = raw.pivot_table(index="Subject", columns="Phase", values="rho_u")
    rho_data = [
        np.asarray(rho_pivot[phase].dropna().values, dtype=float)
        for phase in phases
    ]
    colors = [OI_BLUE, OI_ORANGE, OI_GREEN]
    valid_idx = [
        i for i, vals in enumerate(rho_data)
        if vals.size > 0 and np.any(np.isfinite(vals))
    ]

    if valid_idx:
        bp = ax_c.boxplot(
            [rho_data[i] for i in valid_idx],
            positions=[i + 1 for i in valid_idx],
            labels=[phases[i] for i in valid_idx],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=OI_ORANGE, linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], [colors[i] for i in valid_idx]):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
    else:
        ax_c.text(
            0.5, 0.5, "No valid data", ha="center", va="center",
            transform=ax_c.transAxes,
        )

    for _, row in rho_pivot.iterrows():
        vals = [row.get(phase, np.nan) for phase in phases]
        if np.all(np.isfinite(vals)):
            ax_c.plot([1, 2, 3], vals, color="grey", alpha=0.25,
                      linewidth=0.5, zorder=1)

    rng = np.random.default_rng(42)
    for i in valid_idx:
        vals = rho_data[i]
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax_c.scatter(np.full(len(vals), i + 1) + jitter, vals,
                     color=colors[i], s=14, alpha=0.65, zorder=5,
                     edgecolors="white", linewidth=0.3)

    if valid_idx:
        y_min = min(np.nanmin(rho_data[i]) for i in valid_idx)
        y_max = max(np.nanmax(rho_data[i]) for i in valid_idx)
        span = max(y_max - y_min, np.finfo(float).eps)
        y_top = y_max + span * 0.13
        h = span * 0.04
        upper = y_top
        if 0 in valid_idx and 2 in valid_idx:
            ax_c.plot([1, 1, 3, 3], [y_top, y_top + h, y_top + h, y_top],
                      lw=0.8, color="black")
            ax_c.text(2, y_top + h * 1.25, "**", ha="center", va="bottom",
                      fontsize=12, fontweight="bold")
            upper = y_top + h * 2.5
        ax_c.set_ylim(y_min - span * 0.12, upper)
    ax_c.set_ylabel(r"Residual cross-correlation ($\rho_u$)")
    ax_c.set_title("Residual Cross-Correlation\n"
                   r"($d_z$=$-$0.60, Post$-$Pre $p$=0.003)",
                   fontsize=9, fontweight="bold")
    ax_c.grid(axis="y", alpha=0.2)
    ax_c.text(-0.24, 1.02, "C", transform=ax_c.transAxes,
              fontsize=12, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.18,
                        wspace=0.72)
    fig.savefig(OUT_DIR / "figs7.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'figs7.png'}")


if __name__ == "__main__":
    main()
