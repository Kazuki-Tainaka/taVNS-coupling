"""Generate Figure 2 from bundled figure data.

Data dependencies:
  - figures/data/coupling_raw.csv
  - figures/data/coupling_stats.csv
  - figures/data/bootstrap_ci.csv
  - data/reference/Additional_File_2.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG_DATA = ROOT / "figures" / "data"
REFERENCE_DATA = ROOT / "data" / "reference"
DATA_DIR = FIG_DATA
FIG_SUP = REFERENCE_DATA
OUT_DIR = ROOT / "figures" / "outputs"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
    }
)

W_DOUBLE = 170 / 25.4
NODES = ["RRI", "SBP", "PAT"]
PHASES = ["Pre", "Stim", "Post"]

DIRECTIONS = [
    ("SBP", "RRI", "GC3_F_SBP_to_RRI"),
    ("RRI", "SBP", "GC3_F_RRI_to_SBP"),
    ("PTT", "SBP", "GC3_F_PTT_to_SBP"),
    ("SBP", "PTT", "GC3_F_SBP_to_PTT"),
    ("RRI", "PTT", "GC3_F_RRI_to_PTT"),
    ("PTT", "RRI", "GC3_F_PTT_to_RRI"),
]

DIRECTIONS_ORDERED = [
    ("RRI", "PTT", "GC3_F_RRI_to_PTT"),
    ("RRI", "SBP", "GC3_F_RRI_to_SBP"),
    ("SBP", "RRI", "GC3_F_SBP_to_RRI"),
    ("SBP", "PTT", "GC3_F_SBP_to_PTT"),
    ("PTT", "SBP", "GC3_F_PTT_to_SBP"),
    ("PTT", "RRI", "GC3_F_PTT_to_RRI"),
]

DIRECTION_LABELS_ORDERED = [
    "RRI->PAT",
    "RRI->SBP",
    "SBP->RRI",
    "SBP->PAT",
    "PAT->SBP",
    "PAT->RRI",
]

TRGC_GENUINE = {"RRI_to_SBP", "RRI_to_PTT"}

OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_GREY = "#999999"
GOLD = "#F0C808"

# Hardcoded BCa bootstrap CI from the curated bootstrap CI table.
# These override the normal-approximation CI for metrics with computed BCa CI.
BCA_CI = {
    "GC3_F_RRI_to_PTT": (0.3512, 1.3876),
    "GC3_F_SBP_to_RRI": (-0.2772, 1.2124),
}


def build_matrices(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    node_idx = {"RRI": 0, "SBP": 1, "PTT": 2}
    for phase in PHASES:
        sub = raw[raw["Phase"].eq(phase)]
        mat = np.full((3, 3), np.nan)
        for src, tgt, col in DIRECTIONS:
            mat[node_idx[src], node_idx[tgt]] = sub[col].dropna().median()
        matrices[phase] = mat
    return matrices


def approx_ci(dz: float, n: int) -> tuple[float, float]:
    se = np.sqrt(1.0 / n + dz**2 / (2.0 * n))
    return dz - 1.96 * se, dz + 1.96 * se


def load_bootstrap_ci() -> dict[str, tuple[float, float]]:
    """Load BCa CI from bootstrap_ci.csv if available, falling back to BCA_CI."""
    boot_path = DATA_DIR / "bootstrap_ci.csv"
    if not boot_path.exists():
        return dict(BCA_CI)
    df = pd.read_csv(boot_path)
    ci_dict = {}
    for _, row in df.iterrows():
        metric = row["metric"]
        ci_dict[metric] = (float(row["ci_lower"]), float(row["ci_upper"]))
    ci_dict.update(BCA_CI)
    return ci_dict


def load_forest_stats() -> tuple[pd.DataFrame, Path]:
    """Load canonical Stim-Pre effect sizes for the Panel C forest plot."""
    add2_path = FIG_SUP / "Additional_File_2.csv"
    if add2_path.exists():
        add2 = pd.read_csv(add2_path).set_index("Metric")
        forest = add2.rename(
            columns={
                "dz_Stim_Pre": "dz",
                "p_Stim_Pre": "p_wilcoxon",
            }
        )
        return forest, add2_path

    stats_path = DATA_DIR / "coupling_stats.csv"
    stats = pd.read_csv(stats_path)
    forest = stats[stats["Comparison"].eq("Stim-Pre")].set_index("Metric")
    return forest, stats_path


def draw_network_edge(ax, sx, sy, tx, ty, color, linestyle, lw):
    dx, dy = tx - sx, ty - sy
    dist = np.sqrt(dx**2 + dy**2)
    nx, ny = -dy / dist * 0.03, dx / dist * 0.03
    ax.annotate(
        "",
        xy=(tx + nx, ty + ny),
        xytext=(sx + nx, sy + ny),
        arrowprops=dict(
            arrowstyle="->",
            lw=lw,
            color=color,
            linestyle=linestyle,
            connectionstyle="arc3,rad=0.1",
        ),
    )
    return nx, ny


def main() -> None:
    print("Loading data...")
    raw = pd.read_csv(DATA_DIR / "coupling_raw.csv")
    matrices = build_matrices(raw)
    sp_stats, forest_source = load_forest_stats()
    print(f"Forest plot source: {forest_source}")

    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 1.05))

    gs_top = fig.add_gridspec(
        1, 4, left=0.07, right=0.95, top=0.97, bottom=0.66,
        wspace=0.25, width_ratios=[1, 1, 1, 0.05]
    )
    ax_heatmaps = [fig.add_subplot(gs_top[i]) for i in range(3)]
    ax_cbar = fig.add_subplot(gs_top[3])

    vmax = max(np.nanmax(matrices[phase]) for phase in PHASES)
    last_im = None
    for idx, phase in enumerate(PHASES):
        ax = ax_heatmaps[idx]
        mat = matrices[phase]
        last_im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax,
                            interpolation="nearest")
        for i in range(3):
            for j in range(3):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center",
                            fontsize=8, color="grey")
                elif np.isfinite(mat[i, j]):
                    ax.text(
                        j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if mat[i, j] > vmax * 0.6 else "black",
                    )
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(NODES, fontsize=8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(NODES, fontsize=8)
        ax.set_title(phase, fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("From", fontsize=9)
        ax.set_xlabel("To", fontsize=9)

    cbar = fig.colorbar(last_im, cax=ax_cbar)
    cbar.set_label(r"GC3 $F$-statistic (median)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.text(0.02, 0.97, "A", fontsize=12, fontweight="bold")

    gs_mid = fig.add_gridspec(
        1, 4, left=0.07, right=0.95, top=0.60, bottom=0.34,
        wspace=0.15, width_ratios=[1, 1, 1, 0.6]
    )
    ax_networks = [fig.add_subplot(gs_mid[i]) for i in range(3)]
    ax_legend = fig.add_subplot(gs_mid[3])
    ax_legend.axis("off")

    node_pos = {"RRI": (0.5, 0.9), "SBP": (0.1, 0.2), "PAT": (0.9, 0.2)}
    node_idx = {"RRI": 0, "SBP": 1, "PTT": 2}
    name_map = {"RRI": "RRI", "SBP": "SBP", "PTT": "PAT"}

    for ax, phase in zip(ax_networks, PHASES):
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(phase, fontsize=10, fontweight="bold")
        mat = matrices[phase]

        for src_raw, tgt_raw, _col in DIRECTIONS:
            src = name_map[src_raw]
            tgt = name_map[tgt_raw]
            f_val = mat[node_idx[src_raw], node_idx[tgt_raw]]
            if not np.isfinite(f_val):
                continue
            key = f"{src_raw}_to_{tgt_raw}"
            genuine = key in TRGC_GENUINE
            stim_rri_pat = phase == "Stim" and key == "RRI_to_PTT"
            if stim_rri_pat:
                color, linestyle, text_color, text_weight = OI_ORANGE, "-", OI_ORANGE, "bold"
                lw = max(0.5, min(3.0, f_val / vmax * 3.0))
            elif genuine:
                color, linestyle, text_color, text_weight = "black", "-", "black", "bold"
                lw = max(0.5, min(3.0, f_val / vmax * 3.0))
            else:
                color, linestyle, text_color, text_weight = "#777777", "--", "#777777", "normal"
                lw = max(0.4, min(2.0, f_val / vmax * 2.0))

            sx, sy = node_pos[src]
            tx, ty = node_pos[tgt]
            nx, ny = draw_network_edge(ax, sx, sy, tx, ty, color, linestyle, lw)
            ax.text(
                (sx + tx) / 2 + nx * 3,
                (sy + ty) / 2 + ny * 3,
                f"{f_val:.1f}",
                fontsize=8,
                ha="center",
                va="center",
                color=text_color,
                fontweight=text_weight,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8),
                zorder=6,
            )

        for name, (x, y) in node_pos.items():
            ax.scatter(x, y, s=600, c="#2C91C2", edgecolors="white",
                       linewidth=1.5, zorder=10)
            ax.text(x, y, name, ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", zorder=11)

    ax_legend.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.5, label="TRGC genuine"),
            Line2D([0], [0], color="#777777", lw=1.0, linestyle="--",
                   label="TRGC reverse/inconcl."),
            Line2D([0], [0], color=OI_ORANGE, lw=1.5,
                   label=r"RRI$\to$PAT (Stim enhanced)"),
        ],
        loc="center left",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor="grey",
    )
    fig.text(0.02, 0.61, "B", fontsize=12, fontweight="bold")

    gs_bot = fig.add_gridspec(1, 1, left=0.13, right=0.92,
                              top=0.28, bottom=0.06)
    ax_c = fig.add_subplot(gs_bot[0])

    ci_overrides = load_bootstrap_ci()
    rows = []
    for src, tgt, metric in DIRECTIONS_ORDERED:
        row = sp_stats.loc[metric]
        dz = float(row["dz"])
        p_val = float(row["p_wilcoxon"])
        n_val = int(row.get("n", 16)) if "n" in row.index else 16
        if metric in ci_overrides:
            ci_lo, ci_hi = ci_overrides[metric]
            ci_method = "BCa"
        else:
            ci_lo, ci_hi = approx_ci(dz, n_val)
            ci_method = "normal"
        key = f"{src}_to_{tgt}"
        color = OI_ORANGE if key == "RRI_to_PTT" else GOLD if key == "SBP_to_RRI" else OI_GREY
        rows.append((key, dz, p_val, ci_lo, ci_hi, color, n_val))
        if key in ("RRI_to_PTT", "SBP_to_RRI"):
            print(
                f"  {key}: dz={dz:+.4f}, n={n_val}, "
                f"CI [{ci_method}]=[{ci_lo:+.3f}, {ci_hi:+.3f}]"
            )

    y_pos = np.arange(len(rows))
    ax_c.axvline(0, color="black", linewidth=0.6, linestyle="--")
    for i, (key, dz, p_val, ci_lo, ci_hi, color, _n_val) in enumerate(rows):
        ax_c.plot([ci_lo, ci_hi], [y_pos[i], y_pos[i]], color=color,
                  linewidth=2.2, solid_capstyle="butt")
        # JPG uses filled markers even for non-significant metrics; visual
        # significance is carried by the adjacent * / ** annotations.
        ax_c.plot(dz, y_pos[i], "o", markersize=6.5,
                  markerfacecolor=color, markeredgecolor=color,
                  markeredgewidth=1.5, zorder=5)
        if key == "RRI_to_PTT":
            ax_c.text(ci_hi + 0.04, y_pos[i], "**", fontsize=10,
                      va="center", color=OI_ORANGE, fontweight="bold")
        elif key == "SBP_to_RRI":
            ax_c.text(ci_hi + 0.04, y_pos[i], "*", fontsize=10,
                      va="center", color=GOLD, fontweight="bold")
        if key in TRGC_GENUINE:
            ax_c.text(min(ci_hi + 0.12, 1.50), y_pos[i], "G", fontsize=9,
                      va="center", color=OI_GREEN, fontweight="bold")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(DIRECTION_LABELS_ORDERED, fontsize=8)
    ax_c.set_xlabel(r"Effect size (Cohen's $d_z$, Stim $-$ Pre)", fontsize=9)
    ax_c.set_xlim(-0.75, 1.55)
    ax_c.invert_yaxis()
    ax_c.grid(axis="x", alpha=0.2)
    ax_c.text(-0.12, 1.08, "C", transform=ax_c.transAxes,
              fontsize=12, fontweight="bold", va="top")

    fig.savefig(OUT_DIR / "fig2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'fig2.png'}")


if __name__ == "__main__":
    main()
