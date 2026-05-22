"""Generate Figure 1 from bundled figure data.

Data dependencies:
  - figures/data/coupling_raw.csv
  - figures/data/coupling_coherence_raw.csv
  - figures/data/ar2_damping_results.csv
  - data/reference/Additional_File_2.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG_DATA = ROOT / "figures" / "data"
REFERENCE_DATA = ROOT / "data" / "reference"
DATA_DIR = FIG_DATA
SUP_DATA = FIG_DATA
OUT_DIR = ROOT / "figures" / "outputs"

STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
}
plt.rcParams.update(STYLE)

W_DOUBLE = 170 / 25.4

# Colors
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_RED = "#D55E00"
OI_GREY = "#999999"
PHASE_COLORS = [OI_BLUE, OI_ORANGE, OI_GREEN]

TYPE_COLORS = {"A": OI_RED, "B": OI_ORANGE, "C": "#F0C808", "D": OI_GREY}
TYPE_MARKERS = {"A": "o", "B": "o", "C": "o", "D": "o"}
TYPE_LABEL_TEMPLATES = {
    "A": "A: Persistent",
    "B": "B: Transient",
    "C": "C: Delayed",
    "D": "D: Unchanged",
}


def load_46_metric_data() -> tuple[pd.DataFrame, Path]:
    """Load the canonical 46-metric table for Panel D."""
    candidates = [
        REFERENCE_DATA / "Additional_File_2.csv",
        FIG_DATA / "supplementary_data_1.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return pd.read_csv(cand), cand
    raise FileNotFoundError("No 46-metric CSV found")


def visible_type_counts(sup: pd.DataFrame) -> dict[str, int]:
    """Count finite points visible in Panel D by temporal type."""
    counts = {}
    for ttype in ["A", "B", "C", "D"]:
        sub = sup[sup["Temporal_Type"].eq(ttype)]
        finite = sub.dropna(subset=["dz_Stim_Pre", "dz_Post_Pre"])
        counts[ttype] = len(finite)
    return counts


def draw_boxplot_panel(ax, data_by_condition, subjects, ylabel, title,
                       annotation, panel_label):
    """Draw boxplot with individual paired lines for 3 phases."""
    phases = ["Pre", "Stim", "Post"]
    bp = ax.boxplot(data_by_condition, labels=phases, widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], PHASE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    # Paired lines
    for s_idx in range(len(subjects)):
        vals = [data_by_condition[p][s_idx] if s_idx < len(data_by_condition[p])
                else np.nan for p in range(3)]
        if all(np.isfinite(v) for v in vals):
            ax.plot([1, 2, 3], vals, color='grey', alpha=0.3, linewidth=0.5)

    # Individual dots
    rng = np.random.default_rng(42)
    for p_idx in range(3):
        vals = data_by_condition[p_idx]
        jitter = rng.uniform(-0.1, 0.1, len(vals))
        ax.scatter(np.full(len(vals), p_idx + 1) + jitter, vals,
                   color=PHASE_COLORS[p_idx], s=15, alpha=0.6, zorder=5,
                   edgecolors='white', linewidth=0.3)

    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.2)

    # Bracket annotation between Pre and Stim, matching the Illustrator-finalized layout.
    vals = np.concatenate([np.asarray(v, dtype=float) for v in data_by_condition])
    vals = vals[np.isfinite(vals)]
    y_max = float(np.max(vals))
    y_min = float(np.min(vals))
    y_range = max(y_max - y_min, abs(y_max) * 0.1, 1e-6)
    y_bracket = y_max + 0.06 * y_range
    h = 0.018 * y_range
    ax.plot([1, 1, 2, 2], [y_bracket, y_bracket + h, y_bracket + h, y_bracket],
            lw=0.8, color="black", clip_on=False)
    ax.text(1.5, y_bracket + h * 1.5, annotation, ha="center", va="bottom",
            fontsize=8, color="black")
    ax.set_ylim(top=y_bracket + h * 4.0)

    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')


def main():
    print("Loading data...")

    # BRS_seq_all from the bundled coupling table.
    raw = pd.read_csv(DATA_DIR / "coupling_raw.csv")

    # Coh_mean from the bundled coherence table.
    p0 = pd.read_csv(DATA_DIR / "coupling_coherence_raw.csv")
    p0_gated = p0[p0["Mode"] == "gated"]

    # AR(2) zeta
    ar2 = pd.read_csv(DATA_DIR / "ar2_damping_results.csv")

    # 46-metric scatter data. Prefer the canonical table because the
    # bundled supplementary table has stale Temporal_Type labels for Panel D.
    sup, sup_path = load_46_metric_data()
    print(f"Loaded 46-metric data from: {sup_path}")
    type_counts = visible_type_counts(sup)
    type_labels_count = {
        t: f"{TYPE_LABEL_TEMPLATES[t]} ({type_counts[t]})"
        for t in ["A", "B", "C", "D"]
    }
    print(f"Type counts (both finite): {type_counts}")

    # --- Extract per-phase arrays ---
    phases = ["Pre", "Stim", "Post"]
    subjects_raw = sorted(raw["Subject"].unique())

    # Panel A: BRS_seq_all
    brs_data = []
    for ph in phases:
        sub = raw[raw["Phase"] == ph].sort_values("Subject")
        brs_data.append(sub["BRS_seq_all"].values)

    # Panel B: Coh_mean
    coh_data = []
    subjects_coh = sorted(p0_gated["Subject"].unique())
    for ph in phases:
        sub = p0_gated[p0_gated["Phase"] == ph].sort_values("Subject")
        coh_data.append(sub["Coh_mean"].values)

    # Panel C: zeta
    zeta_data = []
    subjects_ar2 = sorted(ar2["Subject"].unique())
    for ph in phases:
        sub = ar2[ar2["Phase"] == ph].sort_values("Subject")
        zeta_data.append(sub["zeta"].values)

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.8))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # Panel A: BRS_seq_all
    draw_boxplot_panel(
        ax_a, brs_data, subjects_raw,
        r"BRS (ms mmHg$^{-1}$)",
        "BRS (sequence method)",
        r"** $q$ = 5.26 $\times$ 10$^{-4}$",
        "A",
    )

    # Panel B: Coh_mean
    draw_boxplot_panel(
        ax_b, coh_data, subjects_coh,
        "Coherence",
        "Mayer-band coherence",
        "n.s.",
        "B",
    )

    # Panel C: AR(2) zeta
    draw_boxplot_panel(
        ax_c, zeta_data, subjects_ar2,
        r"Damping ratio $\zeta$",
        r"AR(2) damping ratio $\zeta$",
        "n.s.",
        "C",
    )

    # Panel D: 46-metric scatter
    ax = ax_d
    for _, row in sup.iterrows():
        m = row["Metric"]
        dz_sp = row.get("dz_Stim_Pre", np.nan)
        dz_pp = row.get("dz_Post_Pre", np.nan)
        ttype = row.get("Temporal_Type", "D")

        if pd.isna(dz_sp) or pd.isna(dz_pp):
            continue

        color = TYPE_COLORS.get(ttype, OI_GREY)
        marker = TYPE_MARKERS.get(ttype, "o")
        size = 60 if m == "BRS_seq_all" else 20
        zorder = 10 if m == "BRS_seq_all" else 3
        edgecolor = 'black' if m == "BRS_seq_all" else 'none'
        lw = 1.0 if m == "BRS_seq_all" else 0

        ax.scatter(dz_sp, dz_pp, c=color, marker=marker, s=size,
                   alpha=0.8, zorder=zorder, edgecolors=edgecolor,
                   linewidth=lw)

    # Reference lines
    ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.plot([-2, 2], [-2, 2], color='grey', linewidth=0.5, linestyle='--',
            alpha=0.3)

    ax.set_xlabel(r"$d_z$ (Stim $-$ Pre)")
    ax.set_ylabel(r"$d_z$ (Post $-$ Pre)")
    # Legend
    for ttype, label in type_labels_count.items():
        ax.scatter([], [], c=TYPE_COLORS[ttype], marker=TYPE_MARKERS[ttype],
                   s=20, label=label, alpha=0.8)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.2)
    brs_row = sup[sup["Metric"] == "BRS_seq_all"]
    if not brs_row.empty:
        brs = brs_row.iloc[0]
        brs_x = float(brs["dz_Stim_Pre"])
        brs_y = float(brs["dz_Post_Pre"])
        ax.annotate(r"BRS$_{\mathrm{seq}}$", xy=(brs_x, brs_y),
                    xytext=(brs_x - 0.45, brs_y - 0.35),
                    fontsize=8, ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    ax.text(-0.15, 1.05, "D", transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    fig.tight_layout(h_pad=3, w_pad=3)

    fig.savefig(OUT_DIR / "fig1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'fig1.png'}")


if __name__ == "__main__":
    main()
