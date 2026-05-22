"""Generate Supplementary Figure S1 from canonical coupling data.

Data dependencies:
  - data/reference/Additional_File_2.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG_DATA = ROOT / "figures" / "data"
OUT_DIR = ROOT / "figures" / "outputs"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
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


def save_figure(fig, basename: str) -> dict[str, Path]:
    path = OUT_DIR / "figs1.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return {"png": path}


def write_log(name: str, text: str) -> Path:
    return HERE / name


BASENAME = "figs1"

_CANDIDATE_PATHS = [
    ROOT / "data" / "reference" / "Additional_File_2.csv",
    FIG_DATA / "supplementary_data_1.csv",
]
SD1_CSV = next((p for p in _CANDIDATE_PATHS if p.exists()), _CANDIDATE_PATHS[-1])

CATEGORY_ORDER = [
    "Phase Synchrony",
    "Coherence",
    "BRS (Spectral)",
    "BRS (Sequence)",
    "BEI",
    "BPRSA",
    "Granger Causality (P0)",
    "Spectral GC",
    "Trivariate GC",
    "PDC",
    "VAR Residuals",
    "Innovation Variance",
    "CGN",
    "PTT",
]

FILTER_DEPENDENT = {"rhomax_MATLAB"}

METRIC_LABELS = {
    "rhomax_MATLAB": r"$\rho_{\max}$ cross-correlation",
    "Coh_mean": "Mean coherence (Mayer)",
    "BRS_TF_mean": "BRS transfer function mean",
    "BRS_seq_all": "BRS sequence (all ramps)",
    "BRS_seq_down": "BRS sequence (down ramps)",
    "BRS_seq_up": "BRS sequence (up ramps)",
    "BEI_all": "BEI (all)",
    "BEI_down": "BEI (down)",
    "BEI_up": "BEI (up)",
    "AC": "Acceleration capacity",
    "DC": "Deceleration capacity",
    "DC_AC_ratio": "DC/AC ratio",
    "GC_F_BP_to_RRI": r"GC: BP $\to$ RRI",
    "GC_F_RRI_to_SBP": r"GC: RRI $\to$ SBP",
    "SGC_RRI_to_SBP_HF": r"SGC: RRI $\to$ SBP HF",
    "SGC_RRI_to_SBP_LF": r"SGC: RRI $\to$ SBP LF",
    "SGC_RRI_to_SBP_Mayer": r"SGC: RRI $\to$ SBP Mayer",
    "SGC_RRI_to_SBP_VLF": r"SGC: RRI $\to$ SBP VLF",
    "SGC_RRI_to_SBP_total": r"SGC: RRI $\to$ SBP total",
    "SGC_SBP_to_RRI_HF": r"SGC: SBP $\to$ RRI HF",
    "SGC_SBP_to_RRI_LF": r"SGC: SBP $\to$ RRI LF",
    "SGC_SBP_to_RRI_Mayer": r"SGC: SBP $\to$ RRI Mayer",
    "SGC_SBP_to_RRI_VLF": r"SGC: SBP $\to$ RRI VLF",
    "SGC_SBP_to_RRI_total": r"SGC: SBP $\to$ RRI total",
    "GC3_F_PTT_to_RRI": r"GC3: PAT $\to$ RRI",
    "GC3_F_PTT_to_SBP": r"GC3: PAT $\to$ SBP",
    "GC3_F_RRI_to_PTT": r"GC3: RRI $\to$ PAT",
    "GC3_F_RRI_to_SBP": r"GC3: RRI $\to$ SBP",
    "GC3_F_SBP_to_PTT": r"GC3: SBP $\to$ PAT",
    "GC3_F_SBP_to_RRI": r"GC3: SBP $\to$ RRI",
    "PDC_PTT_to_RRI_Mayer": r"PDC: PAT $\to$ RRI",
    "PDC_PTT_to_SBP_Mayer": r"PDC: PAT $\to$ SBP",
    "PDC_RRI_to_SBP_Mayer": r"PDC: RRI $\to$ SBP",
    "PDC_SBP_to_RRI_Mayer": r"PDC: SBP $\to$ RRI",
    "rho_u": "Residual cross-correlation",
    "sigma2_RRI": r"Innovation var. $\sigma^2$ (RRI)",
    "sigma2_SBP": r"Innovation var. $\sigma^2$ (SBP)",
    "Snn_RRI_Mayer": "RRI noise spectrum (Mayer)",
    "innov_RRI_from_SBP_frac_Mayer": "SBP-driven RRI fraction",
    "innov_RRI_own_frac_Mayer": "RRI self-innovation fraction",
    "SNR_openloop_Mayer": "Open-loop SNR (Mayer)",
    "SNR_openloop_Mayer_dB": "Open-loop SNR dB (Mayer)",
    "Snn_Syy_ratio_Mayer": "Noise/output ratio (Mayer)",
    "Snn_openloop_Mayer": "Open-loop noise (Mayer)",
    "gamma2_mean_Mayer": r"$\gamma^2$ mean coherence (VAR)",
    "PTT_mean": "Mean pulse arrival time",
}

CATEGORY_LABELS = {
    "Phase Synchrony": "Phase sync.",
    "Coherence": "Coherence",
    "BRS (Spectral)": "BRS (spectral)",
    "BRS (Sequence)": "BRS (sequence)",
    "BEI": "BEI",
    "BPRSA": "BPRSA",
    "Granger Causality (P0)": "GC (bivariate)",
    "Spectral GC": "Spectral GC",
    "Trivariate GC": "Trivariate GC",
    "PDC": "PDC",
    "VAR Residuals": "VAR residuals",
    "Innovation Variance": "Innovation var.",
    "CGN": "CGN",
    "PTT": "PAT",
}


def main() -> None:
    configure_style()
    sup = pd.read_csv(SD1_CSV)
    sup["_cat_rank"] = sup["Category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)}).fillna(99)
    sup = sup.sort_values(["_cat_rank", "Metric"]).reset_index(drop=True)

    metrics = sup["Metric"].tolist()
    display_labels = [METRIC_LABELS.get(m, m) for m in metrics]
    n_metrics = len(metrics)
    heatmap = sup[["dz_Stim_Pre", "dz_Post_Pre"]].to_numpy(float)
    fdr = sup[["p_FDR_Stim_Pre", "p_FDR_Post_Pre"]].to_numpy(float)
    bf01_sp = sup["BF01_Stim_Pre"].to_numpy(float)
    bf01_pp = (
        sup["BF01_Post_Pre"].to_numpy(float)
        if "BF01_Post_Pre" in sup.columns
        else np.full(n_metrics, np.nan)
    )

    fig_h = max(9.5, n_metrics * 0.21)
    fig, ax = plt.subplots(figsize=(7.1, fig_h))
    norm = TwoSlopeNorm(vmin=-1.05, vcenter=0, vmax=1.05)
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    for i, metric in enumerate(metrics):
        for j in range(2):
            dz = heatmap[i, j]
            if not np.isfinite(dz):
                continue
            parts = [f"{dz:+.2f}"]
            if np.isfinite(fdr[i, j]) and fdr[i, j] < 0.05:
                parts.append("*")
            if j == 0 and np.isfinite(bf01_sp[i]) and bf01_sp[i] > 3:
                parts.append("(o)")
            if j == 1 and np.isfinite(bf01_pp[i]) and bf01_pp[i] > 3:
                parts.append("(o)")
            color = "white" if abs(dz) > 0.50 else "black"
            weight = "bold" if "*" in parts else "normal"
            ax.text(j, i, " ".join(parts), ha="center", va="center", fontsize=6.5, color=color, fontweight=weight)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stim - Pre", "Post - Pre"], fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(display_labels, fontsize=6.6)

    # Category separators and labels.
    categories = sup["Category"].tolist()
    starts = []
    current = None
    for i, cat in enumerate(categories):
        if cat != current:
            starts.append((i, cat))
            if i > 0:
                ax.axhline(i - 0.5, color="white", lw=1.8)
            current = cat

    trans = ax.get_yaxis_transform()
    for idx, (start, cat) in enumerate(starts):
        end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else n_metrics - 1
        mid = (start + end) / 2
        cat_display = CATEGORY_LABELS.get(cat, cat)
        ax.text(1.08, mid, cat_display, transform=trans, va="center", ha="left", fontsize=6.7, color="#333333", fontstyle="italic")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="2%", pad=0.5)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Effect size dz")

    n_fdr_total = int(np.nansum(fdr < 0.05))
    n_fdr_stim = int(np.nansum(fdr[:, 0] < 0.05))
    n_filter_indep = int(sum((sup["p_FDR_Stim_Pre"] < 0.05) & (~sup["Metric"].isin(FILTER_DEPENDENT))))
    n_filter_dep = int(sum((sup["p_FDR_Stim_Pre"] < 0.05) & (sup["Metric"].isin(FILTER_DEPENDENT))))
    ax.text(
        0,
        -0.025,
        "* FDR $q$ < 0.05      (o) BF$_{01}$ > 3 (moderate null evidence)      "
        "$n$ = 18 ($n$ = 12-16 for gated metrics)",
        transform=ax.transAxes,
        fontsize=6.7,
        ha="left",
        va="top",
    )

    paths = save_figure(fig, BASENAME)
    plt.close(fig)

    log = [
        "# Fig S1 change log",
        "",
        "- Source: authoritative supplementary figure script.",
        f"- Data source: {SD1_CSV.name} (canonical CSV with BF01_Stim_Pre + BF01_Post_Pre and p_FDR_Stim_Pre + p_FDR_Post_Pre).",
        f"- Stim-Pre FDR markers: {n_fdr_stim} total = {n_filter_indep} filter-independent + {n_filter_dep} filter-dependent.",
        "- BF01>3 markers are now rendered for both Stim-Pre and Post-Pre columns.",
        "",
        "## Output",
    ]
    log.extend(f"- {fmt}: `{path}`" for fmt, path in paths.items())
    write_log("figS1_change_log.md", "\n".join(log))
    print(f"Saved {BASENAME}: " + ", ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    main()
