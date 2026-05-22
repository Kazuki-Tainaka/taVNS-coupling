"""Generate Figure 3 from bundled filter-comparison data.

Data dependencies:
  - figures/data/rhomax_final_comparison.csv
  - figures/data/fixed_lag_results.csv
  - figures/data/filtfilt_fixed_lag_results.csv
  - figures/data/pdi_causal_results.csv
  - figures/data/filtfilt_pdi_results.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG_DATA = ROOT / "figures" / "data"
DATA_DIR = FIG_DATA
ANALYSIS_SENSITIVITY_DIR = FIG_DATA
LEGACY_OUTPUTS_DIR = FIG_DATA
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
        "svg.fonttype": "none",
        "svg.image_inline": True,
        "svg.hashsalt": None,
    }
)

W_DOUBLE = 170 / 25.4
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREY = "#999999"
PHASES = ["Pre", "Stim", "Post"]
LAGS = np.arange(-10, 11)
LAG_COLS = [f"r_{lag:+d}s" for lag in LAGS]


def sem(vals: np.ndarray) -> float:
    return float(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals))))


def compute_lag_stats(df: pd.DataFrame, phase: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["phase"].eq(phase)]
    means, sems = [], []
    for col in LAG_COLS:
        vals = sub[col].dropna().to_numpy(float)
        means.append(np.mean(vals))
        sems.append(sem(vals))
    return np.asarray(means), np.asarray(sems)


def draw_bracket(ax, x1: float, x2: float, y: float, text: str, fs: int = 9) -> None:
    ymin, ymax = ax.get_ylim()
    h = max((ymax - ymin) * 0.018, 0.005)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color="black")
    ax.text((x1 + x2) / 2, y + h * 1.5, text, ha="center",
            va="bottom", fontsize=fs, color="black", fontweight="bold" if "*" in text else "normal")


def get_delta_pdi(df: pd.DataFrame) -> np.ndarray:
    pre = df[df["phase"].eq("Pre")].set_index("subject")["PDI"]
    stim = df[df["phase"].eq("Stim")].set_index("subject")["PDI"]
    common = pre.index.intersection(stim.index)
    return (stim.loc[common] - pre.loc[common]).to_numpy(float)


def load_taumax_from_lag_profiles(
    df: pd.DataFrame,
    search_lags: np.ndarray = LAGS,
) -> tuple[np.ndarray, np.ndarray]:
    lag_cols = [f"r_{lag:+d}s" for lag in search_lags]
    missing = [col for col in lag_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing lag columns: {missing}")
    df = df.copy()
    df["taumax"] = df[lag_cols].apply(
        lambda row: search_lags[np.nanargmax(row.to_numpy(float))], axis=1
    )
    pre_tau = df[df["phase"].eq("Pre")]["taumax"].to_numpy(float)
    stim_tau = df[df["phase"].eq("Stim")]["taumax"].to_numpy(float)
    return pre_tau, stim_tau


def load_panel_d_tau(df_causal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, Path, str]:
    """Load Panel D tau_max values from the identified legacy wrap source."""
    candidates = [
        ANALYSIS_SENSITIVITY_DIR / "tau_wrap_results.csv",
        LEGACY_OUTPUTS_DIR / "tau_wrap_results.csv",
        DATA_DIR / "tau_wrap_results.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        tau_df = pd.read_csv(path)
        required = {"phase", "taumax_wrap_mean"}
        if not required.issubset(tau_df.columns):
            continue
        pre_tau = tau_df[tau_df["phase"].eq("Pre")]["taumax_wrap_mean"].dropna().to_numpy(float)
        stim_tau = tau_df[tau_df["phase"].eq("Stim")]["taumax_wrap_mean"].dropna().to_numpy(float)
        return pre_tau, stim_tau, path, "tau_wrap"

    pre_tau_raw, stim_tau_raw = load_taumax_from_lag_profiles(df_causal, LAGS)
    pre_tau = pre_tau_raw[(pre_tau_raw >= -6) & (pre_tau_raw <= 6)]
    stim_tau = stim_tau_raw[(stim_tau_raw >= -6) & (stim_tau_raw <= 6)]
    return pre_tau, stim_tau, DATA_DIR / "fixed_lag_results.csv", "branch_b"


def plot_lag_subpanel(
    ax,
    df: pd.DataFrame,
    label: str,
    show_xlabel: bool,
    peak_label: str = "+1 s",
) -> None:
    pre_m, pre_s = compute_lag_stats(df, "Pre")
    stim_m, stim_s = compute_lag_stats(df, "Stim")
    ax.fill_between(LAGS, pre_m - pre_s, pre_m + pre_s, alpha=0.20, color=OI_BLUE)
    ax.plot(LAGS, pre_m, color=OI_BLUE, linewidth=1.5, label="Pre")
    ax.fill_between(LAGS, stim_m - stim_s, stim_m + stim_s, alpha=0.20, color=OI_ORANGE)
    ax.plot(LAGS, stim_m, color=OI_ORANGE, linewidth=1.5, label="Stim")
    ax.axvline(0, color="grey", linewidth=0.4, linestyle=":")
    ax.axhline(0, color="grey", linewidth=0.4)
    valid_positive = np.where((LAGS >= 0) & np.isfinite(stim_m))[0]
    if len(valid_positive) > 0:
        peak_idx = valid_positive[int(np.nanargmax(stim_m[valid_positive]))]
        peak_lag = int(LAGS[peak_idx])
        peak_val = float(stim_m[peak_idx])
        # Label refers to the absolute lag position of the Stim peak per
        # the v12 caption: "lag-specific tests identified the causal-filter
        # peak at +2 s and the zero-phase surviving peak at +1 s". Use the
        # `peak_label` argument to set this per panel from the caller.
        ax.annotate(
            peak_label,
            xy=(peak_lag, peak_val),
            xytext=(min(peak_lag + 2.5, 9.0), peak_val + 0.05),
            fontsize=8,
            color=OI_ORANGE,
            arrowprops=dict(arrowstyle="->", lw=0.8, color=OI_ORANGE),
        )
    ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=8,
            va="top", ha="left", fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.set_xlim(-10.5, 10.5)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_ylabel("Cross-correlation $r$")
    if show_xlabel:
        ax.set_xlabel("Lag (s)")
    else:
        ax.set_xticklabels([])


def main() -> None:
    print("Loading data...")
    rho = pd.read_csv(DATA_DIR / "rhomax_final_comparison.csv")
    df_causal = pd.read_csv(DATA_DIR / "fixed_lag_results.csv")
    df_filtfilt = pd.read_csv(DATA_DIR / "filtfilt_fixed_lag_results.csv")
    pdi_causal = pd.read_csv(DATA_DIR / "pdi_causal_results.csv")
    pdi_filtfilt = pd.read_csv(DATA_DIR / "filtfilt_pdi_results.csv")
    pre_tau, stim_tau, panel_d_source, panel_d_method = load_panel_d_tau(df_causal)
    print(
        f"Panel D tau_max: using {panel_d_method} source {panel_d_source}, "
        f"Pre n={len(pre_tau)}, Stim n={len(stim_tau)}"
    )
    print(
        f"  Pre median={np.median(pre_tau):+.2f} s, mean={np.mean(pre_tau):+.2f} s; "
        f"Stim median={np.median(stim_tau):+.2f} s, mean={np.mean(stim_tau):+.2f} s"
    )

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.85))
    ax_a, ax_b_holder = axes[0]
    ax_c, ax_d = axes[1]

    positions_c = np.array([0.8, 1.8, 2.8])
    positions_f = np.array([1.2, 2.2, 3.2])
    all_c = [rho[f"py_{phase}"].dropna().to_numpy(float) for phase in PHASES]
    all_f = [rho[f"fp_{phase}"].dropna().to_numpy(float) for phase in PHASES]
    means_c = [np.mean(vals) for vals in all_c]
    means_f = [np.mean(vals) for vals in all_f]
    ses_c = [sem(vals) for vals in all_c]
    ses_f = [sem(vals) for vals in all_f]

    ax_a.bar(positions_c, means_c, width=0.35, yerr=ses_c, capsize=3,
             color=OI_BLUE, alpha=0.7, edgecolor=OI_BLUE, linewidth=1.0,
             label="lfilter (causal)")
    ax_a.bar(positions_f, means_f, width=0.35, yerr=ses_f, capsize=3,
             facecolor="white", edgecolor=OI_BLUE, linewidth=1.0,
             label="filtfilt (zero-phase)")
    rng = np.random.default_rng(42)
    for positions, all_vals in ((positions_c, all_c), (positions_f, all_f)):
        for subj_idx in range(len(all_vals[0])):
            vals = [arr[subj_idx] if subj_idx < len(arr) else np.nan for arr in all_vals]
            if all(np.isfinite(vals)):
                ax_a.plot(positions, vals, color="grey", alpha=0.25, linewidth=0.4)
        for p_idx, vals in enumerate(all_vals):
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax_a.scatter(positions[p_idx] + jitter, vals, c="grey", s=8,
                         alpha=0.5, edgecolors="none", zorder=4)
    ax_a.set_xticks([1.0, 2.0, 3.0])
    ax_a.set_xticklabels(PHASES)
    ax_a.set_ylabel(r"$\rho_{\max}$")
    ax_a.grid(axis="y", alpha=0.2)
    ax_a.legend(fontsize=6, loc="upper right", frameon=False)
    ax_a.set_ylim(0.55, max(max(means_c), max(means_f)) + 0.13)
    draw_bracket(ax_a, positions_c[0], positions_c[1], max(means_c[0] + ses_c[0], means_c[1] + ses_c[1]) + 0.03, "***")
    draw_bracket(ax_a, positions_f[0], positions_f[1], max(means_f[0] + ses_f[0], means_f[1] + ses_f[1]) + 0.06, "n.s.", fs=8)
    ax_a.text(-0.15, 1.05, "A", transform=ax_a.transAxes,
              fontsize=12, fontweight="bold", va="top")

    gs_b = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax_b_holder.get_subplotspec(), hspace=0.12)
    ax_b_holder.remove()
    ax_b_top = fig.add_subplot(gs_b[0])
    ax_b_bot = fig.add_subplot(gs_b[1])
    plot_lag_subpanel(ax_b_top, df_causal, "lfilter (causal)", show_xlabel=False, peak_label="+2 s")
    plot_lag_subpanel(ax_b_bot, df_filtfilt, "filtfilt (zero-phase)", show_xlabel=True, peak_label="+1 s")
    ax_b_top.text(-0.15, 1.05, "B", transform=ax_b_top.transAxes,
                  fontsize=12, fontweight="bold", va="top")

    delta_c = get_delta_pdi(pdi_causal)
    delta_f = get_delta_pdi(pdi_filtfilt)
    mean_c, mean_f = np.mean(delta_c), np.mean(delta_f)
    se_c, se_f = sem(delta_c), sem(delta_f)
    ax_c.bar([0], [mean_c], width=0.6, yerr=[se_c], capsize=4,
             color=OI_BLUE, alpha=0.7, edgecolor=OI_BLUE, linewidth=1.0)
    ax_c.bar([1], [mean_f], width=0.6, yerr=[se_f], capsize=4,
             facecolor="white", edgecolor=OI_BLUE, linewidth=1.0)
    for i in range(min(len(delta_c), len(delta_f))):
        ax_c.plot([0, 1], [delta_c[i], delta_f[i]], color="grey",
                  alpha=0.3, lw=0.4, zorder=2)
    ax_c.scatter(np.zeros(len(delta_c)) + rng.uniform(-0.08, 0.08, len(delta_c)),
                 delta_c, c="grey", s=10, alpha=0.5, edgecolors="none", zorder=3)
    ax_c.scatter(np.ones(len(delta_f)) + rng.uniform(-0.08, 0.08, len(delta_f)),
                 delta_f, c="grey", s=10, alpha=0.5, edgecolors="none", zorder=3)
    ax_c.text(0, mean_c + se_c + 0.02, "***", ha="center", fontsize=10, fontweight="bold")
    ax_c.text(1, mean_f + se_f + 0.02, "*", ha="center", fontsize=10, fontweight="bold")
    ax_c.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["lfilter\n(causal)", "filtfilt\n(zero-phase)"], fontsize=8)
    ax_c.set_ylabel(r"$\Delta$PDI (Stim $-$ Pre)")
    ax_c.grid(axis="y", alpha=0.2)
    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes,
              fontsize=12, fontweight="bold", va="top")

    # v2.py-compliant binning and central tendency.
    # v2 uses linspace with 20 edges (~0.6 s width) and mean for the
    # reference lines. The mean of taumax_wrap_mean matches the JPG dashed
    # lines (Pre ~ +0.5, Stim ~ +1.7) whereas the median over-shifts.
    bins = np.linspace(-6, 6, 20)
    ax_d.hist(pre_tau, bins=bins, alpha=0.6, color=OI_BLUE, label="Pre")
    ax_d.hist(stim_tau, bins=bins, alpha=0.6, color=OI_ORANGE, label="Stim")
    ax_d.axvline(np.mean(pre_tau), color=OI_BLUE, linestyle="--", linewidth=1.2)
    ax_d.axvline(np.mean(stim_tau), color=OI_ORANGE, linestyle="--", linewidth=1.2)
    ax_d.axvline(0, color="grey", linewidth=0.5, linestyle=":")
    ax_d.set_xlim(-6.5, 6.5)
    ax_d.set_xlabel(r"$\tau_{\max}$ (s)")
    ax_d.set_ylabel("Count")
    ax_d.legend(fontsize=7, loc="upper left", frameon=False)
    ax_d.grid(axis="y", alpha=0.2)
    ax_d.text(-0.15, 1.05, "D", transform=ax_d.transAxes,
              fontsize=12, fontweight="bold", va="top")

    fig.tight_layout(h_pad=3, w_pad=3)
    fig.savefig(OUT_DIR / "fig3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'fig3.png'}")


if __name__ == "__main__":
    main()
