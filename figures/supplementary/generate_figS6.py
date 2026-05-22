"""Generate Supplementary Figure S6 from bundled tracking data.

Data dependencies:
  - figures/data/tracking_group_rm90.csv
  - figures/data/tracking_individual/*.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "figures" / "data"
OUT_DIR = ROOT / "figures" / "outputs"

# Okabe-Ito palette
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_RED = "#D55E00"
OI_GREY = "#999999"
OI_BLACK = "#000000"

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


def compute_threshold(times, values, pre_start=150.0, pre_end=300.0, k=2.5):
    """Compute threshold from late Pre baseline."""
    mask = (times >= pre_start) & (times < pre_end) & ~np.isnan(values)
    pre_vals = values[mask]
    if len(pre_vals) < 10:
        return np.nan, np.nan, np.nan
    mu = np.mean(pre_vals)
    sd = np.std(pre_vals, ddof=1)
    return mu, sd, mu + k * sd


def load_individual_ts(pattern):
    """Load individual time series files matching glob pattern."""
    ts_dir = DATA_DIR / "tracking_individual"
    all_ts = []
    for f in sorted(ts_dir.glob(pattern)):
        df = pd.read_csv(f)
        if "rm_90s" in df.columns:
            all_ts.append(df[["time_s", "rm_90s"]].values)
    return all_ts


def draw_ts_panel(ax, times, group_mean, group_sem, individual_ts,
                  threshold, title, color, n_label, group_label):
    """Draw time series panel."""
    # Phase shading
    ax.axvspan(0, 300, alpha=0.05, color=OI_BLUE)
    ax.axvspan(300, 600, alpha=0.1, color=OI_RED)
    ax.axvspan(600, 900, alpha=0.05, color=OI_GREEN)

    # Individual traces
    for ts in individual_ts:
        t, v = ts[:, 0], ts[:, 1]
        ax.plot(t, v, color=color, alpha=0.1, linewidth=0.3)

    # Group mean +/- SEM
    valid = ~np.isnan(group_mean)
    ax.fill_between(times[valid],
                    (group_mean - group_sem)[valid],
                    (group_mean + group_sem)[valid],
                    alpha=0.3, color=color)
    ax.plot(times[valid], group_mean[valid], color=color, linewidth=2)

    # Threshold line
    if not np.isnan(threshold):
        ax.axhline(threshold, color=OI_BLACK, linewidth=1.5, linestyle="--",
                   label="Threshold (Pre mean + 2.5 SD)")

    # Phase boundaries
    ax.axvline(300, color=OI_BLACK, linewidth=0.5, linestyle=":")
    ax.axvline(600, color=OI_BLACK, linewidth=0.5, linestyle=":")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("r(+1s) running mean (90s)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    legend_elements = [
        Patch(facecolor=OI_BLUE, alpha=0.15, label="Pre"),
        Patch(facecolor=OI_RED, alpha=0.15, label="Stim"),
        Patch(facecolor=OI_GREEN, alpha=0.15, label="Post"),
        Line2D([0], [0], color=color, lw=2,
               label=f"{group_label} ($n$ = {n_label})"),
        Line2D([0], [0], color=OI_BLACK, lw=1.5, linestyle="--",
               label="Threshold (Pre mean + 2.5 SD)"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="lower right",
              framealpha=0.85)
    ax.set_xlim(60, 840)
    ax.grid(axis="y", alpha=0.2)


def main():
    print("Loading tracking data...")

    # Load group time series
    df_group = pd.read_csv(DATA_DIR / "tracking_group_rm90.csv")
    times = df_group["time_s"].values
    tavns_mean = df_group["tavns_mean"].values
    tavns_sem = df_group["tavns_sem"].values
    vc_mean = df_group["vc_mean"].values
    vc_sem = df_group["vc_sem"].values

    # Load individual time series
    tavns_individual = load_individual_ts("ts_cumul_S*.csv")
    vc_individual = load_individual_ts("ts_cumul_vc_*.csv")

    print(f"  taVNS: {len(tavns_individual)} subjects")
    print(f"  VC: {len(vc_individual)} subjects")

    # Compute thresholds
    _, _, threshold_tavns = compute_threshold(times, tavns_mean, 150, 300, 2.5)
    _, _, threshold_vc = compute_threshold(times, vc_mean, 150, 300, 2.5)

    # Per-subject threshold crossing rates
    rng = np.random.default_rng(42)

    def crossing_rates(ts_list):
        rates = []
        for ts in ts_list:
            t, v = ts[:, 0], ts[:, 1]
            _, _, thr = compute_threshold(t, v, 150, 300, 2.5)
            if np.isnan(thr):
                continue
            s_mask = (t >= 300) & (t < 600) & ~np.isnan(v)
            above = np.sum(v[s_mask] > thr)
            total = np.sum(s_mask)
            rates.append(above / total * 100 if total > 0 else 0)
        return rates

    tavns_rates = crossing_rates(tavns_individual)
    vc_rates = crossing_rates(vc_individual)

    # --- Figure ---
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.75))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                          height_ratios=[1, 1])

    # Panel A: taVNS
    ax_a = fig.add_subplot(gs[0, :])
    draw_ts_panel(ax_a, times, tavns_mean, tavns_sem, tavns_individual,
                  threshold_tavns,
                  f"taVNS cohort ($n$ = {len(tavns_individual)}): causal-filter $r$(+1s)",
                  OI_BLUE, len(tavns_individual), "taVNS")
    ax_a.text(-0.05, 1.05, "A", transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # Panel B: VC
    ax_b = fig.add_subplot(gs[1, 0])
    draw_ts_panel(ax_b, times, vc_mean, vc_sem, vc_individual,
                  threshold_vc,
                  f"Virtual Control ($n$ = {len(vc_individual)}): identical pipeline",
                  OI_ORANGE, len(vc_individual), "VC")
    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # Panel C: Bar chart
    ax_c = fig.add_subplot(gs[1, 1])
    bar_means = [np.mean(tavns_rates) if tavns_rates else 0,
                 np.mean(vc_rates) if vc_rates else 0]
    bar_sems = [np.std(tavns_rates) / np.sqrt(len(tavns_rates))
                if len(tavns_rates) > 1 else 0,
                np.std(vc_rates) / np.sqrt(len(vc_rates))
                if len(vc_rates) > 1 else 0]

    bars = ax_c.bar([0, 1], bar_means, yerr=bar_sems, width=0.5,
                    color=[OI_BLUE, OI_ORANGE], alpha=0.6,
                    edgecolor=[OI_BLUE, OI_ORANGE], linewidth=1.5,
                    capsize=5)

    # Individual dots
    if tavns_rates:
        jitter = rng.uniform(-0.12, 0.12, len(tavns_rates))
        ax_c.scatter(np.zeros(len(tavns_rates)) + jitter, tavns_rates,
                     color=OI_BLUE, alpha=0.7, s=20, zorder=5,
                     edgecolors="white", linewidth=0.5)
    if vc_rates:
        jitter = rng.uniform(-0.12, 0.12, len(vc_rates))
        ax_c.scatter(np.ones(len(vc_rates)) + jitter, vc_rates,
                     color=OI_ORANGE, alpha=0.7, s=20, zorder=5,
                     edgecolors="white", linewidth=0.5)

    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([f"taVNS\n(n={len(tavns_rates)})",
                          f"VC\n(n={len(vc_rates)})"])
    ax_c.set_ylabel("% samples above threshold\n(300-600 s)")
    ax_c.set_title("Threshold-Crossing Rate\n(not significantly different)",
                   fontsize=10, fontweight="bold")
    ax_c.grid(axis="y", alpha=0.3)
    ax_c.set_ylim(0, 105)

    # n.s. annotation
    ax_c.text(0.5, max(bar_means) + max(bar_sems) + 10, "n.s.",
              ha="center", fontsize=11, fontweight="bold", color=OI_GREY)
    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # Save
    fig.savefig(OUT_DIR / "figs6.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'figs6.png'}")
    print(f"  taVNS crossing: {np.mean(tavns_rates):.1f}%")
    print(f"  VC crossing: {np.mean(vc_rates):.1f}%")


if __name__ == "__main__":
    main()
