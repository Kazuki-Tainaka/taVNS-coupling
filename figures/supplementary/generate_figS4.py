"""Generate Supplementary Figure S4 from bundled lag profiles.

Data dependencies:
  - figures/data/fixed_lag_results.csv
  - figures/data/filtfilt_fixed_lag_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "figures" / "data"
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

# Lag columns: r_-10s ... r_+10s
LAGS = list(range(-10, 11))
LAG_COLS = [f"r_{l:+d}s" for l in LAGS]

# Okabe-Ito palette
OI_BLUE = "#0072B2"
OI_RED = "#D55E00"
OI_GREEN = "#009E73"


def load_lag_profiles(filepath: Path) -> pd.DataFrame:
    """Load lag profile CSV, return DataFrame with Subject, Phase, lag columns."""
    df = pd.read_csv(filepath)
    return df


def compute_group_stats(df: pd.DataFrame, phase: str):
    """Compute group mean and SEM for each lag."""
    sub = df[df["phase"] == phase]
    means = []
    sems = []
    for col in LAG_COLS:
        if col in sub.columns:
            vals = sub[col].dropna().values
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    return np.array(means), np.array(sems)


def draw_panel(ax, df, title):
    """Draw lag profile panel for one filter type."""
    pre_mean, pre_sem = compute_group_stats(df, "Pre")
    stim_mean, stim_sem = compute_group_stats(df, "Stim")
    post_mean, post_sem = compute_group_stats(df, "Post")

    lags = np.array(LAGS)

    # Pre
    ax.fill_between(lags, pre_mean - pre_sem, pre_mean + pre_sem,
                    alpha=0.2, color=OI_BLUE)
    ax.plot(lags, pre_mean, color=OI_BLUE, linewidth=1.5,
            marker='o', markersize=3, label="Pre")

    # Stim
    ax.fill_between(lags, stim_mean - stim_sem, stim_mean + stim_sem,
                    alpha=0.2, color=OI_RED)
    ax.plot(lags, stim_mean, color=OI_RED, linewidth=1.5,
            marker='o', markersize=3, label="Stim")

    # Post
    if np.any(np.isfinite(post_mean)):
        ax.fill_between(lags, post_mean - post_sem, post_mean + post_sem,
                        alpha=0.2, color=OI_GREEN)
        ax.plot(lags, post_mean, color=OI_GREEN, linewidth=1.5,
                marker='o', markersize=3, label="Post")

    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)

    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Mean $r$(lag) $\\pm$ SE")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-10.5, 10.5)
    ax.grid(axis='y', alpha=0.2)


def main():
    print("Loading lag profile data...")
    df_causal = load_lag_profiles(DATA_DIR / "fixed_lag_results.csv")
    df_filtfilt = load_lag_profiles(DATA_DIR / "filtfilt_fixed_lag_results.csv")

    # Verify lag columns exist
    for col in LAG_COLS[:3]:
        if col not in df_causal.columns:
            # Try alternative naming
            print(f"  Warning: column {col} not found in causal data")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.45),
                                      sharey=True)

    draw_panel(ax_a, df_causal, "Causal (lfilter)")
    draw_panel(ax_b, df_filtfilt, "Zero-phase (filtfilt)")

    fig.tight_layout(w_pad=3)

    fig.savefig(OUT_DIR / "figs4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'figs4.png'}")


if __name__ == "__main__":
    main()
