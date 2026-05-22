"""Generate Supplementary Figure S3 from bundled WTC arrays.

Data dependencies:
  - figures/data/wavelet_coherence/*.npy
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WTC_DIR = ROOT / "figures" / "data" / "wavelet_coherence"
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


def main():
    print("Loading WTC group-mean arrays...")

    # Check if .npy files exist
    phases = ["Pre", "Stim", "Post"]
    missing = []
    for ph in phases:
        f = WTC_DIR / f"wtc_group_mean_{ph}.npy"
        if not f.exists():
            missing.append(str(f))
    if missing:
        print("ERROR: Missing .npy files. Run the WTC data export first.")
        for m in missing:
            print(f"  Missing: {m}")
        return

    freq = np.load(WTC_DIR / "wtc_freq.npy")
    t_axis = np.load(WTC_DIR / "wtc_time.npy")
    coi = np.load(WTC_DIR / "wtc_coi.npy")
    periods = 1.0 / freq

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, W_DOUBLE * 0.35),
                              sharey=True)

    target_len = len(t_axis)
    coi_axis = coi[:target_len] if len(coi) >= target_len else coi

    for idx, phase_name in enumerate(phases):
        ax = axes[idx]

        mean_wtc = np.load(WTC_DIR / f"wtc_group_mean_{phase_name}.npy")
        mean_sig = np.load(WTC_DIR / f"wtc_group_sigfrac_{phase_name}.npy")

        # Trim to common freq dimension
        nf = min(mean_wtc.shape[0], len(freq))
        mean_wtc = mean_wtc[:nf, :]
        mean_sig = mean_sig[:nf, :]
        periods_trim = periods[:nf]

        im = ax.pcolormesh(
            t_axis, np.log2(periods_trim), mean_wtc,
            cmap="jet", vmin=0, vmax=1, shading="auto",
        )

        # COI
        ax.fill_between(
            t_axis[:len(coi_axis)],
            np.log2(coi_axis[:len(t_axis)]),
            np.log2(periods_trim[-1]),
            alpha=0.3, color="white", hatch="//",
        )

        # Significance contour (>50% of subjects)
        if np.nanmax(mean_sig) > 0:
            try:
                ax.contour(
                    t_axis, np.log2(periods_trim), mean_sig,
                    levels=[0.5], colors="k", linewidths=0.8,
                )
            except Exception:
                pass

        # Mayer band reference lines
        ax.axhline(np.log2(1 / 0.12), color="white", ls="--", lw=0.8,
                   alpha=0.7)
        ax.axhline(np.log2(1 / 0.08), color="white", ls="--", lw=0.8,
                   alpha=0.7)

        ax.set_title(phase_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if idx == 0:
            ax.set_ylabel("Period (s)")

        # Panel label
        label = chr(65 + idx)  # A, B, C
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

        PERIOD_TICKS = [2, 4, 8, 16, 32, 64]
        yticks_log2 = [np.log2(p) for p in PERIOD_TICKS
                       if np.log2(periods_trim[0]) <= np.log2(p) <= np.log2(periods_trim[-1])]
        yticks_labels = [str(int(2**y)) for y in yticks_log2]
        ax.set_yticks(yticks_log2)
        ax.set_yticklabels(yticks_labels)
        ax.set_ylim(np.log2(periods_trim[0]), np.log2(periods_trim[-1]))
        ax.invert_yaxis()

    # Colorbar
    fig.subplots_adjust(right=0.88, wspace=0.08)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("WTC", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Annotation
    fig.text(0.5, -0.02,
             "Mayer band: $d_z$ = +0.04, $p$ = 0.77 (n.s.)    n = 18",
             ha='center', fontsize=7, color='#333333')

    fig.savefig(OUT_DIR / "figs3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'figs3.png'}")


if __name__ == "__main__":
    main()
