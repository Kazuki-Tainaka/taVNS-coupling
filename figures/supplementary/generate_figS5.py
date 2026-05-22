"""Generate Supplementary Figure S5 from bundled segmented time-series data.

Data dependencies:
  - figures/data/its_timeseries.csv
  - figures/data/its_model_coefficients.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    path = OUT_DIR / "figs5.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return {"png": path}


def write_log(name: str, text: str) -> Path:
    return HERE / name


BASENAME = "figs5"
DATA_DIR = FIG_DATA
TS_CSV = DATA_DIR / "its_timeseries.csv"
COEF_CSV = DATA_DIR / "its_model_coefficients.csv"


def main() -> None:
    configure_style()
    ts = pd.read_csv(TS_CSV)
    coefs = pd.read_csv(COEF_CSV)
    ts["time_min"] = ts["window_center_s"] / 60.0

    grp = ts.groupby("time_min")["rhomax_3s"]
    mean_ts = grp.mean()
    se_ts = grp.sem()
    t = mean_ts.index.to_numpy()
    n = int(grp.count().iloc[0])

    rc = coefs[coefs["metric"].eq("rhomax_3s")].set_index("parameter")
    b0 = float(rc.loc["Intercept", "estimate"])
    b1 = float(rc.loc["time_min", "estimate"])
    b2 = float(rc.loc["D_stim", "estimate"])
    b3 = float(rc.loc["time_since_stim", "estimate"])
    b4 = float(rc.loc["D_post", "estimate"])
    b5 = float(rc.loc["time_since_post", "estimate"])
    p_b3 = float(rc.loc["time_since_stim", "p"])

    t_stim = 5.0
    t_post = 10.0
    grid = np.linspace(0, 15, 500)
    d_stim = (grid >= t_stim).astype(float)
    d_post = (grid >= t_post).astype(float)
    y_fit = b0 + b1 * grid + b2 * d_stim + b3 * d_stim * (grid - t_stim) + b4 * d_post + b5 * d_post * (grid - t_post)
    y_cf = b0 + b1 * grid

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.axvspan(0, t_stim, color="#0072B2", alpha=0.06)
    ax.axvspan(t_stim, t_post, color="#D55E00", alpha=0.07)
    ax.axvspan(t_post, 15, color="#009E73", alpha=0.06)
    ax.fill_between(t, mean_ts - se_ts, mean_ts + se_ts, color="#0072B2", alpha=0.18, linewidth=0)
    ax.plot(t, mean_ts, "o", color="#0072B2", ms=3, alpha=0.75)
    ax.plot(grid, y_cf, "--", color="#777777", lw=1.6, label="Counterfactual")
    ax.plot(grid[grid < t_stim], y_fit[grid < t_stim], color="#0072B2", lw=2.2)
    ax.plot(grid[(grid >= t_stim) & (grid < t_post)], y_fit[(grid >= t_stim) & (grid < t_post)], color="#D55E00", lw=2.2, label="ITS fit")
    ax.plot(grid[grid >= t_post], y_fit[grid >= t_post], color="#009E73", lw=2.2)

    for x in (t_stim, t_post):
        ax.axvline(x, color="black", ls=":", lw=0.8, alpha=0.7)
    for label, x, color in (("Pre", 2.5, "#0072B2"), ("Stim", 7.5, "#D55E00"), ("Post", 12.5, "#009E73")):
        ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(), ha="center", va="top", color=color, fontweight="bold")

    ax.annotate(
        rf"$\beta_3$ = {b3:.4f}/min" + "\n" + rf"$p$ = {p_b3:.4f}",
        xy=(7.5, np.interp(7.5, grid, y_fit)),
        xytext=(1.0, float(mean_ts.max()) + 0.05),
        fontsize=9,
        color="#D55E00",
        arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D55E00", "lw": 0.7},
    )

    ax.set_xlim(0, 15)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$\rho_{\max}$ ($\pm$3 s)")
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    paths = save_figure(fig, BASENAME)
    plt.close(fig)

    renumbering = "\n".join(
        [
            "# Supplementary figure renumbering",
            "",
            "- New Fig S5: rho_max ITS analysis (this figure).",
            "- Old Fig S5 -> New Fig S6.",
            "- Old Fig S6 -> New Fig S7.",
            "- Existing Fig S7 Type C effects remains the post-renumbered final supplementary figure if retained after S6/S7 shift.",
        ]
    )
    write_log("S_figures_renumbering.md", renumbering)

    log = [
        "# Fig S5 change log",
        "",
        "- Source: authoritative segmented time-series figure script.",
        "- Rebuilt into repository figure output paths.",
        "- Data read from bundled segmented time-series CSV files.",
        f"- Key ITS coefficient: beta3={b3:+.6f}/min, p={p_b3:.6f}, n={n}.",
        "- Renumbering note retained for traceability.",
        "",
        "## Output",
    ]
    log.extend(f"- {fmt}: `{path}`" for fmt, path in paths.items())
    write_log("figS5_change_log.md", "\n".join(log))
    print(f"Saved {BASENAME}: " + ", ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    main()
