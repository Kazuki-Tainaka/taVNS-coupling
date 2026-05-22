"""Generate Figure 4 from bundled robustness data.

Data dependencies:
  - figures/data/supplementary_data_1.csv
  - figures/data/supplementary_data_2.csv
  - figures/data/coupling_raw.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
FIG_DATA = ROOT / "figures" / "data"
DATA_DIR = FIG_DATA
OUT_DIR = ROOT / "figures" / "outputs"

# Coupling 46 metrics = SD1, HRV 74 metrics = SD2
COUP_CSV = FIG_DATA / "supplementary_data_1.csv"
HRV_CSV = FIG_DATA / "supplementary_data_2.csv"

# ============================================================
# Style (Scientific Reports)
# ============================================================
STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'svg.fonttype': 'none',
}
plt.rcParams.update(STYLE)

W_DOUBLE = 170 / 25.4  # Scientific Reports double-column width (inches)

# Filter-dependent metrics to demote in Panel A
FILTER_DEPENDENT_A = ['rhomax_MATLAB']

# Panel C: 3 filter-independent metrics (display order bottom -> top)
PANEL_C_KEYS = [
    'BRS_seq_all',
    'GC3_F_RRI_to_PTT',
    'GC3_F_SBP_to_RRI',
]
PANEL_C_LABELS = [
    r'BRS$_{seq,all}$',
    r'GC3 RRI$\to$PAT',
    r'GC3 SBP$\to$RRI',
]
# Filter-dependent metrics: none remaining
PANEL_C_FILTER_DEP = set()


# ============================================================
# Helper: Cohen's dz
# ============================================================
def compute_dz(pre, stim):
    diff = stim - pre
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / sd if sd > 0 else 0.0


def compute_loo(pre_all, stim_all):
    n = len(pre_all)
    return np.array([compute_dz(np.delete(pre_all, i), np.delete(stim_all, i))
                     for i in range(n)])


def compute_bca_ci(pre_all, stim_all, B=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(pre_all)
    diff = stim_all - pre_all
    dz_obs = compute_dz(pre_all, stim_all)

    # Bootstrap
    dz_boot = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        d = diff[idx]
        sd = np.std(d, ddof=1)
        dz_boot[b] = np.mean(d) / sd if sd > 0 else 0.0

    # Bias correction z0
    prop = np.clip(np.mean(dz_boot < dz_obs), 1e-10, 1 - 1e-10)
    z0 = scipy_stats.norm.ppf(prop)

    # Acceleration (jackknife)
    dz_jack = compute_loo(pre_all, stim_all)
    d = np.mean(dz_jack) - dz_jack
    a = np.sum(d**3) / (6 * np.sum(d**2)**1.5) if np.sum(d**2) > 0 else 0.0

    # Adjusted percentiles
    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def adj(z):
        denom = 1 - a * (z0 + z)
        if abs(denom) < 1e-10:
            denom = 1e-10
        return scipy_stats.norm.cdf(z0 + (z0 + z) / denom)

    p_lo = np.clip(adj(z_lo), 0.5 / B, 1 - 0.5 / B)
    p_hi = np.clip(adj(z_hi), 0.5 / B, 1 - 0.5 / B)

    return float(np.percentile(dz_boot, p_lo * 100)), float(np.percentile(dz_boot, p_hi * 100))


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Figure 4: Verification of stimulation-associated changes")
    print("=" * 60)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print("\n[1] Loading data...")
    coup_df = pd.read_csv(COUP_CSV)
    print(f"    Coupling: {COUP_CSV.name} ({len(coup_df)} metrics)")

    hrv_df = pd.read_csv(HRV_CSV)
    print(f"    HRV: {HRV_CSV.name} ({len(hrv_df)} metrics)")

    raw = pd.read_csv(DATA_DIR / "coupling_raw.csv")
    print(f"    Raw: coupling_raw.csv ({raw.shape})")

    # ----------------------------------------------------------
    # Figure layout: 1-row, 3-column gridspec (horizontal)
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.45))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.65],
                           wspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # ==========================================================
    # Panel A: HRV vs Coupling dot + box
    # ==========================================================
    print("\n[2] Panel A: HRV vs Coupling contrast...")

    hrv_dz = hrv_df['dz_Stim_Pre'].dropna().values
    coup_dz = coup_df['dz_Stim_Pre'].dropna().values

    brs_mask = coup_df['Metric'] == 'BRS_seq_all'
    brs_dz = coup_df.loc[brs_mask, 'dz_Stim_Pre'].values[0]

    # Filter-independent FDR significance
    coup_fdr_indep = np.array([
        (pd.notna(fdr) and fdr < 0.05 and m not in FILTER_DEPENDENT_A)
        for fdr, m in zip(coup_df['p_FDR_Stim_Pre'].values,
                          coup_df['Metric'].values)
    ])
    n_hrv_fdr = int((hrv_df['p_FDR_Stim_Pre'].dropna() < 0.05).sum())
    n_coup_fdr_indep = int(coup_fdr_indep.sum())

    # Box plots
    bp = ax_a.boxplot(
        [hrv_dz, coup_dz], positions=[1, 2], widths=0.35,
        patch_artist=True, showfliers=False,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='grey', linewidth=0.8),
        capprops=dict(color='grey', linewidth=0.8),
        boxprops=dict(linewidth=0.8), zorder=2)
    bp['boxes'][0].set(facecolor='#DEEBF7', alpha=0.4, edgecolor='#6BAED6')
    bp['boxes'][1].set(facecolor='#FEE0D2', alpha=0.4, edgecolor='#FB6A4A')

    np.random.seed(42)

    # HRV dots (all light blue -- none FDR-significant)
    jh = np.random.uniform(-0.12, 0.12, len(hrv_dz))
    ax_a.scatter(1 + jh, hrv_dz, c='#6BAED6', s=14, alpha=0.5,
                 edgecolors='none', zorder=3)

    # Coupling: light dots (non-FDR or filter-dependent, excluding BRS)
    coup_light = (~coup_fdr_indep) & (~brs_mask.values)
    cl_dz = coup_df.loc[coup_light, 'dz_Stim_Pre'].dropna()
    jcl = np.random.uniform(-0.12, 0.12, len(cl_dz))
    ax_a.scatter(2 + jcl, cl_dz.values, c='#FB6A4A', s=14, alpha=0.4,
                 edgecolors='none', zorder=3)

    # Coupling: dark dots (FDR indep., non-BRS)
    coup_dark = coup_fdr_indep & (~brs_mask.values)
    cd_dz = coup_df.loc[coup_dark, 'dz_Stim_Pre'].dropna()
    jcd = np.random.uniform(-0.12, 0.12, len(cd_dz))
    ax_a.scatter(2 + jcd, cd_dz.values, c='#CB181D', s=35, alpha=0.85,
                 edgecolors='white', linewidth=0.4, zorder=4,
                 label='Filter-indep. FDR $q$ < 0.05')

    # BRS_seq marker and arrow annotation.
    ax_a.scatter(2, brs_dz, marker='o', c='#CB181D', s=70,
                 edgecolors='white', linewidth=0.5, zorder=6)
    ax_a.annotate(
        f'BRS$_{{seq,all}}$\n$d_z$={brs_dz:.2f}**',
        xy=(2.0, brs_dz), xytext=(1.32, -0.45),
        fontsize=6.5, fontweight='bold', color='#111111',
        arrowprops=dict(arrowstyle='->', color='#404040', lw=0.8,
                        connectionstyle='arc3,rad=0.15'),
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='none', alpha=0.75),
        ha='right', va='center', zorder=7)

    rho_mask = coup_df['Metric'] == 'rhomax_MATLAB'
    if rho_mask.any():
        rho_dz = float(coup_df.loc[rho_mask, 'dz_Stim_Pre'].values[0])
        ax_a.scatter(2, rho_dz, marker='o', facecolors='none',
                     edgecolors='#CB181D', linewidth=1.2, s=70, zorder=6)
        ax_a.annotate(
            r'$\rho_{\max}$' + '\n(filter-dep.)',
        xy=(2.0, rho_dz), xytext=(1.55, 0.62),
            fontsize=7, color='#404040',
            arrowprops=dict(arrowstyle='->', color='#404040', lw=0.8,
                            connectionstyle='arc3,rad=-0.15'),
            ha='right', va='center', zorder=7)

    ax_a.axhline(0, color='grey', ls='--', lw=0.5, alpha=0.5, zorder=1)

    # FDR summary text
    ax_a.text(1, 1.18, f'{n_hrv_fdr}/74\nFDR sig.',
              ha='center', va='bottom', fontsize=7, color='#2171B5',
              fontweight='bold')
    ax_a.text(2, 1.18, f'{n_coup_fdr_indep}/46\nFDR sig.\n(filter-indep.)',
              ha='center', va='bottom', fontsize=6.5, color='#CB181D',
              fontweight='bold')

    ax_a.set_xticks([1, 2])
    ax_a.set_xticklabels(['HRV\n(74 metrics)', 'Coupling\n(46 metrics)'])
    ax_a.set_ylabel(r'Effect size ($d_z$, Stim $-$ Pre)')
    ax_a.set_xlim(0.4, 3.1)
    ax_a.set_ylim(-0.95, 1.4)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.legend(loc='lower right', fontsize=5.4, framealpha=0.85,
                edgecolor='grey', handlelength=1.0, borderpad=0.3)
    ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes,
              fontsize=12, fontweight='bold', va='top')

    print(f"    HRV FDR sig: {n_hrv_fdr}/74")
    print(f"    Coupling FDR sig (filter-indep.): {n_coup_fdr_indep}/46")

    # ==========================================================
    # Panel B: BRS_seq spaghetti
    # ==========================================================
    print("\n[3] Panel B: BRS_seq individual trajectories...")

    brs_col = 'BRS_seq_all'
    phases = ['Pre', 'Stim', 'Post']
    subjects = sorted(raw['Subject'].unique())
    n_subj = len(subjects)

    # Individual traces (grey)
    decrease_count = 0
    for subj in subjects:
        vals = []
        for ph in phases:
            row = raw[(raw['Subject'] == subj) & (raw['Phase'] == ph)]
            vals.append(row[brs_col].values[0] if len(row) > 0 else np.nan)
        ax_b.plot([0, 1, 2], vals, color='#BFBFBF', linewidth=0.6,
                  alpha=0.5, zorder=1)
        ax_b.scatter([0, 1, 2], vals, c='#BFBFBF', s=8, alpha=0.4,
                     edgecolors='none', zorder=2)
        if len(vals) >= 2 and not np.isnan(vals[0]) and not np.isnan(vals[1]):
            if vals[1] < vals[0]:
                decrease_count += 1

    # Group mean +/- SE
    means, ses = [], []
    for ph in phases:
        v = raw[raw['Phase'] == ph][brs_col].dropna().values
        means.append(np.mean(v))
        ses.append(np.std(v, ddof=1) / np.sqrt(len(v)))

    ax_b.errorbar([0, 1, 2], means, yerr=ses, color='#E31A1C',
                  linewidth=2.0, capsize=4, capthick=1.5,
                  marker='o', markersize=6, zorder=5,
                  label='Group mean \u00b1 SE')

    # Annotation
    y_top_b = max(raw[raw['Phase'] == ph][brs_col].max() for ph in phases) * 0.82
    ax_b.text(1.0, y_top_b,
              f'{decrease_count}/{n_subj} decreased',
              fontsize=8, color='#E31A1C', fontweight='bold',
              ha='center', va='top')

    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(phases)
    ax_b.set_ylabel(r'BRS$_{seq}$ (ms mmHg$^{-1}$)')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.legend(loc='upper right', fontsize=6, framealpha=0.85)
    ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes,
              fontsize=12, fontweight='bold', va='top')

    print(f"    Subjects: {n_subj}")
    print(f"    Decreased Pre->Stim: {decrease_count}/{n_subj}")
    print(f"    Means: Pre={means[0]:.2f}, Stim={means[1]:.2f}, Post={means[2]:.2f}")

    # ==========================================================
    # Panel C: LOO/BCa (3 filter-independent metrics)
    # ==========================================================
    print("\n[4] Panel C: LOO/BCa robustness (3 filter-independent metrics, vertical)...")

    # Build dict of {metric_key: (pre_values, stim_values)}
    metric_data = {}
    for key in PANEL_C_KEYS:
        pre_s = raw[raw['Phase'] == 'Pre'].set_index('Subject')[key].dropna()
        stim_s = raw[raw['Phase'] == 'Stim'].set_index('Subject')[key].dropna()
        common = pre_s.index.intersection(stim_s.index)
        metric_data[key] = (pre_s.loc[common].values, stim_s.loc[common].values)

    results_c = {}
    for key, label in zip(PANEL_C_KEYS, PANEL_C_LABELS):
        pre_v, stim_v = metric_data[key]
        n = len(pre_v)
        dz = compute_dz(pre_v, stim_v)
        dz_loo = compute_loo(pre_v, stim_v)
        ci_lo, ci_hi = compute_bca_ci(pre_v, stim_v, B=10000)
        excl = (ci_lo > 0) or (ci_hi < 0)
        is_fdep = key in PANEL_C_FILTER_DEP

        results_c[key] = dict(
            label=label, dz=dz, n=n,
            loo_min=float(dz_loo.min()), loo_max=float(dz_loo.max()),
            bca_lo=ci_lo, bca_hi=ci_hi, excl=excl,
            filter_dep=is_fdep)

        dep_tag = ' [FILTER-DEP]' if is_fdep else ''
        print(f"    {label}: dz={dz:+.3f}, BCa=[{ci_lo:+.2f},{ci_hi:+.2f}], "
              f"LOO=[{dz_loo.min():+.2f},{dz_loo.max():+.2f}], "
              f"excl_zero={excl}, n={n}{dep_tag}")

    # Plot Panel C — vertical forest plot
    n_metrics = len(PANEL_C_KEYS)
    x_positions = np.arange(n_metrics)
    loo_color = '#FDAE6B'

    for idx, key in enumerate(PANEL_C_KEYS):
        r = results_c[key]
        x = x_positions[idx]

        # BCa CI bar (vertical)
        color_bar = '#404040' if r['excl'] else '#BFBFBF'
        ax_c.bar(x, r['bca_hi'] - r['bca_lo'], bottom=r['bca_lo'],
                 width=0.45, color=color_bar, alpha=0.85, zorder=3)

        # LOO range bar (narrower, behind)
        ax_c.bar(x, r['loo_max'] - r['loo_min'], bottom=r['loo_min'],
                 width=0.18, color=loo_color, alpha=0.85, zorder=2)

        # Marker (point estimate)
        mkr = 'D' if r['excl'] else 'o'
        mkr_c = '#404040' if r['excl'] else '#BFBFBF'
        mkr_ec = 'white' if r['excl'] else '#404040'
        ax_c.plot(x, r['dz'], marker=mkr, markersize=7, color=mkr_c,
                  markeredgecolor=mkr_ec, markeredgewidth=0.5, zorder=5)

    # Zero line (horizontal)
    ax_c.axhline(0, color='black', linewidth=0.6, zorder=1)

    # x-axis: metric labels
    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels(PANEL_C_LABELS, fontsize=7, rotation=30, ha='right')

    # y-axis: effect size (aligned with Panel A)
    ax_c.set_ylabel(r'$d_z$ (Stim $-$ Pre)')
    ax_c.set_ylim(-1.5, 1.5)
    ax_c.set_xlim(-0.6, n_metrics - 0.4)
    ax_c.grid(axis='y', alpha=0.2, linewidth=0.4, zorder=0)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Legend
    legend_el = [
        Patch(facecolor='#404040', alpha=0.85, label='BCa 95% CI'),
        Patch(facecolor=loo_color, alpha=0.85, label='LOO range'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#404040',
               markersize=6, markeredgecolor='white', label='CI excl. 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#BFBFBF',
               markersize=6, markeredgecolor='#404040', label='CI incl. 0'),
    ]
    ax_c.legend(handles=legend_el, loc='center left', bbox_to_anchor=(1.02, 0.72),
                fontsize=5.5, framealpha=0.85)

    ax_c.text(-0.15, 1.05, 'C', transform=ax_c.transAxes,
              fontsize=12, fontweight='bold', va='top')

    # ==========================================================
    # Save
    # ==========================================================
    png_path = OUT_DIR / "fig4.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print(f"Saved: {png_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
