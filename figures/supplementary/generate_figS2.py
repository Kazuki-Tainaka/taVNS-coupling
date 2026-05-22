"""Generate Supplementary Figure S2 from canonical HRV data.

Data dependencies:
  - figures/data/hrv_type_classification.csv
  - data/reference/Additional_File_3.csv
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ===== GLOBAL STYLE CONFIG (match Figure S3) =====
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
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.5,
}
plt.rcParams.update(STYLE)

W_DOUBLE = 170 / 25.4  # mm to inches

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "figures" / "data"
FIG_OUT = REPO_ROOT / "figures" / "outputs"

AF3_PATH_CANDIDATES = [
    REPO_ROOT / "data" / "reference" / "Additional_File_3.csv",
]

# Expected MD5 of the v2 corrected Additional_File_3.csv.
# Computed once and recorded here as a guard against accidentally loading
# the older canonical CSV (MD5 c6bf4816b45165ef24a458a151c50d54), whose
# BF01 column was generated with the deprecated dz-as-t convention.
AF3_MD5 = '5fd37ceb5269c0558131a02efbb6ba95'


def load_af3_bf01() -> pd.DataFrame:
    """Load corrected BF01 columns from Additional_File_3.csv.

    Returns a DataFrame indexed by Metric with two columns:
        BF01_Stim_Pre, BF01_Post_Pre

    Raises FileNotFoundError if no candidate path exists.
    Emits a warning (not a failure) if the MD5 does not match
    AF3_MD5, so that legacy / mirror copies are still usable
    but flagged for review.
    """
    import hashlib

    for path in AF3_PATH_CANDIDATES:
        if path.exists():
            md5 = hashlib.md5(path.read_bytes()).hexdigest()
            if md5 != AF3_MD5:
                print(
                    f"WARNING: {path.name} MD5 = {md5}\n"
                    f"         expected {AF3_MD5}.\n"
                    f"         The BF01 column may follow the deprecated\n"
                    f"         dz-as-t convention; (o) markers in the figure\n"
                    f"         may not match the published statistics."
                )
            df = pd.read_csv(path)
            return df.set_index('Metric')[['BF01_Stim_Pre', 'BF01_Post_Pre']]

    raise FileNotFoundError(
        "Could not locate Additional_File_3.csv at any of:\n  "
        + "\n  ".join(str(p) for p in AF3_PATH_CANDIDATES)
    )

# ===== Metric order by category (from instructions) =====
METRIC_ORDER = [
    # Time domain (17)
    'Mean_RRI', 'SDNN', 'RMSSD', 'pNN50', 'pNN20', 'Mean_HR', 'SDHR',
    'CV_RRI', 'Median_RRI', 'IQR_RRI', 'SDSD', 'NN50', 'NN20', 'TINN',
    'SI', 'MinHR', 'MaxHR',
    # Frequency domain (14)
    'VLF_power', 'LF_power', 'HF_power', 'Total_power', 'LF_norm',
    'HF_norm', 'LF_HF_ratio', 'LF_peak_freq', 'lnLF', 'lnHF', 'lnTP',
    'LF_pct', 'HF_pct', 'HF_peak_freq',
    # Geometric/Poincare (6)
    'SD1', 'SD2', 'SD1_SD2_ratio', 'CSI', 'CVI', 'Triangular_index',
    # PRSA (4)
    'DC', 'AC', 'DCmod', 'ACmod',
    # Entropy (9)
    'SampEn', 'ApEn', 'FuzzyEn', 'PermEn', 'MSE_slope', 'ShannonEn',
    'DispEn', 'MSE_CI', 'LZC',
    # Fractal/Complexity (9)
    'DFA_alpha1', 'DFA_alpha2', 'DFA_ratio', 'Hurst_exp', 'Lyap_max',
    'Corr_dim', 'HFD', 'KFD', 'beta_1f',
    # RQA (8)
    'RQA_REC', 'RQA_DET', 'RQA_Lmean', 'RQA_Lmax', 'RQA_DIV',
    'RQA_ShanEn', 'RQA_LAM', 'RQA_TT',
    # Symbolic dynamics (4)
    'Symb_0V', 'Symb_1V', 'Symb_2LV', 'Symb_2UV',
    # Irreversibility (3)
    'GI', 'PI', 'EIR',
]

CATEGORY_MAP = {}
_cats = [
    ('Time domain', METRIC_ORDER[0:17]),
    ('Frequency domain', METRIC_ORDER[17:31]),
    ('Geometric/Poincar\u00e9', METRIC_ORDER[31:37]),
    ('PRSA', METRIC_ORDER[37:41]),
    ('Entropy', METRIC_ORDER[41:50]),
    ('Fractal/Complexity', METRIC_ORDER[50:59]),
    ('RQA', METRIC_ORDER[59:67]),
    ('Symbolic dynamics', METRIC_ORDER[67:71]),
    ('Irreversibility', METRIC_ORDER[71:74]),
]
for cat_name, metric_list in _cats:
    for m in metric_list:
        CATEGORY_MAP[m] = cat_name

CATEGORY_ORDER = [c for c, _ in _cats]

METRIC_LABELS_HRV = {
    "Mean_RRI": "Mean RRI",
    "SDNN": "SDNN",
    "RMSSD": "RMSSD",
    "pNN50": "pNN50",
    "pNN20": "pNN20",
    "Mean_HR": "Mean HR",
    "SDHR": "SDHR",
    "CV_RRI": "CV RRI",
    "Median_RRI": "Median RRI",
    "IQR_RRI": "IQR RRI",
    "SDSD": "SDSD",
    "NN50": "NN50",
    "NN20": "NN20",
    "TINN": "TINN",
    "SI": "SI",
    "MinHR": "MinHR",
    "MaxHR": "MaxHR",
    "VLF_power": "VLF power",
    "LF_power": "LF power",
    "HF_power": "HF power",
    "Total_power": "Total power",
    "LF_norm": "LF norm.",
    "HF_norm": "HF norm.",
    "LF_HF_ratio": "LF/HF ratio",
    "LF_peak_freq": "LF peak freq.",
    "lnLF": "lnLF",
    "lnHF": "lnHF",
    "lnTP": "lnTP",
    "LF_pct": "LF%",
    "HF_pct": "HF%",
    "HF_peak_freq": "HF peak freq.",
    "SD1": "SD1",
    "SD2": "SD2",
    "SD1_SD2_ratio": "SD1/SD2 ratio",
    "CSI": "CSI",
    "CVI": "CVI",
    "Triangular_index": "Tri. index",
    "DC": "DC",
    "AC": "AC",
    "DCmod": "DCmod",
    "ACmod": "ACmod",
    "SampEn": "SampEn",
    "ApEn": "ApEn",
    "FuzzyEn": "FuzzyEn",
    "PermEn": "PermEn",
    "MSE_slope": "MSE slope",
    "ShannonEn": "ShannonEn",
    "DispEn": "DispEn",
    "MSE_CI": "MSE CI",
    "LZC": "LZC",
    "DFA_alpha1": r"DFA $\alpha_1$",
    "DFA_alpha2": r"DFA $\alpha_2$",
    "DFA_ratio": r"DFA $\alpha_1/\alpha_2$",
    "Hurst_exp": "Hurst exp.",
    "Lyap_max": "Lyap. max",
    "Corr_dim": "Corr. dim.",
    "HFD": "HFD",
    "KFD": "KFD",
    "beta_1f": r"$\beta$(1/f)",
    "RQA_REC": "RQA REC",
    "RQA_DET": "RQA DET",
    "RQA_Lmean": "RQA Lmean",
    "RQA_Lmax": "RQA Lmax",
    "RQA_DIV": "RQA DIV",
    "RQA_ShanEn": "RQA ShanEn",
    "RQA_LAM": "RQA LAM",
    "RQA_TT": "RQA TT",
    "Symb_0V": "Symb 0V",
    "Symb_1V": "Symb 1V",
    "Symb_2LV": "Symb 2LV",
    "Symb_2UV": "Symb 2UV",
    "GI": "GI",
    "PI": "PI",
    "EIR": "EIR",
}


def main():
    # --- Load data ---
    # dz / p values: per-metric long-format pipeline output (unchanged).
    tc = pd.read_csv(DATA_DIR / 'hrv_type_classification.csv')
    tc_lookup = tc.set_index('Metric')

    # BF01 values: corrected source, indexed by Metric, wide-format.
    # See module docstring for the convention-fix rationale (2026-05-22).
    af3_bf01 = load_af3_bf01()

    # --- Build ordered arrays ---
    n_metrics = len(METRIC_ORDER)
    dz_sp = np.full(n_metrics, np.nan)
    dz_pp = np.full(n_metrics, np.nan)
    p_sp = np.full(n_metrics, 1.0)
    p_pp = np.full(n_metrics, 1.0)
    bf01_sp = np.full(n_metrics, 0.0)
    bf01_pp = np.full(n_metrics, 0.0)

    for i, m in enumerate(METRIC_ORDER):
        if m in tc_lookup.index:
            row = tc_lookup.loc[m]
            dz_sp[i] = row['dz_Stim_Pre']
            dz_pp[i] = row['dz_Post_Pre']
            p_sp[i] = row['p_Stim_Pre']
            p_pp[i] = row['p_Post_Pre']
        if m in af3_bf01.index:
            bf01_sp[i] = af3_bf01.at[m, 'BF01_Stim_Pre']
            bf01_pp[i] = af3_bf01.at[m, 'BF01_Post_Pre']

    heatmap = np.column_stack([dz_sp, dz_pp])

    # --- Figure ---
    fig_h = max(12, n_metrics * 0.24)
    fig, ax = plt.subplots(figsize=(W_DOUBLE, fig_h))

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', norm=norm,
                   interpolation='nearest')

    # --- Cell annotations ---
    # The (o) marker indicates moderate evidence for the null hypothesis
    # (BF01 > 3 per the JZS / Jeffreys convention). The threshold is
    # strictly greater than 3; cells with BF01 = 3.00 do NOT receive (o).
    for i in range(n_metrics):
        for j in range(2):
            dz_val = heatmap[i, j]
            if np.isnan(dz_val):
                continue

            bf_val = bf01_sp[i] if j == 0 else bf01_pp[i]

            txt = f'{dz_val:+.2f}'

            has_null = bf_val > 3

            if has_null:
                txt += ' (o)'

            color = 'white' if abs(dz_val) > 0.5 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7, color=color)

    # --- Y-axis labels ---
    display_labels = [METRIC_LABELS_HRV.get(m, m) for m in METRIC_ORDER]
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(display_labels, fontsize=8)

    # --- X-axis ---
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Stim \u2212 Pre', 'Post \u2212 Pre'], fontsize=10,
                       fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # --- Category separators (white lines) and right-side labels ---
    categories = [CATEGORY_MAP[m] for m in METRIC_ORDER]
    prev_cat = None
    cat_boundaries = []
    for i, cat in enumerate(categories):
        if cat != prev_cat:
            if prev_cat is not None:
                ax.axhline(i - 0.5, color='white', linewidth=2.0)
            cat_boundaries.append((i, cat))
            prev_cat = cat

    # Right-side bracket + label (blended transform)
    blend = ax.get_yaxis_transform()
    bracket_x = 1.01
    tick_w = 0.015
    label_x = bracket_x + tick_w + 0.005

    for idx, (start, cat) in enumerate(cat_boundaries):
        if idx + 1 < len(cat_boundaries):
            end = cat_boundaries[idx + 1][0] - 1
        else:
            end = n_metrics - 1
        mid = (start + end) / 2
        y_top = start - 0.4
        y_bot = end + 0.4

        ax.plot([bracket_x, bracket_x], [y_top, y_bot],
                color='#888888', lw=0.6, clip_on=False, transform=blend)
        ax.plot([bracket_x, bracket_x + tick_w], [y_top, y_top],
                color='#888888', lw=0.6, clip_on=False, transform=blend)
        ax.plot([bracket_x, bracket_x + tick_w], [y_bot, y_bot],
                color='#888888', lw=0.6, clip_on=False, transform=blend)

        ax.text(label_x, mid, cat, ha='left', va='center',
                fontsize=7.5, color='#333333', fontstyle='italic',
                clip_on=False, transform=blend)

    # --- Horizontal colorbar below ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='1.8%', pad=0.6)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(r'Effect size ($d_z$)', fontsize=9, labelpad=4)
    cbar.ax.tick_params(labelsize=8)

    # --- Legend ---
    ax.text(0.0, -0.003,
            r'(o) $\mathrm{BF_{01}}$ > 3 (moderate null evidence)       '
            'n = 18',
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
            color='#333333')

    # --- Save ---
    outpath = FIG_OUT / 'figs2.png'
    fig.savefig(outpath, format='png', dpi=300)
    print(f'Saved: {outpath}')

    # --- Diagnostic summary (printed to stdout for run-time verification) ---
    sp_with_o = int(np.sum(bf01_sp > 3))
    pp_with_o = int(np.sum(bf01_pp > 3))
    print(f'(o) marker counts: Stim-Pre {sp_with_o}/{n_metrics}, '
          f'Post-Pre {pp_with_o}/{n_metrics}, '
          f'total {sp_with_o + pp_with_o}/{2 * n_metrics}')
    print('Expected (AF3, MD5 5fd37c...): 44, 48, 92')

    plt.close(fig)


if __name__ == '__main__':
    main()
