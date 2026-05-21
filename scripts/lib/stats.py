from __future__ import annotations

import numpy as np
import pingouin as pg
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests


def _paired_arrays(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def paired_test(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = _paired_arrays(x, y)
    diff = x - y
    if len(diff) == 0:
        return {"n": 0, "dz": np.nan, "p_t": np.nan, "p_wilcoxon": np.nan, "mean_diff": np.nan}
    sd = np.nanstd(diff, ddof=1)
    dz = float(np.nanmean(diff) / sd) if sd > 0 else np.nan
    p_t = scipy_stats.ttest_rel(x, y, nan_policy="omit").pvalue if len(diff) > 1 else np.nan
    try:
        p_w = scipy_stats.wilcoxon(diff).pvalue
    except ValueError:
        p_w = np.nan
    return {"n": int(len(diff)), "dz": dz, "p_t": float(p_t), "p_wilcoxon": float(p_w), "mean_diff": float(np.nanmean(diff))}


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    _, q, _, _ = multipletests(np.asarray(p_values, dtype=float), method="fdr_bh")
    return q


def bf01_paired(x: np.ndarray, y: np.ndarray, prior_scale: float = 0.707) -> float:
    x, y = _paired_arrays(x, y)
    diff = x - y
    if len(diff) < 2:
        return np.nan
    sd = np.nanstd(diff, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.inf
    t_value = float(np.nanmean(diff) / sd * np.sqrt(len(diff)))
    bf10 = float(pg.bayesfactor_ttest(t_value, len(diff), paired=True, alternative="two-sided", r=prior_scale))
    return np.inf if bf10 == 0 else 1.0 / bf10


def bca_ci(diff: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 20250520) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(diff, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha)))


def loo_robustness(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = _paired_arrays(x, y)
    dz_values = []
    n_significant = 0
    for i in range(len(x)):
        res = paired_test(np.delete(x, i), np.delete(y, i))
        dz_values.append(res["dz"])
        if np.isfinite(res["p_wilcoxon"]) and res["p_wilcoxon"] < 0.05:
            n_significant += 1
    return {"n_iterations": int(len(x)), "n_significant": int(n_significant), "dz_range": (float(np.nanmin(dz_values)), float(np.nanmax(dz_values)))}
