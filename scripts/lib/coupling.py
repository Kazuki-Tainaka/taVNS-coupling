from __future__ import annotations

import numpy as np
from scipy import signal, stats as scipy_stats
from statsmodels.tsa.api import VAR

from .filters import causal_filter, zerophase_filter
from .quality import validate_pat


def _paired_stat(pre_values: np.ndarray, stim_values: np.ndarray) -> dict:
    from . import stats

    result = stats.paired_test(np.asarray(stim_values, dtype=float), np.asarray(pre_values, dtype=float))
    return {
        "pre_brs": float(np.nanmean(pre_values)) if len(pre_values) else np.nan,
        "stim_brs": float(np.nanmean(stim_values)) if len(stim_values) else np.nan,
        "dz": result["dz"],
        "p_value": result["p_wilcoxon"],
        "n_eligible": result["n"],
    }


def brs_seq(rri: np.ndarray, sbp: np.ndarray, slope_min: float = 2.5) -> dict[str, float]:
    rri = np.asarray(rri, dtype=float)
    sbp = np.asarray(sbp, dtype=float)
    n = min(len(rri), len(sbp))
    if n < 4:
        return {"BRS_seq_all": np.nan, "BRS_seq_up": np.nan, "BRS_seq_down": np.nan}
    slopes_up, slopes_down = [], []
    for i in range(n - 2):
        s = sbp[i:i + 3]
        r = rri[i:i + 3]
        if np.all(np.diff(s) > 0) or np.all(np.diff(s) < 0):
            slope = scipy_stats.linregress(s, r).slope
            if np.isfinite(slope) and abs(slope) >= slope_min:
                (slopes_up if np.diff(s).mean() > 0 else slopes_down).append(float(slope))
    all_slopes = slopes_up + slopes_down
    return {
        "BRS_seq_all": float(np.nanmean(all_slopes)) if all_slopes else np.nan,
        "BRS_seq_up": float(np.nanmean(slopes_up)) if slopes_up else np.nan,
        "BRS_seq_down": float(np.nanmean(slopes_down)) if slopes_down else np.nan,
    }


def compute_brsseq(
    rri_pre: np.ndarray,
    sbp_pre: np.ndarray,
    rri_stim: np.ndarray,
    sbp_stim: np.ndarray,
    mode: str = "all",
) -> dict:
    """Sequence-method BRS for Pre versus Stim.

    Wraps the sequence-method check calculation and returns a paired-effect
    summary. `mode` is one of `all`, `up`, or `down`.
    """
    key_map = {"all": "BRS_seq_all", "up": "BRS_seq_up", "down": "BRS_seq_down"}
    if mode not in key_map:
        raise ValueError("mode must be one of 'all', 'up', or 'down'")
    key = key_map[mode]
    pre = brs_seq(rri_pre, sbp_pre, slope_min=0.0)[key]
    stim = brs_seq(rri_stim, sbp_stim, slope_min=0.0)[key]
    return _paired_stat(np.asarray([pre], dtype=float), np.asarray([stim], dtype=float))


def brs_tf(rri: np.ndarray, sbp: np.ndarray, fs: float) -> dict:
    rri = np.asarray(rri, dtype=float)
    sbp = np.asarray(sbp, dtype=float)
    n = min(len(rri), len(sbp))
    if n < 16:
        return {"BRS_TF_mean": np.nan, "Coh_mean": np.nan}
    f, pxy = signal.csd(rri[:n], sbp[:n], fs=fs, nperseg=min(256, n))
    _, pxx = signal.welch(sbp[:n], fs=fs, nperseg=min(256, n))
    _, coh = signal.coherence(rri[:n], sbp[:n], fs=fs, nperseg=min(256, n))
    band = (f >= 0.08) & (f <= 0.12)
    gain = np.abs(pxy[band]) / pxx[band]
    return {"BRS_TF_mean": float(np.nanmean(gain)), "Coh_mean": float(np.nanmean(coh[band]))}


def rhomax(x: np.ndarray, y: np.ndarray, filter_type: str = "causal", lag_range: tuple[int, int] = (-5, 5), fs: float = 4.0) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 10:
        return {"rhomax": np.nan, "lag_at_max": np.nan}
    filt = causal_filter if filter_type == "causal" else zerophase_filter
    xx = filt(x[:n], fc=0.12, fs=fs)
    yy = filt(y[:n], fc=0.12, fs=fs)
    best_r, best_lag = np.nan, 0
    for lag in range(lag_range[0], lag_range[1] + 1):
        if lag < 0:
            a, b = xx[-lag:], yy[:lag]
        elif lag > 0:
            a, b = xx[:-lag], yy[lag:]
        else:
            a, b = xx, yy
        if len(a) > 3:
            r = np.corrcoef(a, b)[0, 1]
            if np.isfinite(r) and (not np.isfinite(best_r) or abs(r) > abs(best_r)):
                best_r, best_lag = float(r), lag
    return {"rhomax": float(best_r), "lag_at_max": int(best_lag)}


def compute_rhomax_mayer(
    rri: np.ndarray,
    sbp: np.ndarray,
    fs: float = 4.0,
    filter_mode: str = "causal",
) -> float:
    """Mayer-band peak cross-correlation check helper.

    `filter_mode='causal'` follows the canonical direction; `zerophase`
    is available for sensitivity analysis.
    """
    mode = "zerophase" if filter_mode in {"zerophase", "filtfilt"} else "causal"
    return float(rhomax(rri, sbp, filter_type=mode, fs=fs)["rhomax"])


def granger_bivariate(x: np.ndarray, y: np.ndarray, max_order: int = 10) -> dict:
    arr = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) <= max_order + 2:
        return {"F": np.nan, "p": np.nan, "order": np.nan}
    fit = VAR(arr).fit(maxlags=max_order, ic="aic")
    test = fit.test_causality(caused=1, causing=[0], kind="f")
    return {"F": float(test.test_statistic), "p": float(test.pvalue), "order": int(fit.k_ar)}


def compute_gc_f_bivariate(x: np.ndarray, y: np.ndarray, order: int | str = "auto") -> dict:
    """Bivariate Granger F-statistics for both directions."""
    max_order = 20 if order == "auto" else int(order)
    xy = granger_bivariate(x, y, max_order=max_order)
    yx = granger_bivariate(y, x, max_order=max_order)
    return {
        "F_x_to_y": xy["F"],
        "p_x_to_y": xy["p"],
        "F_y_to_x": yx["F"],
        "p_y_to_x": yx["p"],
        "order_x_to_y": xy["order"],
        "order_y_to_x": yx["order"],
    }


def granger_trivariate(x: np.ndarray, y: np.ndarray, z: np.ndarray, max_order: int = 10) -> dict:
    if not validate_pat(np.asarray(z, dtype=float)):
        return {"F": np.nan, "p": np.nan, "order": np.nan}
    arr = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)])
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) <= max_order + 3:
        return {"F": np.nan, "p": np.nan, "order": np.nan}
    fit = VAR(arr).fit(maxlags=max_order, ic="aic")
    test = fit.test_causality(caused=1, causing=[0], kind="f")
    return {"F": float(test.test_statistic), "p": float(test.pvalue), "order": int(fit.k_ar)}


def compute_gc3_f_trivariate(
    rri: np.ndarray,
    sbp: np.ndarray,
    pat: np.ndarray,
    order: int | str = "auto",
) -> dict:
    """Trivariate GC3 F-statistics for all six directions."""
    if not validate_pat(pat):
        raise ValueError("PAT values failed quality gate")
    max_order = 20 if order == "auto" else int(order)
    data = {
        "rri": np.asarray(rri, dtype=float),
        "sbp": np.asarray(sbp, dtype=float),
        "pat": np.asarray(pat, dtype=float),
    }
    out: dict[str, float] = {}
    for source in ("rri", "sbp", "pat"):
        for target in ("rri", "sbp", "pat"):
            if source == target:
                continue
            predictors = [data[source], data[target]]
            other = [k for k in data if k not in {source, target}][0]
            res = granger_trivariate(predictors[0], predictors[1], data[other], max_order=max_order)
            key = f"{source}_to_{target}"
            out[f"F_{key}"] = res["F"]
            out[f"p_{key}"] = res["p"]
    return out


def trgc(x: np.ndarray, y: np.ndarray, max_order: int = 10) -> dict:
    fwd = granger_bivariate(x, y, max_order=max_order)
    rev = granger_bivariate(np.asarray(x)[::-1], np.asarray(y)[::-1], max_order=max_order)
    return {"F_forward": fwd["F"], "F_reversed": rev["F"], "delta_F": fwd["F"] - rev["F"]}
