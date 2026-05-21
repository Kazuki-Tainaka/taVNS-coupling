from __future__ import annotations

import numpy as np


def mean_nn(rri_ms: np.ndarray) -> float:
    return float(np.nanmean(rri_ms))


def sdnn(rri_ms: np.ndarray) -> float:
    return float(np.nanstd(rri_ms, ddof=1))


def rmssd(rri_ms: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.diff(np.asarray(rri_ms, dtype=float)) ** 2)))


def pnn50(rri_ms: np.ndarray) -> float:
    return float(np.nanmean(np.abs(np.diff(np.asarray(rri_ms, dtype=float))) > 50.0) * 100.0)


def compute_basic_hrv(rri_ms: np.ndarray) -> dict[str, float]:
    return {"Mean_RRI": mean_nn(rri_ms), "SDNN": sdnn(rri_ms), "RMSSD": rmssd(rri_ms), "pNN50": pnn50(rri_ms)}
