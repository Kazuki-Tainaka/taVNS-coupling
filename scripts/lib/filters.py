from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, lfilter


def make_butter_lowpass(fc: float, fs: float, order: int = 4):
    return butter(order, fc / (fs / 2.0), btype="low")


def causal_filter(x: np.ndarray, fc: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = make_butter_lowpass(fc, fs, order)
    return lfilter(b, a, np.asarray(x, dtype=float))


def zerophase_filter(x: np.ndarray, fc: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = make_butter_lowpass(fc, fs, order)
    return filtfilt(b, a, np.asarray(x, dtype=float))
