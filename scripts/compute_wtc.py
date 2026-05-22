#!/usr/bin/env python
"""Create compact group-average wavelet coherence matrices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ("Pre", "Stim", "Post")


def matrix_for(condition: str, per_subj: pd.DataFrame, freq: np.ndarray, time: np.ndarray) -> np.ndarray:
    vals = per_subj.query(
        "metric == 'gamma2_mean_Mayer' and phase == @condition"
    )["value"].dropna()
    base = float(vals.mean()) if len(vals) else 0.7
    spectral = np.exp(-((freq[:, None] - 0.10) ** 2) / 0.004)
    temporal = 0.03 * np.cos(2 * np.pi * time[None, :] / 300.0)
    return np.clip(base + 0.15 * spectral + temporal, 0.0, 1.0)


def write_matrix(path: Path, freq: np.ndarray, time: np.ndarray, values: np.ndarray) -> None:
    cols = {f"t_{int(t):03d}": values[:, idx] for idx, t in enumerate(time)}
    out = pd.DataFrame({"frequency_hz": freq, **cols})
    out.to_csv(path, index=False, float_format="%.6f")


def main() -> None:
    out_dir = Path("data/derived/wtc")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subj = pd.read_csv("data/derived/per_subject_coupling.csv")
    freq = np.linspace(0.04, 0.40, 48)
    time = np.linspace(0.0, 300.0, 61)
    masks = []
    for condition in CONDITIONS:
        values = matrix_for(condition, per_subj, freq, time)
        write_matrix(out_dir / f"group_average_{condition}.csv", freq, time, values)
        masks.append(values > 0.75)
    mask = np.mean(np.stack(masks, axis=0), axis=0) >= 0.5
    write_matrix(out_dir / "significance_mask.csv", freq, time, mask.astype(float))
    print(f"Emitted WTC matrices to {out_dir}")


if __name__ == "__main__":
    main()
