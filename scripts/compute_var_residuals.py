#!/usr/bin/env python
"""Create residual covariance summaries from beat-to-beat public data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ("Pre", "Stim", "Post")


def residual_cov(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 4:
        return np.full((values.shape[1], values.shape[1]), np.nan)
    centered = values - values.mean(axis=0, keepdims=True)
    return np.cov(np.diff(centered, axis=0), rowvar=False)


def main() -> None:
    subjects = pd.read_csv("data/subjects.csv")
    out_dir = Path("data/derived/var_residuals")
    out_dir.mkdir(parents=True, exist_ok=True)
    bivar_rows: list[dict[str, object]] = []
    trivar_rows: list[dict[str, object]] = []
    for subject_id in subjects["subject_id"].astype(str):
        pat_ok = bool(subjects.loc[subjects["subject_id"] == subject_id, "pat_eligible"].iloc[0])
        for condition in CONDITIONS:
            path = Path("data/beats") / f"{subject_id}_{condition}.csv"
            beats = pd.read_csv(path)
            bi = residual_cov(beats[["RRI_ms", "SBP_mmHg"]].to_numpy(float))
            bivar_rows.append(
                {
                    "subject_id": subject_id,
                    "phase": condition,
                    "cov_RRI": bi[0, 0],
                    "cov_SBP": bi[1, 1],
                    "cov_cross": bi[0, 1],
                }
            )
            if pat_ok:
                tri = residual_cov(beats[["RRI_ms", "SBP_mmHg", "PAT_ms"]].to_numpy(float))
                trivar_rows.append(
                    {
                        "subject_id": subject_id,
                        "phase": condition,
                        "cov_RRI": tri[0, 0],
                        "cov_SBP": tri[1, 1],
                        "cov_PAT": tri[2, 2],
                        "cov_RRI_SBP": tri[0, 1],
                        "cov_RRI_PAT": tri[0, 2],
                        "cov_SBP_PAT": tri[1, 2],
                    }
                )
    pd.DataFrame(bivar_rows).to_csv(out_dir / "bivariate_residual_covariance.csv", index=False, float_format="%.6f")
    pd.DataFrame(trivar_rows).to_csv(out_dir / "trivariate_residual_covariance.csv", index=False, float_format="%.6f")
    print(f"Emitted VAR residual covariance summaries to {out_dir}")


if __name__ == "__main__":
    main()
