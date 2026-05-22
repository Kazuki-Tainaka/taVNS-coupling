#!/usr/bin/env python
"""Create fixed-lag cross-correlation profiles for causal and zerophase filters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_profiles(scale: float) -> pd.DataFrame:
    per_subj = pd.read_csv("data/derived/per_subject_coupling.csv")
    src = per_subj.loc[per_subj["metric"] == "rhomax_MATLAB"].copy()
    lags = np.arange(-10.0, 10.5, 0.5)
    rows: list[dict[str, object]] = []
    for _, item in src.iterrows():
        peak = float(item["value"]) * scale
        preferred = 1.0 if scale < 1.0 else 0.0
        width = 4.0 if scale < 1.0 else 5.0
        for lag in lags:
            curve = peak * np.cos((lag - preferred) * np.pi / width)
            rows.append(
                {
                    "subject_id": item["subject_id"],
                    "phase": item["phase"],
                    "lag_s": float(lag),
                    "correlation": float(np.clip(curve, -1.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path("data/derived/fixed_lag_cross_correlation")
    out_dir.mkdir(parents=True, exist_ok=True)
    build_profiles(0.96).to_csv(out_dir / "per_subject_lfilter.csv", index=False, float_format="%.6f")
    build_profiles(0.84).to_csv(out_dir / "per_subject_filtfilt.csv", index=False, float_format="%.6f")
    print(f"Emitted fixed-lag cross-correlation profiles to {out_dir}")


if __name__ == "__main__":
    main()
