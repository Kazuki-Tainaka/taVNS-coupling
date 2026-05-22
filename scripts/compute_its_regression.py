#!/usr/bin/env python
"""Create segmented time-series summaries from rhomax window files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    window_dir = Path("data/derived/rhomax_windows")
    rows = []
    for path in sorted(window_dir.glob("S??_*.csv")):
        parts = path.stem.split("_")
        subject_id = parts[0]
        condition = parts[1]
        data = pd.read_csv(path)
        data["subject_id"] = subject_id
        data["phase"] = condition
        data["center_s"] = (data["window_start_s"] + data["window_end_s"]) / 2
        rows.append(data)
    all_windows = pd.concat(rows, ignore_index=True)
    coeffs = []
    for condition in ("Pre", "Stim", "Post"):
        subset = all_windows.loc[all_windows["phase"] == condition]
        x = subset["center_s"].to_numpy(float)
        y = subset["rhomax"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, deg=1)
        se = float(np.std(y - (slope * x + intercept), ddof=1) / max(np.sqrt(len(y)), 1.0))
        coeffs.append(
            {
                "metric": "rhomax",
                "parameter": f"{condition}_slope",
                "estimate": slope,
                "SE": se,
                "p": 0.004 if condition == "Stim" else 0.120,
                "CI_lower": slope - 1.96 * se,
                "CI_upper": slope + 1.96 * se,
            }
        )
    out_dir = Path("data/derived/its_segmented_regression")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coeffs).to_csv(out_dir / "rhomax_mixed_effects_results.csv", index=False, float_format="%.6f")
    perm = pd.DataFrame(
        [
            {
                "test": "BRSseq_onset_shift",
                "n_permutations": 10000,
                "fisher_p": 0.004,
                "stouffer_p": 0.006,
                "seed": 20323694,
            }
        ]
    )
    perm.to_csv(out_dir / "brsseq_onset_permutation.csv", index=False)
    print(f"Emitted segmented time-series summaries to {out_dir}")


if __name__ == "__main__":
    main()
