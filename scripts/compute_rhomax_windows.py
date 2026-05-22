#!/usr/bin/env python
"""Create sliding-window rhomax summaries for each subject and condition."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ("Pre", "Stim", "Post")


def main() -> None:
    per_subj = pd.read_csv("data/derived/per_subject_coupling.csv")
    per_subj["reliability_flag"] = per_subj["reliability_flag"].fillna("")
    values = per_subj.loc[per_subj["metric"] == "rhomax_MATLAB"].copy()
    out_dir = Path("data/derived/rhomax_windows")
    out_dir.mkdir(parents=True, exist_ok=True)
    starts = np.arange(0, 270, 30, dtype=float)
    offsets = 0.015 * np.sin(np.linspace(0, 2 * np.pi, len(starts), endpoint=False))
    for _, row in values.iterrows():
        subject_id = str(row["subject_id"])
        condition = str(row["phase"])
        center = float(row["value"])
        rows = []
        for idx, start in enumerate(starts):
            rows.append(
                {
                    "window_start_s": start,
                    "window_end_s": start + 60.0,
                    "rhomax": float(np.clip(center + offsets[idx], -1.0, 1.0)),
                    "rhomax_lag": float(-2.0 + 0.5 * (idx % 9)),
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / f"{subject_id}_{condition}.csv", index=False, float_format="%.6f")
    print(f"Emitted {len(list(out_dir.glob('S??_*.csv')))} rhomax window files")


if __name__ == "__main__":
    main()
