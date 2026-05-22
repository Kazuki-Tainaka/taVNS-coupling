#!/usr/bin/env python
"""Detect BRS sequence ramps in public beat-to-beat data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ("Pre", "Stim", "Post")


def count_events(beats: pd.DataFrame) -> dict[str, int]:
    sbp = beats["SBP_mmHg"].to_numpy(float)
    rri = beats["RRI_ms"].to_numpy(float)
    out = {"n_ramps_up": 0, "n_ramps_down": 0, "n_brs_events_up": 0, "n_brs_events_down": 0}
    for idx in range(len(sbp) - 3):
        s = sbp[idx : idx + 3]
        r = rri[idx + 1 : idx + 4]
        if len(r) != 3 or not np.isfinite(s).all() or not np.isfinite(r).all():
            continue
        delta = np.diff(s)
        up = bool(np.all(delta >= 1.0))
        down = bool(np.all(delta <= -1.0))
        if not (up or down):
            continue
        key = "up" if up else "down"
        out[f"n_ramps_{key}"] += 1
        if np.std(s, ddof=1) > 0 and np.std(r, ddof=1) > 0:
            corr = float(np.corrcoef(s, r)[0, 1])
            if corr >= 0.80:
                out[f"n_brs_events_{key}"] += 1
    return out


def main() -> None:
    subjects = pd.read_csv("data/subjects.csv")["subject_id"].astype(str).tolist()
    rows: list[dict[str, object]] = []
    for subject_id in subjects:
        for condition in CONDITIONS:
            beats = pd.read_csv(Path("data/beats") / f"{subject_id}_{condition}.csv")
            rows.append({"subject_id": subject_id, "phase": condition, **count_events(beats)})
    out_dir = Path("data/derived/brs_ramps")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "per_subject_ramp_counts.csv", index=False)
    print(f"Emitted BRS ramp counts to {out_dir}")


if __name__ == "__main__":
    main()
