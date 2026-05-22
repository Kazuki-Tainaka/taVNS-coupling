#!/usr/bin/env python
"""Persist temporal classification assignments for coupling metrics."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def classify(stim_p: float, post_p: float, stim_dz: float, post_dz: float) -> str:
    stim = stim_p < 0.05
    post = post_p < 0.05
    same_direction = stim_dz == 0 or post_dz == 0 or (stim_dz > 0) == (post_dz > 0)
    if stim and post and same_direction:
        return "A"
    if stim and not post:
        return "B"
    if post and not stim:
        return "C"
    return "D"


def main() -> None:
    af2 = pd.read_csv("data/reference/Additional_File_2.csv")
    rows = []
    for _, item in af2.iterrows():
        rows.append(
            {
                "metric": item["Metric"],
                "p_Stim_Pre": item["p_Stim_Pre"],
                "p_Post_Pre": item["p_Post_Pre"],
                "dz_Stim_Pre": item["dz_Stim_Pre"],
                "dz_Post_Pre": item["dz_Post_Pre"],
                "temporal_type": classify(
                    float(item["p_Stim_Pre"]),
                    float(item["p_Post_Pre"]),
                    float(item["dz_Stim_Pre"]),
                    float(item["dz_Post_Pre"]),
                ),
                "canonical_temporal_type": item["Temporal_Type"],
            }
        )
    out_dir = Path("data/derived/temporal_classification")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "coupling_type_assignments.csv", index=False)
    print(f"Emitted temporal classification to {out_dir}")


if __name__ == "__main__":
    main()
