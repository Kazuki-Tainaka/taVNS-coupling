#!/usr/bin/env python
"""Create deterministic bootstrap replicate arrays for selected paired effects."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {
    "BRSseq_all": "BRS_seq_all",
    "BRSseq_up": "BRS_seq_up",
    "GC_F_BP_to_RRI": "GC_F_BP_to_RRI",
    "GC3_F_RRI_to_PAT": "GC3_F_RRI_to_PTT",
    "GC3_F_SBP_to_RRI": "GC3_F_SBP_to_RRI",
}


def seed_for(name: str) -> int:
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)


def dz(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / sd) if sd else float("nan")


def values_for(per_subj: pd.DataFrame, metric: str) -> np.ndarray:
    wide = (
        per_subj.loc[(per_subj["metric"] == metric) & (per_subj["reliability_flag"] == "")]
        .pivot(index="subject_id", columns="phase", values="value")
        .dropna(subset=["Pre", "Stim"])
    )
    return (wide["Stim"] - wide["Pre"]).to_numpy(float)


def main() -> None:
    per_subj = pd.read_csv("data/derived/per_subject_coupling.csv")
    per_subj["reliability_flag"] = per_subj["reliability_flag"].fillna("")
    out_dir = Path("data/derived/bootstrap")
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_lines = [
        "# Bootstrap Replicates",
        "",
        "Replicates are deterministic paired-subject resamples of Stim minus Pre differences.",
        "",
        "| Output | Source metric | Seed |",
        "|---|---|---:|",
    ]
    for file_key, metric in TARGETS.items():
        diffs = values_for(per_subj, metric)
        rng = np.random.default_rng(seed_for(file_key))
        reps = []
        for _ in range(10000):
            sample = rng.choice(diffs, size=len(diffs), replace=True)
            reps.append(dz(sample))
        out_path = out_dir / f"replicates_{file_key}.csv"
        pd.DataFrame({"dz": reps}).to_csv(out_path, index=False, float_format="%.6f")
        readme_lines.append(f"| `{out_path.name}` | `{metric}` | {seed_for(file_key)} |")
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"Emitted bootstrap replicates to {out_dir}")


if __name__ == "__main__":
    main()
