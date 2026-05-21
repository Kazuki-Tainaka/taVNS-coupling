from __future__ import annotations

import numpy as np
import pandas as pd

PAT_NEAR_ZERO_THRESHOLD = 1e-10


def validate_pat(pat_array: np.ndarray) -> bool:
    if pat_array is None or len(pat_array) == 0:
        return False
    arr = np.asarray(pat_array, dtype=float)
    if np.any(np.isnan(arr)):
        return False
    if np.any(np.abs(arr) < PAT_NEAR_ZERO_THRESHOLD):
        return False
    return True


def filter_eligible_subjects(subjects_df, metric_class: str) -> list[str]:
    col = f"{metric_class}_eligible"
    if col not in subjects_df.columns:
        raise KeyError(f"missing eligibility column: {col}")
    return subjects_df.loc[subjects_df[col].astype(bool), "subject_id"].tolist()


def is_brsseq_down_eligible(subject_id: str, subjects_df: pd.DataFrame) -> bool:
    """Return whether a subject belongs to the BRSseq,down n=15 pool.

    Subjects S06, S12, and S17 are excluded because no Stim-phase descending sequence passed the Pearson r >= 0.80 correlation criterion
    used by the sequence-method BRS detector. Per-channel ECG/BP source attribution
    cannot be uniquely assigned from archived materials.
    """
    row = subjects_df.loc[subjects_df["subject_id"] == subject_id]
    if row.empty:
        raise ValueError(f"Unknown subject_id: {subject_id}")
    if "brsseq_down_eligible" not in subjects_df.columns:
        raise KeyError("missing eligibility column: brsseq_down_eligible")
    return bool(row["brsseq_down_eligible"].iloc[0])
