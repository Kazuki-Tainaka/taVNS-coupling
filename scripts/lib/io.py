from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

BEAT_COLUMNS = ["beat_idx", "RRI_ms", "SBP_mmHg", "PAT_ms"]
CONDITIONS = ("Pre", "Stim", "Post")


def load_beats(subject_id: str, condition: str, data_dir: Path) -> pd.DataFrame:
    subject_id = subject_id.upper()
    condition = condition.capitalize()
    path = Path(data_dir) / f"{subject_id}_{condition}.csv"
    df = pd.read_csv(path)
    missing = [col for col in BEAT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[BEAT_COLUMNS].copy()


def load_subjects(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "subjects.csv")


def iter_subject_conditions(data_dir: Path) -> Iterator[tuple[str, str, pd.DataFrame, dict]]:
    data_dir = Path(data_dir)
    subjects = load_subjects(data_dir)
    for _, row in subjects.iterrows():
        subject_id = str(row["subject_id"])
        for condition in CONDITIONS:
            yield subject_id, condition, load_beats(subject_id, condition, data_dir / "beats"), row.to_dict()
