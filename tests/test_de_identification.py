from pathlib import Path
import re

import pandas as pd


BEAT_SCHEMA = ["beat_idx", "RRI_ms", "SBP_mmHg", "PAT_ms"]
PII_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b(patient|hospital|name|dob|birth|medical_record|mrn)\b", re.IGNORECASE),
]


def test_public_beats_are_de_identified():
    paths = sorted(Path("data/beats").glob("S??_*.csv"))
    assert len(paths) == 54
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(pattern.search(text) for pattern in PII_PATTERNS), path
        df = pd.read_csv(path)
        assert df.columns.tolist() == BEAT_SCHEMA
        assert df["beat_idx"].is_monotonic_increasing
        assert 200 < df["RRI_ms"].mean() < 1500
        assert 70 < df["SBP_mmHg"].mean() < 200
        assert "timestamp" not in {c.lower() for c in df.columns}
        assert "time" not in {c.lower() for c in df.columns}


def test_dbp_removed_from_all_public_beat_files():
    paths = sorted(Path("data/beats").glob("S??_*.csv"))
    assert len(paths) == 54
    for path in paths:
        df = pd.read_csv(path)
        assert "DBP_mmHg" not in df.columns
        assert df.columns.tolist() == BEAT_SCHEMA
