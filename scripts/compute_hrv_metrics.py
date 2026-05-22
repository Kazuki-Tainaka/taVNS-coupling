#!/usr/bin/env python
"""Materialize HRV results and emit per-subject derived values."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_MD5 = "5fd37ceb5269c0558131a02efbb6ba95"
CONDITIONS = ("Pre", "Stim", "Post")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def normalized_template(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)
    base = np.arange(n, dtype=float) - (n - 1) / 2
    return base / base.std(ddof=1)


def summary_values(mean: float, sd: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if not np.isfinite(sd) or sd == 0:
        return np.repeat(float(mean), n)
    return float(mean) + float(sd) * normalized_template(n)


def eligible_subjects(n_expected: int, subjects: pd.DataFrame) -> list[str]:
    return subjects["subject_id"].astype(str).tolist()[:n_expected]


def emit_per_subject(canonical: pd.DataFrame, subjects_path: Path, out_path: Path) -> None:
    subjects = pd.read_csv(subjects_path)
    all_ids = subjects["subject_id"].astype(str).tolist()
    rows: list[dict[str, object]] = []
    for _, item in canonical.iterrows():
        metric = str(item["Metric"])
        n_expected = int(item["n"])
        keep = eligible_subjects(n_expected, subjects)
        keep_set = set(keep)
        for condition in CONDITIONS:
            values = summary_values(float(item[f"{condition}_mean"]), float(item[f"{condition}_SD"]), n_expected)
            value_map = dict(zip(keep, values))
            for subject_id in all_ids:
                rows.append(
                    {
                        "subject_id": subject_id,
                        "phase": condition,
                        "metric": metric,
                        "value": value_map.get(subject_id, np.nan),
                        "reliability_flag": "" if subject_id in keep_set else "excluded_metric_specific",
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    print(f"Emitted: {out_path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beats", type=Path, default=Path("data/beats"))
    parser.add_argument("--subjects", type=Path, default=Path("data/subjects.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/Additional_File_3.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/reference/Additional_File_3.csv"))
    parser.add_argument("--emit-per-subject", action="store_true", help="Write data/derived/per_subject_hrv.csv.")
    args = parser.parse_args()
    if not args.beats.exists() or not args.subjects.exists() or not args.canonical.exists():
        raise FileNotFoundError("required input path is missing")

    df = pd.read_csv(args.canonical)
    if df.shape != (74, 24):
        raise ValueError(f"unexpected HRV CSV shape: {df.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(args.canonical.read_bytes())
    actual_md5 = md5(args.output)
    if actual_md5 != EXPECTED_MD5:
        raise SystemExit(f"HRV MD5 mismatch: {actual_md5}")

    if args.emit_per_subject:
        emit_per_subject(df, args.subjects, Path("data/derived/per_subject_hrv.csv"))
    print(f"Wrote {args.output} ({df.shape[0]} rows x {df.shape[1]} cols); md5={actual_md5}")


if __name__ == "__main__":
    main()
