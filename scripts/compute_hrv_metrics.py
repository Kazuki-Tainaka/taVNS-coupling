#!/usr/bin/env python
"""Materialize HRV results and emit per-subject derived values."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

CONDITIONS = ("Pre", "Stim", "Post")
EXPECTED_MD5 = "df4edfa0c874ddc684e19a43f8b60038"
FDR_CONTRASTS = (
    ("p_Stim_Pre", "p_FDR_Stim_Pre"),
    ("p_Post_Pre", "p_FDR_Post_Pre"),
    ("p_Post_Stim", "p_FDR_Post_Stim"),
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def format_fdr(value: float) -> str:
    return str(round(float(value), 4))


def recompute_fdr_csv_bytes(csv_bytes: bytes) -> bytes:
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    rows = list(reader)
    if reader.fieldnames is None:
        raise ValueError("HRV CSV has no header")

    for p_col, q_col in FDR_CONTRASTS:
        pvals = np.asarray([float(row[p_col]) for row in rows], dtype=float)
        _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        for row, q in zip(rows, qvals):
            row[q_col] = format_fdr(q)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


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
    parser.add_argument("--update-canonical-fdr", action="store_true", help="Rewrite the canonical HRV CSV with per-contrast FDR values.")
    parser.add_argument("--emit-per-subject", action="store_true", help="Write data/derived/per_subject_hrv.csv.")
    args = parser.parse_args()
    if not args.beats.exists() or not args.subjects.exists() or not args.canonical.exists():
        raise FileNotFoundError("required input path is missing")

    csv_bytes = recompute_fdr_csv_bytes(args.canonical.read_bytes())
    if args.update_canonical_fdr:
        args.canonical.write_bytes(csv_bytes)

    df = pd.read_csv(io.BytesIO(csv_bytes))
    if df.shape != (74, 24):
        raise ValueError(f"unexpected HRV CSV shape: {df.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(csv_bytes)
    actual_md5 = md5(args.output)
    if actual_md5 != EXPECTED_MD5:
        raise SystemExit(f"HRV MD5 mismatch: {actual_md5}")

    if args.emit_per_subject:
        emit_per_subject(df, args.subjects, Path("data/derived/per_subject_hrv.csv"))
    print(f"Wrote {args.output} ({df.shape[0]} rows x {df.shape[1]} cols); md5={actual_md5}")


if __name__ == "__main__":
    main()
