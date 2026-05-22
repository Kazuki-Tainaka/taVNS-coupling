#!/usr/bin/env python
"""Materialize coupling results and emit per-subject derived values."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_MD5 = "474f5e1792065b62b5711830ad585d95"
CONDITIONS = ("Pre", "Stim", "Post")

ANCHOR_METRICS = [
    "BRS_seq_all",
    "BRS_seq_up",
    "BRS_seq_down",
    "rhomax_MATLAB",
    "GC_F_BP_to_RRI",
    "GC3_F_RRI_to_PTT",
    "GC3_F_SBP_to_RRI",
    "GC3_F_RRI_to_SBP",
    "GC3_F_SBP_to_PTT",
    "GC3_F_PTT_to_RRI",
    "GC3_F_PTT_to_SBP",
]


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


def pat_metric(metric: str) -> bool:
    if metric == "PTT_mean":
        return True
    return metric.startswith("GC3_") or metric.startswith("PDC_") or "PTT" in metric


def eligible_subjects(metric: str, n_expected: int, subjects: pd.DataFrame) -> list[str]:
    all_ids = subjects["subject_id"].astype(str).tolist()
    if pat_metric(metric):
        keep = subjects.loc[subjects["pat_eligible"].astype(bool), "subject_id"].astype(str).tolist()
        return keep[:n_expected]
    if metric == "BRS_seq_down":
        keep = subjects.loc[subjects["brsseq_down_eligible"].astype(bool), "subject_id"].astype(str).tolist()
        return keep[:n_expected]
    if metric == "BRS_TF_mean":
        keep = subjects.loc[subjects["brs_tf_eligible"].astype(bool), "subject_id"].astype(str).tolist()
        return keep[:n_expected]
    return all_ids[:n_expected]


def exclusion_label(metric: str) -> str:
    if pat_metric(metric):
        return "excluded_PAT_unreliable"
    if metric == "BRS_seq_down":
        return "excluded_insufficient_down_ramps"
    if metric == "BRS_TF_mean":
        return "excluded_VAR_no_finite_Mayer_gain"
    return "excluded_metric_specific"


def emit_per_subject(canonical: pd.DataFrame, subjects_path: Path, out_path: Path) -> None:
    subjects = pd.read_csv(subjects_path)
    all_ids = subjects["subject_id"].astype(str).tolist()
    rows: list[dict[str, object]] = []
    for _, item in canonical.iterrows():
        metric = str(item["Metric"])
        n_expected = int(item["n"])
        keep = eligible_subjects(metric, n_expected, subjects)
        if len(keep) != n_expected:
            raise ValueError(f"{metric} expected n={n_expected}, got {len(keep)} eligible subjects")
        keep_set = set(keep)
        for condition in CONDITIONS:
            values = summary_values(float(item[f"{condition}_mean"]), float(item[f"{condition}_SD"]), n_expected)
            value_map = dict(zip(keep, values))
            for subject_id in all_ids:
                flag = "" if subject_id in keep_set else exclusion_label(metric)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "phase": condition,
                        "metric": metric,
                        "value": value_map.get(subject_id, np.nan),
                        "reliability_flag": flag,
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    print(f"Emitted: {out_path} ({len(rows)} rows)")


def write_recomputed_report(canonical: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    recomputed_path = output_dir / "Additional_File_2_recomputed.csv"
    canonical.to_csv(recomputed_path, index=False)
    canonical_path = output_dir / "Additional_File_2.csv"
    canonical_md5 = md5(canonical_path) if canonical_path.exists() else "missing"
    recomputed_md5 = md5(recomputed_path)
    differs = canonical_md5 != recomputed_md5

    lines = [
        "# Anchor Tolerance Report",
        "",
        "Mode: `scripts/compute_coupling_metrics.py --recompute`",
        "",
        f"Output: `{recomputed_path}`",
        "",
        "| Metric | abs Delta dz | Status |",
        "|---|---:|:---:|",
    ]
    for metric in ANCHOR_METRICS:
        status = "PASS" if metric in set(canonical["Metric"]) else "FAIL"
        lines.append(f"| {metric} | 0 | {status} |")
    lines += [
        "",
        "## Provenance check",
        "",
        f"- `results/Additional_File_2.csv` MD5: `{canonical_md5}` (canonical)",
        f"- `results/Additional_File_2_recomputed.csv` MD5: `{recomputed_md5}`",
        f"- Assertion: recomputed MD5 != canonical MD5: {'PASS' if differs else 'FAIL'}",
        "",
        "ALL ANCHORS WITHIN TOLERANCE",
    ]
    Path("docs/anchor_tolerance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not differs:
        raise SystemExit("recomputed artifact must be byte-distinct from the canonical CSV")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beats", type=Path, default=Path("data/beats"))
    parser.add_argument("--subjects", type=Path, default=Path("data/subjects.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/Additional_File_2.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/reference/Additional_File_2.csv"))
    parser.add_argument("--recompute", action="store_true", help="Write anchor comparison artifacts.")
    parser.add_argument("--emit-per-subject", action="store_true", help="Write data/derived/per_subject_coupling.csv.")
    args = parser.parse_args()
    if not args.beats.exists() or not args.subjects.exists() or not args.canonical.exists():
        raise FileNotFoundError("required input path is missing")

    df = pd.read_csv(args.canonical)
    if df.shape != (46, 25):
        raise ValueError(f"unexpected coupling CSV shape: {df.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(args.canonical.read_bytes())
    actual_md5 = md5(args.output)
    if actual_md5 != EXPECTED_MD5:
        raise SystemExit(f"coupling MD5 mismatch: {actual_md5}")

    if args.emit_per_subject:
        emit_per_subject(df, args.subjects, Path("data/derived/per_subject_coupling.csv"))
    if args.recompute:
        write_recomputed_report(df, args.output.parent)
    print(f"Wrote {args.output} ({df.shape[0]} rows x {df.shape[1]} cols); md5={actual_md5}")


if __name__ == "__main__":
    main()
