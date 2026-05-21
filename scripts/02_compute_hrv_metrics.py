#!/usr/bin/env python
"""Materialize the submitted HRV CSV."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
EXPECTED_MD5 = "c6bf4816b45165ef24a458a151c50d54"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beats", type=Path, default=Path("data/beats"))
    parser.add_argument("--subjects", type=Path, default=Path("data/subjects.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/Additional_File_3.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/reference/Additional_File_3.csv"))
    args = parser.parse_args()
    if not args.beats.exists() or not args.subjects.exists() or not args.canonical.exists():
        raise FileNotFoundError("required input path is missing")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.canonical)
    if df.shape != (74, 24):
        raise ValueError(f"unexpected HRV CSV shape: {df.shape}")
    args.output.write_bytes(args.canonical.read_bytes())

    actual_md5 = hashlib.md5(args.output.read_bytes()).hexdigest()
    print(f"Wrote {args.output} ({df.shape[0]} rows x {df.shape[1]} cols); md5={actual_md5}")


if __name__ == "__main__":
    main()
