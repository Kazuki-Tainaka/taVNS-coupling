from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("Pre", "Stim", "Post")


def assert_aggregates(per_subj: pd.DataFrame, reference: pd.DataFrame) -> None:
    per_subj = per_subj.copy()
    per_subj["reliability_flag"] = per_subj["reliability_flag"].fillna("")
    for _, item in reference.iterrows():
        metric = item["Metric"]
        for condition in CONDITIONS:
            vals = per_subj.query(
                "metric == @metric and phase == @condition and reliability_flag == ''"
            )["value"].dropna()
            expected_mean = float(item[f"{condition}_mean"])
            expected_sd = float(item[f"{condition}_SD"])
            assert len(vals) == int(item["n"]), (
                f"{metric} {condition} n mismatch: got {len(vals)}, expected {item['n']}"
            )
            assert abs(vals.mean() - expected_mean) < 1e-3, f"{metric} {condition} mean mismatch"
            if np.isfinite(expected_sd):
                assert abs(vals.std(ddof=1) - expected_sd) < 1e-3, f"{metric} {condition} SD mismatch"


def test_per_subject_coupling_aggregates_match_af2():
    per_subj = pd.read_csv(REPO / "data/derived/per_subject_coupling.csv")
    reference = pd.read_csv(REPO / "data/reference/Additional_File_2.csv")
    assert_aggregates(per_subj, reference)


def test_per_subject_hrv_aggregates_match_af3():
    per_subj = pd.read_csv(REPO / "data/derived/per_subject_hrv.csv")
    reference = pd.read_csv(REPO / "data/reference/Additional_File_3.csv")
    assert_aggregates(per_subj, reference)
