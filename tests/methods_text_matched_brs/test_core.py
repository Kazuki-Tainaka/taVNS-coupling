"""Automated validation for the locked Antonino harmonization workflow."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "methods_text_matched_brs"))

from core import (  # noqa: E402
    ASSOCIATION_PERMUTATION_SEED,
    BRANCHES,
    PHASES,
    SUBJECTS,
    SUBPHASES,
    bca_paired_correlation,
    clean_and_segment,
    detect_maximal_ramps,
    evaluate_branch,
    load_subject,
    permutation_spearman,
)


def synthetic_frame(
    sbp: list[float],
    rri: list[float],
    times: list[float] | None = None,
) -> pd.DataFrame:
    n = len(sbp)
    if times is None:
        times = list(np.arange(n, dtype=float))
    return pd.DataFrame(
        {
            "subject": 99,
            "source_row": np.arange(n),
            "R_wave_timing_ms": np.asarray(times) * 1000.0,
            "RRI_ms": rri,
            "SBP_peak_timing_ms": np.asarray(times) * 1000.0 + 100.0,
            "SBP_stored_V": np.asarray(sbp) / 100.0,
            "PAT_ms": np.full(n, 100.0),
            "R_wave_time_s": times,
            "SBP_peak_time_s": np.asarray(times) + 0.1,
            "SBP_mmHg": sbp,
        }
    )


def test_three_beat_ascending_and_descending_sequences() -> None:
    up = detect_maximal_ramps(np.array([100.0, 102.0, 104.0]), "up", 3, 1.0, True)
    down = detect_maximal_ramps(np.array([104.0, 102.0, 100.0]), "down", 3, 1.0, True)
    assert up == [(0, 2)]
    assert down == [(0, 2)]
    for sbp, rri, direction in (
        ([100.0, 102.0, 104.0], [800.0, 810.0, 820.0], "up"),
        ([104.0, 102.0, 100.0], [820.0, 810.0, 800.0], "down"),
    ):
        summaries, sequences = evaluate_branch(
            synthetic_frame(sbp, rri), 99, "synthetic", 0.0, 3.0, BRANCHES["A0_MAX"]
        )
        seq = pd.DataFrame(sequences)
        assert len(seq) == 1
        assert seq.iloc[0]["direction"] == direction
        assert bool(seq.iloc[0]["qualifying_sequence"])
        all_summary = next(row for row in summaries if row["direction"] == "all")
        assert all_summary["n_qualifying_sequences"] == 1


def test_strict_step_threshold_rejects_exactly_one_and_accepts_above() -> None:
    exact = synthetic_frame([100.0, 101.0, 102.0], [800.0, 801.0, 802.0])
    summaries, sequences = evaluate_branch(
        exact, 99, "synthetic", 0.0, 3.0, BRANCHES["A0_MAX"]
    )
    assert sequences == []
    assert next(row for row in summaries if row["direction"] == "all")[
        "no_valid_sequence_flag"
    ]
    above = synthetic_frame(
        [100.0, 101.0001, 102.0002],
        [800.0, 801.0001, 802.0002],
    )
    _, sequences = evaluate_branch(
        above, 99, "synthetic", 0.0, 3.0, BRANCHES["A0_MAX"]
    )
    assert len(sequences) == 1
    assert bool(sequences[0]["qualifying_sequence"])


def test_r_squared_strict_boundary_and_above() -> None:
    x = np.array([98.0, 100.0, 102.0])
    slope = 10.0
    residual_scale = np.sqrt(slope**2 * 8.0 * 0.15 / (6.0 * 0.85))
    y = slope * (x - 100.0) + residual_scale * np.array([1.0, -2.0, 1.0]) + 800.0
    computed_r2 = float(stats.linregress(x, y).rvalue**2)
    assert computed_r2 == pytest.approx(0.85, abs=2e-15)
    boundary_config = replace(BRANCHES["A0_MAX"], r2_threshold=computed_r2)
    _, boundary = evaluate_branch(
        synthetic_frame(x.tolist(), y.tolist()),
        99,
        "synthetic",
        0.0,
        3.0,
        boundary_config,
    )
    assert len(boundary) == 1
    assert not bool(boundary[0]["r2_gate"])
    above_config = replace(BRANCHES["A0_MAX"], r2_threshold=computed_r2 - 1e-10)
    _, above = evaluate_branch(
        synthetic_frame(x.tolist(), y.tolist()),
        99,
        "synthetic",
        0.0,
        3.0,
        above_config,
    )
    assert bool(above[0]["r2_gate"])


def test_lag_zero_and_lag_one_alignment() -> None:
    frame = synthetic_frame(
        [100.0, 102.0, 104.0, 106.0],
        [790.0, 800.0, 810.0, 820.0],
    )
    _, lag0 = evaluate_branch(frame, 99, "synthetic", 0.0, 4.0, BRANCHES["A0_MAX"])
    _, lag1 = evaluate_branch(frame, 99, "synthetic", 0.0, 4.0, BRANCHES["A1_MAX"])
    assert lag0[0]["sbp_source_row_start"] == lag0[0]["rri_source_row_start"] == 0
    assert lag1[0]["sbp_source_row_start"] == 0
    assert lag1[0]["rri_source_row_start"] == 1
    assert lag1[0]["rri_source_row_end"] == 3


@pytest.mark.parametrize("boundary", [300.0, 450.0, 600.0, 750.0])
def test_half_open_boundary_exclusion(boundary: float) -> None:
    times = [boundary - 0.3, boundary - 0.2, boundary - 0.1, boundary, boundary + 0.1, boundary + 0.2]
    frame = synthetic_frame(
        [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        [800.0, 810.0, 820.0, 830.0, 840.0, 850.0],
        times,
    )
    left = clean_and_segment(frame, boundary - 1.0, boundary)
    right = clean_and_segment(frame, boundary, boundary + 1.0)
    assert list(left["R_wave_time_s"]) == times[:3]
    assert list(right["R_wave_time_s"]) == times[3:]


@pytest.mark.parametrize("boundary", [300.0, 450.0, 600.0, 750.0])
def test_no_sequence_crosses_any_boundary(boundary: float) -> None:
    times = [boundary - 0.3, boundary - 0.2, boundary - 0.1, boundary, boundary + 0.1, boundary + 0.2]
    frame = synthetic_frame(
        [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        [800.0, 810.0, 820.0, 830.0, 840.0, 850.0],
        times,
    )
    for start, end in ((boundary - 1.0, boundary), (boundary, boundary + 1.0)):
        segment = clean_and_segment(frame, start, end)
        _, sequences = evaluate_branch(
            segment, 99, "synthetic", start, end, BRANCHES["A0_MAX"]
        )
        assert len(sequences) == 1
        assert sequences[0]["sbp_rwave_time_start_s"] >= start
        assert sequences[0]["sbp_rwave_time_end_s"] < end
        assert sequences[0]["rri_rwave_time_end_s"] < end


def test_sequence_slope_units_ms_per_mmhg() -> None:
    frame = synthetic_frame([100.0, 102.0, 104.0], [800.0, 810.0, 820.0])
    _, sequences = evaluate_branch(
        frame, 99, "synthetic", 0.0, 3.0, BRANCHES["A0_MAX"]
    )
    assert sequences[0]["slope_ms_per_mmHg"] == pytest.approx(5.0)


def test_no_valid_sequence_is_nan_not_zero() -> None:
    frame = synthetic_frame([100.0, 100.4, 100.8], [800.0, 805.0, 810.0])
    summaries, sequences = evaluate_branch(
        frame, 99, "synthetic", 0.0, 3.0, BRANCHES["A0_MAX"]
    )
    assert sequences == []
    all_summary = next(row for row in summaries if row["direction"] == "all")
    assert np.isnan(all_summary["gain_ms_per_mmHg"])
    assert all_summary["no_valid_sequence_reason"] == "NO_CANDIDATE_SBP_RAMP"


def test_paired_bootstrap_keeps_records_together() -> None:
    x = np.arange(1.0, 19.0)
    y = -3.0 * x
    result = bca_paired_correlation(x, y, "spearman", seed=12345, n_resamples=1_000)
    assert result["estimate"] == pytest.approx(-1.0)
    assert result["ci_low"] == pytest.approx(-1.0)
    assert result["ci_high"] == pytest.approx(-1.0)


def test_permutation_reproducibility_with_fixed_seed() -> None:
    x = np.arange(18.0)
    y = np.array([8, 5, 12, 1, 15, 3, 10, 0, 17, 4, 13, 7, 16, 2, 14, 6, 11, 9], dtype=float)
    first = permutation_spearman(x, y, ASSOCIATION_PERMUTATION_SEED, 5_000)
    second = permutation_spearman(x, y, ASSOCIATION_PERMUTATION_SEED, 5_000)
    assert first == second

