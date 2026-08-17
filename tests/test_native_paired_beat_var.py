"""Synthetic unit tests for the native paired-beat bivariate VAR module."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from native_paired_beat_var import FitInput, fit_native_var


def independent_ar_fixture(length: int = 800, seed: int = 20_260_806) -> FitInput:
    """Return two independent stationary AR processes."""
    rng = np.random.default_rng(seed)
    sbp = np.zeros(length, dtype=float)
    rri = np.zeros(length, dtype=float)
    for index in range(1, length):
        sbp[index] = 0.55 * sbp[index - 1] + rng.normal(scale=1.0)
        rri[index] = 0.45 * rri[index - 1] + rng.normal(scale=1.0)
    return FitInput(
        subject_id="SYNTH_UNCOUPLED",
        phase="Synthetic",
        sbp=120.0 + sbp,
        rri=800.0 + rri,
        source_file="synthetic_fixture",
        source_row_or_record=f"generated_rows_1-{length}",
    )


def directional_fixture(length: int = 800, seed: int = 20_260_807) -> FitInput:
    """Return a process with past SBP driving RRI but no reverse pathway."""
    rng = np.random.default_rng(seed)
    sbp = np.zeros(length, dtype=float)
    rri = np.zeros(length, dtype=float)
    for index in range(2, length):
        sbp[index] = 0.62 * sbp[index - 1] + rng.normal(scale=0.8)
        rri[index] = (
            0.48 * rri[index - 1]
            + 0.85 * sbp[index - 1]
            + rng.normal(scale=0.8)
        )
    return FitInput(
        subject_id="SYNTH_SBP_TO_RRI",
        phase="Synthetic",
        sbp=120.0 + sbp,
        rri=800.0 + rri,
        source_file="synthetic_fixture",
        source_row_or_record=f"generated_rows_1-{length}",
    )


class NativePairedBeatVarSyntheticTests(unittest.TestCase):
    """Minimum false-positive and direction-recovery checks."""

    def test_uncoupled_fixture_has_no_nominal_direction(self) -> None:
        rows, order_trace = fit_native_var(independent_ar_fixture())
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(order_trace), 10)
        self.assertTrue(all(row["model_fit_status"] == "fit_succeeded" for row in rows))
        self.assertTrue(all(float(row["p_value"]) >= 0.05 for row in rows))

    def test_directional_fixture_recovers_past_sbp_to_rri(self) -> None:
        rows, order_trace = fit_native_var(directional_fixture())
        by_direction = {row["direction"]: row for row in rows}
        self.assertEqual(len(order_trace), 10)
        self.assertLess(float(by_direction["past_SBP_to_RRI"]["p_value"]), 0.001)
        self.assertGreater(float(by_direction["past_RRI_to_SBP"]["p_value"]), 0.05)
        self.assertEqual(by_direction["past_SBP_to_RRI"]["stability_status"], "stable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
