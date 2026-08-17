from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TAVNS_DATA_ROOT", str(ROOT / "controlled_data_not_included"))
os.environ.setdefault("TAVNS_OUTPUT_ROOT", str(ROOT / "_unit_test_output"))

from brs_core import BRSSetting, _detect_ramps, evaluate_brs  # noqa: E402


class BRSRriMonotonicityTests(unittest.TestCase):
    @staticmethod
    def evaluate(
        sbp: list[float],
        rri: list[float],
        **setting_overrides: object,
    ) -> tuple[dict[str, float | int], list[dict[str, object]]]:
        return evaluate_brs(
            np.asarray(sbp, dtype=float),
            np.asarray(rri, dtype=float),
            BRSSetting(lag_beats=0, **setting_overrides),
        )

    def test_ascending_sbp_and_rri_are_eligible(self) -> None:
        summary, rows = self.evaluate(
            [100.0, 102.0, 104.0, 90.0],
            [700.0, 710.0, 720.0, 700.0],
        )
        self.assertEqual(summary["n_brs_up"], 1)
        self.assertTrue(rows[0]["rri_direction_monotonic"])

    def test_reference_accepts_ascending_sbp_with_nonmonotonic_rri(self) -> None:
        summary, rows = self.evaluate(
            [100.0, 102.0, 104.0, 90.0],
            [700.0, 712.0, 711.0, 700.0],
        )
        self.assertGreater(rows[0]["pearson_r"], 0.80)
        self.assertFalse(rows[0]["rri_direction_monotonic"])
        self.assertEqual(summary["n_brs_up"], 1)

    def test_directional_rri_gate_is_an_explicit_sensitivity(self) -> None:
        summary, rows = self.evaluate(
            [100.0, 102.0, 104.0, 90.0],
            [700.0, 712.0, 711.0, 700.0],
            require_directional_rri_monotonicity=True,
        )
        self.assertFalse(rows[0]["rri_direction_monotonic"])
        self.assertEqual(summary["n_brs_up"], 0)

    def test_descending_sbp_and_rri_are_eligible(self) -> None:
        summary, rows = self.evaluate(
            [104.0, 102.0, 100.0, 110.0],
            [720.0, 710.0, 700.0, 730.0],
        )
        self.assertEqual(summary["n_brs_down"], 1)
        self.assertTrue(rows[0]["rri_direction_monotonic"])

    def test_reference_accepts_descending_sbp_with_nonmonotonic_rri(self) -> None:
        summary, rows = self.evaluate(
            [104.0, 102.0, 100.0, 110.0],
            [720.0, 708.0, 709.0, 730.0],
        )
        self.assertGreater(rows[0]["pearson_r"], 0.80)
        self.assertFalse(rows[0]["rri_direction_monotonic"])
        self.assertEqual(summary["n_brs_down"], 1)

    def test_sub_1_ms_directional_rri_changes_remain_eligible(self) -> None:
        summary, rows = self.evaluate(
            [100.0, 102.0, 104.0, 90.0],
            [700.0, 700.2, 700.4, 699.0],
        )
        self.assertEqual(summary["n_brs_up"], 1)
        self.assertTrue(rows[0]["rri_direction_monotonic"])
        self.assertFalse(rows[0]["rri_concordance_ge_1ms_descriptive"])

    def test_correlation_below_threshold_is_ineligible(self) -> None:
        summary, rows = self.evaluate(
            [100.0, 102.0, 104.0, 106.0, 90.0],
            [700.0, 700.001, 700.002, 800.0, 700.0],
            minimum_sequence_length=4,
        )
        self.assertTrue(rows[0]["rri_direction_monotonic"])
        self.assertLess(rows[0]["pearson_r"], 0.80)
        self.assertEqual(summary["n_brs_up"], 0)

    def test_nonfinite_slope_is_ineligible(self) -> None:
        summary, rows = self.evaluate(
            [100.0, 102.0, 104.0, 90.0],
            [700.0, 710.0, np.inf, 700.0],
        )
        self.assertFalse(np.isfinite(rows[0]["slope_ms_per_mmHg"]))
        self.assertEqual(summary["n_brs_up"], 0)

    def test_lag_zero_one_two_alignment(self) -> None:
        sbp = np.asarray([100.0, 102.0, 104.0, 90.0, 90.0, 90.0])
        rri = np.asarray([500.0, 600.0, 700.0, 710.0, 720.0, 500.0])
        for lag in (0, 1, 2):
            _, rows = evaluate_brs(sbp, rri, BRSSetting(lag_beats=lag))
            up = next(row for row in rows if row["direction"] == "up")
            self.assertEqual(up["rri_start_ms"], rri[lag])
            self.assertEqual(up["lag_beats"], lag)

    def test_minimum_length_three_and_four(self) -> None:
        sbp = [100.0, 102.0, 104.0, 90.0]
        rri = [700.0, 710.0, 720.0, 700.0]
        summary_three, _ = self.evaluate(sbp, rri, minimum_sequence_length=3)
        summary_four, _ = self.evaluate(sbp, rri, minimum_sequence_length=4)
        self.assertEqual(summary_three["n_brs_up"], 1)
        self.assertEqual(summary_four["n_brs_up"], 0)

    def test_exact_minimum_aligned_length_is_evaluable(self) -> None:
        summary, rows = evaluate_brs(
            np.asarray([100.0, 102.0, 104.0, 90.0]),
            np.asarray([600.0, 700.0, 710.0, 720.0]),
            BRSSetting(lag_beats=1, minimum_sequence_length=3),
        )
        self.assertEqual(summary["n_brs_up"], 1)
        self.assertEqual(len(rows), 1)

    def test_maximal_same_direction_ramps_do_not_overlap(self) -> None:
        ramps = _detect_ramps(
            np.asarray([100.0, 102.0, 104.0, 100.0, 102.0, 104.0, 90.0]),
            direction="up",
            minimum_length=3,
            minimum_increment=1.0,
        )
        self.assertEqual(ramps, [(0, 2), (3, 5)])
        self.assertLess(ramps[0][1], ramps[1][0])


if __name__ == "__main__":
    unittest.main()
