"""Regression tests for degenerate three-phase binary prevalence tests."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from native_paired_beat_var import summarize_prevalence
from stats_core import cochran_q_test


class DegenerateCochranQTests(unittest.TestCase):
    """Ensure an all-identical matrix is not reported as Q=0, p=1."""

    def test_core_classifies_all_true_matrix_as_not_estimable(self) -> None:
        result = cochran_q_test(np.ones((18, 3), dtype=int))
        self.assertEqual(result["n"], 18)
        self.assertEqual(result["k"], 3)
        self.assertTrue(np.isnan(float(result["statistic"])))
        self.assertTrue(np.isnan(float(result["p_value"])))
        self.assertEqual(result["status"], "not_estimable")
        self.assertEqual(result["exact_or_asymptotic"], "NA")
        self.assertEqual(
            result["NA_reason"],
            "no_within_participant_variation_across_phases",
        )

    def test_non_degenerate_matrix_remains_estimable(self) -> None:
        matrix = np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
            ],
            dtype=int,
        )
        result = cochran_q_test(matrix)
        self.assertEqual(result["status"], "estimable")
        self.assertEqual(result["exact_or_asymptotic"], "asymptotic_chi_square")
        self.assertEqual(result["NA_reason"], "NA")
        self.assertTrue(np.isfinite(float(result["statistic"])))
        self.assertTrue(np.isfinite(float(result["p_value"])))

    def test_native_prevalence_keeps_mcnemar_p_one_estimable(self) -> None:
        records = []
        for direction in ("past_SBP_to_RRI", "past_RRI_to_SBP"):
            for subject_index in range(1, 19):
                for phase in ("Pre", "Stim", "Post"):
                    records.append(
                        {
                            "subject_id": f"S{subject_index:02d}",
                            "phase": phase,
                            "direction": direction,
                            "nominal_significant": True,
                            "fdr_significant": True,
                        }
                    )
        summary = summarize_prevalence(pd.DataFrame(records))
        rows = summary.loc[
            summary["direction"].eq("past_RRI_to_SBP")
            & summary["summary_type"].eq("paired_prevalence_comparison")
        ]
        q_rows = rows.loc[rows["phase_or_contrast"].eq("Pre-Stim-Post")]
        self.assertEqual(len(q_rows), 2)
        self.assertTrue(q_rows["test_statistic"].isna().all())
        self.assertTrue(q_rows["p_value"].isna().all())
        self.assertTrue(q_rows["status"].eq("not_estimable").all())
        self.assertTrue(q_rows["exact_or_asymptotic"].eq("NA").all())
        self.assertTrue(
            q_rows["NA_reason"]
            .eq("no_within_participant_variation_across_phases")
            .all()
        )

        mcnemar = rows.loc[rows["phase_or_contrast"].eq("Stim-vs-Pre")]
        self.assertEqual(len(mcnemar), 2)
        self.assertTrue(mcnemar["p_value"].eq(1.0).all())
        self.assertTrue(mcnemar["status"].eq("estimable").all())
        self.assertTrue(mcnemar["exact_or_asymptotic"].eq("exact").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
