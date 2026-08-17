from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from coupling_core import (  # noqa: E402
    REFERENCE_COHERENCE_CONFIG,
    SEGMENT_LENGTH_SENSITIVITY_CONFIG,
    coherence_statistic,
    effective_welch_segment_count,
)


class AuthorReviewFixTests(unittest.TestCase):
    def test_segment_length_is_only_estimator_change(self) -> None:
        reference = asdict(REFERENCE_COHERENCE_CONFIG)
        sensitivity = asdict(SEGMENT_LENGTH_SENSITIVITY_CONFIG)
        differences = {
            key for key in reference if reference[key] != sensitivity[key]
        }
        self.assertEqual(differences, {"nperseg", "noverlap"})
        self.assertEqual(reference["nperseg"], 512)
        self.assertEqual(reference["noverlap"], 256)
        self.assertEqual(sensitivity["nperseg"], 256)
        self.assertEqual(sensitivity["noverlap"], 128)

    def test_effective_welch_segment_count(self) -> None:
        self.assertEqual(
            effective_welch_segment_count(1196, REFERENCE_COHERENCE_CONFIG),
            3,
        )
        self.assertEqual(
            effective_welch_segment_count(
                1196, SEGMENT_LENGTH_SENSITIVITY_CONFIG
            ),
            8,
        )
        self.assertEqual(
            effective_welch_segment_count(255, SEGMENT_LENGTH_SENSITIVITY_CONFIG),
            0,
        )

    def test_both_configs_return_finite_coherence(self) -> None:
        rng = np.random.default_rng(20260805)
        shared = rng.normal(size=1200)
        sbp = shared + 0.5 * rng.normal(size=1200)
        rri = 0.8 * shared + 0.5 * rng.normal(size=1200)
        reference = coherence_statistic(
            sbp,
            rri,
            config=REFERENCE_COHERENCE_CONFIG,
        )
        sensitivity = coherence_statistic(
            sbp,
            rri,
            config=SEGMENT_LENGTH_SENSITIVITY_CONFIG,
        )
        self.assertTrue(np.isfinite(reference))
        self.assertTrue(np.isfinite(sensitivity))
        self.assertGreaterEqual(reference, 0.0)
        self.assertLessEqual(reference, 1.0)
        self.assertGreaterEqual(sensitivity, 0.0)
        self.assertLessEqual(sensitivity, 1.0)


if __name__ == "__main__":
    unittest.main()
