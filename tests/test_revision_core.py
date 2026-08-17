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

from brs_core import BRSSetting, evaluate_brs  # noqa: E402
from coupling_core import phase_randomize  # noqa: E402


class RevisionCoreTests(unittest.TestCase):
    def test_known_linear_up_sequence(self) -> None:
        sbp = np.array([100.0, 102.0, 104.0, 100.0])
        rri = np.array([700.0, 710.0, 720.0, 730.0])
        summary, _ = evaluate_brs(
            sbp,
            rri,
            BRSSetting(lag_beats=0),
        )
        self.assertEqual(summary["n_brs_up"], 1)
        self.assertAlmostEqual(summary["BRS_seq_up"], 5.0, places=12)

    def test_fourier_phase_randomization_preserves_magnitude(self) -> None:
        rng = np.random.default_rng(20260805)
        source = rng.normal(size=1200)
        randomized = phase_randomize(source, rng)
        delta = np.max(np.abs(np.abs(np.fft.rfft(source)) - np.abs(np.fft.rfft(randomized))))
        self.assertLess(delta, 1e-10)

    def test_release_contains_no_hardcoded_user_profile_path(self) -> None:
        forbidden = ("C:" + "\\Users\\",)
        for path in (ROOT / "src").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, msg=f"{token} in {path.name}")


if __name__ == "__main__":
    unittest.main()
