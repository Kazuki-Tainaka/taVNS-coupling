"""Regression gate for the v1.1.2 publication-output synchronization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from synchronize_v1_1_2_publication_outputs import verify_repository_outputs


class PublicationOutputSyncTests(unittest.TestCase):
    """Verify the pinned public copies and central REF provenance."""

    def test_reviewed_outputs_match_v1_1_2_configuration(self) -> None:
        config = json.loads(
            (ROOT / "config" / "v1_1_2_publication_output_sync.json").read_text(
                encoding="utf-8"
            )
        )
        hashes = verify_repository_outputs(config)
        self.assertEqual(
            hashes["supplementary_data_1_sha256"],
            config["supplementary_data_1"]["target_sha256"],
        )
        self.assertEqual(
            hashes["supplementary_data_3_sha256"],
            config["supplementary_data_3"]["target_sha256"],
        )
        self.assertEqual(
            hashes["canonical_brs_contrasts_sha256"],
            config["canonical_brs_contrasts"]["target_sha256"],
        )
        self.assertEqual(
            hashes["methods_text_matched_brs_summary_sha256"],
            config["methods_text_matched_brs_summary"]["target_sha256"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
