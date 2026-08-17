"""Tests for the public allow-list of author-approved derived beat tables.

These tests cover the 54 files under ``data/beats/`` together with
``data/LICENSE``, ``data/README.md``, and
``config/approved_derived_beats_manifest.csv``. They also exercise the
validator's ``check_author_approved_derived_beats`` and ``check_public_tree_safety``
gates so that the allow-list is enforced end-to-end.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BEATS_DIR = ROOT / "data" / "beats"
MANIFEST = ROOT / "config" / "approved_derived_beats_manifest.csv"
DATA_LICENSE = ROOT / "data" / "LICENSE"
DATA_README = ROOT / "data" / "README.md"

EXPECTED_HEADER = ("beat_idx", "RRI_ms", "SBP_mmHg", "PAT_ms")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _load_validator_module():
    spec = importlib.util.spec_from_file_location(
        "validate_public_release_under_test",
        str(ROOT / "scripts" / "validate_public_release.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ApprovedDerivedBeatsInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.beats_files = sorted(p for p in BEATS_DIR.iterdir() if p.is_file())

    def test_exactly_54_beat_csv_files(self) -> None:
        self.assertEqual(len(self.beats_files), 54)
        non_csv = [p.name for p in self.beats_files if p.suffix != ".csv"]
        self.assertEqual(non_csv, [])

    def test_filename_pattern(self) -> None:
        pattern = re.compile(r"^S(0[1-9]|1[0-8])_(Pre|Stim|Post)\.csv$")
        for path in self.beats_files:
            self.assertRegex(path.name, pattern)

    def test_full_18x3_matrix(self) -> None:
        subjects = {f"S{n:02d}" for n in range(1, 19)}
        phases = {"Pre", "Stim", "Post"}
        expected = {f"{s}_{ph}.csv" for s in subjects for ph in phases}
        actual = {p.name for p in self.beats_files}
        self.assertEqual(actual, expected)

    def test_data_license_and_readme_present(self) -> None:
        self.assertTrue(DATA_LICENSE.is_file())
        self.assertTrue(DATA_README.is_file())
        license_text = DATA_LICENSE.read_text(encoding="utf-8")
        self.assertIn("CC BY 4.0", license_text)
        readme_text = DATA_README.read_text(encoding="utf-8")
        self.assertIn("beat_idx,RRI_ms,SBP_mmHg,PAT_ms", readme_text)
        self.assertIn("54", readme_text)


class ApprovedDerivedBeatsManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)

    def test_manifest_row_count(self) -> None:
        self.assertEqual(len(self.manifest), 54)

    def test_manifest_columns(self) -> None:
        self.assertEqual(
            list(self.manifest.columns),
            [
                "repository_path",
                "subject_id",
                "phase",
                "header",
                "column_count",
                "data_row_count",
                "size_bytes",
                "sha256",
                "classification",
            ],
        )

    def test_manifest_classification_is_uniform(self) -> None:
        self.assertEqual(
            set(self.manifest["classification"]),
            {"author_approved_public_pseudonymised_participant_level_derived_beat_table"},
        )

    def test_manifest_header_is_uniform(self) -> None:
        self.assertEqual(set(self.manifest["header"]), {"beat_idx,RRI_ms,SBP_mmHg,PAT_ms"})

    def test_manifest_hashes_match_files(self) -> None:
        for _, row in self.manifest.iterrows():
            path = ROOT / row["repository_path"]
            with self.subTest(file=row["repository_path"]):
                self.assertTrue(path.is_file())
                self.assertEqual(_sha256(path), row["sha256"].lower())
                self.assertEqual(path.stat().st_size, int(row["size_bytes"]))

    def test_manifest_matches_beat_directory(self) -> None:
        beat_paths = {f"data/beats/{p.name}" for p in BEATS_DIR.iterdir() if p.is_file()}
        manifest_paths = set(self.manifest["repository_path"])
        self.assertEqual(beat_paths, manifest_paths)


class ApprovedDerivedBeatsSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.beats_files = sorted(p for p in BEATS_DIR.iterdir() if p.is_file())

    def test_header_is_exact(self) -> None:
        for path in self.beats_files:
            with self.subTest(file=path.name):
                with path.open(encoding="utf-8", newline="") as fh:
                    reader = csv.reader(fh)
                    header = tuple(next(reader))
                self.assertEqual(header, EXPECTED_HEADER)

    def test_all_data_fields_are_numeric_finite(self) -> None:
        for path in self.beats_files:
            with self.subTest(file=path.name):
                raw = pd.read_csv(path, dtype=str, keep_default_na=False)
                self.assertEqual(tuple(raw.columns), EXPECTED_HEADER)
                for col in EXPECTED_HEADER:
                    empties = int((raw[col].str.len() == 0).sum())
                    self.assertEqual(empties, 0, f"empty field in {col}")
                numeric = raw.apply(pd.to_numeric, errors="raise")
                for col in EXPECTED_HEADER:
                    arr = numeric[col].to_numpy(float)
                    self.assertTrue(
                        bool(np.isfinite(arr).all()),
                        f"non-finite value in {col}",
                    )

    def test_beat_idx_is_integer_dense_and_strictly_increasing(self) -> None:
        for path in self.beats_files:
            with self.subTest(file=path.name):
                raw = pd.read_csv(path, dtype=str, keep_default_na=False)
                beat_idx_float = pd.to_numeric(raw["beat_idx"], errors="raise").to_numpy(float)
                self.assertTrue(
                    bool(np.all(beat_idx_float == np.floor(beat_idx_float))),
                    "beat_idx values not integer",
                )
                beat_idx = beat_idx_float.astype(np.int64)
                self.assertEqual(int(beat_idx[0]), 1)
                self.assertTrue(
                    bool(np.all(np.diff(beat_idx) == 1)),
                    "beat_idx not dense strictly increasing",
                )
                self.assertEqual(int(beat_idx[-1]), len(beat_idx))

    def test_physiological_ranges(self) -> None:
        for path in self.beats_files:
            with self.subTest(file=path.name):
                raw = pd.read_csv(path, dtype=str, keep_default_na=False)
                rri = pd.to_numeric(raw["RRI_ms"], errors="raise")
                sbp = pd.to_numeric(raw["SBP_mmHg"], errors="raise")
                self.assertTrue(bool(((rri > 200) & (rri < 3000)).all()))
                self.assertTrue(bool(((sbp > 40) & (sbp < 260)).all()))

    def test_no_date_or_identifier_columns(self) -> None:
        forbidden = {
            "date",
            "datetime",
            "timestamp",
            "session_start",
            "acquisition_time",
            "mrn",
            "patient_id",
            "dob",
            "birthdate",
            "birth_date",
            "name",
        }
        for path in self.beats_files:
            with self.subTest(file=path.name):
                frame = pd.read_csv(path, nrows=1)
                lowered = {c.lower() for c in frame.columns}
                self.assertTrue(forbidden.isdisjoint(lowered), lowered & forbidden)


class ValidatorAllowListGateTests(unittest.TestCase):
    def test_check_author_approved_derived_beats_passes(self) -> None:
        module = _load_validator_module()
        detail = module.check_author_approved_derived_beats()
        self.assertIn("54", detail)

    def test_check_public_tree_safety_passes(self) -> None:
        module = _load_validator_module()
        detail = module.check_public_tree_safety()
        self.assertIn("no", detail.lower())


if __name__ == "__main__":
    unittest.main()
