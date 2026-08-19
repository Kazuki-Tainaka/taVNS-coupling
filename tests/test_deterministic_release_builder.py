"""Unit test the deterministic ZIP writer without requiring Git metadata."""

from __future__ import annotations

from hashlib import sha256
import sys
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_deterministic_release import ArchiveEntry, write_archive


class DeterministicReleaseBuilderTests(unittest.TestCase):
    """Prove fixed ordering and metadata produce identical archive bytes."""

    def test_two_builds_are_byte_identical(self) -> None:
        entries = [
            ArchiveEntry("z.txt", 0o100644, b"last\n"),
            ArchiveEntry("a.txt", 0o100644, b"first\n"),
        ]
        with tempfile.TemporaryDirectory(prefix="tavns_release_builder_") as temp:
            first = Path(temp) / "first.zip"
            second = Path(temp) / "second.zip"
            write_archive(sorted(entries, key=lambda item: item.path), first, "release")
            write_archive(sorted(entries, key=lambda item: item.path), second, "release")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                sha256(first.read_bytes()).hexdigest(),
                sha256(second.read_bytes()).hexdigest(),
            )
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.namelist(), ["release/a.txt", "release/z.txt"])
                self.assertEqual(
                    {info.date_time for info in archive.infolist()},
                    {(1980, 1, 1, 0, 0, 0)},
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
