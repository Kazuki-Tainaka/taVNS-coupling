"""Build a byte-deterministic release ZIP from the committed Git tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ArchiveEntry:
    """One committed path, mode, and immutable blob payload."""

    path: str
    mode: int
    data: bytes


def run_git(*arguments: str, text: bool = False) -> subprocess.CompletedProcess:
    """Run Git in the repository and fail with captured diagnostics."""
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
    )


def committed_entries() -> list[ArchiveEntry]:
    """Read sorted HEAD blobs and their Git modes without EOL conversion."""
    status = run_git("status", "--porcelain=v1", "-uall", text=True).stdout
    if status:
        raise RuntimeError("release tree must be clean before deterministic build")
    stage_lines = run_git("ls-files", "--stage", "-z").stdout.split(b"\0")
    entries: list[ArchiveEntry] = []
    for raw_line in stage_lines:
        if not raw_line:
            continue
        metadata, raw_path = raw_line.split(b"\t", maxsplit=1)
        mode_text, _, _ = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        blob = run_git("show", f"HEAD:{path}").stdout
        entries.append(ArchiveEntry(path=path, mode=int(mode_text, 8), data=blob))
    return sorted(entries, key=lambda entry: entry.path)


def write_archive(
    entries: Iterable[ArchiveEntry],
    output: Path,
    prefix: str,
) -> None:
    """Write entries with fixed metadata, ordering, mode, and compression."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for entry in entries:
            info = zipfile.ZipInfo(
                filename=f"{prefix}/{entry.path}",
                date_time=FIXED_TIMESTAMP,
            )
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = entry.mode << 16
            archive.writestr(info, entry.data)


def parse_args() -> argparse.Namespace:
    """Parse output and optional archive-root override."""
    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    version = metadata["version"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default=f"taVNS-coupling-v{version}",
        help="single top-level archive directory",
    )
    return parser.parse_args()


def main() -> None:
    """Build the deterministic release asset."""
    args = parse_args()
    expected_prefix = "taVNS-coupling-v1.1.2"
    if args.prefix != expected_prefix:
        raise ValueError(f"release prefix must be {expected_prefix}")
    write_archive(committed_entries(), args.output, args.prefix)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
