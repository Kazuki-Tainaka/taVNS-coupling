"""Shared I/O and boundary utilities for the Scientific Reports revision."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


RELEASE_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PROJECT_ROOT: Final[Path] = Path(
    os.environ.get(
        "TAVNS_DATA_ROOT",
        RELEASE_ROOT / "controlled_data_not_included",
    )
).resolve()
REVISION_ROOT: Final[Path] = Path(
    os.environ.get("TAVNS_OUTPUT_ROOT", RELEASE_ROOT / "outputs")
).resolve()
RESULTS_DIR: Final[Path] = REVISION_ROOT / "02_analysis" / "results"
TABLES_DIR: Final[Path] = REVISION_ROOT / "02_analysis" / "tables"
FIGURE_DATA_DIR: Final[Path] = REVISION_ROOT / "02_analysis" / "figure_source_data"
REPORTS_DIR: Final[Path] = REVISION_ROOT / "02_analysis" / "reports"
PAIRED_DIR: Final[Path] = PROJECT_ROOT / "paired"
RAW_DIR: Final[Path] = PROJECT_ROOT / "raw"

SUBJECTS: Final[tuple[int, ...]] = tuple(range(1, 19))
PHASES: Final[dict[str, tuple[float, float]]] = {
    "Pre": (0.0, 300.0),
    "Stim": (300.0, 600.0),
    "Post": (600.0, 900.0),
}
PHASE_ORDER: Final[tuple[str, ...]] = ("Pre", "Stim", "Post")
FS_4HZ: Final[float] = 4.0
PAIRED_COLUMNS: Final[list[str]] = [
    "R_wave_timing_ms",
    "RRI_ms",
    "sBP_timing_ms",
    "sBP_mmHg100",
    "PAT_ms",
]


def normcase(path: Path) -> str:
    """Return a resolved, case-normalized Windows path string."""
    return os.path.normcase(str(path.resolve()))


def assert_boundaries() -> None:
    """Fail before writes if inputs are absent or outputs escape their root."""
    if not PROJECT_ROOT.exists():
        raise RuntimeError(f"Input project root does not exist: {PROJECT_ROOT}")
    if REVISION_ROOT == PROJECT_ROOT or normcase(PROJECT_ROOT).startswith(
        normcase(REVISION_ROOT) + os.sep
    ):
        raise RuntimeError("Output root must be separate from the read-only project input")
    if REVISION_ROOT.is_symlink():
        raise RuntimeError("Output root must not be a symbolic link")


def ensure_output_dirs() -> None:
    """Create only the approved output directories."""
    assert_boundaries()
    for path in (RESULTS_DIR, TABLES_DIR, FIGURE_DATA_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
        if not normcase(path).startswith(normcase(REVISION_ROOT) + os.sep):
            raise RuntimeError(f"Unsafe output path: {path}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_paired_full(subject: int) -> pd.DataFrame:
    """Load one authoritative paired-beat file without modifying the source."""
    path = PAIRED_DIR / f"paired_beats_{subject:02d}.csv"
    frame = pd.read_csv(path, header=None, names=PAIRED_COLUMNS)
    frame["subject"] = subject
    frame["beat_time_s"] = frame["R_wave_timing_ms"].astype(float) / 1000.0
    frame["SBP_mmHg"] = frame["sBP_mmHg100"].astype(float) * 100.0
    return frame


def load_paired_phase(subject: int, phase: str) -> pd.DataFrame:
    """Load valid RRI/SBP paired beats for one prespecified phase."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    t0, t1 = PHASES[phase]
    frame = load_paired_full(subject)
    mask = (frame["beat_time_s"] >= t0) & (frame["beat_time_s"] < t1)
    phase_frame = frame.loc[mask].copy()
    valid = (
        np.isfinite(phase_frame["RRI_ms"].to_numpy(float))
        & np.isfinite(phase_frame["SBP_mmHg"].to_numpy(float))
        & (phase_frame["RRI_ms"].to_numpy(float) > 0.0)
    )
    return phase_frame.loc[valid].reset_index(drop=True)


def resample_phase_4hz(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Natural-cubic-spline resampling within a single phase only."""
    times = frame["beat_time_s"].to_numpy(float)
    sbp = frame["SBP_mmHg"].to_numpy(float)
    rri = frame["RRI_ms"].to_numpy(float)
    if len(times) < 4:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    if np.any(np.diff(times) <= 0):
        raise ValueError("Beat times are not strictly increasing")
    grid = np.arange(times[0], times[-1], 1.0 / FS_4HZ)
    if len(grid) < 10:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    sbp_spline = CubicSpline(times, sbp, bc_type="natural")
    rri_spline = CubicSpline(times, rri, bc_type="natural")
    return grid, sbp_spline(grid), rri_spline(grid)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a machine-readable CSV with explicit NA encoding."""
    assert_boundaries()
    resolved_parent = path.resolve().parent
    if not normcase(resolved_parent).startswith(normcase(REVISION_ROOT) + os.sep):
        raise RuntimeError(f"Refusing write outside revision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", na_rep="NA", lineterminator="\n")


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    """Write a machine-readable TSV with explicit NA encoding."""
    assert_boundaries()
    resolved_parent = path.resolve().parent
    if not normcase(resolved_parent).startswith(normcase(REVISION_ROOT) + os.sep):
        raise RuntimeError(f"Refusing write outside revision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
        na_rep="NA",
        lineterminator="\n",
    )
