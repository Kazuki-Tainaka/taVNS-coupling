"""Test configuration for the frozen nonlinear-coupling pipeline."""

from __future__ import annotations

from pathlib import Path
import sys


CODE_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "nonlinear_sensitivity"
)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
