from pathlib import Path
import hashlib
import numpy as np
import subprocess
import sys

from scripts.lib import filters, hrv, quality


def test_anchor_recompute_report_passes():
    result = subprocess.run(
        [sys.executable, "scripts/01_compute_coupling_metrics.py", "--recompute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = Path("docs/anchor_tolerance_report.md")
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "ALL ANCHORS WITHIN TOLERANCE" in text
    assert "ANCHOR TOLERANCE FAILURE" not in text
    assert "abs Delta dz" in text
    assert "Provenance check" in text
    assert "Assertion: recomputed MD5 != canonical MD5: PASS" in text
    assert Path("results/Additional_File_2_recomputed.csv").exists()


def test_recompute_is_not_materialization():
    canonical = Path("results/Additional_File_2.csv").read_bytes()
    recomputed = Path("results/Additional_File_2_recomputed.csv").read_bytes()
    canonical_md5 = hashlib.md5(canonical).hexdigest()
    recomputed_md5 = hashlib.md5(recomputed).hexdigest()
    assert canonical_md5 != recomputed_md5, (
        "Recomputed CSV is byte-identical to canonical. --recompute is not "
        "producing a distinct verification artifact."
    )


def test_pat_gate_and_filter_provenance():
    assert not quality.validate_pat(np.array([1e-13, 300.0]))
    assert quality.validate_pat(np.array([250.0, 300.0]))
    x = np.linspace(0, 1, 100) + 0.1 * np.sin(np.linspace(0, 10, 100))
    causal = filters.causal_filter(x, 0.2, 4.0)
    zero = filters.zerophase_filter(x, 0.2, 4.0)
    assert np.max(np.abs(causal - zero)) > 1e-9


def test_basic_hrv_functions():
    rri = np.array([800.0, 810.0, 790.0, 805.0])
    out = hrv.compute_basic_hrv(rri)
    assert out["Mean_RRI"] == 801.25
    assert out["SDNN"] > 0
    assert out["RMSSD"] > 0
