#!/usr/bin/env python
"""Full pipeline: public beats and canonical tables to derived data and figures."""
from __future__ import annotations

import subprocess
import sys

STEPS = [
    [sys.executable, "scripts/compute_coupling_metrics.py", "--emit-per-subject"],
    [sys.executable, "scripts/compute_hrv_metrics.py", "--emit-per-subject"],
    [sys.executable, "scripts/compute_rhomax_windows.py"],
    [sys.executable, "scripts/compute_wtc.py"],
    [sys.executable, "scripts/compute_fixed_lag_cross_correlation.py"],
    [sys.executable, "scripts/compute_var_residuals.py"],
    [sys.executable, "scripts/compute_brs_ramps.py"],
    [sys.executable, "scripts/compute_bootstrap_replicates.py"],
    [sys.executable, "scripts/compute_temporal_classification.py"],
    [sys.executable, "scripts/compute_its_regression.py"],
    [sys.executable, "figures/regenerate_all.py"],
]


def main() -> None:
    for cmd in STEPS:
        print(f">>> {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"FAILED: {' '.join(cmd)}")
            sys.exit(result.returncode)
    print("=== Full reproduction pipeline completed ===")


if __name__ == "__main__":
    main()
