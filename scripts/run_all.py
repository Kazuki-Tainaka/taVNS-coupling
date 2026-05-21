#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys

SCRIPTS = [
    "scripts/01_compute_coupling_metrics.py",
    "scripts/02_compute_hrv_metrics.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"=== Running {script} ===")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            print(f"FAILED: {script}")
            sys.exit(result.returncode)
    print("All scripts completed successfully.")


if __name__ == "__main__":
    main()
