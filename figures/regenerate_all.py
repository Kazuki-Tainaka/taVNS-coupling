#!/usr/bin/env python
"""Regenerate every published figure and record output MD5 checksums."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

FIGURE_SCRIPTS = [
    "figures/main/generate_fig1.py",
    "figures/main/generate_fig2.py",
    "figures/main/generate_fig3.py",
    "figures/main/generate_fig4.py",
    "figures/supplementary/generate_figS1.py",
    "figures/supplementary/generate_figS2.py",
    "figures/supplementary/generate_figS3.py",
    "figures/supplementary/generate_figS4.py",
    "figures/supplementary/generate_figS5.py",
    "figures/supplementary/generate_figS6.py",
    "figures/supplementary/generate_figS7.py",
]

OUTPUT_DIR = Path("figures/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    results = []
    failed = False
    for script in FIGURE_SCRIPTS:
        print(f">>> {script}")
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, check=False)
        fig_name = Path(script).stem.replace("generate_", "").lower() + ".png"
        fig_path = OUTPUT_DIR / fig_name
        if proc.returncode == 0 and fig_path.exists():
            digest = hashlib.md5(fig_path.read_bytes()).hexdigest()
            results.append(f"PASS {fig_name} md5={digest}")
        else:
            failed = True
            results.append(f"FAIL {fig_name}: {(proc.stderr or proc.stdout)[:500]}")
    manifest = OUTPUT_DIR / "_md5_manifest.txt"
    manifest.write_text("\n".join(results) + "\n", encoding="utf-8")
    print("\n".join(results))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
