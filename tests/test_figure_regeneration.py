from pathlib import Path
import subprocess
import sys

import pytest

FIGURE_SCRIPTS = [
    ("Fig1", "figures/main/generate_fig1.py"),
    ("Fig2", "figures/main/generate_fig2.py"),
    ("Fig3", "figures/main/generate_fig3.py"),
    ("Fig4", "figures/main/generate_fig4.py"),
    ("FigS1", "figures/supplementary/generate_figS1.py"),
    ("FigS2", "figures/supplementary/generate_figS2.py"),
    ("FigS3", "figures/supplementary/generate_figS3.py"),
    ("FigS4", "figures/supplementary/generate_figS4.py"),
    ("FigS5", "figures/supplementary/generate_figS5.py"),
    ("FigS6", "figures/supplementary/generate_figS6.py"),
    ("FigS7", "figures/supplementary/generate_figS7.py"),
]


@pytest.mark.parametrize("name,script", FIGURE_SCRIPTS)
def test_figure_runs(name: str, script: str) -> None:
    result = subprocess.run([sys.executable, script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{name} script failed: {result.stderr[:500]}"
    out = Path(f"figures/outputs/{name.lower()}.png")
    assert out.exists(), f"{name} did not produce {out}"
    assert out.stat().st_size > 1000, f"{name} produced suspiciously small file"
