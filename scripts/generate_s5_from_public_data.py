"""Regenerate Supplementary Figure S5 from public Supplementary Data 3."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_summary_figures import (  # noqa: E402
    configure_style,
    make_supplementary_figure_s5,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data3",
        type=Path,
        default=(
            ROOT
            / "expected_outputs"
            / "publication_source_data"
            / "supplementary_data_3_brs_sensitivity_and_coupling_significance.csv"
        ),
    )
    parser.add_argument(
        "--stem",
        default="supplementary_figure_s5_native_var_diagnostics",
    )
    args = parser.parse_args()
    configure_style()
    paths, _, _ = make_supplementary_figure_s5(
        data3_path=args.data3,
        output_dir=args.output_dir,
        stem=args.stem,
    )
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
