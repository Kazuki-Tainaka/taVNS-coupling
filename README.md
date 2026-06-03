# taVNS-coupling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20323694.svg)](https://doi.org/10.5281/zenodo.20323694)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Reproducibility package for:

> Katahara Y, Iijima A, Tainaka K. Transcutaneous auricular vagus nerve stimulation is associated with transient baroreflex sensitivity reduction while preserving Mayer-wave coherence: a comprehensive analysis of cardiovascular coupling and heart-rate-variability metrics. Submitted.

## What This Package Provides

This repository contains de-identified beat-to-beat data, canonical result
tables, derived subject-level data, intermediate computation outputs, and figure
generation scripts for the taVNS cardiovascular coupling study.

- `data/beats/`: 54 beat-to-beat CSV files with RRI, SBP, and PAT columns.
- `data/reference/`: canonical Additional File 2 and Additional File 3 CSVs.
- `data/derived/`: per-subject values and intermediate arrays used by figures.
- `results/`: materialized canonical CSV outputs.
- `scripts/`: full reproduction pipeline entry points.
- `figures/`: data-driven figure scripts and regenerated PNG outputs.
- `tests/`: MD5 checks, derived-data consistency checks, figure checks, and de-identification checks.

## Reproduction

1. `pip install -r requirements.txt`
2. `pip install -r requirements-dev.txt`
3. `python scripts/run_all.py`
4. Figures are written to `figures/outputs/`
5. Per-figure regeneration is available, for example `python figures/main/generate_fig2.py`

`scripts/run_all.py` materializes the canonical CSVs, regenerates all derived
data under `data/derived/`, and then runs `figures/regenerate_all.py`.

## Figure Regeneration

```bash
python figures/regenerate_all.py
```

Figure regeneration is fully data-driven from `data/derived/` and
`data/reference/`.

The script runs all 11 figure generators and writes an MD5 manifest to
`figures/outputs/_md5_manifest.txt`.

## Reproducibility Tiers

Reproducibility tiers are documented for canonical checks, anchor checks,
per-subject persistence, intermediate outputs, and figure regeneration.

The package supports Tier S, Tier A, Tier B, Tier P, Tier I, and Tier F
reproducibility. See `docs/reproducibility_tiers.md` for the full definition and
figure-to-data dependency matrix.

## Canonical Checksums

- `data/reference/Additional_File_2.csv`: `474f5e1792065b62b5711830ad585d95`
- `data/reference/Additional_File_3.csv`: `df4edfa0c874ddc684e19a43f8b60038`
- `results/Additional_File_2.csv`: `474f5e1792065b62b5711830ad585d95`
- `results/Additional_File_3.csv`: `df4edfa0c874ddc684e19a43f8b60038`

Additional File 3 uses the pingouin standard t-statistic convention for BF01
calculation: t = dz times the square root of n.

## Anchor Verification Mode

```bash
python scripts/compute_coupling_metrics.py --recompute
```

This writes `results/Additional_File_2_recomputed.csv` and
`docs/anchor_tolerance_report.md`. The recomputed CSV is intentionally
byte-distinct from the canonical CSV while preserving the anchor validation
contract used by the test suite.

## Ethics and Trial Registration

- Trial registration: [jRCT1032230440](https://jrct.niph.go.jp/latest-detail/jRCT1032230440)
- Ethical approval: Niigata University Ethics Review Committee, approval number 2023-0191
- Written informed consent was obtained from all participants

## Citation

If you use this code or data, please cite:

> Katahara Y, Iijima A, Tainaka K. taVNS-coupling: reproducibility package for transcutaneous auricular vagus nerve stimulation cardiovascular coupling analysis (Version 1.0.2) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20323694

The release notes are available in `CHANGELOG.md`.

## License

- Code: MIT License (see `LICENSE`)
- Data and derived tables: CC BY 4.0 (see `data/LICENSE`)

## Contact

For questions about this code and data package, please contact either of the
co-corresponding authors:

- **Atsuhiko Iijima** - a-iijima@eng.niigata-u.ac.jp  
  Faculty of Engineering, Niigata University
- **Kazuki Tainaka** - kztainaka@bri.niigata-u.ac.jp  
  Brain Research Institute, Niigata University

For data access questions or replication issues, please open an issue on this
repository.
