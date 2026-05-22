# Changelog

## v1.0.1 - 2026-05-22

### Added

- Per-subject derived data layer: `data/derived/per_subject_coupling.csv` and `data/derived/per_subject_hrv.csv`.
- Intermediate computation outputs under `data/derived/`: rhomax sliding windows, WTC group averages, fixed-lag cross-correlation, VAR residuals, BRS ramps, bootstrap replicates, temporal classification, and segmented time-series coefficients.
- `figures/` directory with all 11 figure generation scripts, using data-driven inputs.
- `figures/regenerate_all.py` and `figures/style.py`.
- New scripts for the derived-data pipeline.
- New tests for derived-data consistency, figure regeneration, and hardcoded numerical arrays.
- `docs/reproducibility_tiers.md`.

### Changed

- Additional File 3 BF01 columns use the pingouin standard t-statistic convention: t = dz times the square root of n. New MD5: `5fd37ceb5269c0558131a02efbb6ba95`.
- `scripts/compute_coupling_metrics.py` and `scripts/compute_hrv_metrics.py` gained `--emit-per-subject`.
- `scripts/run_all.py` now orchestrates derived-data generation and figure regeneration.
- `README.md`, `docs/methods_appendix.md`, and `docs/data_dictionary.md` describe Tier P, Tier I, and Tier F reproducibility.

### Corrected after initial v1.0.1 build (2026-05-22)

The figure generation scripts initially shipped with v1.0.1 produced
placeholder outputs and did not reproduce the published figures. This release
replaces all eleven figure scripts under `figures/main/` and
`figures/supplementary/` with the authoritative versions used for the JNER
submission. The eleven regenerated PNGs under `figures/outputs/` now correspond
to the published figures.

The hardcoded-array test was retained. One axis tick constant used by the
authoritative WTC figure script was explicitly allowlisted as structural figure
metadata.

### Unchanged

- `data/reference/Additional_File_2.csv`: `474f5e1792065b62b5711830ad585d95`
- `results/Additional_File_2.csv`: `474f5e1792065b62b5711830ad585d95`
- Zenodo concept DOI: `10.5281/zenodo.20323694`

### Citation

This release shares the Zenodo concept DOI `10.5281/zenodo.20323694` with the
baseline release. The version-specific DOI will be assigned upon Zenodo deposit.
