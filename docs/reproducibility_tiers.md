# Reproducibility Tiers

This document defines the reproducibility levels supported by this package and
lists what each level covers.

## Tier S

Canonical byte-level preservation for `data/reference/Additional_File_2.csv`
and `data/reference/Additional_File_3.csv`. MD5 checksums are verified by
`tests/test_md5_match.py`.

## Tier A

Anchor metric verification for 11 coupling metrics: BRS sequence metrics,
rhomax, bivariate GC F statistics, and trivariate GC3 F statistics.

## Tier B

Materialized canonical tables are preserved for metrics whose full
beat-to-beat recomputation is outside the public package scope.

## Tier P

Per-subject derived values for all 46 coupling metrics and all 74 HRV metrics
are available in `data/derived/per_subject_coupling.csv` and
`data/derived/per_subject_hrv.csv`.

## Tier I

Intermediate outputs are persisted under `data/derived/`:

- rhomax sliding-window values per subject and condition
- Wavelet transform coherence group averages
- Fixed-lag cross-correlation profiles for causal and zerophase filters
- Bivariate and trivariate VAR residual covariances
- BRS sequence ramp counts and event counts
- Bootstrap replicate arrays
- Temporal classification assignments
- Segmented time-series coefficients

## Tier F

All figures are regenerable via `python figures/regenerate_all.py`. Figure
scripts declare data dependencies in module docstrings and read numerical
values from `data/derived/` or `data/reference/`. The regenerated PNGs are
committed under `figures/outputs/`.

## Dependency Matrix

| Figure | Data dependency |
|---|---|
| Fig1 | `data/derived/per_subject_coupling.csv` and `data/reference/Additional_File_2.csv` |
| Fig2 | `data/derived/per_subject_coupling.csv` and bootstrap replicates |
| Fig3 | `data/derived/fixed_lag_cross_correlation/` |
| Fig4 | `data/derived/bootstrap/` and temporal classification |
| FigS1 | `data/reference/Additional_File_2.csv` |
| FigS2 | `data/reference/Additional_File_3.csv` |
| FigS3 | `data/derived/wtc/` |
| FigS4 | `data/derived/fixed_lag_cross_correlation/` |
| FigS5 | `data/derived/rhomax_windows/` and segmented summaries |
| FigS6 | temporal classification and per-subject coupling data |
| FigS7 | `data/derived/per_subject_coupling.csv` |
