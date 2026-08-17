# Scientific Reports taVNS analysis code and public derived beat tables

Analysis code, fixed configurations, synthetic validation, tests, publication
source tables, and the approved pseudonymised participant-level derived
beat-to-beat tables for “Acute transcutaneous auricular vagus nerve stimulation
is associated with lower baroreflex sensitivity without detectable Mayer-wave
coherence change: an exploratory study”.

The study used a fixed-order, within-participant Pre–Stim–Post design in 18
healthy adult males, without a sham condition. The findings are exploratory and
do not establish a causal stimulation effect.

## Reproducibility tiers

### Public tier A — syntax, tests, synthetic validation, plotting

The public package supports syntax checks, deterministic unit tests, synthetic
BRS/coherence/VAR/nonlinear validation, metadata checks, and figure-generation
checks without participant recordings.

```powershell
python -m pip install -r environment/requirements-lock.txt
./validate_public_release.ps1
```

Equivalent commands are:

```powershell
python -m pytest tests -q
python scripts/validate_public_release.py
```

Typical validation takes under two minutes on a desktop computer and uses
modest memory. It does not read raw participant recordings.

### Public tier B — approved public derived beat tables

The 54 approved pseudonymised participant-level derived beat tables under
`data/beats/S01_Pre.csv … data/beats/S18_Post.csv` are part of the public
package. Each file has four numeric finite columns
(`beat_idx, RRI_ms, SBP_mmHg, PAT_ms`), no direct identifier, and no absolute
acquisition date/time. The data license is Creative Commons Attribution 4.0
International, stated in `data/LICENSE` and `data/README.md`; the same CC BY
4.0 license also covers the aggregate publication tables under
`expected_outputs/`. The exact filename set and expected SHA-256 hashes for
these 54 files are enumerated in
`config/approved_derived_beats_manifest.csv` and enforced by the validator's
`author_approved_derived_beats` gate.

These tables enable independent, beats-only reimplementations of
beat-index-native analyses that operate directly on
`(beat_idx, RRI_ms, SBP_mmHg, PAT_ms)` sequences — for example, secondary
sequence-BRS estimates, native paired-beat VAR-based directional diagnostics,
methods-text-matched BRS sanity comparators, and native-beat nonlinear
sensitivity. The v1.1.0 automated gate validates the exact integrity of these
54 tables (filename set, header, SHA-256, size, row count, numeric-finite
schema, and beat-index density) alongside the aggregate publication-source
anchors under `expected_outputs/`.

Two reproducibility limits are stated up front:

1. The shipped `src/` analysis pipeline reads the upstream paired-beat source
   layout (`paired/paired_beats_XX.csv` with `R_wave_timing_ms` and
   `sBP_timing_ms`), not the phase-separated `data/beats/S??_(Pre|Stim|Post).csv`
   layout. No adapter and no beats-only execution entry point is shipped in
   this release, and no beat-table-to-anchor numerical test is exercised.
   The package therefore does not itself directly re-run sequence-BRS, VAR,
   methods-text-matched BRS, or nonlinear analyses from `data/beats/`.
   Independent researchers can build such adapters or beats-only
   implementations from the four beat-level scalars supplied here; the public
   gate exercises the aggregate publication source tables under
   `expected_outputs/` (see tier C).
2. Analyses that require the original continuous time axis — the Fourier /
   wavelet coherence pipelines and any 4-Hz-resampled analyses — cannot be
   reproduced exactly from `data/beats/` alone, because the beat-timing
   columns (`R_wave_timing_ms`, `sBP_timing_ms`) are intentionally not part
   of the approved public schema. Exact raw-to-beat and exact
   continuous-time/coherence reproduction are unsupported by this release.
   This is an honest scope limitation, not a validation failure; the
   aggregate results for those analyses are preserved and audited under
   `expected_outputs/`.

### Public tier C — manuscript-output audit

`expected_outputs/publication_source_data/` contains publication-approved
derived tables. `scripts/validate_public_release.py` checks central BRS, the
72-setting BRS landscape, coherence and surrogate prevalence, haemodynamic
context, native paired-beat VAR diagnostics, methods-text-matched BRS
branches, and targeted nonlinear sensitivity.

Supplementary Figure S3 is represented by two complementary public artifacts:

- `figures/supplementary_figure_s3_brs_specification_landscape.jpg` is the
  author-finished publication artwork. Its scientific values and layout were
  audited; typography, spacing, and label placement include authorial figure
  finishing.
- `scripts/generate_supplementary_figure_s3.py` is the underlying data-driven
  generator. It reads Supplementary Data 3 and generates the specification
  curve, evaluable-sample-size point plot, parameter matrix, and direction
  summary. Small typographic or layout differences from the publication
  artwork are not scientific discrepancies.

Regenerate the data-driven S3 preview with:

```powershell
python scripts/generate_supplementary_figure_s3.py `
  --supplementary-data-3 expected_outputs/publication_source_data/supplementary_data_3_brs_sensitivity_and_coupling_significance.csv `
  --output-dir generated_outputs
```

Regenerate Supplementary Figure S5 from the public summary table with:

```powershell
python scripts/generate_s5_from_public_data.py --output-dir generated_outputs
```

SVG bytes can vary across valid Matplotlib/font installations. Validation checks
source-data anchors and semantic rendering instead of requiring cross-platform
byte identity.

### Controlled tier — full raw-waveform-to-result pipeline

Exact rerunning from raw ECG or continuous blood-pressure waveforms requires
controlled access under the article’s Data Availability statement and applicable
ethics approval. Raw ECG, continuous blood-pressure waveforms, upstream
paired-source paired-beat streams, and manually reviewed participant-level
intermediates are not included. The public repository alone does not reproduce
the raw-waveform-to-result pipeline because MATLAB preprocessing and manual
event review require non-public inputs.

Set `TAVNS_DATA_ROOT` to an authorized data directory outside the repository.
The expected controlled layout is documented in
`config/data_paths.example.yaml` and `docs/CONTROLLED_REPRODUCTION.md`.

## Reference implementation

The output-generating sequence-BRS reference uses maximal non-overlapping
monotonic SBP ramps of at least three beats, successive absolute SBP steps of
at least 1.0 mmHg, a one-beat SBP–RRI lag, Pearson correlation at least 0.80, a
finite OLS slope of RRI on SBP, and arithmetic-mean aggregation. Beat-by-beat
monotonic RRI change is not an eligibility requirement. A directional-RRI gate
remains available only as an explicitly labelled sensitivity.

The machine-readable configuration is `config/revision_analysis_config.json`.
Scientific estimators, thresholds, lags, sequence enumeration, coherence
settings, model-order limits, and fixed seeds are locked.

## Package map

- `src/`: central BRS, coherence, haemodynamic, statistics, native VAR,
  methods-matched BRS, and nonlinear sensitivity source.
- `scripts/`: public validation and plotting entry points.
- `tests/`: public synthetic and unit tests; tests requiring controlled
  participant data are excluded.
- `preprocessing_matlab/`: non-data MATLAB source for the historical /
  semi-manual preprocessing stage.
- `expected_outputs/`: publication-approved source tables and aggregate expected
  outputs (CC BY 4.0).
- `figures/`: non-sensitive publication artwork included for audit.
- `docs/`: provenance, privacy boundary, software versions, and controlled-
  reproduction notes.
- `data/beats/`: the 54 approved pseudonymised participant-level derived
  beat tables (`beat_idx`, `RRI_ms`, `SBP_mmHg`, `PAT_ms`).
- `data/LICENSE`, `data/README.md`: data-license text and schema description.
- `config/approved_derived_beats_manifest.csv`: exhaustive filename, SHA-256,
  size, and row-count manifest for the approved 54 files.

## Data and privacy boundary

The package contains the 54 approved pseudonymised participant-level derived
beat tables under `data/beats/` (columns `beat_idx`, `RRI_ms`, `SBP_mmHg`,
`PAT_ms`; no direct identifiers and no absolute acquisition date/time).
It contains no raw ECG or continuous blood-pressure waveform, no upstream
paired-source paired-beat stream, no credential, cookie, private key, or
environment file, and no host-specific absolute path. See
`docs/DATA_BOUNDARY.md` for the detailed boundary and reproducibility scope.

## Citation, archive series, and license

Repository: <https://github.com/Kazuki-Tainaka/taVNS-coupling>

Zenodo concept DOI: `10.5281/zenodo.20323694`

Release metadata are in `.zenodo.json`. The software and analysis source code
(`src/`, `scripts/`, `tests/`, `preprocessing_matlab/`, `synthetic_fixtures/`,
and top-level `LICENSE`) are distributed under the existing author-approved
MIT License. All derived and publication data tables — the 54 approved public
derived beat tables under `data/beats/` and the publication-approved source
tables under `expected_outputs/` — are distributed under the Creative Commons
Attribution 4.0 International License, as stated in `data/LICENSE`. The two
licenses apply to distinct file classes; the derived data tables are not
covered by MIT.
