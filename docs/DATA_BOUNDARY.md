# Public data boundary

## Included

- Analysis and plotting source code.
- Fixed configuration and seeds.
- Synthetic fixtures and public unit tests.
- Publication-approved Supplementary Data 1–3 and Figure 2 scalar source tables.
- Aggregate native-VAR, methods-text-matched BRS, and nonlinear sensitivity outputs.
- Author-finished Supplementary Figure S3 artwork and its data-driven generator.
- The 54 approved pseudonymised participant-level derived beat tables under
  `data/beats/S01_Pre.csv … data/beats/S18_Post.csv`, with columns
  `beat_idx`, `RRI_ms`, `SBP_mmHg`, `PAT_ms` (all fields numeric and
  finite). The data license is Creative Commons Attribution 4.0
  International, stated explicitly in `data/LICENSE` and `data/README.md`;
  the same license covers the aggregate publication tables under
  `expected_outputs/`. The software / analysis source in the surrounding
  repository (`src/`, `scripts/`, `tests/`, `preprocessing_matlab/`,
  `synthetic_fixtures/`, top-level `LICENSE`) is separately licensed under
  MIT; the two licenses apply to distinct file classes and the derived data
  are not covered by MIT. The 54 files are exhaustively enumerated, with
  expected SHA-256 hashes, in `config/approved_derived_beats_manifest.csv`.

Supplementary Data 3 contains pseudonymous participant labels and scalar participant/phase summaries intended for journal publication. It contains no continuous signal or beat sequence.

## Excluded

- Raw ECG and continuous blood-pressure waveforms.
- Upstream paired-source paired-beat streams and any acquisition-format binaries
  (`*.mat`, `*.acq`, `*.edf`, `paired_beats_*.csv`).
- Controlled intermediate physiological time series and selected nonlinear-analysis series.
- Manual-review workspaces, semi-manual event-detection intermediates, and
  original local data directories.
- Direct participant identifiers, absolute acquisition dates/times, and any
  linkage keys.
- Supplementary Data 4, Word documents, and journal upload packages.
- Credentials, tokens, cookies, private keys, environment files, caches, and host-specific paths.

## Reproducibility scope of the approved derived beat tables

The public `data/beats/` layer publishes 54 four-column beat-level tables
(`beat_idx`, `RRI_ms`, `SBP_mmHg`, `PAT_ms`) whose exact integrity is
verified by the v1.1.0 automated gate together with the aggregate
publication-source anchors under `expected_outputs/`. The tables enable
independent researchers to build beats-only implementations and secondary
analyses of beat-index-native quantities such as:

- baroreflex-sensitivity sequence analysis on the public beat streams,
- native paired-beat VAR-based directional diagnostics,
- native-beat nonlinear sensitivity,
- methods-text-matched BRS sanity comparators.

This release does not itself directly re-run those analyses from
`data/beats/`. No adapter from the phase-separated public layout to the
upstream paired-source paired-beat layout is shipped, no beats-only execution
entry point is provided, and no beat-table-to-anchor numerical test is
exercised. The public gate exercises the aggregate publication source tables
under `expected_outputs/` for numerical verification.

The public package cannot by itself reproduce raw ECG or continuous
blood-pressure preprocessing, nor the semi-manual event-detection stage that
produced the derived tables. Exact raw-to-beat reproduction and exact
continuous-time / coherence reproduction (Fourier / wavelet coherence
pipelines and any 4-Hz-resampled analyses) are unsupported by this release;
the aggregate audit numbers for those analyses are preserved under
`expected_outputs/` and verified by the tier C anchors. These are honest
scope limitations, not validation failures.

Access to excluded data is governed by the article’s Data Availability
statement, institutional requirements, and ethics approval.
