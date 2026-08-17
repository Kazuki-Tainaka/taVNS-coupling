# Changelog

## 1.1.0

- Prepared the Scientific Reports major-revision analysis package from a
  clean allow-listed tree.
- Included the 54 approved pseudonymised participant-level derived beat
  tables under `data/beats/S01_Pre.csv … data/beats/S18_Post.csv` (columns
  `beat_idx`, `RRI_ms`, `SBP_mmHg`, `PAT_ms`; 18 participants x 3 phases;
  all fields numeric and finite), distributed under CC BY 4.0 as stated in
  `data/LICENSE`.
- Added `data/README.md` documenting the beat-table schema, pseudonymisation,
  phase separation, units, and the derived-vs-raw boundary.
- Added `config/approved_derived_beats_manifest.csv`, an exhaustive
  filename / SHA-256 / size / row-count manifest for the 54 approved files.
- Replaced the blanket-prohibition validator on `data/` with an exact
  allow-list validator that checks the 54 filenames, the four-column schema,
  numeric-finite fields, dense integer `beat_idx` starting at 1, expected row
  counts, and the SHA-256 set; the validator still hard-fails on raw waveform
  binaries, direct identifiers, secrets, private keys, absolute host paths,
  and now on release-text placeholder markers.
- Extended forbidden raw / acquisition binary extensions in the validator to
  include `.npy`, `.npz`, `.h5`, `.hdf5`, `.wav`, `.bin`, `.dat`, `.xdf`,
  `.set`, `.eeg`, `.cnt`, `.vhdr`, `.vmrk` in addition to existing formats,
  and made the public-tree scan ignore `.git` internals.
- Metadata gate now verifies the Zenodo concept DOI
  `10.5281/zenodo.20323694` and the trial identifier `jRCT1032230440`.
- Refactored `.gitignore` to permit `data/beats/S??_(Pre|Stim|Post).csv` while
  keeping raw waveform formats and upstream paired-source streams out.
- Documented the dual-license boundary consistently across `README.md`,
  `data/LICENSE`, `data/README.md`, `.zenodo.json`, `RELEASE_NOTES.md`,
  `docs/DATA_BOUNDARY.md`, and `docs/CONTROLLED_REPRODUCTION.md`: MIT covers
  the software / analysis source; CC BY 4.0 covers all derived and
  publication data tables under `data/beats/` and `expected_outputs/`. The
  two licenses apply to distinct file classes; derived data are not covered
  by MIT.
- Tightened reproducibility wording: this release does not itself directly
  re-run sequence-BRS, VAR, methods-text-matched BRS, or nonlinear analyses
  from `data/beats/`; no adapter or beats-only execution entry point is
  shipped, and no beat-table-to-anchor numerical test is exercised. The
  tables enable independent beats-only implementations and secondary
  analyses; v1.1.0 automatically validates their exact integrity plus the
  aggregate publication-source anchors. Exact raw-to-beat and exact
  continuous-time / coherence reproduction remain unsupported.
- Excluded raw ECG and continuous blood-pressure waveforms, upstream
  paired-source streams, semi-manual event-detection intermediates, and
  direct identifiers.
- Added unified controlled-data configuration through `TAVNS_DATA_ROOT`.
- Added the author-finished Supplementary Figure S3 artwork under a clean
  public filename.
- Added the corresponding data-driven S3 generator and scientific anchor
  validation.
- Retained the candidate-order 1–10, 54-fit native paired-beat VAR
  implementation and aggregate diagnostics.
- Retained methods-text-matched BRS and targeted nonlinear sensitivity
  validation.
- Retained restricted-path, secret, local-path, metadata, and regression
  gates.
