# Changelog

## 1.1.1

- Forward-only cross-platform portability hotfix. v1.1.0 tag, release, and
  its published asset are preserved and not modified.
- No scientific values, analysis logic, seeds, parameters, figures, or
  Supplementary Data content bytes are changed by this hotfix.
- The 54 author-approved pseudonymised participant-level derived beat
  tables under `data/beats/` remain byte-for-byte identical to v1.1.0
  (54/54 filenames, sizes, raw SHA-256, Git blob object IDs, headers, and
  content bytes preserved). `config/approved_derived_beats_manifest.csv`
  is unchanged.
- Added `.gitattributes`: canonical LF for repository-authored text;
  `data/beats/*.csv` and common binary asset types declared `binary -eol`
  so `text`, `diff`, `merge`, AND `eol` are all explicitly unset on
  byte-frozen paths (the `binary` macro alone leaves the default
  `eol=lf` in place). `git check-attr --all -- data/beats/S01_Pre.csv`
  reports both `text: unset` and `eol: unset`. Corrected type
  classification: `*.mlx` (MATLAB Live Script, ZIP-packed) is declared
  binary rather than text; `*.svg` (XML) is declared text rather than
  binary.
- Added `docs/EOL_POLICY.md` documenting the canonical LF policy, the four
  named `eol_normalized_lf_bytes` text anchors, raw-mode hashing for all
  binary artefacts and for `data/beats/*.csv`, the extracted GitHub
  source-archive per-file inventory gate (with an explicit note that the
  outer container SHA of GitHub-generated archives is not a hard gate),
  and the whole-asset SHA convention for deterministic manually-uploaded
  release assets.
- `scripts/validate_public_release.py`:
    * `sha256_file(path)` continues to compute raw byte SHA-256; it is the
      authoritative hash for beat tables and binary artefacts.
    * Added `sha256_file_eol_normalized(path)` — an in-memory
      CRLF → LF then bare CR → LF normalisation before SHA-256, used only
      for the four reviewed text anchors listed in the EOL policy.
    * `check_public_artifact_hashes(anchors, root=None)` now accepts an
      optional `root` so the same integrity gate can be re-executed against
      an extracted GitHub source archive; the CLI entry point is unchanged.
    * Each artefact anchor now carries an explicit `mode`
      (`raw_bytes` | `eol_normalized_lf_bytes`); modes are honoured
      per-artefact.
    * Metadata gate expects `.zenodo.json` `version == "1.1.1"`.
- `config/release_anchors.json`:
    * Introduced a nested `public_artifact_sha256` schema whose entries
      declare an explicit hash mode per artefact.
    * Updated only the four reviewed text anchors
      (`supplementary_data_1.csv`, `supplementary_data_2.csv`,
      `supplementary_data_3.csv`,
      `supplementary_figure_s3_generator.py`) to their EOL-normalised
      SHA-256 values so the artefact-hash gate passes on Linux, macOS,
      and both Windows `core.autocrlf` modes.
    * Left `supplementary_figure_s3.jpg` at its raw-bytes SHA-256 value.
    * Left every scientific numeric anchor (`central_brs`,
      `brs_landscape`, `coherence`, `haemodynamics`, `native_var`) exactly
      as in v1.1.0.
- Added `tests/test_archive_eol_portability.py`: builds a GitHub-style ZIP
  (single top-level directory) from the current in-tree bytes, extracts
  it to a temporary directory, verifies the extraction inventory
  round-trips exactly, and proves that both the LF and CRLF variants of
  the four reviewed text anchors pass
  `check_public_artifact_hashes(root=<extracted>)`. Simultaneously proves
  the raw-byte hash mode used for `data/beats/*.csv` is not EOL-normalised.
- `.zenodo.json`: bumped `version` to `1.1.1`; all other fields unchanged.
- Added a v1.1.1 hotfix release notes section.

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
