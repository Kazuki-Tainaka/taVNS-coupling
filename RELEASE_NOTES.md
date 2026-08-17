# Release notes — 1.1.1

This is a forward-only cross-platform portability hotfix over v1.1.0. The
v1.1.0 tag, GitHub release, and its published release asset are preserved
and not modified. No scientific values, analysis logic, seeds, parameters,
figures, or Supplementary Data content bytes are changed by this release.

## What changed

- Added `.gitattributes` declaring canonical LF for repository-authored
  text and `binary -eol` for `data/beats/*.csv` and for common binary
  asset types, so byte-frozen paths have `text`, `diff`, `merge`, AND
  `eol` all explicitly unset (the `binary` macro alone leaves the default
  `eol=lf` from `* text=auto eol=lf` in place). `git check-attr --all --
  data/beats/S01_Pre.csv` therefore reports both `text: unset` and
  `eol: unset`. Corrected two obvious type-classification errors while
  in the same file: `*.mlx` (MATLAB Live Script, ZIP-packed) is declared
  binary rather than text; `*.svg` (XML) is declared text rather than
  binary. See `docs/EOL_POLICY.md` for the full policy.
- Added `docs/EOL_POLICY.md` documenting: (i) canonical LF for
  repository-authored text; (ii) the four named text anchors that are
  hashed with an in-memory EOL-normalisation helper
  (`supplementary_data_1.csv`, `supplementary_data_2.csv`,
  `supplementary_data_3.csv`, `supplementary_figure_s3_generator.py`);
  (iii) raw-byte hashing for `data/beats/*.csv` and for binary artefacts;
  (iv) the extracted GitHub source-archive per-file inventory gate — with
  the explicit note that the outer container SHA of a GitHub-generated
  source archive is not a hard gate; and (v) the deterministic whole-asset
  SHA convention for manually-uploaded release assets.
- `scripts/validate_public_release.py` now honours an explicit hash mode
  per artefact (`raw_bytes` or `eol_normalized_lf_bytes`) and accepts an
  optional archive root so the artefact-hash gate can be exercised against
  an extracted GitHub source archive. The CLI signature and behaviour when
  invoked without arguments remain unchanged.
- `config/release_anchors.json` was migrated to a nested schema whose
  entries declare an explicit hash mode. Only the four reviewed text
  anchors were updated to their EOL-normalised SHA-256 values;
  `supplementary_figure_s3.jpg` continues to be gated on its raw-bytes
  SHA-256; every scientific numeric anchor (`central_brs`,
  `brs_landscape`, `coherence`, `haemodynamics`, `native_var`) is
  unchanged from v1.1.0.
- Added `tests/test_archive_eol_portability.py`, an archive-level
  regression test that is independent of `.git`. It builds a GitHub-style
  ZIP (single top-level directory), extracts it, verifies the extraction
  inventory round-trips exactly, and proves that both LF and CRLF variants
  of the four reviewed text anchors pass the artefact-hash gate.
  Simultaneously it proves that the raw-byte hash mode used for
  `data/beats/*.csv` is not EOL-normalised.
- `.zenodo.json` was bumped to `version: "1.1.1"`; all other fields
  unchanged.
- `CHANGELOG.md` was updated with a v1.1.1 hotfix entry.

## What did not change

- The 54 author-approved pseudonymised participant-level derived beat
  tables under `data/beats/S01_Pre.csv … data/beats/S18_Post.csv` remain
  byte-for-byte identical to v1.1.0 (54/54 filenames, sizes, raw SHA-256,
  Git blob object IDs, content bytes, and CSV headers preserved).
- `config/approved_derived_beats_manifest.csv` is unchanged.
- All Supplementary Data content bytes are unchanged. The stored anchor
  hashes for the four text anchors were relabelled as
  `eol_normalized_lf_bytes` so that a Linux, macOS, or Windows
  `core.autocrlf=false` checkout — as well as a Windows
  `core.autocrlf=true` checkout — all pass the artefact-hash gate on
  the same underlying content.
- All scientific numeric anchors, analysis logic, seeds, parameters,
  figures, and the Supplementary Figure S3 generator bytes are unchanged.
- The v1.1.0 tag, GitHub release, and its published release asset are
  preserved and not modified.

## Verification

Run either

    validate_public_release.ps1

on Windows PowerShell, or

    python -m pytest tests -q
    python scripts/validate_public_release.py

on any platform. The archive-level portability regression is exercised by
the new `tests/test_archive_eol_portability.py` module.

---

# Release notes — 1.1.0

This is the Scientific Reports major-revision release. It contains the central
analysis code and fixed configurations, the BRS implementation / specification
landscape, coherence and surrogate analysis, native paired-beat directional
diagnostics, targeted sensitivity analyses, synthetic fixtures, and public
tests. It also contains the 54 approved pseudonymised participant-level derived
beat-to-beat tables under `data/beats/`
(`beat_idx`, `RRI_ms`, `SBP_mmHg`, `PAT_ms`; 18 participants x 3 phases; all
fields numeric and finite). All derived and publication data tables — the 54
tables under `data/beats/` and the aggregate publication tables under
`expected_outputs/` — are distributed under the Creative Commons Attribution
4.0 International License as stated in `data/LICENSE`. The software and
analysis source code (`src/`, `scripts/`, `tests/`, `preprocessing_matlab/`,
`synthetic_fixtures/`, top-level `LICENSE`) is distributed under the MIT
License. The two licenses apply to distinct file classes; derived data are
not covered by MIT. Repository:
<https://github.com/Kazuki-Tainaka/taVNS-coupling>; Zenodo concept DOI
`10.5281/zenodo.20323694`.

Supplementary Figure S3 includes author-finished publication artwork and a
separate data-driven generator. The artwork may contain minor typography and
spacing refinements; the values and scientific layout are audited against
Supplementary Data 3.

Raw ECG and continuous blood-pressure waveforms, upstream paired-source
paired-beat streams, and controlled intermediate physiological time series
are not included. The semi-manual raw-waveform event-detection stage that
produced the derived beat tables is not reproducible from the public package
alone. This release does not itself directly re-run sequence-BRS, VAR,
methods-text-matched BRS, or nonlinear analyses from `data/beats/`; no
adapter or beats-only execution entry point is shipped, and no
beat-table-to-anchor numerical test is exercised. The four-column beat tables
enable independent beats-only implementations and secondary analyses.
Analyses that require the original continuous time axis (for example the
Fourier / wavelet coherence pipelines) are audited via the aggregate
publication tables under `expected_outputs/`; exact raw-to-beat and exact
continuous-time / coherence reproduction remain unsupported by this release.
This is an honest scope limitation, not a validation failure. The v1.1.0
automated gate validates the exact integrity of the 54 beat tables plus the
aggregate publication-source anchors.

Run `validate_public_release.ps1` on Windows PowerShell, or run
`python -m pytest tests -q` followed by
`python scripts/validate_public_release.py`.
