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
