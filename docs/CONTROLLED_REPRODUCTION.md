# Reproduction tiers

The repository supports three tiers of reproduction. Choose the tier that
matches your available data.

## Public tier A — syntax, tests, synthetic validation, plotting

No participant data required. The bundled synthetic fixture is used by the
public unit tests, and no controlled recordings are touched.

```powershell
python -m pip install -r environment/requirements-lock.txt
./validate_public_release.ps1
```

## Public tier B — approved public derived beat tables

The 54 approved pseudonymised participant-level derived beat tables under
`data/beats/S01_Pre.csv … data/beats/S18_Post.csv` are part of the public
package. Each file has the exact columns
`beat_idx, RRI_ms, SBP_mmHg, PAT_ms` (all fields numeric and finite), no
direct identifier, and no absolute acquisition date / time. The exact
filename set and expected SHA-256 hashes are enforced by the validator's
`author_approved_derived_beats` gate against
`config/approved_derived_beats_manifest.csv`. The v1.1.0 automated gate also
verifies the numeric-finite four-column schema, dense integer `beat_idx`
starting at 1, per-file row counts and sizes, and the aggregate
publication-source anchors under `expected_outputs/`.

Independent researchers may use these four-column tables to build
beats-only implementations and secondary analyses of beat-index-native
quantities such as sequence-BRS on beats, native paired-beat VAR-based
directional diagnostics, methods-text-matched BRS comparators, and
native-beat nonlinear sensitivity.

### Reproducibility limits

1. **No shipped beats-only entry point.** The shipped `src/` analysis
   pipeline reads the upstream paired-beat source layout
   (`paired/paired_beats_XX.csv` with `R_wave_timing_ms`,
   `RRI_ms`, `sBP_timing_ms`, `sBP_mmHg100`, `PAT_ms`), not the
   phase-separated `data/beats/S??_(Pre|Stim|Post).csv` layout. No adapter
   from the phase-separated public layout to the upstream layout is
   included in this release, and no beats-only execution entry point is
   provided. This release therefore does not itself directly re-run
   sequence-BRS, VAR, methods-text-matched BRS, or nonlinear analyses from
   `data/beats/`, and no beat-table-to-anchor numerical test is exercised.
   Independent researchers can construct such adapters or beats-only
   implementations from the four beat-level scalars supplied here. The
   public gate exercises the aggregate publication source tables under
   `expected_outputs/` for numerical verification (see tier C).
2. **Exact time-axis not reconstructable from beats-only public data.**
   Analyses that require the original continuous time axis — the
   Fourier / wavelet coherence pipelines and any 4-Hz-resampled analyses —
   cannot be reproduced exactly from `data/beats/` alone, because the
   beat-timing columns (`R_wave_timing_ms`, `sBP_timing_ms`) are
   intentionally not part of the approved public schema. Exact raw-to-beat
   and exact continuous-time / coherence reproduction remain unsupported
   by this release. The aggregate results for those analyses are preserved
   and audited via the publication source tables under `expected_outputs/`.
   These are honest scope limitations, not validation failures.

## Public tier C — publication-source-table audit

`expected_outputs/publication_source_data/` contains the publication-approved
derived tables. `scripts/validate_public_release.py` checks central BRS, the
72-setting BRS landscape, coherence and surrogate prevalence, haemodynamic
context, native paired-beat VAR diagnostics, methods-text-matched BRS
branches, and targeted nonlinear sensitivity against fixed anchors, so all
reported numerical results are independently verifiable from the public
package.

## Controlled tier — full raw-waveform-to-result pipeline

Exact rerunning from raw ECG or continuous blood-pressure waveforms requires
controlled access under the article's Data Availability statement and
applicable ethics approval. Raw ECG, continuous blood-pressure waveforms,
upstream paired-source paired-beat streams, and manually reviewed
participant-level intermediates are not included.

```powershell
$env:TAVNS_DATA_ROOT = (Resolve-Path ../controlled-data).Path
$env:TAVNS_OUTPUT_ROOT = (Resolve-Path ../controlled-output).Path
python src/run_baseline_reproduction.py
python src/run_revision_analysis.py
python src/run_author_review_fix_analysis.py
python src/run_sequence_mixed_model.py
python src/native_paired_beat_var.py
python src/methods_text_matched_brs/run_all.py
python src/nonlinear_sensitivity/run_nonlinear_coupling_analysis.py --jobs 4
```

Comparator workflows may also require `revision_reference/` and
`nonlinear_reference/`, as shown in `config/data_paths.example.yaml`. Use a
separate writable output root.

Do not place controlled raw waveforms in a public Git worktree; the
`.gitignore` in this repository blocks them by construction, but the
responsibility for handling controlled data remains with the operator.
