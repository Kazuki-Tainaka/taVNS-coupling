# `data/` — approved public derived beat tables

## Scope

This directory contains the author-approved, pseudonymised
participant-level derived beat-to-beat tables that accompany the Scientific
Reports major-revision analysis package. It intentionally does not contain
any raw ECG or continuous blood-pressure waveform.

## Contents

- `data/beats/S01_Pre.csv … data/beats/S18_Post.csv` — 54 files.
  - 18 pseudonymised participants (`S01` – `S18`)
  - three phases per participant (`Pre`, `Stim`, `Post`)
  - each file has four columns with the exact header
    `beat_idx,RRI_ms,SBP_mmHg,PAT_ms`
  - every data field in all 54 current tables is a complete numeric finite
    value; there are no empty fields, no non-numeric-non-empty tokens, and no
    NaN / Inf values, and there is no timestamp field
- `data/LICENSE` — Creative Commons Attribution 4.0 International license
  text that applies to all derived and publication data tables in this
  repository (the 54 beat tables under `data/beats/` and the aggregate
  publication tables under `../expected_outputs/`). The software / analysis
  source in the surrounding repository is separately licensed under MIT
  (`../LICENSE`); the two licenses apply to distinct file classes and the
  derived data are not covered by MIT.

The exhaustive filename / SHA-256 / size / row-count manifest for the 54
files lives outside this directory, at
`../config/approved_derived_beats_manifest.csv`, so that this directory
contains exactly the 54 CSV files plus this `README.md` and `LICENSE`.

## Column definitions

| Column      | Unit    | Meaning                                                                    |
|-------------|---------|----------------------------------------------------------------------------|
| `beat_idx`  | integer | Dense, strictly increasing sequential beat index within the participant-phase segment, starting at 1.|
| `RRI_ms`    | ms      | R-R interval associated with that beat.                                    |
| `SBP_mmHg`  | mmHg    | Systolic blood-pressure value for the same beat, in engineering mmHg.      |
| `PAT_ms`    | ms      | Pulse arrival time. Includes the pre-ejection period (see manuscript).     |

Every field in all four columns of the 54 currently released tables is a
complete numeric finite value. The public schema does not permit empty
fields, sentinel non-numeric tokens, `NaN`, or `Inf`; participant/phase
segments for which a stable PAT could not be extracted are not represented
in this release.

## Provenance and boundary

- These tables are the phase-separated pseudonymised derived beat streams
  used for the manuscript’s beat-index-native analyses. They are not raw
  physiological recordings.
- No direct participant identifier, absolute acquisition date / time, or
  linkage key is present in any file.
- Raw ECG waveforms, continuous blood-pressure waveforms, upstream
  paired-source paired-beat streams, semi-manual event-detection
  intermediates, and manual-review workspaces are not part of the public
  package. They are governed by the article’s Data Availability statement
  and applicable ethics approval.

## Citation

Katahara Y, Iijima A, Tainaka K. (2026). taVNS cardiovascular coupling —
pseudonymised participant-level derived beat tables (Scientific Reports
major-revision release, v1.1.0). Archived in the Zenodo record whose
version-specific DOI is recorded in `../.zenodo.json` and in
`../RELEASE_NOTES.md`.
