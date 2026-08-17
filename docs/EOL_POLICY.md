# End-of-line (EOL) policy and cross-platform hash portability

*Applies to v1.1.1 and later. v1.1.0 remains published as-is; no v1.1.0
artefacts are re-signed, re-tagged, or overwritten by this policy.*

## 1. Scope and motivation

The taVNS-coupling public release ships with two categories of SHA-256
integrity anchors:

1. **Scientific numeric anchors** — analytical values under
   `config/release_anchors.json` keys such as `central_brs`, `brs_landscape`,
   `coherence`, `haemodynamics`, `native_var`. These are numeric quantities
   with explicit floating-point tolerances and are unaffected by EOL.
2. **Reviewed public artefact anchors** — per-file SHA-256 values under the
   `public_artifact_sha256` key that gate the byte-identical shape of the
   five reviewed public artefacts.

In v1.1.0 the artefact-hash gate compared *raw byte* SHA-256 values against
anchors that had been computed on a Windows checkout with Git's default
`core.autocrlf=true`. On any Linux, macOS, or `core.autocrlf=false` Windows
checkout, the text artefacts materialise with LF line endings and the raw
SHA-256 changes even though the *content* is bit-identical after EOL
normalisation. This produced a cross-platform integrity-gate false alarm.

The v1.1.1 hotfix eliminates this cross-platform sensitivity while
**leaving all scientific values, analysis logic, figures, parameters, seeds,
Supplementary Data bytes, and the 54 author-approved derived beat tables
byte-for-byte identical to v1.1.0** (54/54 raw SHA-256 and Git blob OIDs
unchanged).

## 2. Canonical repository storage

The repository authoritatively stores all repository-authored text with
**canonical LF** line endings. This is enforced by `.gitattributes`:

```
* text=auto eol=lf
*.py *.md *.txt *.rst *.json *.yaml *.yml *.toml *.cfg *.ini *.ps1 *.sh
*.m *.tex *.bib *.csv *.svg                                      text eol=lf

data/beats/*.csv                                                 binary -eol
```

Byte-frozen artefact patterns are declared with `binary -eol`. The `binary`
macro expands to `-text -diff -merge` but does not clear the `eol` attribute
inherited from the `* text=auto eol=lf` default; the explicit `-eol` unsets
`eol` as well, so on any byte-frozen file
`git check-attr --all -- <path>` reports both `text: unset` and
`eol: unset`. This makes the byte-frozen guarantee explicit rather than
relying on the fact that EOL conversion is inert when `text` is unset.

Common binary asset types (`*.jpg`, `*.jpeg`, `*.png`, `*.tif`, `*.tiff`,
`*.gif`, `*.pdf`, `*.ico`, `*.mp4`, `*.mov`, `*.zip`, `*.gz`, `*.tar`,
`*.7z`, `*.xz`, `*.bz2`, `*.mat`, `*.mlx`, `*.npy`, `*.npz`, `*.h5`,
`*.hdf5`, `*.wav`, `*.bin`, `*.dat`, `*.xdf`, `*.set`, `*.eeg`, `*.cnt`,
`*.vhdr`, `*.vmrk`, `*.acq`, `*.edf`, …) are declared `binary -eol` so
they are never EOL-converted at any layer. Note that MATLAB Live Script
(`*.mlx`) is a ZIP-packed container and is classified as binary; SVG is
XML and is classified as text (`*.svg text eol=lf`).

## 3. Hash modes (explicit, per anchor)

`config/release_anchors.json` declares each artefact anchor with an explicit
hash-mode field so the validator behaviour is auditable per artefact.
Two modes are used:

| Mode                     | What is hashed                                                             | Applies to                                                                                         |
|--------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `raw_bytes`              | The file's on-disk bytes, unmodified                                       | All binary artefacts and all `data/beats/*.csv` participant tables                                 |
| `eol_normalized_lf_bytes`| The file's on-disk bytes with CRLF → LF, then any bare CR → LF, then SHA-256 | The four reviewed repository-authored text artefacts enumerated below                              |

### 3.1 The four named `eol_normalized_lf_bytes` anchors

The EOL-normalized hash-mode is used for exactly these four repository-authored
text artefacts, and only these four:

1. `supplementary_data_1.csv`
   (`expected_outputs/publication_source_data/supplementary_data_1.csv`)
2. `supplementary_data_2.csv`
   (`expected_outputs/publication_source_data/supplementary_data_2.csv`)
3. `supplementary_data_3.csv`
   (`expected_outputs/publication_source_data/supplementary_data_3_brs_sensitivity_and_coupling_significance.csv`)
4. `supplementary_figure_s3_generator.py`
   (`scripts/generate_supplementary_figure_s3.py`)

The normalisation is applied to the *in-memory copy* of the file's bytes for
hashing only. It never rewrites the file on disk. The order is:

```
bytes -> bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n") -> sha256
```

This makes the anchor identical for any working-tree encoding of the same
logical text content: LF-only (Linux / macOS / `core.autocrlf=false`
Windows), CRLF (Windows `core.autocrlf=true`), or bare-CR variants.

### 3.2 `raw_bytes` anchors — never normalised

`supplementary_figure_s3.jpg` and the 54 `data/beats/S??_(Pre|Stim|Post).csv`
participant tables are always hashed on raw bytes. In particular:

- `data/beats/*.csv` are declared `binary -eol` in `.gitattributes` so no
  EOL munging occurs at checkout, commit, or hash time, and `git check-attr
  --all -- data/beats/S01_Pre.csv` reports both `text: unset` and
  `eol: unset`. Their raw SHA-256 values, sizes, and Git blob OIDs remain
  exactly identical to v1.1.0, 54 out of 54.
  `config/approved_derived_beats_manifest.csv` is unchanged.
- The JPG is a binary artefact and coincidentally contains byte sequences
  such as `0x0D 0x0A` inside the JPEG payload; hashing those as text would
  destroy the image bytes.

## 4. GitHub extracted-source archive verification

GitHub auto-generated source archives (the `Source code (zip)` and
`Source code (tar.gz)` links attached to every release, and the archives
served from `/archive/refs/tags/vX.Y.Z.zip`) are re-materialised by
`git archive` on GitHub's infrastructure at request time. As a result:

- The **outer container SHA-256** (the SHA of the ZIP / tar.gz file) is
  **not** a stable, deterministic gate: it can differ between requests due
  to compressor version, mtime, and container-format encoding choices.
- The **extracted inventory** (file list, per-file bytes) *is* the stable,
  auditable object.

The v1.1.1 archive-portability regression test therefore:

1. Builds a synthetic GitHub-style ZIP whose sole top-level directory
   matches GitHub's `<repo>-<sha>/` convention.
2. Extracts it into a temporary tree.
3. Verifies the extracted inventory round-trips exactly (no files added,
   removed, or renamed by the extraction).
4. Runs the hotfix validator's `check_public_artifact_hashes(...)` against
   the extracted tree — with both LF-only and CRLF variants of all four
   reviewed text anchors — and requires it to pass in both cases.
5. Simultaneously proves that the raw `data/beats/*.csv` participant-table
   hashes are **not** subject to EOL normalisation.

The outer ZIP SHA is therefore explicitly **not** used as a hard gate.

## 5. Deterministic manual assets

Any manually-uploaded release asset (for example, an author-prepared
`taVNS-coupling-vX.Y.Z.zip` uploaded through the GitHub Releases UI) is a
deterministic, byte-fixed object: its whole-asset SHA-256 is published in
the release notes and can be verified directly. This whole-asset SHA is a
supplement to — not a replacement for — the extracted per-file inventory
gate described in §4.

## 6. Backward compatibility

- v1.1.0 tag, release, and its published asset are preserved and never
  modified. This is a forward-only hotfix; consumers who already validated
  v1.1.0 against the CRLF-based anchors under Git-for-Windows defaults are
  unaffected.
- The 54 author-approved derived beat tables under `data/beats/` remain
  byte-for-byte identical to v1.1.0 (54/54 filenames, sizes, raw SHA-256,
  Git blob OIDs, headers, and content bytes preserved).
- `config/approved_derived_beats_manifest.csv` is unchanged.
- No scientific values, analysis logic, seeds, parameters, figures, or
  Supplementary Data bytes are changed by this hotfix.
