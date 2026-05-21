# taVNS-coupling

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Reproducibility package for:

> Katahara Y, Iijima A, Tainaka K. Transcutaneous auricular vagus nerve stimulation is associated with transient baroreflex sensitivity reduction while preserving Mayer-wave coherence: a comprehensive analysis of 120 cardiovascular metrics. Submitted to Journal of NeuroEngineering and Rehabilitation.

## What this package provides

This repository contains the de-identified derived data, canonical result tables,
and validation harness accompanying the taVNS cardiovascular coupling study.

- `data/beats/`: 54 beat-to-beat CSV files (18 participants x 3 phases) with RRI, SBP, and PAT columns.
- `data/subjects.csv`: de-identified subject-level eligibility metadata.
- `data/reference/`: byte-identical reference copies of the submitted Additional File 2 and 3 CSVs.
- `results/Additional_File_2.csv`: 46 cardiovascular coupling metrics.
- `results/Additional_File_3.csv`: 74 HRV metrics.
- `scripts/`: entry points for materializing canonical results and running anchor checks.
- `lib/` and `scripts/lib/`: public helper modules.
- `tests/`: MD5 checks, anchor-value checks, de-identification checks, and repository-structure checks.

## Reproducibility scope

This public release is designed to accompany the canonical results and verify a
representative subset of metrics, while avoiding redistribution of restricted
raw waveform data.

### Default mode: canonical byte-identical preservation

```bash
python scripts/run_all.py
```

This materializes `results/Additional_File_2.csv` and
`results/Additional_File_3.csv` byte-for-byte identical to the submitted
canonical CSVs. The MD5 hashes are checked by `tests/test_md5_match.py`.

### Anchor verification mode

```bash
python scripts/01_compute_coupling_metrics.py --recompute
```

This regenerates `results/Additional_File_2_recomputed.csv` through the
public-package serialization path and produces a tolerance comparison report
against the canonical `results/Additional_File_2.csv`. The report covers 11
anchor metrics: the BRSseq family, rhomax, bivariate GC F, and the six
directions of trivariate GC3 F.

The recomputed CSV is intentionally byte-distinct from the canonical CSV. Their
MD5 hashes differ, and this is checked by
`tests/test_anchor_tolerance.py::test_recompute_is_not_materialization`.

Tolerance bands:

- Cohen's dz: absolute tolerance +/-0.02
- p-values: relative tolerance +/-5%
- FDR-corrected q-values: relative tolerance +/-5%

**Current scope and intentional limitation.** In this public release, the
recomputed anchor values are produced from the canonical-stable metric table
rather than from a beat-from-source re-derivation in `lib/coupling.py`.
Consequently, the displayed anchor deltas in the tolerance report are zero. The
entry point, tolerance test, separate output CSV, and CI workflow are in place so
that full beat-from-source implementations for these 11 metrics can be added
without changing the validation contract.

### What is intentionally not provided

- Raw waveforms (continuous ECG and tonometric blood pressure recordings) are
  not redistributed due to the original IRB approval scope.
- First-class beat-from-source implementation for the remaining 35 coupling
  metrics and 74 HRV metrics is not included in this v1.0.0 package. These
  metrics are currently materialized from the canonical CSVs.

## Quick start

```bash
git clone https://github.com/USERNAME/taVNS-coupling.git
cd taVNS-coupling
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/run_all.py
python scripts/01_compute_coupling_metrics.py --recompute
python -m pytest tests -v
```

## Repository layout

```text
taVNS-coupling/
  data/
    beats/
    reference/
    subjects.csv
  docs/
    data_dictionary.md
    methods_appendix.md
  lib/
  scripts/
  results/
    Additional_File_2.csv
    Additional_File_3.csv
  tests/
```

## Ethics and trial registration

- Trial registration: [jRCT1032230440](https://jrct.niph.go.jp/latest-detail/jRCT1032230440) (Japan Registry of Clinical Trials, prospectively registered 2023-11-07)
- Ethical approval: Niigata University Ethics Review Committee, approval number 2023-0191
- Written informed consent was obtained from all participants

## Citation

If you use this code or data, please cite both:

**Software (this package):**
> Katahara Y, Iijima A, Tainaka K. taVNS-coupling: reproducibility package for "Transcutaneous auricular vagus nerve stimulation is associated with transient baroreflex sensitivity reduction while preserving Mayer-wave coherence: a comprehensive analysis of 120 cardiovascular metrics" (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

**Underlying study:**
> Katahara Y, Iijima A, Tainaka K. Transcutaneous auricular vagus nerve stimulation is associated with transient baroreflex sensitivity reduction while preserving Mayer-wave coherence: a comprehensive analysis of 120 cardiovascular metrics. Submitted to *Journal of NeuroEngineering and Rehabilitation*.

## License

- Code: MIT License (see `LICENSE`)
- Data and derived tables: CC BY 4.0 (see `data/LICENSE`)

## Contact

For questions about this code and data package, please contact either of the
co-corresponding authors:

- **Atsuhiko Iijima** - a-iijima@eng.niigata-u.ac.jp  
  Faculty of Engineering, Niigata University
- **Kazuki Tainaka** - kztainaka@bri.niigata-u.ac.jp  
  Brain Research Institute, Niigata University

For data access questions or replication issues, please open an issue on this
repository.
