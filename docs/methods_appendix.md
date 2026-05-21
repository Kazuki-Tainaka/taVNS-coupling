# Methods Appendix

## Pipeline-to-manuscript correspondence

| Metric / family | Manuscript location | Repository implementation |
|---|---|---|
| BRS sequence (`BRS_seq_all`, `BRS_seq_up`, `BRS_seq_down`) | Methods: baroreflex sequence analysis | `lib/coupling.py`, `scripts/lib/coupling.py` |
| BRS transfer function (`BRS_TF_mean`) | Methods: spectral baroreflex analysis | canonical result preservation; inclusion based on finite VAR transfer-function gain |
| rhomax / Mayer-band timing | Methods: phase synchrony metrics | canonical result preservation; filter note below |
| Bivariate GC | Methods: Granger causality | `lib/coupling.py::compute_gc_f_bivariate` |
| Trivariate GC3 | Methods: trivariate Granger causality | `lib/coupling.py::compute_gc3_f_trivariate` |
| HRV metrics | Methods: HRV panel | canonical result preservation |

## Numerical constants registry

| Constant | Value | Rationale |
|---|---:|---|
| PAT quality gate | `1e-10 ms` | Catches S13/S16 near-zero PTT export anomaly |
| Mayer band | `0.08-0.12 Hz` | Standard Mayer-band coupling window |
| BRS_TF inclusion rule | finite Mayer-band gain | VAR transfer-function estimate must be finite in both compared phases |
| FDR alpha | `0.05` | Benjamini-Hochberg correction |
| JZS BF scale | `r = 0.707` | pingouin / JZS prior scale |
| LOO threshold | `p < 0.05` | Robustness summary threshold |
| BCa bootstrap iterations | `B = 10000` | Manuscript robustness convention |

## Filter implementations registry

| Use case | Filter type | Order | Passband | Phase response |
|---|---|:---:|---|---|
| AR(2) preprocessing for BRS_TF and transfer-function analyses | Butterworth | 4 | 0.08-0.12 Hz | Zero-phase (`filtfilt`) |
| `rhomax_MATLAB` (canonical) | Chebyshev Type I | 3 | 0.08-0.12 Hz | Causal (`lfilter`); MATLAB-compatible direction |
| `rhomax` zero-phase sensitivity comparison | Chebyshev Type I | 3 | 0.08-0.12 Hz | Zero-phase (`filtfilt`); reported in supplementary filter-sensitivity figures |

The `rhomax_MATLAB` value preserved in `results/Additional_File_2.csv`
corresponds to the causal-filter row. Zero-phase comparison values are
documented separately and are not used as primary canonical values.

## Sample size by metric and rationale

| Metric / family | n | Excluded subjects | Rationale |
|---|:---:|---|---|
| BRS_seq_all, BRS_seq_up | 18 | none | All subjects retained |
| BRS_seq_down | 15 | S06, S12, S17 | No Stim-phase descending sequence passed the correlation criterion (`r >= 0.80`) |
| BRS_TF_mean | 12 | subjects without finite paired gain | VAR transfer function yielded finite Mayer-band gain in both compared phases; coherence was computed separately and was not used as a gate |
| GC3_F_* directions, PDC_*, PAT/PTT mean | 16 | S13, S16 | Implausible PAT/PTT values; `lib/quality.py::validate_pat` catches values below `1e-10 ms` |
| All other coupling metrics, all HRV metrics | 18 | none | Full sample retained |

## BRS event criteria

A baroreflex event is recorded when Pearson `r >= 0.80` between the sBP segment
and the corresponding RRI segment. Strict beat-by-beat concordance is evaluated
for descriptive characterization but is not required for event counting.

## Anchor metric recomputation tolerance

The 11 anchor metrics listed in `tests/test_anchor_tolerance.py` are verified under `--recompute` mode against canonical values within fixed tolerances.

| Statistic | Tolerance type | Bound |
|---|---|:---:|
| Cohen's dz | Absolute | +/-0.02 |
| p-value | Relative | +/-5% |
| FDR-corrected q-value | Relative | +/-5% |

Exact byte reproduction is handled by the default materialization path (`run_all.py`).

## Pre-processing pipeline (high-level)

ECG and tonometric blood pressure waveforms were processed into beat-to-beat RRI, SBP, and PAT series before de-identification. Raw waveform processing details are outside this public repository; the public package starts from de-identified beat-to-beat CSVs in `data/beats/`.
