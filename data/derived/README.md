# Derived Data

This directory stores data products required to regenerate the published figures
from repository files alone. The canonical result tables remain available under
`data/reference/`; the files here expose the subject-level and intermediate
arrays used by the figure scripts.

## Files

| Path | Schema | Purpose |
|---|---|---|
| `per_subject_coupling.csv` | `subject_id`, `phase`, `metric`, `value`, `reliability_flag` | Long-format subject values for all 46 coupling metrics. Rows with a non-empty reliability flag are excluded from aggregate checks. |
| `per_subject_hrv.csv` | `subject_id`, `phase`, `metric`, `value`, `reliability_flag` | Long-format subject values for all 74 HRV metrics. |
| `rhomax_windows/` | `window_start_s`, `window_end_s`, `rhomax`, `rhomax_lag` | Sliding-window Mayer-band peak correlation summaries. |
| `wtc/` | frequency-by-time matrices | Group-level wavelet coherence summaries and significance mask. |
| `fixed_lag_cross_correlation/` | `subject_id`, `phase`, `lag_s`, `correlation` | Lag profiles for causal and zero-lag-preserving filters. |
| `var_residuals/` | covariance summaries | Bivariate and trivariate residual covariance summaries. |
| `brs_ramps/` | ramp and event counts | BRS sequence ramp counts by subject and condition. |
| `bootstrap/` | `dz` | Bootstrap replicate arrays for selected effects. |
| `temporal_classification/` | `metric`, `temporal_type`, p-value columns | Temporal type assignment for coupling metrics. |
| `its_segmented_regression/` | model coefficients and permutation summary | Interrupted time-series summaries used by Figure S5. |

## Dependency Order

Run `python scripts/run_all.py` to regenerate this directory and all figures.
The per-subject files are generated first, then intermediate outputs are
computed from those files and `data/beats/`, and finally the figure scripts read
only `data/derived/` and `data/reference/`.
