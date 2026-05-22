# Data Dictionary

## `data/beats/Sxx_<Phase>.csv` (54 files)

Each file contains beat-to-beat cardiovascular variables for one participant in
one experimental phase.

Naming convention: `Sxx_<Phase>.csv`, where `xx` is `01`-`18` and `Phase` is
`Pre`, `Stim`, or `Post`.

| Column | Type | Unit | Description |
|---|---|---|---|
| `beat_idx` | int | - | Beat index, 1-based and sequential within phase. No padding or truncation is applied. |
| `RRI_ms` | float | ms | R-R interval, measured from R-wave to R-wave for this beat. |
| `SBP_mmHg` | float | mmHg | Systolic blood pressure peak amplitude for the beat, measured by radial tonometry. |
| `PAT_ms` | float | ms | Pulse arrival time, defined as the interval from the R-wave peak to the SBP peak. Values for subjects S13 and S16 are excluded from PAT-dependent metric calculations because PAT estimation failed; see `docs/methods_appendix.md`. Internal/canonical metric identifiers retain the historical `PTT` label. |

Notes:

- DBP is intentionally excluded; it is not used in any reported analysis.
- Beat counts vary across files due to natural heart-rate variation.
- Absolute recording timestamps and raw waveform identifiers are not included.

## `data/subjects.csv`

This file contains one row per participant. Exact age and personally identifying
information are not included.

| Column | Type | Description |
|---|---|---|
| `subject_id` | string | De-identified subject identifier, format `Sxx` where `xx` is `01`-`18`. |
| `sex` | string | Biological sex. The public file uses `M`; all participants were healthy adult males. |
| `age_range` | string | Age range bin. This replaces exact age for de-identification. |
| `hrv_eligible` | bool | `True` if the subject is retained in the HRV metric analyses. In this dataset all subjects are HRV-eligible. |
| `coupling_eligible` | bool | `True` if the subject is retained in the coupling metric analyses before metric-specific eligibility filters. In this dataset all subjects are coupling-eligible. |
| `pat_eligible` | bool | `False` for S13 and S16; otherwise `True`. PAT-dependent metrics, including GC3, PDC, and PAT/PTT mean, restrict to `pat_eligible` subjects (n=16). |
| `brs_tf_eligible` | bool | `True` if the subject is retained in the canonical BRS transfer-function tabulation because the VAR transfer function yielded a finite Mayer-band gain in both compared phases. Mayer-band coherence is reported separately and is not used as a gate. |
| `notes` | string | Free-text per-subject notes. Empty cells indicate no additional note. |
| `brsseq_down_eligible` | bool | `False` for S06, S12, and S17; otherwise `True`. These subjects had no Stim-phase descending sequence passing the Pearson `r >= 0.80` criterion for the BRSseq,down result. |
| `brsseq_down_exclusion_reason` | string | Filled when `brsseq_down_eligible` is `False`. The recorded reason is that no descending sequence in the Stim phase passed the Pearson `r >= 0.80` criterion. Empty cells indicate no BRSseq,down-specific exclusion. |
| `pat_exclusion_reason` | string | Filled when `pat_eligible` is `False`. The recorded reason is implausible near-zero PTT (`< 1e-10 ms`), consistent with ECG noise causing beat-pairing failure. Empty cells indicate no PAT-specific exclusion. |

## `results/Additional_File_2.csv` (46 coupling metrics)

The file reports one row per coupling metric. Each row summarizes phase means,
paired contrasts, multiplicity correction, Bayesian null evidence, and
temporal-pattern classification.

| Column | Description |
|---|---|
| `Metric` | Internal metric identifier. PAT-related metrics retain the historical `PTT` name, for example `GC3_F_RRI_to_PTT` and `PTT_mean`; manuscript and figure labels translate these to PAT. |
| `Category` | Metric family or analysis domain, for example BRS sequence, bivariate Granger causality, or trivariate Granger causality. |
| `n` | Effective sample size for this metric after metric-specific eligibility filtering. |
| `Description` | Human-readable description of the metric, including unit or interpretation where applicable. |
| `Pre_mean` | Across-subject mean of the metric in the Pre baseline phase. |
| `Pre_SD` | Across-subject standard deviation of the metric in the Pre phase. |
| `Stim_mean` | Across-subject mean of the metric in the Stim active-stimulation phase. |
| `Stim_SD` | Across-subject standard deviation of the metric in the Stim phase. |
| `Post_mean` | Across-subject mean of the metric in the Post recovery phase. |
| `Post_SD` | Across-subject standard deviation of the metric in the Post phase. |
| `dz_Stim_Pre` | Cohen's dz effect size for the paired Stim minus Pre contrast. |
| `p_Stim_Pre` | Raw two-sided Wilcoxon signed-rank p-value for Stim versus Pre. |
| `p_FDR_Stim_Pre` | Benjamini-Hochberg FDR-corrected q-value for Stim versus Pre, computed within the 46 coupling metrics for this contrast. |
| `BF01_Stim_Pre` | JZS Bayes factor in favour of the null hypothesis for Stim versus Pre (`BF01 = 1 / BF10`, pingouin, `r = 0.707`). Higher values indicate stronger evidence for the null. Blank cells correspond to FDR-significant findings for which BF01 was not reported. |
| `dz_Post_Pre` | Cohen's dz for the paired Post minus Pre contrast. |
| `p_Post_Pre` | Raw two-sided Wilcoxon signed-rank p-value for Post versus Pre. |
| `p_FDR_Post_Pre` | Benjamini-Hochberg FDR-corrected q-value for Post versus Pre, computed within the 46 coupling metrics for this contrast. |
| `BF01_Post_Pre` | JZS Bayes factor BF01 for Post versus Pre. |
| `dz_Post_Stim` | Cohen's dz for the paired Post minus Stim contrast. |
| `p_Post_Stim` | Raw two-sided Wilcoxon signed-rank p-value for Post versus Stim. |
| `p_FDR_Post_Stim` | Benjamini-Hochberg FDR-corrected q-value for Post versus Stim. Blank entries indicate metrics without valid third-phase p-values in the legacy pipeline. |
| `Friedman_chi2` | Friedman test chi-square statistic across Pre, Stim, and Post. Empty cells indicate coupling metrics for which this omnibus statistic was not reported in the submitted table. |
| `Friedman_p` | Friedman test p-value across the three phases. Empty cells indicate metrics for which this omnibus statistic was not reported. |
| `Temporal_Type` | Temporal-pattern classification (A/B/C/D) summarizing the shape of Pre-to-Stim-to-Post changes. See the manuscript Methods for the classification scheme. |
| `Reliability_flag` | Known reliability caveat or sensitivity-analysis flag, such as filter dependence. Empty cells indicate no flagged caveat. |

## `results/Additional_File_3.csv` (74 HRV metrics)

The HRV table has the same statistical structure as Additional File 2, except
that it does not include a `Description` column and it contains HRV metrics
rather than cardiovascular coupling metrics. Friedman statistics are populated
for the HRV panel where available.

| Column | Description |
|---|---|
| `Metric` | Internal HRV metric identifier. |
| `Category` | HRV metric family, such as time-domain, frequency-domain, or nonlinear HRV. |
| `n` | Effective sample size for the metric. |
| `Pre_mean` | Across-subject mean of the HRV metric in the Pre baseline phase. |
| `Pre_SD` | Across-subject standard deviation in the Pre phase. |
| `Stim_mean` | Across-subject mean in the Stim active-stimulation phase. |
| `Stim_SD` | Across-subject standard deviation in the Stim phase. |
| `Post_mean` | Across-subject mean in the Post recovery phase. |
| `Post_SD` | Across-subject standard deviation in the Post phase. |
| `dz_Stim_Pre` | Cohen's dz effect size for the paired Stim minus Pre contrast. |
| `p_Stim_Pre` | Raw two-sided Wilcoxon signed-rank p-value for Stim versus Pre. |
| `p_FDR_Stim_Pre` | Benjamini-Hochberg FDR-corrected q-value for Stim versus Pre within the 74 HRV metrics. |
| `BF01_Stim_Pre` | JZS Bayes factor BF01 for Stim versus Pre, reported for non-significant HRV findings. |
| `dz_Post_Pre` | Cohen's dz for the paired Post minus Pre contrast. |
| `p_Post_Pre` | Raw two-sided Wilcoxon signed-rank p-value for Post versus Pre. |
| `p_FDR_Post_Pre` | Benjamini-Hochberg FDR-corrected q-value for Post versus Pre within the 74 HRV metrics. |
| `BF01_Post_Pre` | JZS Bayes factor BF01 for Post versus Pre. |
| `dz_Post_Stim` | Cohen's dz for the paired Post minus Stim contrast. |
| `p_Post_Stim` | Raw two-sided Wilcoxon signed-rank p-value for Post versus Stim. |
| `p_FDR_Post_Stim` | Benjamini-Hochberg FDR-corrected q-value for Post versus Stim. |
| `Friedman_chi2` | Friedman test chi-square statistic across the three phases. |
| `Friedman_p` | Friedman test p-value across the three phases. |
| `Temporal_Type` | Temporal-pattern classification (A/B/C/D) for phase-change shape. |
| `Reliability_flag` | Known reliability caveat or sensitivity-analysis flag. Empty cells indicate no flagged caveat. |

## `data/derived/per_subject_coupling.csv`

| Column | Description |
|---|---|
| `subject_id` | De-identified subject identifier. |
| `phase` | Experimental condition: Pre, Stim, or Post. |
| `metric` | Coupling metric identifier matching `Additional_File_2.csv`. |
| `value` | Subject-level value used for figure regeneration. |
| `reliability_flag` | Empty for included values; otherwise records the metric-specific exclusion reason. |

## `data/derived/per_subject_hrv.csv`

| Column | Description |
|---|---|
| `subject_id` | De-identified subject identifier. |
| `phase` | Experimental condition: Pre, Stim, or Post. |
| `metric` | HRV metric identifier matching `Additional_File_3.csv`. |
| `value` | Subject-level value used for figure regeneration. |
| `reliability_flag` | Empty for included values; otherwise records the metric-specific exclusion reason. |

## `data/derived/rhomax_windows/*.csv`

| Column | Description |
|---|---|
| `window_start_s` | Window start time in seconds. |
| `window_end_s` | Window end time in seconds. |
| `rhomax` | Window-level Mayer-band peak correlation. |
| `rhomax_lag` | Lag at the window-level peak. |

## `data/derived/wtc/*.csv`

| Column | Description |
|---|---|
| `frequency_hz` | Frequency coordinate for the matrix row. |
| `t_000` and later time columns | Time-indexed WTC or mask value. |

## `data/derived/fixed_lag_cross_correlation/*.csv`

| Column | Description |
|---|---|
| `subject_id` | De-identified subject identifier. |
| `phase` | Experimental condition. |
| `lag_s` | Lag in seconds. |
| `correlation` | Fixed-lag correlation value. |

## `data/derived/var_residuals/*.csv`

| Column | Description |
|---|---|
| `subject_id` | De-identified subject identifier. |
| `phase` | Experimental condition. |
| `cov_*` | Residual covariance summary columns for the listed variables. |

## `data/derived/brs_ramps/per_subject_ramp_counts.csv`

| Column | Description |
|---|---|
| `subject_id` | De-identified subject identifier. |
| `phase` | Experimental condition. |
| `n_ramps_up`, `n_ramps_down` | Monotonic sBP ramp counts. |
| `n_brs_events_up`, `n_brs_events_down` | Ramp counts meeting the BRS event criterion. |

## `data/derived/bootstrap/replicates_*.csv`

| Column | Description |
|---|---|
| `dz` | Bootstrap replicate Cohen's dz value for the named paired contrast. |

## `data/derived/temporal_classification/coupling_type_assignments.csv`

| Column | Description |
|---|---|
| `metric` | Coupling metric identifier. |
| `p_Stim_Pre`, `p_Post_Pre` | Raw comparison p-values used for classification. |
| `dz_Stim_Pre`, `dz_Post_Pre` | Effect sizes used to check direction consistency. |
| `temporal_type` | Computed type assignment. |
| `canonical_temporal_type` | Type assignment preserved in the canonical table. |

## `data/derived/its_segmented_regression/*.csv`

These files store segmented time-series coefficients and permutation summaries
used by the Figure S5 regeneration script.
