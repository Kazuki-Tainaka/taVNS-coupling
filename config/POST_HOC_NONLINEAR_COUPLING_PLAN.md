# Frozen full-cohort post hoc nonlinear RRI–SBP coupling rerun plan

**Plan version:** 2.0 (author-requested n=18 rerun)  
**Frozen date:** 2026-08-06 (Asia/Tokyo)  
**Random seed:** `20260806`  
**Purpose:** narrowly scoped response to Reviewer 1, Comment 6  
**Status:** this document must be hashed before any n=18 real-data phase contrast is calculated

## 1. Scientific scope and design constraints

This is an author-requested full-cohort rerun of a post hoc sensitivity
analysis, not a new primary analysis and not a preregistered confirmatory
analysis. A previous n=16 run has already been reviewed internally. This rerun
plan is therefore frozen before viewing any n=18 real-data output, not before
all prior knowledge. It asks one narrow question:

> Does a small, prespecified set of validated nonlinear RRI–SBP coupling
> estimators reveal a robust Stim–Pre change that was missed by the linear
> coherence/VAR framework?

The study is a small exploratory pilot with 18 male protocol completers, a
fixed-order Pre (0–300 s) → Stim (300–600 s) → Post (600–900 s) design, no
sham condition, spontaneous breathing, and no recorded respiratory signal.
All 18 protocol completers (participants 01–18) will be included in this
full-cohort nonlinear RRI–SBP sensitivity rerun. The only change from the prior
production plan is the subject inclusion set:

```python
ALL_SUBJECTS = tuple(range(1, 19))
EXCLUDED_SUBJECTS = tuple()
ANALYSIS_SUBJECTS = ALL_SUBJECTS
```

Estimator definitions, parameters, windows, preprocessing, validation gate,
statistics, six-test multiplicity family, random seed, sensitivity settings,
surrogates, and adoption rules remain identical. Participant-level signal
provenance and QC adjudication are outside this rerun's scope and will not be
reopened. The previous n=16 output is a read-only internal comparison source;
the new n=18 output will be the authoritative result for this sensitivity
analysis.

Only native beat-to-beat paired RRI and systolic blood pressure (SBP) will be
analysed. PAT, respiratory proxies, external controls, band-pass filtering,
and uniformly interpolated series are outside scope. The primary contrast is
Stim minus Pre. Post minus Pre and Post minus Stim are descriptive and
exploratory only.

The main manuscript's central analysis and figures will not be changed by this
pipeline. No DOCX, existing CSV, existing figure, manuscript, Supplementary
Information, or response-letter file will be edited automatically.

## 2. Frozen input and preprocessing rules

### 2.1 Canonical source

The retained provider-prepared paired-beat files exposed at
`data/paired/paired_beats_XX.csv` are the canonical downstream input. Before
analysis, all 18 SHA-256 values must match the paired files recorded in the
previous production package. An input mismatch stops the run as
`BLOCKED_INPUT_DRIFT`. The `data/paired` directory resolves to the same file
identity as the previously recorded provider-retained source. Columns are:

1. R-wave timing (ms)
2. RRI (ms)
3. systolic-pressure peak timing (ms)
4. SBP stored in mmHg/100
5. PAT (ms; not used here)

SBP will be multiplied by 100 to obtain mmHg. This analysis preserves the
provider-retained pairing without inventing a new beat-rejection rule, then
excludes only non-finite RRI/SBP pairs and non-positive RRI values. Every
subject is handled by the same deterministic inclusion, segmentation,
detrending, scaling, and estimability rules.

Phase membership is assigned by R-wave time using half-open intervals
`[0,300)`, `[300,600)`, and `[600,900)` seconds. Data from different phases are
never concatenated for an estimator. No interpolation or imputation is used.

### 2.2 Segment selection

The primary segment is the centred 256-valid-pair segment within each phase.
For every possible consecutive 256-beat window, its midpoint is defined as the
mean of its first and last R-wave times. The window whose midpoint is closest
to the fixed phase midpoint (150, 450, or 750 s) is selected; an exact tie is
resolved in favour of the earlier window. A phase with fewer than 256 valid
pairs is not estimable and is not filled by interpolation.

Prespecified segment sensitivities are centred 192 valid pairs and the full
valid phase. They are not used to select a preferred result.

### 2.3 Detrending, scaling, and stationarity flag

For each subject × phase × segment × signal, an ordinary least-squares linear
trend against beat index is removed, followed by sample-SD z-normalisation
(mean 0, SD 1). Zero or non-finite SD makes the estimate non-finite. Raw-unit
series and trends are retained only in QC summaries.

For reporting only, KPSS level-stationarity tests (`regression='c'`,
`nlags='auto'`) are applied separately to the detrended, normalised RRI and SBP
primary segments. `PASS_BOTH` requires p ≥ 0.05 for both signals; otherwise the
flag identifies the signal(s) with p < 0.05. Numerical inability to run the
test is `INDETERMINATE`. The flag never triggers window replacement or
exclusion.

## 3. Frozen estimators

All direction labels describe lag-dependent prediction or information
transfer, not physiological causation.

### 3.1 LP: nonlinear local-predictability ratio

Directions are SBP→RRI and RRI→SBP. At the primary setting, target and source
candidate components are lags 1–8 beats, `k = 30`, Chebyshev/maximum-norm
distance, leave-one-out local prediction, and a Theiler exclusion window of
±8 beats. The zero-order local predictor is the inverse-distance-weighted mean
of the k neighbour target values. If one or more distances are exactly zero,
only the zero-distance neighbours receive equal weight.

A Porta-style nonuniform embedding is built greedily. Starting from no
components, every remaining candidate is provisionally added and the
candidate giving the largest squared correlation between observed and locally
predicted target values is selected. Ties use the fixed candidate order
(target before source, then increasing lag). After selecting a component, that
component and all more-recent or equal-lag candidates from the same signal are
removed. The process continues until no candidate remains. The earliest prefix
attaining the largest squared correlation is the optimal full embedding.

`NCI_full = 1 - r_full²`. The reduced embedding is formed by removing all
source components from the optimal full embedding, as in Porta et al. (2014),
and is evaluated on the same target indices and neighbour rules;
`NCI_reduced = 1 - r_reduced²`. An empty reduced embedding has r² = 0 and NCI
= 1. The raw Porta sign convention and the reoriented strength are:

`CR_Porta = (NCI_full - NCI_reduced) / NCI_reduced`

`LP_strength = -CR_Porta = (NCI_reduced - NCI_full) / NCI_reduced`

No negative strength is truncated. Reduced/full NCI, reduced/full r², raw
ratio, reoriented strength, and selected components will all be exported.

### 3.2 CE: conditional-entropy ratio

Directions, candidates, `k = 30`, Chebyshev distance, Theiler ±8 beats,
deterministic tie handling, and nonuniform candidate-removal rules match LP.
The target tolerance is frozen as
`epsilon = 0.10 × (target 84th percentile - target 16th percentile)`.

For a conditioning vector at target index n, the k eligible nearest embedding
vectors define k associated target values. Let `C_n` be the fraction of all
unordered distinct pairs among those k values whose absolute difference is
strictly smaller than epsilon. The local conditional entropy is
`-ln(C_n)`. A zero `C_n` is a non-finite numerical estimate; no pseudocount is
introduced. CE is the mean finite local entropy over all target indices only
when every local term is finite. Target Shannon entropy `H_target` is estimated
by the same correlation-probability definition over all unordered distinct
pairs of target values. A non-positive or non-finite `H_target` is a failure.

The embedding is grown one component at a time by selecting the candidate
that minimises CE. The earliest prefix attaining the minimum CE is the optimal
full embedding. The reduced embedding removes source components from that
optimal full embedding. Empty conditioning gives CE = `H_target`.

`NCI_full = CE_full / H_target`

`NCI_reduced = CE_reduced / H_target`

`CR_Porta = (NCI_full - NCI_reduced) / NCI_reduced`

`CE_strength = -CR_Porta`

Raw entropy, normalised reduced/full scores, raw ratio, positive-strength
orientation, epsilon, and selected components will be exported. This is termed
conditional information transfer, not causal influence.

### 3.3 SSC/KNNCP candidate

SSC/KNNCP will be implemented only from the exact public method described by
Porta et al. (2024), not by substituting CCM or generic cross mapping. For a
driver X and target Y, consecutive strictly lagged driver patterns of length
`m - 1` predict current Y. `m` is evaluated from 1 through 15, `k = 20`,
Euclidean distance is used, and only the reference vector itself is excluded
(the published method does not apply a Theiler window). Neighbour weights are
the normalised simplex weights `exp(-d_i / d_min)`; exact zero-distance
neighbours receive equal weight. The cross-predictability function is the
squared correlation between observed and predicted Y; CPI is its maximum over
m, with the smallest m breaking a tie. The direction label X→Y means better
prediction of future Y from past X. This estimator is unconditional on the
target's own past and is therefore reported as state-space
cross-predictability, not unique conditional information transfer.

If implementation details cannot be reconciled with the publication or the
validation gate is not passed, the fixed status will be
`NOT_IMPLEMENTED_EXACTLY` or `IMPLEMENTED_BUT_NOT_VALIDATED`, and SSC will be
excluded from real-data inference and the multiplicity family.

## 4. Prespecified synthetic validation

Each scenario has 200 independently seeded replicates, N = 256 retained after
a 512-sample burn-in, and low or moderate additive measurement noise (0.10 or
0.35 times each latent series SD). Innovations are independent standard
Gaussian variables unless a common driver is specified. Series are detrended
and z-normalised exactly like real inputs.

1. **Uncoupled linear:** `X_t=0.60X_(t-1)+eX`,
   `Y_t=0.50Y_(t-1)+eY`.
2. **Unidirectional linear X→Y:** the X equation above and
   `Y_t=0.45Y_(t-1)+0.45X_(t-1)+eY`.
3. **Unidirectional nonlinear X→Y:** the X equation above and
   `Y_t=0.45Y_(t-1)+0.65tanh(1.25X_(t-1))+eY`.
4. **Common driver:** `Z_t=0.60Z_(t-1)+eZ`,
   `X_t=0.45X_(t-1)+0.55Z_(t-1)+eX`, and
   `Y_t=0.45Y_(t-1)+0.55Z_(t-1)+eY`, with no direct X↔Y term.
5. **Bidirectional:** `X_t=0.45X_(t-1)+0.25Y_(t-1)+eX` and
   `Y_t=0.45Y_(t-1)+0.50X_(t-1)+eY`; X→Y is the stronger path.

For empirical false-positive assessment in the uncoupled scenario, each
observed direction is compared with 39 circular-shift source surrogates. Shift
offsets are sampled with replacement from values at least
`max(32, ceil(N/10))` beats from zero/wrap-around. The finite-sample p value is
`(1 + count(surrogate >= observed)) / 40`; p ≤ 0.05 is a positive detection.
The common-driver scenario is a documented bivariate-confounding stress test,
not an additional pass/fail criterion.

A method is `VALIDATED` only if all of the following hold:

- uncoupled false-positive rate is ≤10% for each direction and noise level;
- in both unidirectional scenarios at moderate noise, X→Y strength exceeds
  Y→X strength in at least 80% of replicates;
- non-finite or numerical-failure frequency is ≤5% in every scenario/noise
  cell;
- no unidirectional scenario shows systematic reversal; and
- for the asymmetric bidirectional scenario, X→Y exceeds Y→X in at least 65%
  of moderate-noise replicates (reported as a secondary direction check).

Failure of any mandatory item makes the method
`METHOD_NOT_VALIDATED`. Real-data values may be computed for software
diagnosis but cannot enter inference, figures of real-data findings,
manuscript/SI claims, or the Reviewer-response numerical conclusion.

## 5. Frozen parameter sensitivities

LP and CE use a one-factor-at-a-time grid; no best setting is selected:

- primary: k=30, lag depth=8, centred 256, strictly lagged;
- neighbour sensitivity: k=20 and k=40;
- lag sensitivity: depth 4 and depth 12;
- segment sensitivity: centred 192 and full valid phase;
- same-beat-inclusive sensitivity only: SBP→RRI source lags 0–8 and
  RRI→SBP source lags 1–8; target lags remain 1–8.

All other settings remain primary while one factor changes. SSC retains its
publication-defined parameters and is not tuned against the real data.

## 6. Subject-level surrogate analysis

Only validated methods are eligible. For each subject, phase, direction, and
method at its primary setting, 199 source-series circular shifts are drawn with
replacement using the shift restriction in Section 4. The entire estimator,
including embedding selection, is recomputed for each surrogate. Observed
strength must strictly exceed the surrogate 95th percentile calculated with
the `higher` quantile convention to be labelled significant. The exact Monte
Carlo p value is `(1 + count(surrogate >= observed)) / 200`.

For each phase and direction, prevalence is significant/evaluable n and an
exact Clopper–Pearson 95% CI. Pre versus Stim uses an exact two-sided McNemar
test; all three phases use Cochran's Q. These address prevalence, not the
group-level mean-strength contrast, and are exploratory.

## 7. Group-level inference

The primary nonlinear sensitivity family comprises the Stim–Pre contrasts for
both directions of every validated method: LP, CE, and SSC only if validated.
Pairwise complete cases are used without imputation. Two-sided Wilcoxon
signed-rank p values are adjusted together by Benjamini–Hochberg FDR. A zero
difference uses the standard `wilcox` zero method; all-zero differences return
p=1. Evaluable paired n is reported. n < 15 is labelled `LOW_ESTIMABILITY` and
cannot support manuscript adoption.

For every outcome the pipeline reports Pre, Stim, and Post mean ± sample SD;
Stim–Pre mean and median paired difference; participant-paired BCa 95% CI for
the mean difference; Cohen's dz; negative/zero/positive difference counts;
Wilcoxon p; BH q; and paired n. BCa resampling uses 10,000 participant-level
paired resamples and seed 20260806. Post contrasts are descriptive and do not
enter the BH family.

The optional nonlinear-over-linear diagnostic is not used for hypothesis
testing. To avoid adding another under-specified analysis, it is frozen as
`NOT_RUN_OPTIONAL`; the required CSV will contain this status and rationale.

## 8. Adoption rules

- `SI_BRIEF_NULL_SENSITIVITY`: all validated primary q ≥ 0.05, intervals
  include zero, estimates are small or directionally inconsistent across
  methods/settings, and prevalence has no clear phase shift.
- `RESPONSE_ONLY_INCONCLUSIVE`: nominal p < 0.05 without q < 0.05,
  parameter/method instability, or limited finite n.
- `SI_EXPLORATORY_NON_NULL`: validated method, q < 0.05, BCa interval excludes
  zero, paired n ≥15, direction agrees across most prespecified sensitivities,
  and the finding is not driven by missingness/artifact QC.
- `DO_NOT_REPORT_REAL_DATA_INFERENCE`: no candidate method passes validation
  or estimability requirements.

No result will be described as proof of absence, equivalence, physiological
causation, efferent drive, vagal causation, or a mechanism. Null wording will
be “no robust Stim–Pre difference was detected.” Any non-null result remains
post hoc exploratory and SI-centred. The main manuscript's central conclusion
will change only if the prespecified robust-non-null rule is met and even then
only an optional single cautious sentence will be drafted, not inserted.

## 9. Reproducibility and stopping rules

All input and plan hashes, package versions, warnings, errors, tests, validation
cells, QC cells, and generated-file hashes will be retained under the dedicated
output root. Tests must cover deterministic seeding, direction labels,
surrogate p values, phase-boundary isolation, absence of interpolation,
paired-subject bootstrap, BH correction, known simulation direction, and
missing-data handling.

Real-data group comparison must not begin unless this file has a recorded
SHA-256. After freezing, parameters, windows, directions, validation thresholds,
and adoption criteria cannot be changed in response to the observed phase
contrasts. A software defect may be corrected only with a logged code change;
the plan file and original plan hash remain immutable.
