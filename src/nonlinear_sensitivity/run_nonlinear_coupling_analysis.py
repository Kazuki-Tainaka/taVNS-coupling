"""Gate-first production pipeline for Reviewer 1 Comment 6.

The program asserts the immutable plan hash, inventories inputs, executes or
strictly verifies the prespecified synthetic validation, and only then computes
real-data estimates. All writes are confined to the dedicated output root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd

from analysis_utils import (
    ALL_SUBJECTS,
    ANALYSIS_SUBJECTS,
    OUTPUT_ROOT,
    PAIRED_DIR,
    PHASE_ORDER,
    PHASES,
    PROJECT_ROOT,
    SegmentSetting,
    assert_frozen_plan,
    clopper_pearson,
    cochran_q,
    cohens_dz,
    direction_arrays,
    exact_mcnemar,
    load_valid_phase,
    phase_contrast_rows,
    prepare_primary_qc,
    select_segment,
    sensitivity_settings,
    sha256_file,
    wilcoxon_p,
)
from nonlinear_estimators import (
    circular_shift_offsets,
    detrend_zscore,
    estimate_direction,
    surrogate_significant,
)
from plot_nonlinear_results import make_all_figures
from validation_simulations import (
    METHOD_CONFIGS,
    run_validation,
    write_validation_figure,
)


SEED = 20260806
DIRECTIONS = ("SBP→RRI", "RRI→SBP")
REFERENCE_N16_PACKAGE = Path(
    PROJECT_ROOT / "nonlinear_reference"
).resolve()
REFERENCE_INPUT_HASHES = (
    REFERENCE_N16_PACKAGE / "01_input_inventory" / "INPUT_SHA256.tsv"
)
VALIDATION_FILENAMES = (
    "nonlinear_method_validation.csv",
    "validation_scenario_summary.csv",
    "validation_method_gate.csv",
    "validation_failures.csv",
)
PRIMARY_SETTING = SegmentSetting(
    "primary", "centered", 256, 30, 8, False, True
)
REQUIRED_DIRS = (
    "00_plan",
    "01_input_inventory",
    "02_code",
    "03_validation",
    "04_qc",
    "05_results",
    "06_figures",
    "07_reports",
    "08_response_and_manuscript_snippets",
    "logs",
)


@dataclass(frozen=True)
class MetricJob:
    method: str
    direction: str
    subject: int
    phase: str
    setting: SegmentSetting
    validation_status: str


@dataclass(frozen=True)
class SurrogateJob:
    method: str
    direction: str
    subject: int
    phase: str
    target: np.ndarray
    source: np.ndarray
    observed_strength: float
    count: int
    validation_status: str


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("nonlinear_coupling")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(
        OUTPUT_ROOT / "logs" / "execution.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        na_rep="NA",
        lineterminator="\n",
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _input_candidates() -> list[tuple[str, Path, str, bool]]:
    analysis_code = (
        PROJECT_ROOT
        / "SciRep_submission"
        / "revision"
        / "02_analysis"
        / "code"
    )
    source_snapshot = (
        PROJECT_ROOT
        / "SciRep_submission"
        / "revision"
        / "01_source_snapshot"
        / "submission_original"
    )
    items: list[tuple[str, Path, str, bool]] = []
    for subject in ALL_SUBJECTS:
        items.append(
            (
                "canonical_native_pair",
                PAIRED_DIR / f"paired_beats_{subject:02d}.csv",
                "read for RRI, SBP, R-wave time, and phase assignment",
                True,
            )
        )
    items.extend(
        [
            (
                "input_identity_reference",
                REFERENCE_INPUT_HASHES,
                "previous production input hashes used only to lock file identity",
                True,
            ),
            (
                "current_loader_code",
                analysis_code / "revision_utils.py",
                "phase boundaries, finite-pair rules, and 4-Hz helper",
                True,
            ),
            (
                "current_brs_code",
                analysis_code / "brs_core.py",
                "BRS sequence implementation context",
                True,
            ),
            (
                "current_coupling_code",
                analysis_code / "coupling_core.py",
                "linear coupling and 4-Hz resampling context",
                True,
            ),
            (
                "supplement_generator",
                analysis_code / "build_submission_data.py",
                "Supplementary Data 1/2 generator provenance",
                False,
            ),
            (
                "supplement_source",
                source_snapshot / "Supplementary_Data_1.csv",
                "existing submission data source; not used in nonlinear estimates",
                False,
            ),
            (
                "supplement_source",
                source_snapshot / "Supplementary_Data_2.csv",
                "existing submission data source; not used in nonlinear estimates",
                False,
            ),
        ]
    )
    return items


def verify_paired_input_identity() -> pd.DataFrame:
    """Require all 18 paired inputs to match the previous production hashes."""

    if not REFERENCE_INPUT_HASHES.is_file():
        raise RuntimeError(
            f"BLOCKED_INPUT_DRIFT: reference inventory missing: {REFERENCE_INPUT_HASHES}"
        )
    reference = pd.read_csv(REFERENCE_INPUT_HASHES, sep="\t", dtype=str)
    reference = reference[
        reference["path"].str.contains(r"paired_beats_\d{2}\.csv$", regex=True)
    ].copy()
    if len(reference) != len(ALL_SUBJECTS):
        raise RuntimeError(
            "BLOCKED_INPUT_DRIFT: reference inventory does not contain 18 paired files"
        )
    expected_by_name = {
        Path(row.path).name: str(row.sha256).lower()
        for row in reference.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for subject in ALL_SUBJECTS:
        path = PAIRED_DIR / f"paired_beats_{subject:02d}.csv"
        name = path.name
        exists = path.is_file()
        observed = sha256_file(path) if exists else "NA"
        expected = expected_by_name.get(name, "NA")
        rows.append(
            {
                "subject_id": f"{subject:02d}",
                "file_name": name,
                "analysis_path": _relative(path),
                "reference_sha256": expected,
                "observed_sha256": observed,
                "hash_match": bool(exists and observed.lower() == expected.lower()),
            }
        )
    identity = pd.DataFrame(rows)
    if not bool(identity["hash_match"].all()):
        failures = identity[~identity["hash_match"]][
            ["subject_id", "file_name", "reference_sha256", "observed_sha256"]
        ].to_dict(orient="records")
        raise RuntimeError(f"BLOCKED_INPUT_DRIFT: {failures}")
    return identity


def write_input_inventory() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identity = verify_paired_input_identity()
    inventory_rows: list[dict[str, object]] = []
    hash_rows: list[dict[str, object]] = []
    for category, path, role, used in _input_candidates():
        exists = path.is_file()
        inventory_rows.append(
            {
                "category": category,
                "path": _relative(path),
                "exists": exists,
                "used_for_nonlinear_estimation": used,
                "role": role,
                "size_bytes": path.stat().st_size if exists else np.nan,
                "modified_local": (
                    datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                    if exists
                    else "NA"
                ),
            }
        )
        hash_rows.append(
            {
                "path": _relative(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else np.nan,
                "sha256": sha256_file(path) if exists else "NA",
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    hashes = pd.DataFrame(hash_rows)
    phase_rows: list[dict[str, object]] = []
    for subject in ALL_SUBJECTS:
        for phase, (start, end) in PHASES.items():
            _, counts = load_valid_phase(subject, phase)
            phase_rows.append(
                {
                    "subject_id": f"{subject:02d}",
                    "phase": phase,
                    "phase_start_s_inclusive": start,
                    "phase_end_s_exclusive": end,
                    **counts,
                    "included_in_full_cohort_analysis": True,
                    "exclusion_reason": "NA",
                    "source_file": _relative(
                        PAIRED_DIR / f"paired_beats_{subject:02d}.csv"
                    ),
                }
            )
    phase_map = pd.DataFrame(phase_rows)
    inventory_dir = OUTPUT_ROOT / "01_input_inventory"
    _write_csv(inventory, inventory_dir / "INPUT_INVENTORY.tsv")
    _write_csv(hashes, inventory_dir / "INPUT_SHA256.tsv")
    _write_csv(phase_map, inventory_dir / "SUBJECT_PHASE_MAP.tsv")
    # Convert the three required tabular inventories to actual TSV encoding.
    for name, frame in (
        ("INPUT_INVENTORY.tsv", inventory),
        ("INPUT_SHA256.tsv", hashes),
        ("SUBJECT_PHASE_MAP.tsv", phase_map),
    ):
        frame.to_csv(
            inventory_dir / name,
            sep="\t",
            index=False,
            encoding="utf-8",
            na_rep="NA",
            lineterminator="\n",
        )
    identity.to_csv(
        inventory_dir / "PAIRED_INPUT_IDENTITY_CHECK.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
        na_rep="NA",
        lineterminator="\n",
    )

    selection = f"""# Canonical data selection

## Decision

The analysis reuses the provider-retained native beat-to-beat paired files in
`{_relative(PAIRED_DIR)}`. No raw peak re-extraction was required. These files
contain R-wave timing, RRI, systolic-pressure peak timing, stored SBP, and PAT;
only R-wave timing, RRI, and SBP are read here. SBP is converted from the stored
scale by multiplication by 100, matching the current revision loader.

All 18 protocol completers were included in this author-requested full-cohort
rerun of the nonlinear RRI–SBP sensitivity analysis. All 18 paired-file SHA-256
values matched the previous production input inventory before analysis. The
subject inclusion set was the only analysis change; estimator definitions,
parameters, windows, statistics, multiplicity, seed, surrogate procedure, and
adoption rules were unchanged.

## Phase and pair rules

- Pre: [0, 300) s; Stim: [300, 600) s; Post: [600, 900) s, based on R-wave time.
- Finite paired RRI/SBP observations with positive RRI are retained.
- Existing provider-retained pairing is preserved; no new peak detector,
  rematching, interpolation, or imputation is applied.
- `artifact_excluded_count` is reported as unavailable rather than inferred.
- Every subject-phase is evaluated by the same deterministic rule; all 54
  subject-phase cells are expected to provide at least 256 valid pairs.

## Why native beats were selected

The current revision code constructs 4-Hz series on demand for spectral and
linear coupling analyses; no canonical stored 4-Hz beat table was identified.
Those interpolated samples are not treated as independent observations here.
All nonlinear estimators use deterministic windows of native paired beats.

## Supplementary Data provenance

Supplementary Data 1 and 2 in the source snapshot are generated/reformatted by
`SciRep_submission/revision/02_analysis/code/build_submission_data.py`. They are
inventory references only and are not nonlinear-analysis inputs.
"""
    (inventory_dir / "CANONICAL_DATA_SELECTION.md").write_text(
        selection, encoding="utf-8"
    )
    return inventory, hashes, phase_map


def _method_parameters(method: str, setting: SegmentSetting) -> dict[str, int]:
    if method == "SSC":
        return {"k": 20, "lag_depth": 15, "theiler": 0}
    return {
        "k": setting.k,
        "lag_depth": setting.lag_depth,
        "theiler": 8,
    }


def run_metric_job(job: MetricJob) -> dict[str, object]:
    valid, counts = load_valid_phase(job.subject, job.phase)
    selected, selected_start = select_segment(valid, job.phase, job.setting)
    common: dict[str, object] = {
        "subject_id": f"{job.subject:02d}",
        "phase": job.phase,
        "method": job.method,
        "direction": job.direction,
        "setting_id": job.setting.setting_id,
        "segment_kind": job.setting.segment_kind,
        "segment_length": len(selected) if selected is not None else 0,
        "selected_valid_pair_start_1based": (
            selected_start + 1 if selected_start is not None else np.nan
        ),
        "original_valid_pair_count": counts["valid_pair_count"],
        "validation_status": job.validation_status,
    }
    parameters = _method_parameters(job.method, job.setting)
    if selected is None:
        return {
            **common,
            **parameters,
            "same_beat_included": False,
            "finite": False,
            "failure_reason": "segment_not_estimable",
            "reduced_score": np.nan,
            "full_score": np.nan,
            "directed_strength": np.nan,
            "selected_embedding": "[]",
        }
    rri, _, _ = detrend_zscore(selected["RRI_ms"].to_numpy(float))
    sbp, _, _ = detrend_zscore(selected["SBP_mmHg"].to_numpy(float))
    target, source, target_name, source_name = direction_arrays(
        job.direction, rri, sbp
    )
    source_lag_zero = bool(
        job.setting.same_beat_convention and job.direction == "SBP→RRI"
    )
    result = estimate_direction(
        job.method,
        target,
        source,
        k=parameters["k"],
        lag_depth=parameters["lag_depth"],
        theiler=parameters["theiler"],
        source_lag_zero=source_lag_zero,
    )
    output = {
        **common,
        "target_signal": target_name,
        "source_signal": source_name,
        **result.to_dict(),
    }
    return output


def _run_jobs(
    jobs: Sequence[MetricJob], n_jobs: int
) -> list[dict[str, object]]:
    if n_jobs == 1:
        return [run_metric_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(run_metric_job, jobs, chunksize=4))


def _verified_validation_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_dir = OUTPUT_ROOT / "03_validation"
    validation_path = validation_dir / "nonlinear_method_validation.csv"
    summary_path = validation_dir / "validation_scenario_summary.csv"
    gate_path = validation_dir / "validation_method_gate.csv"
    if not validation_path.is_file() or not gate_path.is_file():
        raise RuntimeError("Requested validation reuse, but production files are missing")
    hash_rows: list[dict[str, object]] = []
    for name in VALIDATION_FILENAMES:
        source = REFERENCE_N16_PACKAGE / "03_validation" / name
        destination = validation_dir / name
        source_hash = sha256_file(source) if source.is_file() else "NA"
        destination_hash = sha256_file(destination) if destination.is_file() else "NA"
        hash_rows.append(
            {
                "file_name": name,
                "reference_sha256": source_hash,
                "copied_sha256": destination_hash,
                "hash_match": source_hash != "NA" and source_hash == destination_hash,
            }
        )
    hash_check = pd.DataFrame(hash_rows)
    if not bool(hash_check["hash_match"].all()):
        raise RuntimeError(
            "Validation reuse rejected: copied validation hashes differ from reference"
        )
    hash_check.to_csv(
        validation_dir / "VALIDATION_REUSE_SHA256.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    validation = pd.read_csv(validation_path)
    summary = pd.read_csv(summary_path)
    gate = pd.read_csv(gate_path)
    expected_rows = 5 * 2 * 200 * 3
    if len(validation) != expected_rows:
        raise RuntimeError(
            f"Validation reuse rejected: {len(validation)} rows != {expected_rows}"
        )
    counts = validation.groupby(["method", "scenario", "noise"]).size()
    if not bool((counts == 200).all()):
        raise RuntimeError("Validation reuse rejected: a cell is not 200 replicates")
    uncoupled = validation[validation["scenario"].eq("uncoupled_linear")]
    if not bool((uncoupled["surrogate_count"] == 39).all()):
        raise RuntimeError("Validation reuse rejected: uncoupled count is not 39")
    write_validation_figure(
        summary,
        gate,
        OUTPUT_ROOT / "06_figures" / "nonlinear_method_validation.svg",
    )
    return validation, gate


def compute_primary_metrics(
    validated_methods: Sequence[str],
    gate: pd.DataFrame,
    n_jobs: int,
) -> pd.DataFrame:
    status = gate.set_index("method")["validation_status"].to_dict()
    jobs = [
        MetricJob(method, direction, subject, phase, PRIMARY_SETTING, status[method])
        for method in validated_methods
        for direction in DIRECTIONS
        for subject in ANALYSIS_SUBJECTS
        for phase in PHASE_ORDER
    ]
    rows = _run_jobs(jobs, n_jobs)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["method", "direction", "subject_id", "phase"]
        )
    return frame


def _sensitivity_subject_metrics(
    validated_methods: Sequence[str],
    gate: pd.DataFrame,
    n_jobs: int,
) -> pd.DataFrame:
    eligible = [method for method in validated_methods if method in {"LP", "CE"}]
    status = gate.set_index("method")["validation_status"].to_dict()
    jobs = [
        MetricJob(method, direction, subject, phase, setting, status[method])
        for method in eligible
        for direction in DIRECTIONS
        for setting in sensitivity_settings()
        for subject in ANALYSIS_SUBJECTS
        for phase in PHASE_ORDER
    ]
    rows = _run_jobs(jobs, n_jobs)
    return pd.DataFrame(rows)


def summarize_sensitivity(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "setting_id",
        "method",
        "direction",
        "segment_kind",
        "segment_length_rule",
        "k",
        "lag_depth",
        "same_beat_included",
        "evaluable_paired_n",
        "paired_mean_difference",
        "paired_median_difference",
        "wilcoxon_statistic",
        "wilcoxon_p_two_sided",
        "cohens_dz",
        "negative_difference_n",
        "zero_difference_n",
        "positive_difference_n",
        "direction_sign",
        "primary_sign",
        "sign_consistent_with_primary",
        "finite_Pre_n",
        "finite_Stim_n",
        "finite_Post_n",
        "validation_status",
    ]
    if subject_metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_columns = ["method", "direction", "setting_id"]
    for (method, direction, setting_id), group in subject_metrics.groupby(
        group_columns, sort=False
    ):
        finite = group[group["finite"].eq(True)]
        pivot = finite.pivot(
            index="subject_id", columns="phase", values="directed_strength"
        )
        paired = pivot[["Stim", "Pre"]].dropna()
        differences = paired["Stim"].to_numpy(float) - paired["Pre"].to_numpy(float)
        statistic, p_value = wilcoxon_p(differences)
        sample = group.iloc[0]
        actual_lengths = group.groupby("phase")["segment_length"].median()
        segment_rule = (
            "full_valid_phase"
            if sample["segment_kind"] == "full"
            else str(int(sample["segment_length"]))
        )
        mean_difference = float(np.mean(differences)) if len(differences) else np.nan
        sign = int(np.sign(mean_difference)) if np.isfinite(mean_difference) else 0
        rows.append(
            {
                "setting_id": setting_id,
                "method": method,
                "direction": direction,
                "segment_kind": sample["segment_kind"],
                "segment_length_rule": segment_rule,
                "median_actual_segment_length": float(actual_lengths.median()),
                "k": int(sample["k"]),
                "lag_depth": int(sample["lag_depth"]),
                "same_beat_included": bool(
                    group["same_beat_included"].astype(bool).any()
                ),
                "evaluable_paired_n": len(differences),
                "paired_mean_difference": mean_difference,
                "paired_median_difference": (
                    float(np.median(differences)) if len(differences) else np.nan
                ),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_two_sided": p_value,
                "cohens_dz": cohens_dz(differences),
                "negative_difference_n": int(np.count_nonzero(differences < 0)),
                "zero_difference_n": int(np.count_nonzero(differences == 0)),
                "positive_difference_n": int(np.count_nonzero(differences > 0)),
                "direction_sign": sign,
                "finite_Pre_n": int(finite["phase"].eq("Pre").sum()),
                "finite_Stim_n": int(finite["phase"].eq("Stim").sum()),
                "finite_Post_n": int(finite["phase"].eq("Post").sum()),
                "validation_status": sample["validation_status"],
            }
        )
    output = pd.DataFrame(rows)
    primary_signs = (
        output[output["setting_id"].eq("primary")]
        .set_index(["method", "direction"])["direction_sign"]
        .to_dict()
    )
    output["primary_sign"] = [
        primary_signs.get((row.method, row.direction), 0)
        for row in output.itertuples(index=False)
    ]
    output["sign_consistent_with_primary"] = (
        output["direction_sign"].eq(output["primary_sign"])
        & output["direction_sign"].ne(0)
    )
    return output[columns[:5] + ["median_actual_segment_length"] + columns[5:]]


def run_surrogate_job(job: SurrogateJob) -> dict[str, object]:
    method_index = tuple(METHOD_CONFIGS).index(job.method)
    direction_index = DIRECTIONS.index(job.direction)
    phase_index = PHASE_ORDER.index(job.phase)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [SEED, 700, method_index, direction_index, job.subject, phase_index]
        )
    )
    offsets = circular_shift_offsets(len(job.source), job.count, rng)
    config = METHOD_CONFIGS[job.method]
    surrogate_values: list[float] = []
    failure_reasons: list[str] = []
    for offset in offsets:
        result = estimate_direction(
            job.method,
            job.target,
            np.roll(job.source, int(offset)),
            k=config["k"],
            lag_depth=config["lag_depth"],
            theiler=config["theiler"],
            source_lag_zero=False,
        )
        surrogate_values.append(result.directed_strength)
        if not result.finite:
            failure_reasons.append(result.failure_reason)
    significant, threshold, p_value = surrogate_significant(
        job.observed_strength, surrogate_values
    )
    return {
        "subject_id": f"{job.subject:02d}",
        "phase": job.phase,
        "method": job.method,
        "direction": job.direction,
        "observed_strength": job.observed_strength,
        "surrogate_count_requested": job.count,
        "surrogate_finite_n": int(np.isfinite(surrogate_values).sum()),
        "surrogate_95th_percentile_higher": threshold,
        "surrogate_monte_carlo_p": p_value,
        "significant_coupling": significant if not failure_reasons else False,
        "evaluable": not failure_reasons and np.isfinite(job.observed_strength),
        "failure_count": len(failure_reasons),
        "failure_reason": (
            "NA" if not failure_reasons else ";".join(sorted(set(failure_reasons)))
        ),
        "validation_status": job.validation_status,
    }


def compute_surrogates(
    metrics: pd.DataFrame,
    segments: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    count: int,
    n_jobs: int,
) -> pd.DataFrame:
    jobs: list[SurrogateJob] = []
    for row in metrics.itertuples(index=False):
        if not bool(row.finite):
            continue
        subject = int(row.subject_id)
        rri, sbp = segments[(subject, row.phase)]
        target, source, _, _ = direction_arrays(row.direction, rri, sbp)
        jobs.append(
            SurrogateJob(
                row.method,
                row.direction,
                subject,
                row.phase,
                target,
                source,
                float(row.directed_strength),
                count,
                row.validation_status,
            )
        )
    if n_jobs == 1:
        rows = [run_surrogate_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            rows = list(executor.map(run_surrogate_job, jobs, chunksize=1))
    columns = [
        "subject_id",
        "phase",
        "method",
        "direction",
        "observed_strength",
        "surrogate_count_requested",
        "surrogate_finite_n",
        "surrogate_95th_percentile_higher",
        "surrogate_monte_carlo_p",
        "significant_coupling",
        "evaluable",
        "failure_count",
        "failure_reason",
        "validation_status",
    ]
    return pd.DataFrame(rows, columns=columns)


def prevalence_summary(surrogates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "direction",
        "phase",
        "significant_n",
        "evaluable_n",
        "percentage",
        "Clopper_Pearson_95_low",
        "Clopper_Pearson_95_high",
        "Pre_to_Stim_loss_n",
        "Pre_to_Stim_gain_n",
        "exact_McNemar_p_Pre_vs_Stim",
        "McNemar_paired_n",
        "Cochran_Q_statistic_Pre_Stim_Post",
        "Cochran_Q_p_Pre_Stim_Post",
        "Cochran_Q_complete_n",
        "analysis_role",
    ]
    if surrogates.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (method, direction), group in surrogates.groupby(
        ["method", "direction"], sort=False
    ):
        evaluable = group[group["evaluable"].eq(True)]
        pivot = evaluable.pivot(
            index="subject_id", columns="phase", values="significant_coupling"
        )
        paired = pivot[["Pre", "Stim"]].dropna()
        loss, gain, mcnemar_p = exact_mcnemar(
            paired["Pre"].astype(bool), paired["Stim"].astype(bool)
        )
        complete = pivot[["Pre", "Stim", "Post"]].dropna()
        if len(complete):
            q_stat, q_p = cochran_q(complete.astype(int).to_numpy())
        else:
            q_stat, q_p = np.nan, np.nan
        for phase in PHASE_ORDER:
            phase_values = evaluable[evaluable["phase"].eq(phase)]
            successes = int(phase_values["significant_coupling"].astype(bool).sum())
            total = len(phase_values)
            lower, upper = clopper_pearson(successes, total)
            rows.append(
                {
                    "method": method,
                    "direction": direction,
                    "phase": phase,
                    "significant_n": successes,
                    "evaluable_n": total,
                    "percentage": 100.0 * successes / total if total else np.nan,
                    "Clopper_Pearson_95_low": lower,
                    "Clopper_Pearson_95_high": upper,
                    "Pre_to_Stim_loss_n": loss,
                    "Pre_to_Stim_gain_n": gain,
                    "exact_McNemar_p_Pre_vs_Stim": mcnemar_p,
                    "McNemar_paired_n": len(paired),
                    "Cochran_Q_statistic_Pre_Stim_Post": q_stat,
                    "Cochran_Q_p_Pre_Stim_Post": q_p,
                    "Cochran_Q_complete_n": len(complete),
                    "analysis_role": "exploratory_subject_level_prevalence",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def missingness_summary(
    gate: pd.DataFrame, metrics: pd.DataFrame, contrasts: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    contrast_lookup = {}
    if not contrasts.empty:
        contrast_lookup = (
            contrasts[contrasts["contrast"].eq("Stim-Pre")]
            .set_index(["method", "direction"])["evaluable_paired_n"]
            .to_dict()
        )
    for gate_row in gate.itertuples(index=False):
        for direction in DIRECTIONS:
            for phase in PHASE_ORDER:
                subset = metrics[
                    metrics["method"].eq(gate_row.method)
                    & metrics["direction"].eq(direction)
                    & metrics["phase"].eq(phase)
                ] if not metrics.empty else pd.DataFrame()
                finite = int(subset["finite"].eq(True).sum()) if not subset.empty else 0
                status = gate_row.validation_status
                rows.append(
                    {
                        "method": gate_row.method,
                        "direction": direction,
                        "phase": phase,
                        "eligible_analysis_subject_n": len(ANALYSIS_SUBJECTS),
                        "computed_subject_n": len(subset),
                        "finite_subject_n": finite,
                        "nonfinite_subject_n": len(subset) - finite,
                        "not_computed_due_to_validation_n": (
                            len(ANALYSIS_SUBJECTS)
                            if status != "VALIDATED"
                            else 0
                        ),
                        "Stim_Pre_evaluable_paired_n": contrast_lookup.get(
                            (gate_row.method, direction), 0
                        ),
                        "validation_status": status,
                        "group_inference_status": (
                            "EXCLUDED_METHOD_NOT_VALIDATED"
                            if status != "VALIDATED"
                            else (
                                "ESTIMABLE"
                                if contrast_lookup.get((gate_row.method, direction), 0)
                                >= 15
                                else "LOW_ESTIMABILITY"
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _sensitivity_stability(sensitivity: pd.DataFrame) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    if sensitivity.empty:
        return output
    for key, group in sensitivity.groupby(["method", "direction"], sort=False):
        evaluable = group[
            group["setting_id"].ne("primary")
            & group["paired_mean_difference"].notna()
        ]
        output[key] = (
            float(evaluable["sign_consistent_with_primary"].mean())
            if len(evaluable)
            else np.nan
        )
    return output


def decide_adoption(
    gate: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    prevalence: pd.DataFrame,
) -> tuple[str, pd.DataFrame, list[str]]:
    validated = gate[gate["validation_status"].eq("VALIDATED")]["method"].tolist()
    primary = contrasts[contrasts["contrast"].eq("Stim-Pre")].copy()
    stability = _sensitivity_stability(sensitivity)
    audit_rows: list[dict[str, object]] = []
    robust_outcomes: list[str] = []
    for row in primary.itertuples(index=False):
        key = (row.method, row.direction)
        interval_excludes_zero = bool(
            np.isfinite(row.mean_difference_BCa95_low)
            and np.isfinite(row.mean_difference_BCa95_high)
            and (
                row.mean_difference_BCa95_low > 0
                or row.mean_difference_BCa95_high < 0
            )
        )
        stability_fraction = stability.get(key, np.nan)
        stable_majority = bool(
            row.method == "SSC"
            or (
                np.isfinite(stability_fraction)
                and stability_fraction > 0.50
            )
        )
        robust = bool(
            row.BH_q_primary_family < 0.05
            and interval_excludes_zero
            and row.evaluable_paired_n >= 15
            and stable_majority
        )
        if robust:
            robust_outcomes.append(f"{row.method} {row.direction}")
        audit_rows.append(
            {
                "method": row.method,
                "direction": row.direction,
                "BH_q_lt_0p05": bool(row.BH_q_primary_family < 0.05),
                "BCa_CI_excludes_zero": interval_excludes_zero,
                "paired_n_ge_15": bool(row.evaluable_paired_n >= 15),
                "nonprimary_sensitivity_sign_agreement_fraction": stability_fraction,
                "majority_sensitivity_sign_agreement": stable_majority,
                "robust_non_null_rule_met": robust,
            }
        )
    audit = pd.DataFrame(audit_rows)
    reasons: list[str] = []
    if not validated or primary.empty or bool((primary["evaluable_paired_n"] < 15).all()):
        verdict = "DO_NOT_REPORT_REAL_DATA_INFERENCE"
        reasons.append("No validated, estimable primary nonlinear family was available.")
    elif robust_outcomes:
        verdict = "SI_EXPLORATORY_NON_NULL"
        reasons.append(
            "At least one validated outcome met q<0.05, BCa-CI, n, and prespecified stability requirements: "
            + ", ".join(robust_outcomes)
            + "."
        )
    else:
        nominal = bool((primary["wilcoxon_p_two_sided"] < 0.05).any())
        any_low_n = bool((primary["evaluable_paired_n"] < 15).any())
        all_q_null = bool((primary["BH_q_primary_family"] >= 0.05).all())
        all_ci_cross = bool(
            (
                (primary["mean_difference_BCa95_low"] <= 0)
                & (primary["mean_difference_BCa95_high"] >= 0)
            ).all()
        )
        effects_small = bool((primary["cohens_dz"].abs() < 0.50).all())
        instability = any(
            np.isfinite(value) and value < 0.75 for value in stability.values()
        )
        prevalence_shift = bool(
            not prevalence.empty
            and (
                (prevalence["exact_McNemar_p_Pre_vs_Stim"] < 0.05).any()
                or (prevalence["Cochran_Q_p_Pre_Stim_Post"] < 0.05).any()
            )
        )
        if (
            all_q_null
            and all_ci_cross
            and not nominal
            and not any_low_n
            and (effects_small or instability)
            and not prevalence_shift
        ):
            verdict = "SI_BRIEF_NULL_SENSITIVITY"
            reasons.extend(
                [
                    "All validated primary Stim–Pre q values were at least 0.05.",
                    "All paired-mean BCa intervals included zero.",
                    "No nominal primary finding or clear subject-prevalence phase shift was detected.",
                    "Effects were small and/or directionally inconsistent across frozen settings.",
                ]
            )
        else:
            verdict = "RESPONSE_ONLY_INCONCLUSIVE"
            reasons.append("No outcome met the complete robust non-null rule.")
            if all_q_null:
                reasons.append(
                    "All validated primary Stim–Pre contrasts had BH q≥0.05."
                )
            if not nominal:
                reasons.append(
                    "No primary two-sided Wilcoxon test was nominally significant."
                )
            if not all_ci_cross:
                excluded_zero = primary[
                    (primary["mean_difference_BCa95_low"] > 0)
                    | (primary["mean_difference_BCa95_high"] < 0)
                ]
                labels = ", ".join(
                    f"{row.method} {row.direction}"
                    for row in excluded_zero.itertuples(index=False)
                )
                reasons.append(
                    "The paired-mean BCa interval excluded zero for "
                    f"{labels}, precluding the frozen brief-null category even "
                    "though its rank-based p and BH q were not significant."
                )
            if any_low_n:
                reasons.append("At least one primary contrast had paired n<15.")
            if instability:
                reasons.append(
                    "At least one LP/CE direction had less than 75% sign agreement across non-primary frozen settings."
                )
            if prevalence_shift:
                reasons.append(
                    "At least one exploratory prevalence phase test had p<0.05."
                )
    return verdict, audit, reasons


def _empty_required_results() -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.DataFrame(
            columns=[
                "subject_id", "phase", "method", "direction", "segment_length",
                "k", "lag_depth", "same_beat_included", "reduced_score",
                "full_score", "directed_strength", "selected_embedding", "finite",
                "validation_status",
            ]
        ),
        "contrasts": pd.DataFrame(),
        "surrogates": pd.DataFrame(),
        "prevalence": pd.DataFrame(),
        "sensitivity": pd.DataFrame(),
    }


def write_optional_diagnostic() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "status": "NOT_RUN_OPTIONAL",
                "analysis": "nonlinear_over_linear_diagnostic",
                "reason": (
                    "Frozen plan designated this diagnostic optional. It was not run to avoid expanding the post hoc analysis battery."
                ),
                "hypothesis_family_included": False,
            }
        ]
    )
    _write_csv(
        frame,
        OUTPUT_ROOT / "05_results" / "nonlinear_vs_linear_diagnostic.csv",
    )
    return frame


def write_package_versions() -> None:
    packages = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pytest"]
    lines = [
        f"generated_local={datetime.now().isoformat()}",
        f"python={platform.python_version()}",
        f"python_executable={sys.executable}",
        f"platform={platform.platform()}",
    ]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        lines.append(f"{package}={version}")
    (OUTPUT_ROOT / "logs" / "package_versions.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_manifest() -> None:
    manifest_path = OUTPUT_ROOT / "logs" / "git_diff_or_file_manifest.txt"
    rows = [
        "# Dedicated-output file manifest",
        "# Existing project files were not edited; the repository had pre-existing unrelated changes.",
        "relative_path\tsize_bytes\tsha256",
    ]
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            f"{path.relative_to(OUTPUT_ROOT).as_posix()}\t{path.stat().st_size}\t{sha256_file(path)}"
        )
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reuse-validation", action="store_true")
    parser.add_argument("--validation-replicates", type=int, default=200)
    parser.add_argument("--validation-surrogates", type=int, default=39)
    parser.add_argument("--subject-surrogates", type=int, default=199)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--non-production-test-run",
        action="store_true",
        help="Allow reduced counts; outputs are explicitly non-production.",
    )
    return parser.parse_args(argv)


def _assert_production_counts(args: argparse.Namespace) -> str:
    exact = (
        args.validation_replicates == 200
        and args.validation_surrogates == 39
        and args.subject_surrogates == 199
        and args.bootstrap_resamples == 10_000
    )
    if exact:
        return "PRODUCTION_FROZEN_COUNTS"
    if not args.non_production_test_run:
        raise ValueError(
            "Reduced counts require --non-production-test-run and cannot support inference"
        )
    return "NON_PRODUCTION_TEST_RUN_DO_NOT_USE_FOR_INFERENCE"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in REQUIRED_DIRS:
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)
    logger = _configure_logging()
    warnings_path = OUTPUT_ROOT / "logs" / "warnings.log"
    errors_path = OUTPUT_ROOT / "logs" / "errors.log"
    warnings_path.write_text("", encoding="utf-8")
    errors_path.write_text("", encoding="utf-8")
    run_status = _assert_production_counts(args)
    logger.info("Run status: %s", run_status)
    logger.info("Git HEAD: %s", _git_head())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert_frozen_plan()
            logger.info("Frozen plan hash verified before inventory")
            write_package_versions()
            inventory, hashes, phase_map = write_input_inventory()
            logger.info(
                "Input inventory complete: %d files, %d subject-phase rows",
                len(inventory),
                len(phase_map),
            )
            logger.info("Paired input identity verified: 18/18 SHA-256 matches")
            qc, selected_series, segments = prepare_primary_qc()
            _write_csv(qc, OUTPUT_ROOT / "04_qc" / "subject_phase_qc.csv")
            _write_csv(
                selected_series,
                OUTPUT_ROOT / "04_qc" / "selected_primary_series.csv",
            )
            logger.info(
                "QC complete: %d/%d full-cohort subject-phase cells estimable",
                int(qc["primary_estimable"].sum()),
                len(ANALYSIS_SUBJECTS) * len(PHASE_ORDER),
            )

            if args.reuse_validation:
                validation, gate = _verified_validation_outputs()
                logger.info("Verified and reused complete production validation")
            else:
                logger.info("Starting prespecified synthetic validation gate")
                validation, _, _, gate = run_validation(
                    replicates=args.validation_replicates,
                    surrogate_count=args.validation_surrogates,
                    n_jobs=args.jobs,
                    write_outputs=True,
                )
                logger.info("Synthetic validation completed")
            assert_frozen_plan()
            logger.info("Frozen plan hash reverified before real-data calculation")
            validated_methods = gate[
                gate["validation_status"].eq("VALIDATED")
            ]["method"].tolist()
            logger.info("Validated methods: %s", validated_methods or "none")

            empty = _empty_required_results()
            if validated_methods:
                metrics = compute_primary_metrics(validated_methods, gate, args.jobs)
                _write_csv(
                    metrics,
                    OUTPUT_ROOT / "05_results" / "nonlinear_subject_phase_metrics.csv",
                )
                contrasts = phase_contrast_rows(
                    metrics, bootstrap_resamples=args.bootstrap_resamples
                )
                _write_csv(
                    contrasts,
                    OUTPUT_ROOT / "05_results" / "nonlinear_phase_contrasts.csv",
                )
                logger.info("Primary real-data contrasts complete")

                sensitivity_subject = _sensitivity_subject_metrics(
                    validated_methods, gate, args.jobs
                )
                sensitivity = summarize_sensitivity(sensitivity_subject)
                _write_csv(
                    sensitivity,
                    OUTPUT_ROOT
                    / "05_results"
                    / "nonlinear_parameter_sensitivity.csv",
                )
                logger.info("Frozen parameter sensitivity complete")

                surrogates = compute_surrogates(
                    metrics, segments, args.subject_surrogates, args.jobs
                )
                _write_csv(
                    surrogates,
                    OUTPUT_ROOT
                    / "05_results"
                    / "nonlinear_surrogate_significance.csv",
                )
                prevalence = prevalence_summary(surrogates)
                _write_csv(
                    prevalence,
                    OUTPUT_ROOT
                    / "05_results"
                    / "nonlinear_prevalence_summary.csv",
                )
                logger.info("Subject-level surrogate prevalence complete")
            else:
                metrics = empty["metrics"]
                contrasts = empty["contrasts"]
                sensitivity = empty["sensitivity"]
                surrogates = empty["surrogates"]
                prevalence = empty["prevalence"]
                for filename, frame in (
                    ("nonlinear_subject_phase_metrics.csv", metrics),
                    ("nonlinear_phase_contrasts.csv", contrasts),
                    ("nonlinear_parameter_sensitivity.csv", sensitivity),
                    ("nonlinear_surrogate_significance.csv", surrogates),
                    ("nonlinear_prevalence_summary.csv", prevalence),
                ):
                    _write_csv(frame, OUTPUT_ROOT / "05_results" / filename)
                logger.warning("No method passed validation; real-data inference skipped")

            missingness = missingness_summary(gate, metrics, contrasts)
            _write_csv(
                missingness,
                OUTPUT_ROOT / "05_results" / "nonlinear_missingness_summary.csv",
            )
            write_optional_diagnostic()
            verdict, adoption_audit, decision_reasons = decide_adoption(
                gate, contrasts, sensitivity, prevalence
            )
            _write_csv(
                adoption_audit,
                OUTPUT_ROOT / "07_reports" / "adoption_rule_audit.csv",
            )

            from reporting import write_all_reports

            write_all_reports(
                run_status=run_status,
                inventory=inventory,
                hashes=hashes,
                phase_map=phase_map,
                qc=qc,
                validation=validation,
                gate=gate,
                metrics=metrics,
                contrasts=contrasts,
                sensitivity=sensitivity,
                surrogates=surrogates,
                prevalence=prevalence,
                missingness=missingness,
                verdict=verdict,
                adoption_audit=adoption_audit,
                decision_reasons=decision_reasons,
                git_head=_git_head(),
            )
            if not metrics.empty and not contrasts.empty:
                figure_paths = make_all_figures()
                logger.info("Result figures written: %s", [str(p) for p in figure_paths])
            else:
                logger.info("Result figures skipped because no validated real-data family")

            if caught:
                warning_lines = [
                    f"{item.category.__name__}\t{item.filename}:{item.lineno}\t{item.message}"
                    for item in caught
                ]
                warnings_path.write_text(
                    "\n".join(warning_lines) + "\n", encoding="utf-8"
                )
            else:
                warnings_path.write_text("NO_CAPTURED_WARNINGS\n", encoding="utf-8")
            logger.info("Overall verdict: %s", verdict)
            logger.info("Main manuscript central conclusion change: NO")
            write_manifest()
            final_report = OUTPUT_ROOT / "07_reports" / "FINAL_EXECUTION_SUMMARY.md"
            summary_text = final_report.read_text(encoding="utf-8")
            console_encoding = sys.stdout.encoding or "utf-8"
            console_safe = summary_text.encode(
                console_encoding, errors="backslashreplace"
            ).decode(console_encoding)
            print(console_safe)
        return 0
    except Exception:
        errors_path.write_text(traceback.format_exc(), encoding="utf-8")
        logger.exception("Pipeline failed")
        write_manifest()
        return 1


if __name__ == "__main__":
    sys.exit(main())
