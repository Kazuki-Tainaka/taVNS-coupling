"""Prespecified synthetic validation for nonlinear coupling estimators."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from nonlinear_estimators import (
    circular_shift_offsets,
    detrend_zscore,
    estimate_direction,
    surrogate_p_value,
)


SEED = 20260806
N_RETAINED = 256
BURN_IN = 512
PLAN_SHA256 = "d3ea3d785ff3615eb3370e8fb60daf6fed37440593ae7501e2892be86bd1a0b4"
RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    os.environ.get(
        "TAVNS_NONLINEAR_OUTPUT_ROOT",
        RELEASE_ROOT / "controlled_outputs" / "nonlinear_sensitivity",
    )
).resolve()
PLAN_PATH = Path(
    os.environ.get(
        "TAVNS_NONLINEAR_PLAN_PATH",
        RELEASE_ROOT / "config" / "POST_HOC_NONLINEAR_COUPLING_PLAN.md",
    )
).resolve()
VALIDATION_DIR = OUTPUT_ROOT / "03_validation"
FIGURE_DIR = OUTPUT_ROOT / "06_figures"
REPORT_DIR = OUTPUT_ROOT / "07_reports"

SCENARIOS = (
    "uncoupled_linear",
    "unidirectional_linear_X_to_Y",
    "unidirectional_nonlinear_X_to_Y",
    "common_driver_no_direct_XY",
    "bidirectional_X_to_Y_stronger",
)
NOISE_LEVELS = {"low": 0.10, "moderate": 0.35}
METHOD_CONFIGS = {
    "LP": {"k": 30, "lag_depth": 8, "theiler": 8},
    "CE": {"k": 30, "lag_depth": 8, "theiler": 8},
    "SSC": {"k": 20, "lag_depth": 15, "theiler": 0},
}


@dataclass(frozen=True)
class ValidationJob:
    scenario: str
    noise_name: str
    replicate: int
    surrogate_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_plan() -> None:
    """Stop if the prereal-data plan is absent or changed."""

    if not PLAN_PATH.is_file():
        raise RuntimeError(f"Frozen plan is missing: {PLAN_PATH}")
    observed = sha256_file(PLAN_PATH)
    if observed != PLAN_SHA256:
        raise RuntimeError(
            f"Frozen plan hash mismatch: {observed} != {PLAN_SHA256}"
        )


def _seed_sequence(*parts: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([SEED, *parts])


def simulate_pair(
    scenario: str,
    noise_name: str,
    replicate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one frozen ground-truth pair and apply real-data preprocessing."""

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if noise_name not in NOISE_LEVELS:
        raise ValueError(f"Unknown noise level: {noise_name}")
    scenario_index = SCENARIOS.index(scenario)
    noise_index = tuple(NOISE_LEVELS).index(noise_name)
    rng = np.random.default_rng(
        _seed_sequence(100, scenario_index, noise_index, replicate)
    )
    length = BURN_IN + N_RETAINED
    x = np.zeros(length, dtype=float)
    y = np.zeros(length, dtype=float)
    z = np.zeros(length, dtype=float)
    innovations = rng.normal(size=(3, length))

    for index in range(2, length):
        ex, ey, ez = innovations[:, index]
        if scenario == "uncoupled_linear":
            x[index] = 0.60 * x[index - 1] + ex
            y[index] = 0.50 * y[index - 1] + ey
        elif scenario == "unidirectional_linear_X_to_Y":
            x[index] = 0.60 * x[index - 1] + ex
            y[index] = (
                0.45 * y[index - 1] + 0.45 * x[index - 1] + ey
            )
        elif scenario == "unidirectional_nonlinear_X_to_Y":
            x[index] = 0.60 * x[index - 1] + ex
            y[index] = (
                0.45 * y[index - 1]
                + 0.65 * np.tanh(1.25 * x[index - 1])
                + ey
            )
        elif scenario == "common_driver_no_direct_XY":
            z[index] = 0.60 * z[index - 1] + ez
            x[index] = 0.45 * x[index - 1] + 0.55 * z[index - 1] + ex
            y[index] = 0.45 * y[index - 1] + 0.55 * z[index - 1] + ey
        elif scenario == "bidirectional_X_to_Y_stronger":
            x[index] = (
                0.45 * x[index - 1] + 0.25 * y[index - 1] + ex
            )
            y[index] = (
                0.45 * y[index - 1] + 0.50 * x[index - 1] + ey
            )

    x = x[BURN_IN:]
    y = y[BURN_IN:]
    measurement_fraction = NOISE_LEVELS[noise_name]
    x = x + rng.normal(scale=measurement_fraction * np.std(x, ddof=1), size=len(x))
    y = y + rng.normal(scale=measurement_fraction * np.std(y, ddof=1), size=len(y))
    x, _, _ = detrend_zscore(x)
    y, _, _ = detrend_zscore(y)
    return x, y


def _estimate(method: str, target: np.ndarray, source: np.ndarray):
    config = METHOD_CONFIGS[method]
    return estimate_direction(
        method,
        target,
        source,
        k=config["k"],
        lag_depth=config["lag_depth"],
        theiler=config["theiler"],
        source_lag_zero=False,
    )


def _surrogate_validation(
    method: str,
    target: np.ndarray,
    source: np.ndarray,
    scenario_index: int,
    noise_index: int,
    replicate: int,
    direction_index: int,
    count: int,
) -> tuple[float, int, str]:
    rng = np.random.default_rng(
        _seed_sequence(
            900,
            scenario_index,
            noise_index,
            replicate,
            tuple(METHOD_CONFIGS).index(method),
            direction_index,
        )
    )
    offsets = circular_shift_offsets(len(source), count, rng)
    values: list[float] = []
    failures: list[str] = []
    for offset in offsets:
        result = _estimate(method, target, np.roll(source, int(offset)))
        values.append(result.directed_strength)
        if not result.finite:
            failures.append(result.failure_reason)
    if failures:
        return np.nan, len(failures), ";".join(sorted(set(failures)))
    return surrogate_p_value(_estimate(method, target, source).directed_strength, values), 0, "NA"


def run_validation_job(job: ValidationJob) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run all three methods for one simulation replicate."""

    x, y = simulate_pair(job.scenario, job.noise_name, job.replicate)
    scenario_index = SCENARIOS.index(job.scenario)
    noise_index = tuple(NOISE_LEVELS).index(job.noise_name)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for method in METHOD_CONFIGS:
        forward = _estimate(method, y, x)
        reverse = _estimate(method, x, y)
        forward_p = np.nan
        reverse_p = np.nan
        forward_surrogate_failures = 0
        reverse_surrogate_failures = 0
        if job.scenario == "uncoupled_linear":
            forward_p, forward_surrogate_failures, forward_reason = _surrogate_validation(
                method,
                y,
                x,
                scenario_index,
                noise_index,
                job.replicate,
                0,
                job.surrogate_count,
            )
            reverse_p, reverse_surrogate_failures, reverse_reason = _surrogate_validation(
                method,
                x,
                y,
                scenario_index,
                noise_index,
                job.replicate,
                1,
                job.surrogate_count,
            )
            if forward_surrogate_failures:
                failures.append(
                    {
                        "scenario": job.scenario,
                        "noise": job.noise_name,
                        "replicate": job.replicate,
                        "method": method,
                        "direction": "X_to_Y",
                        "stage": "surrogate",
                        "failure_count": forward_surrogate_failures,
                        "reason": forward_reason,
                    }
                )
            if reverse_surrogate_failures:
                failures.append(
                    {
                        "scenario": job.scenario,
                        "noise": job.noise_name,
                        "replicate": job.replicate,
                        "method": method,
                        "direction": "Y_to_X",
                        "stage": "surrogate",
                        "failure_count": reverse_surrogate_failures,
                        "reason": reverse_reason,
                    }
                )

        for direction, result in (("X_to_Y", forward), ("Y_to_X", reverse)):
            if not result.finite:
                failures.append(
                    {
                        "scenario": job.scenario,
                        "noise": job.noise_name,
                        "replicate": job.replicate,
                        "method": method,
                        "direction": direction,
                        "stage": "observed",
                        "failure_count": 1,
                        "reason": result.failure_reason,
                    }
                )

        expected = {
            "uncoupled_linear": "none",
            "unidirectional_linear_X_to_Y": "X_to_Y",
            "unidirectional_nonlinear_X_to_Y": "X_to_Y",
            "common_driver_no_direct_XY": "none_direct",
            "bidirectional_X_to_Y_stronger": "bidirectional_X_to_Y_stronger",
        }[job.scenario]
        rows.append(
            {
                "scenario": job.scenario,
                "noise": job.noise_name,
                "replicate": job.replicate,
                "method": method,
                "expected_relationship": expected,
                "x_to_y_strength": forward.directed_strength,
                "y_to_x_strength": reverse.directed_strength,
                "x_to_y_finite": forward.finite,
                "y_to_x_finite": reverse.finite,
                "x_to_y_gt_y_to_x": bool(
                    forward.finite
                    and reverse.finite
                    and forward.directed_strength > reverse.directed_strength
                ),
                "x_to_y_surrogate_p": forward_p,
                "y_to_x_surrogate_p": reverse_p,
                "x_to_y_false_positive": bool(
                    np.isfinite(forward_p) and forward_p <= 0.05
                ),
                "y_to_x_false_positive": bool(
                    np.isfinite(reverse_p) and reverse_p <= 0.05
                ),
                "surrogate_count": (
                    job.surrogate_count
                    if job.scenario == "uncoupled_linear"
                    else 0
                ),
                "surrogate_failure_count": (
                    forward_surrogate_failures + reverse_surrogate_failures
                ),
            }
        )
    return rows, failures


def _iter_jobs(replicates: int, surrogate_count: int) -> Iterable[ValidationJob]:
    for scenario in SCENARIOS:
        for noise_name in NOISE_LEVELS:
            for replicate in range(replicates):
                yield ValidationJob(
                    scenario=scenario,
                    noise_name=noise_name,
                    replicate=replicate,
                    surrogate_count=surrogate_count,
                )


def summarize_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build scenario summaries and apply the frozen method gate."""

    summary_rows: list[dict[str, object]] = []
    for (method, scenario, noise), group in frame.groupby(
        ["method", "scenario", "noise"], sort=False
    ):
        directional_total = 2 * len(group)
        finite_total = int(group["x_to_y_finite"].sum() + group["y_to_x_finite"].sum())
        row: dict[str, object] = {
            "method": method,
            "scenario": scenario,
            "noise": noise,
            "replicates": len(group),
            "directional_estimates": directional_total,
            "finite_directional_estimates": finite_total,
            "failure_rate": 1.0 - finite_total / directional_total,
            "x_to_y_mean_strength": group["x_to_y_strength"].mean(),
            "y_to_x_mean_strength": group["y_to_x_strength"].mean(),
            "x_to_y_gt_y_to_x_fraction": group["x_to_y_gt_y_to_x"].mean(),
            "x_to_y_false_positive_rate": np.nan,
            "y_to_x_false_positive_rate": np.nan,
            "common_driver_warning": scenario == "common_driver_no_direct_XY",
        }
        if scenario == "uncoupled_linear":
            row["x_to_y_false_positive_rate"] = group[
                "x_to_y_false_positive"
            ].mean()
            row["y_to_x_false_positive_rate"] = group[
                "y_to_x_false_positive"
            ].mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    gate_rows: list[dict[str, object]] = []
    for method in METHOD_CONFIGS:
        method_summary = summary[summary["method"].eq(method)]
        uncoupled = method_summary[
            method_summary["scenario"].eq("uncoupled_linear")
        ]
        fpr_pass = bool(
            (uncoupled["x_to_y_false_positive_rate"] <= 0.10).all()
            and (uncoupled["y_to_x_false_positive_rate"] <= 0.10).all()
        )
        moderate_linear = method_summary[
            method_summary["scenario"].eq("unidirectional_linear_X_to_Y")
            & method_summary["noise"].eq("moderate")
        ].iloc[0]
        moderate_nonlinear = method_summary[
            method_summary["scenario"].eq("unidirectional_nonlinear_X_to_Y")
            & method_summary["noise"].eq("moderate")
        ].iloc[0]
        direction_pass = bool(
            moderate_linear["x_to_y_gt_y_to_x_fraction"] >= 0.80
            and moderate_nonlinear["x_to_y_gt_y_to_x_fraction"] >= 0.80
        )
        failure_pass = bool((method_summary["failure_rate"] <= 0.05).all())
        bidirectional = method_summary[
            method_summary["scenario"].eq("bidirectional_X_to_Y_stronger")
            & method_summary["noise"].eq("moderate")
        ].iloc[0]
        bidirectional_pass = bool(
            bidirectional["x_to_y_gt_y_to_x_fraction"] >= 0.65
        )
        reversal_pass = bool(
            moderate_linear["x_to_y_gt_y_to_x_fraction"] > 0.50
            and moderate_nonlinear["x_to_y_gt_y_to_x_fraction"] > 0.50
        )
        validated = bool(
            fpr_pass
            and direction_pass
            and failure_pass
            and bidirectional_pass
            and reversal_pass
        )
        gate_rows.append(
            {
                "method": method,
                "uncoupled_fpr_pass": fpr_pass,
                "moderate_linear_direction_fraction": moderate_linear[
                    "x_to_y_gt_y_to_x_fraction"
                ],
                "moderate_nonlinear_direction_fraction": moderate_nonlinear[
                    "x_to_y_gt_y_to_x_fraction"
                ],
                "unidirectional_direction_pass": direction_pass,
                "all_failure_rates_le_0p05": failure_pass,
                "bidirectional_moderate_direction_fraction": bidirectional[
                    "x_to_y_gt_y_to_x_fraction"
                ],
                "bidirectional_direction_pass": bidirectional_pass,
                "no_systematic_reversal_pass": reversal_pass,
                "validation_status": (
                    "VALIDATED" if validated else "METHOD_NOT_VALIDATED"
                ),
            }
        )
    return summary, pd.DataFrame(gate_rows)


def write_validation_figure(summary: pd.DataFrame, gate: pd.DataFrame, path: Path) -> None:
    """Write editable-text SVG validation overview."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "Arial"
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {"LP": "#0072B2", "CE": "#D55E00", "SSC": "#009E73"}
    x_labels = [
        "Linear\nX→Y",
        "Nonlinear\nX→Y",
        "Bidirectional\nX→Y stronger",
    ]
    scenarios = [
        "unidirectional_linear_X_to_Y",
        "unidirectional_nonlinear_X_to_Y",
        "bidirectional_X_to_Y_stronger",
    ]
    x = np.arange(len(scenarios), dtype=float)
    width = 0.22
    for method_index, method in enumerate(METHOD_CONFIGS):
        subset = summary[
            summary["method"].eq(method)
            & summary["noise"].eq("moderate")
            & summary["scenario"].isin(scenarios)
        ].set_index("scenario")
        values = [subset.loc[scenario, "x_to_y_gt_y_to_x_fraction"] for scenario in scenarios]
        axes[0].bar(
            x + (method_index - 1) * width,
            values,
            width,
            label=method,
            color=colors[method],
        )
    axes[0].axhline(0.80, color="0.35", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x, x_labels)
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_ylabel("Correct directional ordering fraction")
    axes[0].set_title("a  Direction recovery at moderate noise", loc="left")
    axes[0].legend(frameon=False)

    fpr_labels: list[str] = []
    fpr_values: list[float] = []
    fpr_colors: list[str] = []
    for method in METHOD_CONFIGS:
        for noise in NOISE_LEVELS:
            row = summary[
                summary["method"].eq(method)
                & summary["scenario"].eq("uncoupled_linear")
                & summary["noise"].eq(noise)
            ].iloc[0]
            fpr_labels.append(f"{method}\n{noise}")
            fpr_values.append(
                max(
                    row["x_to_y_false_positive_rate"],
                    row["y_to_x_false_positive_rate"],
                )
            )
            fpr_colors.append(colors[method])
    axes[1].bar(np.arange(len(fpr_values)), fpr_values, color=fpr_colors)
    axes[1].axhline(0.10, color="0.35", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(np.arange(len(fpr_values)), fpr_labels)
    axes[1].set_ylim(0.0, max(0.12, max(fpr_values) * 1.15))
    axes[1].set_ylabel("Maximum empirical false-positive rate")
    axes[1].set_title("b  Uncoupled-process calibration", loc="left")
    status_text = ", ".join(
        f"{row.method}: {row.validation_status}"
        for row in gate.itertuples(index=False)
    )
    figure.text(0.01, 0.01, status_text, fontsize=8, color="0.25")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def run_validation(
    *,
    replicates: int = 200,
    surrogate_count: int = 39,
    n_jobs: int = 4,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute all frozen validation cells."""

    assert_frozen_plan()
    jobs = list(_iter_jobs(replicates, surrogate_count))
    all_rows: list[dict[str, object]] = []
    all_failures: list[dict[str, object]] = []
    if n_jobs == 1:
        iterator = map(run_validation_job, jobs)
        for rows, failures in iterator:
            all_rows.extend(rows)
            all_failures.extend(failures)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for rows, failures in executor.map(
                run_validation_job, jobs, chunksize=max(1, replicates // 20)
            ):
                all_rows.extend(rows)
                all_failures.extend(failures)

    validation = pd.DataFrame(all_rows).sort_values(
        ["method", "scenario", "noise", "replicate"]
    )
    failures = pd.DataFrame(all_failures)
    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "scenario",
                "noise",
                "replicate",
                "method",
                "direction",
                "stage",
                "failure_count",
                "reason",
            ]
        )
    else:
        failures = failures.sort_values(
            ["method", "scenario", "noise", "replicate", "direction"]
        )
    summary, gate = summarize_validation(validation)

    if write_outputs:
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        validation.to_csv(
            VALIDATION_DIR / "nonlinear_method_validation.csv",
            index=False,
            encoding="utf-8",
            na_rep="NA",
            lineterminator="\n",
        )
        summary.to_csv(
            VALIDATION_DIR / "validation_scenario_summary.csv",
            index=False,
            encoding="utf-8",
            na_rep="NA",
            lineterminator="\n",
        )
        failures.to_csv(
            VALIDATION_DIR / "validation_failures.csv",
            index=False,
            encoding="utf-8",
            na_rep="NA",
            lineterminator="\n",
        )
        gate.to_csv(
            VALIDATION_DIR / "validation_method_gate.csv",
            index=False,
            encoding="utf-8",
            na_rep="NA",
            lineterminator="\n",
        )
        write_validation_figure(
            summary,
            gate,
            FIGURE_DIR / "nonlinear_method_validation.svg",
        )
    return validation, summary, failures, gate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--surrogates", type=int, default=39)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validation, summary, failures, gate = run_validation(
        replicates=args.replicates,
        surrogate_count=args.surrogates,
        n_jobs=args.jobs,
        write_outputs=True,
    )
    print(
        json.dumps(
            {
                "validation_rows": len(validation),
                "summary_rows": len(summary),
                "failure_rows": len(failures),
                "method_gate": gate.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
