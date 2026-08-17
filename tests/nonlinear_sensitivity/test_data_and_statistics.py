"""Data-boundary, no-interpolation, and statistical utility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_utils import (
    PHASES,
    SegmentSetting,
    bca_mean_ci,
    bh_adjust,
    load_valid_phase,
    phase_contrast_rows,
    select_segment,
)
from plot_nonlinear_results import DIRECTION_MARKERS, _label


def test_paired_subject_bca_is_deterministic_and_translation_equivariant() -> None:
    differences = np.array([-0.2, 0.1, 0.4, 0.3, -0.1, 0.7])
    first = bca_mean_ci(differences, resamples=2_000, seed_parts=(9,))
    second = bca_mean_ci(differences, resamples=2_000, seed_parts=(9,))
    shifted = bca_mean_ci(
        differences + 2.0, resamples=2_000, seed_parts=(9,)
    )
    assert first == second
    assert np.allclose(np.asarray(shifted) - np.asarray(first), 2.0)


def test_bh_correction_known_values_and_na_preservation() -> None:
    observed = bh_adjust([0.01, 0.04, 0.03, 0.002, np.nan])
    expected = np.array([0.02, 0.04, 0.04, 0.008, np.nan])
    assert np.allclose(observed[:4], expected[:4])
    assert np.isnan(observed[4])


def test_pairwise_complete_missing_data_count() -> None:
    rows = []
    values = {
        "01": {"Pre": 0.1, "Stim": 0.2, "Post": 0.3},
        "02": {"Pre": 0.2, "Stim": np.nan, "Post": 0.4},
        "03": {"Pre": 0.3, "Stim": 0.5, "Post": 0.6},
    }
    for subject, phases in values.items():
        for phase, value in phases.items():
            rows.append(
                {
                    "subject_id": subject,
                    "phase": phase,
                    "method": "LP",
                    "direction": "SBP→RRI",
                    "directed_strength": value,
                    "finite": np.isfinite(value),
                }
            )
    contrasts = phase_contrast_rows(
        pd.DataFrame(rows), bootstrap_resamples=200
    )
    primary = contrasts[contrasts["contrast"].eq("Stim-Pre")].iloc[0]
    assert primary["evaluable_paired_n"] == 2
    assert primary["estimability_status"] == "LOW_ESTIMABILITY"


def test_figure_direction_labels_and_markers_preserve_csv_schema() -> None:
    assert _label("LP", "SBP→RRI") == "LP SBP→RRI"
    assert _label("LP", "RRI→SBP") == "LP RRI→SBP"
    assert DIRECTION_MARKERS["SBP→RRI"] != DIRECTION_MARKERS["RRI→SBP"]
