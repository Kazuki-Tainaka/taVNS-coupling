"""Estimator, direction, surrogate, and simulation unit tests."""

from __future__ import annotations

import numpy as np

from analysis_utils import direction_arrays
from nonlinear_estimators import (
    estimate_direction,
    estimate_porta_lp,
    surrogate_p_value,
    surrogate_significant,
)
from validation_simulations import simulate_pair


def test_deterministic_output_under_fixed_seed() -> None:
    x_first, y_first = simulate_pair(
        "unidirectional_nonlinear_X_to_Y", "moderate", 7
    )
    x_second, y_second = simulate_pair(
        "unidirectional_nonlinear_X_to_Y", "moderate", 7
    )
    assert np.array_equal(x_first, x_second)
    assert np.array_equal(y_first, y_second)
    first = estimate_porta_lp(y_first, x_first)
    second = estimate_porta_lp(y_second, x_second)
    assert first.to_dict() == second.to_dict()


def test_direction_label_maps_source_to_target() -> None:
    rri = np.array([1.0, 2.0, 3.0])
    sbp = np.array([4.0, 5.0, 6.0])
    target, source, target_name, source_name = direction_arrays(
        "SBP→RRI", rri, sbp
    )
    assert np.array_equal(target, rri)
    assert np.array_equal(source, sbp)
    assert (target_name, source_name) == ("RRI", "SBP")
    target, source, target_name, source_name = direction_arrays(
        "RRI→SBP", rri, sbp
    )
    assert np.array_equal(target, sbp)
    assert np.array_equal(source, rri)
    assert (target_name, source_name) == ("SBP", "RRI")


def test_surrogate_p_value_and_strict_threshold() -> None:
    surrogates = np.arange(1.0, 200.0)
    assert surrogate_p_value(200.0, surrogates) == 1.0 / 200.0
    significant, threshold, p_value = surrogate_significant(
        200.0, surrogates
    )
    assert significant
    assert threshold == np.quantile(surrogates, 0.95, method="higher")
    assert p_value == 1.0 / 200.0
    equal_significant, _, _ = surrogate_significant(threshold, surrogates)
    assert not equal_significant


def test_known_direction_simulation() -> None:
    ordering = []
    for replicate in range(8):
        x, y = simulate_pair(
            "unidirectional_linear_X_to_Y", "moderate", replicate
        )
        forward = estimate_direction(
            "LP", y, x, k=30, lag_depth=8, theiler=8
        )
        reverse = estimate_direction(
            "LP", x, y, k=30, lag_depth=8, theiler=8
        )
        assert forward.finite and reverse.finite
        ordering.append(forward.directed_strength > reverse.directed_strength)
    assert np.mean(ordering) >= 0.75


def test_missing_input_is_not_imputed() -> None:
    target = np.linspace(-1.0, 1.0, 256)
    source = np.sin(np.linspace(0.0, 10.0, 256))
    target[50] = np.nan
    result = estimate_porta_lp(target, source)
    assert not result.finite
    assert result.failure_reason == "invalid_or_short_input"

