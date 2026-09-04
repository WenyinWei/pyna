"""Tests for the exact-label JAX/Cyna wall-strike parity gate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyna.topo.manifold_strike_contracts import (
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
)
from pyna.topo.manifold_strike_correspondence import (
    compare_jax_manifold_strike,
)


def _match(distance_mode):
    return ManifoldStrikeMatch(
        label=ManifoldStrikeLabel("orbit7:P0.unstable.plus", "+", 5),
        point_RZPhi_m_rad=np.array([1.5, 0.2, 0.2]),
        target_distance_m=0.0,
        connection_length_m=12.0,
        bundle_seed_index=0,
        distance_mode=distance_mode,
    )


def _jax_state(
    point_RZPhi,
    *,
    periodic_converged=True,
    intersection_converged=True,
    within_window=True,
    wall_residual=0.0,
    transversality=0.4,
    phi_shift=0.01,
):
    return SimpleNamespace(
        periodic_point=SimpleNamespace(converged=periodic_converged),
        intersection=SimpleNamespace(
            point_RZPhi=np.asarray(point_RZPhi),
            converged=intersection_converged,
            within_window=within_window,
            wall_residual=wall_residual,
            transversality=transversality,
            phi_shift_from_guess=phi_shift,
        ),
    )


def test_axisymmetric_correspondence_ignores_toroidal_phase():
    report = compare_jax_manifold_strike(
        _match("rz"),
        _jax_state([1.5003, 0.1998, 1.2]),
        absolute_tolerance_m=5.0e-4,
    )

    assert report.accepted
    assert report.rejection_reason is None
    assert report.distance_mode == "rz"
    np.testing.assert_allclose(
        report.deviation_m,
        np.hypot(3.0e-4, 2.0e-4),
    )


def test_three_dimensional_correspondence_retains_toroidal_phase():
    report = compare_jax_manifold_strike(
        _match("xyz"),
        _jax_state([1.5, 0.2, 1.2]),
        absolute_tolerance_m=5.0e-4,
    )

    assert not report.accepted
    assert report.rejection_reason == "strike_deviation_exceeds_tolerance"
    assert report.deviation_m > 1.0


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (
            _jax_state([1.5, 0.2, 0.2], periodic_converged=False),
            "periodic_point_not_converged",
        ),
        (
            _jax_state([np.nan, 0.2, 0.2], intersection_converged=False),
            "jax_strike_nonfinite",
        ),
        (
            _jax_state([1.5, 0.2, 0.2], within_window=False),
            "wall_intersection_left_event_window",
        ),
        (
            _jax_state([1.5, 0.2, 0.2], intersection_converged=False),
            "wall_intersection_not_converged",
        ),
    ],
)
def test_correspondence_rejects_failed_jax_diagnostics(state, reason):
    report = compare_jax_manifold_strike(
        _match("rz"),
        state,
        absolute_tolerance_m=1.0e-3,
    )

    assert not report.accepted
    assert report.rejection_reason == reason


def test_correspondence_validates_tolerance_and_state_contract():
    with pytest.raises(ValueError, match="non-negative"):
        compare_jax_manifold_strike(
            _match("rz"),
            _jax_state([1.5, 0.2, 0.2]),
            absolute_tolerance_m=-1.0,
        )
    with pytest.raises(TypeError, match="periodic-point"):
        compare_jax_manifold_strike(
            _match("rz"),
            object(),
            absolute_tolerance_m=1.0e-3,
        )
