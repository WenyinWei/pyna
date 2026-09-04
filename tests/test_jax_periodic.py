"""Analytic tests for implicitly differentiable periodic points."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pyna.topo.jax_periodic import periodic_point_state, solve_periodic_point


def _shifted_hyperbolic_field(xyz, params):
    rate, center_r, center_z = params
    x, y, z = xyz
    radius = jnp.sqrt(x * x + y * y)
    cos_phi = x / radius
    sin_phi = y / radius
    b_r = rate * (radius - center_r)
    b_z = -rate * (z - center_z)
    b_phi = radius
    return jnp.array(
        [
            b_r * cos_phi - b_phi * sin_phi,
            b_r * sin_phi + b_phi * cos_phi,
            b_z,
        ]
    )


def _solve(params, *, map_power=1):
    return solve_periodic_point(
        _shifted_hyperbolic_field,
        params,
        jnp.array([0.91, -0.08]),
        phi_start=0.23,
        phi_span=0.7,
        map_power=map_power,
        n_steps_per_span=128,
        newton_iterations=4,
    )


def test_periodic_point_solver_finds_shifted_hyperbolic_root():
    params = jnp.array([0.32, 1.04, 0.035])
    root = jax.jit(_solve)(params)
    np.testing.assert_allclose(root, params[1:], atol=2e-14)


def test_periodic_point_uses_implicit_parameter_derivative():
    params = jnp.array([0.32, 1.04, 0.035])
    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(jax.jacfwd(_solve)(params), expected, atol=3e-13)
    np.testing.assert_allclose(jax.jacrev(_solve)(params), expected, atol=3e-13)


def test_periodic_point_state_reports_monodromy_and_classification_data():
    params = jnp.array([0.32, 1.04, 0.035])
    phi_span = 0.7
    state = periodic_point_state(
        _shifted_hyperbolic_field,
        params,
        jnp.array([0.91, -0.08]),
        phi_start=0.23,
        phi_span=phi_span,
        n_steps_per_span=256,
        newton_iterations=4,
    )
    expected_monodromy = np.diag(
        [
            np.exp(params[0] * phi_span),
            np.exp(-params[0] * phi_span),
        ]
    )

    assert bool(state.converged)
    np.testing.assert_allclose(state.residual, 0.0, atol=2e-14)
    np.testing.assert_allclose(
        state.monodromy,
        expected_monodromy,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(state.determinant, 1.0, atol=2e-12)
    assert float(state.hyperbolicity_margin) > 0.0
    assert float(state.greene_residue) < 0.0


def test_periodic_point_state_trace_supports_reverse_mode():
    params = jnp.array([0.32, 1.04, 0.035])
    phi_span = 0.7

    def trace(dynamic_params):
        return periodic_point_state(
            _shifted_hyperbolic_field,
            dynamic_params,
            jnp.array([0.91, -0.08]),
            phi_span=phi_span,
            n_steps_per_span=128,
            newton_iterations=4,
        ).trace

    expected = jnp.array(
        [2.0 * phi_span * jnp.sinh(params[0] * phi_span), 0.0, 0.0]
    )
    np.testing.assert_allclose(
        jax.grad(trace)(params),
        expected,
        rtol=2e-11,
        atol=1e-14,
    )


def test_map_power_tracks_same_root_and_accumulates_monodromy():
    params = jnp.array([0.25, 1.02, -0.02])
    state = periodic_point_state(
        _shifted_hyperbolic_field,
        params,
        jnp.array([0.95, 0.05]),
        phi_span=0.6,
        map_power=2,
        n_steps_per_span=96,
        newton_iterations=4,
    )
    expected = np.diag(
        [
            np.exp(params[0] * 1.2),
            np.exp(-params[0] * 1.2),
        ]
    )
    np.testing.assert_allclose(state.position, params[1:], atol=3e-14)
    np.testing.assert_allclose(state.monodromy, expected, rtol=2e-11, atol=2e-14)


def test_periodic_point_validates_static_solver_configuration():
    params = jnp.array([0.25, 1.0, 0.0])
    with pytest.raises(ValueError, match="map_power must be positive"):
        _solve(params, map_power=0)
    with pytest.raises(ValueError, match="initial_guess"):
        solve_periodic_point(
            _shifted_hyperbolic_field,
            params,
            jnp.ones(3),
            phi_span=0.5,
        )
