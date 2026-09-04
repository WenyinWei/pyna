"""Analytic tests for implicitly differentiable local wall crossings."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pyna.topo.jax_strike import solve_wall_intersection, wall_intersection_state


def _constant_poloidal_slope_field(xyz, params):
    radial_slope, vertical_slope = params
    x, y, z = xyz
    del z
    radius = jnp.sqrt(x * x + y * y)
    cos_phi = x / radius
    sin_phi = y / radius
    b_r = radial_slope
    b_z = vertical_slope
    b_phi = radius
    return jnp.array(
        [
            b_r * cos_phi - b_phi * sin_phi,
            b_r * sin_phi + b_phi * cos_phi,
            b_z,
        ]
    )


def _radial_wall(rz, phi, wall_params):
    del phi
    return rz[0] - wall_params[0]


def _solve(params, wall_params, rz0, phi_start):
    return solve_wall_intersection(
        _constant_poloidal_slope_field,
        params,
        rz0,
        _radial_wall,
        wall_params,
        phi_start=phi_start,
        phi_guess=0.9,
        n_steps=32,
        newton_iterations=3,
        maximum_phi_shift=0.5,
    )


def test_wall_intersection_matches_analytic_crossing_and_state():
    params = jnp.array([0.4, -0.15])
    wall_params = jnp.array([1.3])
    rz0 = jnp.array([1.0, 0.2])
    phi_start = jnp.asarray(0.1)
    expected_phi = phi_start + (wall_params[0] - rz0[0]) / params[0]
    expected_z = rz0[1] + params[1] * (expected_phi - phi_start)

    state = jax.jit(
        lambda active_params: wall_intersection_state(
            _constant_poloidal_slope_field,
            active_params,
            rz0,
            _radial_wall,
            wall_params,
            phi_start=phi_start,
            phi_guess=0.9,
            n_steps=32,
            newton_iterations=3,
            maximum_phi_shift=0.5,
        )
    )(params)

    assert bool(state.converged)
    np.testing.assert_allclose(state.phi, expected_phi, atol=2.0e-14)
    np.testing.assert_allclose(
        state.point_RZPhi,
        [wall_params[0], expected_z, expected_phi],
        atol=2.0e-14,
    )
    np.testing.assert_allclose(state.wall_residual, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(state.transversality, params[0], atol=2.0e-14)


def test_wall_intersection_uses_implicit_forward_and_reverse_derivatives():
    params = jnp.array([0.4, -0.15])
    wall_params = jnp.array([1.3])
    rz0 = jnp.array([1.0, 0.2])
    phi_start = jnp.asarray(0.1)
    distance = float(wall_params[0] - rz0[0])
    expected_params = np.array([-distance / params[0] ** 2, 0.0])
    expected_wall = np.array([1.0 / params[0]])
    expected_rz0 = np.array([-1.0 / params[0], 0.0])

    np.testing.assert_allclose(
        jax.jacfwd(_solve, argnums=0)(params, wall_params, rz0, phi_start),
        expected_params,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        jax.jacrev(_solve, argnums=0)(params, wall_params, rz0, phi_start),
        expected_params,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        jax.jacfwd(_solve, argnums=1)(params, wall_params, rz0, phi_start),
        expected_wall,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        jax.jacfwd(_solve, argnums=2)(params, wall_params, rz0, phi_start),
        expected_rz0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        jax.jacfwd(_solve, argnums=3)(params, wall_params, rz0, phi_start),
        1.0,
        atol=2.0e-13,
    )


def test_wall_hit_xyz_derivative_includes_moving_event_time():
    params = jnp.array([0.4, -0.15])
    wall_params = jnp.array([1.3])
    rz0 = jnp.array([1.0, 0.2])
    phi_start = 0.1

    def hit_xyz(active_params):
        return wall_intersection_state(
            _constant_poloidal_slope_field,
            active_params,
            rz0,
            _radial_wall,
            wall_params,
            phi_start=phi_start,
            phi_guess=0.9,
            n_steps=32,
            newton_iterations=3,
            maximum_phi_shift=0.5,
        ).point_xyz

    forward = jax.jacfwd(hit_xyz)(params)
    reverse = jax.jacrev(hit_xyz)(params)
    delta = 1.0e-5
    finite_difference = (
        hit_xyz(params + jnp.array([delta, 0.0]))
        - hit_xyz(params - jnp.array([delta, 0.0]))
    ) / (2.0 * delta)

    np.testing.assert_allclose(reverse, forward, atol=2.0e-13)
    np.testing.assert_allclose(
        forward[:, 0],
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_wall_intersection_rejects_window_boundary_and_tangency():
    params = jnp.array([0.4, -0.15])
    state = wall_intersection_state(
        _constant_poloidal_slope_field,
        params,
        jnp.array([1.0, 0.2]),
        _radial_wall,
        jnp.array([1.3]),
        phi_start=0.1,
        phi_guess=0.2,
        n_steps=32,
        newton_iterations=3,
        maximum_phi_shift=0.1,
    )

    assert not bool(state.within_window)
    assert not bool(state.converged)

    def tangent_wall(rz, phi, wall_params):
        del phi
        return (rz[0] - wall_params[0]) ** 2

    tangent = wall_intersection_state(
        _constant_poloidal_slope_field,
        params,
        jnp.array([1.0, 0.2]),
        tangent_wall,
        jnp.array([1.3]),
        phi_start=0.1,
        phi_guess=0.85,
        n_steps=32,
        newton_iterations=16,
        maximum_phi_shift=0.5,
        residual_tolerance=1.0e-10,
        minimum_abs_transversality=1.0e-5,
    )
    assert not bool(tangent.converged)


def test_wall_intersection_validates_static_configuration():
    params = jnp.array([0.4, -0.15])
    kwargs = {
        "field_fn": _constant_poloidal_slope_field,
        "params": params,
        "rz0": jnp.array([1.0, 0.2]),
        "wall_level_set_fn": _radial_wall,
        "wall_params": jnp.array([1.3]),
        "phi_start": 0.1,
        "phi_guess": 0.85,
    }
    with pytest.raises(ValueError, match="n_steps must be positive"):
        solve_wall_intersection(**kwargs, n_steps=0)
    with pytest.raises(ValueError, match="maximum_phi_shift"):
        solve_wall_intersection(**kwargs, maximum_phi_shift=0.0)
    with pytest.raises(ValueError, match="rz0"):
        solve_wall_intersection(**{**kwargs, "rz0": jnp.ones(3)})
    with pytest.raises(ValueError, match="scalar"):
        solve_wall_intersection(
            **{**kwargs, "wall_level_set_fn": lambda rz, phi, wall: rz}
        )
