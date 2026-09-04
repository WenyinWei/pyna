"""Analytic tests for the optional-JAX fixed-phi return map."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pyna.toroidal.flt.jax_poincare import (
    cylindrical_field_components,
    fixed_phi_trajectory,
    poincare_map,
    poincare_map_jacobian,
)


def _helical_field(xyz, pitch):
    """B_phi=R and B_z=pitch, giving R'=0 and Z'=pitch."""

    x, y, _ = xyz
    return jnp.array([-y, x, pitch])


def _hyperbolic_field(xyz, rate):
    """Field with exact poloidal flow R'=a(R-R0), Z'=-a Z."""

    x, y, z = xyz
    radius = jnp.sqrt(x * x + y * y)
    cos_phi = x / radius
    sin_phi = y / radius
    b_r = rate * (radius - 1.0)
    b_phi = radius
    b_z = -rate * z
    return jnp.array(
        [
            b_r * cos_phi - b_phi * sin_phi,
            b_r * sin_phi + b_phi * cos_phi,
            b_z,
        ]
    )


def test_cartesian_field_is_converted_to_pyna_component_order():
    rz = jnp.array([1.4, -0.2])
    components = cylindrical_field_components(
        _helical_field,
        0.3,
        rz,
        0.7,
    )
    np.testing.assert_allclose(components, [0.0, 0.3, 1.4], atol=1e-14)


def test_helical_map_is_jittable_and_differentiable_in_field_parameter():
    rz0 = jnp.array([1.4, -0.2])
    phi_span = 0.8

    def mapped(rz, pitch):
        return poincare_map(
            _helical_field,
            pitch,
            rz,
            phi_span=phi_span,
            n_steps=32,
        )

    result = jax.jit(mapped)(rz0, 0.3)
    np.testing.assert_allclose(result, [1.4, 0.04], atol=1e-13)
    np.testing.assert_allclose(
        jax.jacfwd(mapped, argnums=0)(rz0, 0.3),
        np.eye(2),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        jax.jacfwd(mapped, argnums=1)(rz0, 0.3),
        [0.0, phi_span],
        atol=1e-13,
    )


def test_fixed_phi_trajectory_matches_map_and_retains_parameter_gradient():
    rz0 = jnp.array([1.4, -0.2])
    phi_start = 0.17
    phi_span = 0.8

    def traced(pitch):
        return fixed_phi_trajectory(
            _helical_field,
            pitch,
            rz0,
            phi_start=phi_start,
            phi_span=phi_span,
            n_steps=16,
        )

    trajectory = jax.jit(traced)(0.3)
    expected_phi = np.linspace(phi_start, phi_start + phi_span, 17)
    expected_rz = np.column_stack(
        (
            np.full(17, float(rz0[0])),
            float(rz0[1]) + 0.3 * (expected_phi - phi_start),
        )
    )
    np.testing.assert_allclose(trajectory.phi, expected_phi, atol=1e-15)
    np.testing.assert_allclose(trajectory.rz, expected_rz, atol=2e-15)
    np.testing.assert_allclose(
        trajectory.rz[-1],
        poincare_map(
            _helical_field,
            0.3,
            rz0,
            phi_start=phi_start,
            phi_span=phi_span,
            n_steps=16,
        ),
        atol=0.0,
    )
    np.testing.assert_allclose(
        jax.jacfwd(lambda pitch: traced(pitch).rz[:, 1])(0.3),
        expected_phi - phi_start,
        atol=2e-15,
    )


def test_negative_phi_span_reverses_the_map():
    rz0 = jnp.array([1.2, 0.4])
    forward = poincare_map(
        _helical_field,
        -0.25,
        rz0,
        phi_span=1.1,
        phi_start=0.2,
        n_steps=32,
    )
    recovered = poincare_map(
        _helical_field,
        -0.25,
        forward,
        phi_span=-1.1,
        phi_start=1.3,
        n_steps=32,
    )
    np.testing.assert_allclose(recovered, rz0, atol=1e-13)


def test_hyperbolic_map_has_expected_symplectic_monodromy():
    rate = 0.35
    phi_span = 0.9
    rz0 = jnp.array([1.15, 0.2])

    monodromy = poincare_map_jacobian(
        _hyperbolic_field,
        rate,
        rz0,
        phi_span=phi_span,
        n_steps=256,
    )
    expected = np.diag(
        [np.exp(rate * phi_span), np.exp(-rate * phi_span)]
    )
    np.testing.assert_allclose(monodromy, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.linalg.det(monodromy), 1.0, atol=2e-12)


def test_invalid_fixed_phi_map_inputs_fail_explicitly():
    with pytest.raises(ValueError, match="n_steps must be positive"):
        poincare_map(
            _helical_field,
            0.2,
            jnp.array([1.0, 0.0]),
            phi_span=1.0,
            n_steps=0,
        )

    def zero_field(xyz, params):
        del xyz, params
        return jnp.zeros(3)

    result = poincare_map(
        zero_field,
        None,
        jnp.array([1.0, 0.0]),
        phi_span=0.1,
        n_steps=1,
    )
    assert bool(jnp.isnan(result).all())
