"""Analytic tests for differentiable invariant-manifold generations."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pyna.topo.jax_manifold import (
    fundamental_segment_distances,
    hyperbolic_directions,
    manifold_seed_segment,
    trace_manifold_generations,
)


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


def test_hyperbolic_directions_are_oriented_outer_data():
    expansion = 2.5
    monodromy = jnp.diag(jnp.array([expansion, 1.0 / expansion]))
    directions = hyperbolic_directions(
        monodromy,
        stable_reference=jnp.array([0.0, -1.0]),
        unstable_reference=jnp.array([-1.0, 0.0]),
    )

    assert bool(directions.valid)
    np.testing.assert_allclose(directions.stable_direction, [0.0, -1.0])
    np.testing.assert_allclose(directions.unstable_direction, [-1.0, 0.0])
    np.testing.assert_allclose(directions.stable_backward_expansion, expansion)
    np.testing.assert_allclose(directions.unstable_expansion, expansion)

    def selected_data(matrix):
        selected = hyperbolic_directions(
            matrix,
            stable_reference=jnp.array([0.0, -1.0]),
            unstable_reference=jnp.array([-1.0, 0.0]),
        )
        return jnp.concatenate(
            (
                selected.stable_direction,
                selected.unstable_direction,
                jnp.array(
                    [
                        selected.stable_backward_expansion,
                        selected.unstable_expansion,
                    ]
                ),
            )
        )

    np.testing.assert_allclose(jax.jacfwd(selected_data)(monodromy), 0.0)


def test_elliptic_monodromy_is_not_a_valid_manifold_source():
    angle = 0.4
    rotation = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ]
    )
    assert not bool(hyperbolic_directions(rotation).valid)


def test_fundamental_segment_matches_production_carousel_spacing():
    expansion = 4.0
    distances = fundamental_segment_distances(
        -expansion,
        minimum_distance=1.0e-5,
        maximum_distance=1.0e-2,
        n_seeds=4,
    )
    ratio = expansion**0.25

    np.testing.assert_allclose(distances[1:] / distances[:-1], ratio)
    np.testing.assert_allclose(expansion * distances[0], ratio * distances[-1])
    assert float(distances[-1]) <= 1.0e-2


def test_seed_direction_and_distances_are_frozen_but_origin_is_active():
    origin = jnp.array([1.1, -0.04])
    direction = jnp.array([2.0, 0.0])
    distances = jnp.array([1.0e-4, 2.0e-4])

    direction_jacobian = jax.jacfwd(
        lambda active_direction: manifold_seed_segment(
            origin,
            active_direction,
            distances,
            side=1,
        )
    )(direction)
    distance_jacobian = jax.jacfwd(
        lambda active_distances: manifold_seed_segment(
            origin,
            direction,
            active_distances,
            side=1,
        )
    )(distances)
    origin_jacobian = jax.jacfwd(
        lambda active_origin: manifold_seed_segment(
            active_origin,
            direction,
            distances,
            side=1,
        )
    )(origin)

    np.testing.assert_allclose(direction_jacobian, 0.0)
    np.testing.assert_allclose(distance_jacobian, 0.0)
    np.testing.assert_allclose(
        origin_jacobian,
        np.broadcast_to(np.eye(2), (len(distances), 2, 2)),
    )


def test_analytic_stable_and_unstable_generations_follow_invariant_axes():
    params = jnp.array([0.3, 1.1, -0.04])
    phi_span = 0.6
    origin = params[1:]
    distances = jnp.array([1.0e-4, 2.0e-4, 4.0e-4])
    unstable_seeds = manifold_seed_segment(
        origin,
        jnp.array([1.0, 0.0]),
        distances,
        side=1,
    )
    stable_seeds = manifold_seed_segment(
        origin,
        jnp.array([0.0, 1.0]),
        distances,
        side=-1,
    )

    unstable = jax.jit(
        lambda active_params: trace_manifold_generations(
            _shifted_hyperbolic_field,
            active_params,
            unstable_seeds,
            stability="unstable",
            phi_span=phi_span,
            n_generations=3,
            n_steps_per_span=128,
        )
    )(params)
    stable = trace_manifold_generations(
        _shifted_hyperbolic_field,
        params,
        stable_seeds,
        stability="stable",
        phi_span=phi_span,
        n_generations=3,
        n_steps_per_span=128,
    )

    factors = np.exp(float(params[0]) * phi_span * np.arange(4))
    np.testing.assert_allclose(
        unstable[:, :, 0] - float(origin[0]),
        factors[:, None] * np.asarray(distances)[None, :],
        rtol=3.0e-12,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(unstable[:, :, 1], float(origin[1]), atol=2.0e-14)
    np.testing.assert_allclose(stable[:, :, 0], float(origin[0]), atol=2.0e-14)
    np.testing.assert_allclose(
        stable[:, :, 1] - float(origin[1]),
        -factors[:, None] * np.asarray(distances)[None, :],
        rtol=3.0e-12,
        atol=2.0e-14,
    )


def test_unstable_branch_endpoint_has_exact_parameter_gradient():
    phi_span = 0.5
    generations = 4
    distance = 2.0e-4

    def endpoint_r(rate):
        params = jnp.array([rate, 1.1, -0.04])
        seeds = manifold_seed_segment(
            params[1:],
            jnp.array([1.0, 0.0]),
            jnp.array([distance]),
            side=1,
        )
        traced = trace_manifold_generations(
            _shifted_hyperbolic_field,
            params,
            seeds,
            stability="unstable",
            phi_span=phi_span,
            n_generations=generations,
            n_steps_per_span=128,
        )
        return traced[-1, 0, 0]

    rate = 0.25
    exponential = np.exp(rate * phi_span * generations)
    expected = distance * phi_span * generations * exponential
    np.testing.assert_allclose(jax.grad(endpoint_r)(rate), expected, rtol=5.0e-12)


def test_manifold_api_validates_static_configuration():
    with pytest.raises(ValueError, match="imaginary_tolerance"):
        hyperbolic_directions(jnp.eye(2), imaginary_tolerance=-1.0)
    with pytest.raises(ValueError, match="side"):
        manifold_seed_segment(
            jnp.ones(2),
            jnp.ones(2),
            jnp.ones(2),
            side=0,
        )
    with pytest.raises(ValueError, match="stability"):
        trace_manifold_generations(
            _shifted_hyperbolic_field,
            jnp.array([0.3, 1.0, 0.0]),
            jnp.ones((2, 2)),
            stability="center",
            phi_span=0.5,
        )
    with pytest.raises(ValueError, match="n_generations must be positive"):
        trace_manifold_generations(
            _shifted_hyperbolic_field,
            jnp.array([0.3, 1.0, 0.0]),
            jnp.ones((2, 2)),
            stability="unstable",
            phi_span=0.5,
            n_generations=0,
        )

