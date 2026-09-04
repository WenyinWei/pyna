"""Analytic tests for implicitly differentiable local wall crossings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pyna.topo.jax_strike as jax_strike_module
from pyna.topo.jax_strike import (
    solve_wall_intersection,
    trace_manifold_strike,
    trace_manifold_strike_bundle,
    wall_intersection_state,
)
from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_strike_contracts import (
    LocalWallPlane,
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
)
from pyna.topo.manifold_strike_correspondence import (
    compare_jax_manifold_strike_bundle,
    compare_jax_manifold_strike,
)


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


def _xyz(R, Z, phi):
    return np.array([R * np.cos(phi), R * np.sin(phi), Z])


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


def _manifold_strike_reference():
    rate = 0.4
    center = np.array([1.0, -0.05])
    distance = 0.01
    map_span = 0.5
    wall_R = 1.2
    phi_start = 0.1
    hit_phi = phi_start + np.log((wall_R - center[0]) / distance) / rate
    hit_xyz = _xyz(wall_R, center[1], hit_phi)
    label = ManifoldStrikeLabel(
        branch_label="1:P0.unstable.plus",
        direction="+",
        seed_order=7,
    )
    branch = ManifoldBranchReference(
        origin_label="1:P0",
        stability="unstable",
        seed_side=1,
        section_phi=phi_start,
        map_span=map_span,
        anchor_RZ_m=center,
        direction_RZ=np.array([1.0, 0.0]),
        expansion=np.exp(rate * map_span),
        orientation_reversing=False,
        seed_orders=np.array([7]),
        seed_distances_m=np.array([distance]),
        points_RZ_m=np.array([[[center[0] + distance, center[1]]]]),
        available=np.ones((1, 1), dtype=bool),
    )
    match = ManifoldStrikeMatch(
        label=label,
        point_RZPhi_m_rad=np.array([wall_R, center[1], hit_phi]),
        target_distance_m=0.0,
        connection_length_m=12.0,
        bundle_seed_index=0,
    )
    plane = LocalWallPlane(
        strike_label=label,
        point_xyz_m=hit_xyz,
        normal_xyz=np.array([np.cos(hit_phi), np.sin(hit_phi), 0.0]),
        wall_phi_rad=hit_phi,
        wall_s=0.25,
        projection_distance_m=0.0,
    )
    return branch, match, plane


def _trace_labelled_strike(params):
    branch, match, plane = _manifold_strike_reference()
    return trace_manifold_strike(
        _shifted_hyperbolic_field,
        params,
        branch,
        match,
        plane,
        n_steps_per_span=64,
        xline_newton_iterations=4,
        wall_n_steps=512,
        wall_newton_iterations=5,
        maximum_phi_shift=0.2,
    )


def _manifold_strike_bundle_reference():
    rate = 0.4
    center = np.array([1.0, -0.05])
    distances = np.array([0.01, 0.02])
    orders = np.array([7, 11])
    map_span = 0.5
    wall_R = 1.2
    phi_start = 0.1
    branch = ManifoldBranchReference(
        origin_label="1:P0",
        stability="unstable",
        seed_side=1,
        section_phi=phi_start,
        map_span=map_span,
        anchor_RZ_m=center,
        direction_RZ=np.array([1.0, 0.0]),
        expansion=np.exp(rate * map_span),
        orientation_reversing=False,
        seed_orders=orders,
        seed_distances_m=distances,
        points_RZ_m=np.array(
            [[[center[0] + distance, center[1]] for distance in distances]]
        ),
        available=np.ones((1, len(distances)), dtype=bool),
    )
    matches = []
    planes = []
    for index, (order, distance) in enumerate(zip(orders, distances, strict=True)):
        hit_phi = phi_start + np.log((wall_R - center[0]) / distance) / rate
        label = ManifoldStrikeLabel(
            branch_label=branch.label,
            direction="+",
            seed_order=int(order),
        )
        matches.append(
            ManifoldStrikeMatch(
                label=label,
                point_RZPhi_m_rad=np.array([wall_R, center[1], hit_phi]),
                target_distance_m=0.0,
                connection_length_m=12.0,
                bundle_seed_index=index,
            )
        )
        planes.append(
            LocalWallPlane(
                strike_label=label,
                point_xyz_m=_xyz(wall_R, center[1], hit_phi),
                normal_xyz=np.array([np.cos(hit_phi), np.sin(hit_phi), 0.0]),
                wall_phi_rad=hit_phi,
                wall_s=0.25 + 0.1 * index,
                projection_distance_m=0.0,
            )
        )
    return branch, tuple(matches), tuple(planes)


def _trace_labelled_bundle(params):
    branch, matches, planes = _manifold_strike_bundle_reference()
    return trace_manifold_strike_bundle(
        _shifted_hyperbolic_field,
        params,
        branch,
        matches,
        planes,
        n_steps_per_span=64,
        xline_newton_iterations=4,
        wall_n_steps=512,
        wall_newton_iterations=5,
        maximum_phi_shift=0.2,
    )


def test_labelled_manifold_strike_tracks_moving_xline_and_exact_seed():
    params = jnp.array([0.4, 1.0, -0.05])
    branch, match, _plane = _manifold_strike_reference()
    state = jax.jit(_trace_labelled_strike)(params)

    assert bool(state.periodic_point.converged)
    assert bool(state.intersection.converged)
    np.testing.assert_allclose(state.periodic_point.position, params[1:], atol=2.0e-13)
    np.testing.assert_allclose(state.seed_RZ, [1.01, -0.05], atol=2.0e-13)
    np.testing.assert_allclose(
        state.intersection.point_xyz,
        match.point_xyz_m,
        rtol=0.0,
        atol=1.0e-10,
    )
    correspondence = compare_jax_manifold_strike(
        match,
        state,
        absolute_tolerance_m=1.0e-10,
    )
    assert correspondence.accepted

    moved_center = params.at[1:].add(jnp.array([2.0e-4, -3.0e-4]))
    moved = _trace_labelled_strike(moved_center)
    np.testing.assert_allclose(
        moved.periodic_point.position - state.periodic_point.position,
        [2.0e-4, -3.0e-4],
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        moved.seed_RZ - state.seed_RZ,
        [2.0e-4, -3.0e-4],
        atol=2.0e-12,
    )


def test_labelled_manifold_strike_gradient_matches_finite_difference():
    params = jnp.array([0.4, 1.0, -0.05])

    def hit_xyz(active_params):
        return _trace_labelled_strike(active_params).intersection.point_xyz

    forward = jax.jacfwd(hit_xyz)(params)
    reverse = jax.jacrev(hit_xyz)(params)
    delta = 2.0e-6
    direction = jnp.array([0.3, -0.2, 0.4])
    finite_difference = (
        hit_xyz(params + delta * direction)
        - hit_xyz(params - delta * direction)
    ) / (2.0 * delta)

    np.testing.assert_allclose(reverse, forward, rtol=2.0e-9, atol=2.0e-10)
    np.testing.assert_allclose(
        forward @ direction,
        finite_difference,
        rtol=2.0e-7,
        atol=2.0e-8,
    )


def test_labelled_manifold_strike_rejects_mismatched_outer_identity():
    params = jnp.array([0.4, 1.0, -0.05])
    branch, match, plane = _manifold_strike_reference()
    wrong_label = ManifoldStrikeLabel(
        branch_label=match.label.branch_label,
        direction=match.label.direction,
        seed_order=9,
    )
    wrong_match = ManifoldStrikeMatch(
        label=wrong_label,
        point_RZPhi_m_rad=match.point_RZPhi_m_rad,
        target_distance_m=0.0,
        connection_length_m=match.connection_length_m,
        bundle_seed_index=0,
    )

    with pytest.raises(ValueError, match="different strike label"):
        trace_manifold_strike(
            _shifted_hyperbolic_field,
            params,
            branch,
            wrong_match,
            plane,
            maximum_phi_shift=0.2,
        )


def test_labelled_strike_bundle_matches_ordered_single_strikes():
    params = jnp.array([0.4, 1.0, -0.05])
    branch, matches, planes = _manifold_strike_bundle_reference()
    bundle = jax.jit(_trace_labelled_bundle)(params)

    assert bool(bundle.periodic_point.converged)
    assert np.all(np.asarray(bundle.intersections.converged))
    np.testing.assert_allclose(
        bundle.seed_RZ,
        [[1.01, -0.05], [1.02, -0.05]],
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        bundle.intersections.point_xyz,
        [match.point_xyz_m for match in matches],
        rtol=0.0,
        atol=1.0e-10,
    )

    for index, (match, plane) in enumerate(zip(matches, planes, strict=True)):
        single = trace_manifold_strike(
            _shifted_hyperbolic_field,
            params,
            branch,
            match,
            plane,
            n_steps_per_span=64,
            xline_newton_iterations=4,
            wall_n_steps=512,
            wall_newton_iterations=5,
            maximum_phi_shift=0.2,
        )
        np.testing.assert_allclose(bundle.seed_RZ[index], single.seed_RZ, atol=0.0)
        np.testing.assert_allclose(
            bundle.intersections.point_xyz[index],
            single.intersection.point_xyz,
            atol=2.0e-13,
        )


def test_labelled_strike_bundle_gradient_matches_finite_difference():
    params = jnp.array([0.4, 1.0, -0.05])

    def hit_xyz(active_params):
        return _trace_labelled_bundle(active_params).intersections.point_xyz

    forward = jax.jacfwd(hit_xyz)(params)
    reverse = jax.jacrev(hit_xyz)(params)
    delta = 2.0e-6
    direction = jnp.array([0.3, -0.2, 0.4])
    finite_difference = (
        hit_xyz(params + delta * direction)
        - hit_xyz(params - delta * direction)
    ) / (2.0 * delta)

    np.testing.assert_allclose(reverse, forward, rtol=2.0e-9, atol=2.0e-10)
    np.testing.assert_allclose(
        jnp.einsum("sxp,p->sx", forward, direction),
        finite_difference,
        rtol=2.0e-7,
        atol=2.0e-8,
    )


def test_labelled_strike_bundle_solves_one_xline_and_validates_labels(monkeypatch):
    params = jnp.array([0.4, 1.0, -0.05])
    branch, matches, planes = _manifold_strike_bundle_reference()
    original_periodic_point_state = jax_strike_module.periodic_point_state
    call_count = 0

    def counting_periodic_point_state(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_periodic_point_state(*args, **kwargs)

    monkeypatch.setattr(
        jax_strike_module,
        "periodic_point_state",
        counting_periodic_point_state,
    )
    trace_manifold_strike_bundle(
        _shifted_hyperbolic_field,
        params,
        branch,
        matches,
        planes,
        n_steps_per_span=64,
        xline_newton_iterations=4,
        wall_n_steps=512,
        wall_newton_iterations=5,
        maximum_phi_shift=0.2,
    )
    assert call_count == 1

    with pytest.raises(ValueError, match="equal length"):
        trace_manifold_strike_bundle(
            _shifted_hyperbolic_field,
            params,
            branch,
            matches,
            planes[:1],
            maximum_phi_shift=0.2,
        )
    with pytest.raises(ValueError, match="unique"):
        trace_manifold_strike_bundle(
            _shifted_hyperbolic_field,
            params,
            branch,
            (matches[0], matches[0]),
            (planes[0], planes[0]),
            maximum_phi_shift=0.2,
        )


def test_labelled_strike_bundle_has_ordered_outer_correspondence_gate():
    params = jnp.array([0.4, 1.0, -0.05])
    _branch, matches, _planes = _manifold_strike_bundle_reference()
    bundle = _trace_labelled_bundle(params)

    accepted = compare_jax_manifold_strike_bundle(
        matches,
        bundle,
        absolute_tolerance_m=1.0e-10,
    )
    assert accepted.accepted
    assert accepted.rejection_reason is None
    assert accepted.max_deviation_m < 1.0e-10
    assert accepted.labels == tuple(match.label for match in matches)
    assert all(report.accepted for report in accepted.strike_reports)

    moved_point = matches[1].point_RZPhi_m_rad.copy()
    moved_point[0] += 1.0e-3
    moved_matches = (
        matches[0],
        replace(matches[1], point_RZPhi_m_rad=moved_point),
    )
    rejected = compare_jax_manifold_strike_bundle(
        moved_matches,
        bundle,
        absolute_tolerance_m=1.0e-6,
    )
    assert not rejected.accepted
    assert rejected.strike_reports[0].accepted
    assert not rejected.strike_reports[1].accepted
    assert rejected.max_deviation_m == pytest.approx(1.0e-3)
    assert matches[1].label.key in rejected.rejection_reason
