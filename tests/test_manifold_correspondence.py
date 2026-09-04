"""Tests for production-to-JAX manifold sample correspondence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyna.topo.manifold_correspondence import (
    ManifoldSampleLabel,
    compare_jax_manifold_branch,
    manifold_branch_reference_from_trace,
    nearest_available_manifold_sample,
    refresh_manifold_sample_match,
)


def _synthetic_unstable_payload(
    *,
    raw_direction=(-1.0, 0.0),
    orientation_reversing=False,
    include_explicit_seed_side=True,
):
    anchor = np.array([1.0, 0.0])
    direction = np.asarray(raw_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    expansion = 2.0
    distances = np.array([0.01, 0.02])
    seed_sides = np.repeat([-1.0, 1.0], len(distances))
    seed_distances = np.tile(distances, 2)
    seed_orders = np.tile(np.arange(len(distances)), 2)
    seed_points = (
        anchor[None, :]
        + seed_sides[:, None] * seed_distances[:, None] * direction[None, :]
    )

    point_r = []
    point_z = []
    point_generation = []
    point_order = []
    point_physical_side = []
    point_seed_side = []
    sign = -1 if orientation_reversing else 1
    for seed_side in (-1, 1):
        for generation in range(3):
            for seed_order, distance in enumerate(distances):
                displacement = (
                    seed_side * distance * (sign * expansion) ** generation
                )
                point = anchor + displacement * direction
                point_r.append(point[0])
                point_z.append(point[1])
                point_generation.append(generation)
                point_order.append(seed_order)
                point_physical_side.append(seed_side * sign**generation)
                point_seed_side.append(seed_side)

    payload = {
        "origin_R": anchor[0],
        "origin_Z": anchor[1],
        "origin_phi": 0.2,
        "manifold_field_period": 0.6,
        "manifold_field_period_source": "synthetic",
        "manifold_origin_label": "orbit7:P0",
        "orbit_id": 7,
        "point_index": 0,
        "map_order_index": 0,
        "unstable_eigenvector": direction,
        "unstable_expansion": expansion,
        "unstable_orientation_reversing": orientation_reversing,
        "u_seed_R": seed_points[:, 0],
        "u_seed_Z": seed_points[:, 1],
        "u_seed_distance": seed_distances,
        "u_seed_side": seed_sides,
        "u_seed_order": seed_orders,
        "u_R": np.asarray(point_r),
        "u_Z": np.asarray(point_z),
        "u_generation": np.asarray(point_generation),
        "u_point_seed_order": np.asarray(point_order),
        "u_point_side": np.asarray(point_physical_side),
    }
    if include_explicit_seed_side:
        payload["u_point_seed_side"] = np.asarray(point_seed_side)
    return payload


def test_reference_locks_direction_and_selects_a_fixed_target_label():
    payload = _synthetic_unstable_payload(raw_direction=(-1.0, 0.0))
    branch = manifold_branch_reference_from_trace(
        payload,
        stability="unstable",
        seed_side=1,
        n_generations=2,
        direction_reference=np.array([1.0, 0.0]),
        minimum_direction_alignment=0.9,
    )

    assert branch.label == "orbit7:P0.unstable.plus"
    assert branch.complete
    np.testing.assert_allclose(branch.direction_RZ, [1.0, 0.0])
    np.testing.assert_array_equal(branch.seed_orders, [0, 1])
    np.testing.assert_allclose(branch.seed_distances_m, [0.01, 0.02])
    np.testing.assert_allclose(
        branch.points_RZ_m[:, :, 0] - 1.0,
        np.array([1.0, 2.0, 4.0])[:, None] * np.array([0.01, 0.02]),
    )
    assert branch.metadata["orientation_sign"] == -1
    assert branch.metadata["raw_seed_side"] == -1

    match = nearest_available_manifold_sample(
        branch,
        np.array([1.079, 0.0]),
    )
    assert match.label == ManifoldSampleLabel(
        origin_label="orbit7:P0",
        stability="unstable",
        seed_side=1,
        generation_index=2,
        seed_order=1,
    )
    assert match.label.key == "orbit7:P0.unstable.plus.g2.s1"
    assert match.jax_seed_index == 1
    np.testing.assert_allclose(branch.point(match.label), [1.08, 0.0])


def test_orientation_reversing_payload_recovers_initial_seed_side():
    payload = _synthetic_unstable_payload(
        raw_direction=(1.0, 0.0),
        orientation_reversing=True,
        include_explicit_seed_side=False,
    )
    branch = manifold_branch_reference_from_trace(
        payload,
        stability="unstable",
        seed_side=1,
        n_generations=2,
    )

    assert branch.complete
    assert branch.orientation_reversing
    assert branch.metadata["point_seed_side_source"] == (
        "reconstructed_from_physical_side"
    )
    np.testing.assert_allclose(
        branch.points_RZ_m[:, :, 0] - 1.0,
        np.array([1.0, -2.0, 4.0])[:, None] * np.array([0.01, 0.02]),
    )


def test_correspondence_report_gates_tolerance_and_missing_samples():
    branch = manifold_branch_reference_from_trace(
        _synthetic_unstable_payload(),
        stability="unstable",
        seed_side=1,
        n_generations=2,
        direction_reference=np.array([1.0, 0.0]),
    )
    exact = compare_jax_manifold_branch(
        branch,
        branch.points_RZ_m,
        absolute_tolerance_m=1.0e-12,
    )
    assert exact.accepted
    assert exact.n_missing == 0

    perturbed = branch.points_RZ_m.copy()
    perturbed[-1, -1, 0] += 1.0e-4
    rejected = compare_jax_manifold_branch(
        branch,
        perturbed,
        absolute_tolerance_m=1.0e-6,
    )
    assert not rejected.accepted
    assert rejected.max_deviation_m == pytest.approx(1.0e-4)

    availability = branch.available.copy()
    availability[-1, -1] = False
    incomplete = replace(branch, available=availability)
    strict = compare_jax_manifold_branch(
        incomplete,
        branch.points_RZ_m,
        absolute_tolerance_m=1.0e-12,
    )
    permissive = compare_jax_manifold_branch(
        incomplete,
        branch.points_RZ_m,
        absolute_tolerance_m=1.0e-12,
        require_complete=False,
    )
    assert not strict.accepted
    assert permissive.accepted
    assert permissive.n_missing == 1


def test_reference_rejects_a_branch_direction_identity_change():
    with pytest.raises(ValueError, match="not aligned"):
        manifold_branch_reference_from_trace(
            _synthetic_unstable_payload(raw_direction=(1.0, 0.0)),
            stability="unstable",
            seed_side=1,
            n_generations=2,
            direction_reference=np.array([0.0, 1.0]),
            minimum_direction_alignment=0.8,
        )


def test_refresh_preserves_exact_label_and_rejects_disappearance_or_jump():
    branch = manifold_branch_reference_from_trace(
        _synthetic_unstable_payload(),
        stability="unstable",
        seed_side=1,
        n_generations=2,
        direction_reference=np.array([1.0, 0.0]),
    )
    previous = nearest_available_manifold_sample(branch, np.array([1.08, 0.0]))
    shift = np.array([1.0e-3, -2.0e-3])
    candidate = replace(
        branch,
        anchor_RZ_m=branch.anchor_RZ_m + shift,
        points_RZ_m=branch.points_RZ_m + shift,
    )
    accepted = refresh_manifold_sample_match(
        previous,
        candidate,
        np.array([1.08, 0.0]),
        maximum_sample_displacement_m=5.0e-3,
    )

    assert accepted.accepted
    assert accepted.candidate_match is not None
    assert accepted.candidate_match.label == previous.label
    assert accepted.candidate_match.jax_seed_index == previous.jax_seed_index
    assert accepted.sample_displacement_m == pytest.approx(np.linalg.norm(shift))

    jumped = replace(
        branch,
        anchor_RZ_m=branch.anchor_RZ_m + 0.1,
        points_RZ_m=branch.points_RZ_m + 0.1,
    )
    rejected_jump = refresh_manifold_sample_match(
        previous,
        jumped,
        np.array([1.08, 0.0]),
        maximum_sample_displacement_m=5.0e-3,
    )
    assert not rejected_jump.accepted
    assert rejected_jump.rejection_reason == "sample_displacement_exceeds_limit"

    availability = branch.available.copy()
    availability[
        previous.label.generation_index,
        previous.jax_seed_index,
    ] = False
    missing = replace(branch, available=availability)
    rejected_missing = refresh_manifold_sample_match(
        previous,
        missing,
        np.array([1.08, 0.0]),
        maximum_sample_displacement_m=5.0e-3,
    )
    assert not rejected_missing.accepted
    assert rejected_missing.candidate_match is None
    assert rejected_missing.rejection_reason == "sample_unavailable"
