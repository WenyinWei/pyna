import numpy as np
import pytest

from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_strike import (
    local_wall_plane_from_strike,
    manifold_branch_strike_seed_bundle,
    refresh_manifold_strike_match,
    select_manifold_strike_match,
)
from pyna.toroidal.control.strike_heat import WallStrikeSamples
from pyna.toroidal.geometry import ToroidalWall


def _branch(*, stability="unstable", seed_orders=(2, 5), shift=0.0):
    orders = np.asarray(seed_orders, dtype=int)
    distances = np.array([1.0e-3, 3.0e-3])
    anchor = np.array([1.5 + shift, -0.1])
    direction = np.array([1.0, 0.0])
    seeds = anchor + distances[:, None] * direction
    points = np.stack((seeds, seeds + np.array([0.1, -0.05])))
    return ManifoldBranchReference(
        origin_label="1:P0",
        stability=stability,
        seed_side=1,
        section_phi=0.0,
        map_span=np.pi / 4.0,
        anchor_RZ_m=anchor,
        direction_RZ=direction,
        expansion=2.0,
        orientation_reversing=False,
        seed_orders=orders,
        seed_distances_m=distances,
        points_RZ_m=points,
        available=np.ones(points.shape[:2], dtype=bool),
    )


def _strikes(bundle, *, seed_index=(0, 1), r_shift=0.0):
    indices = np.asarray(seed_index, dtype=int)
    orders = np.asarray(bundle.metadata["seed_order"], dtype=int)
    base_R = np.where(orders == 2, 2.0, 2.1)
    base_Z = np.where(orders == 2, -0.4, -0.5)
    base_phi = np.where(orders == 2, 0.2, 0.4)
    base_length = np.where(orders == 2, 8.0, 12.0)
    return WallStrikeSamples(
        label=bundle.label,
        mode="chaotic_manifold",
        R=base_R[indices] + r_shift,
        Z=base_Z[indices],
        phi=base_phi[indices],
        weights=bundle.weights[indices],
        connection_length=base_length[indices],
        seed_index=indices,
        direction=bundle.direction,
        unresolved_weight=float(np.sum(bundle.weights) - np.sum(bundle.weights[indices])),
        metadata=bundle.metadata,
    )


def _xyz(R, Z, phi):
    return np.array([R * np.cos(phi), R * np.sin(phi), Z])


def test_branch_reference_builds_ordered_cyna_strike_bundle():
    unstable = manifold_branch_strike_seed_bundle(_branch())
    stable = manifold_branch_strike_seed_bundle(_branch(stability="stable"))

    assert unstable.label == "1:P0.unstable.plus"
    assert unstable.direction == "+"
    assert stable.direction == "-"
    np.testing.assert_array_equal(unstable.metadata["seed_order"], [2, 5])
    np.testing.assert_array_equal(unstable.metadata["jax_seed_index"], [0, 1])
    np.testing.assert_allclose(unstable.source_coordinate, [1.0e-3, 3.0e-3])
    np.testing.assert_allclose(unstable.R, [1.501, 1.503])
    assert unstable.metadata["quantitative"] is False


def test_strike_selection_uses_physical_xyz_and_sparse_seed_order():
    bundle = manifold_branch_strike_seed_bundle(_branch())
    strikes = _strikes(bundle)
    target = _xyz(2.1, -0.5, 0.4) + np.array([1.0e-3, 0.0, 0.0])

    match = select_manifold_strike_match(bundle, strikes, target)

    assert match.label.seed_order == 5
    assert match.label.branch_label == bundle.label
    assert match.label.key.endswith("phi_plus.s5.first_wall_hit")
    assert match.bundle_seed_index == 1
    assert match.connection_length_m == pytest.approx(12.0)
    assert match.target_distance_m == pytest.approx(1.0e-3)


def test_axisymmetric_strike_metric_uses_only_poloidal_coordinates():
    bundle = manifold_branch_strike_seed_bundle(_branch())
    strikes = _strikes(bundle)

    match = select_manifold_strike_match(
        bundle,
        strikes,
        np.array([2.1, -0.5]),
        distance_mode="rz",
    )
    candidate = WallStrikeSamples(
        label=bundle.label,
        mode="chaotic_manifold",
        R=np.array([2.1]),
        Z=np.array([-0.5]),
        phi=np.array([2.4]),
        weights=np.array([1.0]),
        connection_length=np.array([12.0]),
        seed_index=np.array([1]),
        direction=bundle.direction,
        unresolved_weight=1.0,
        metadata=bundle.metadata,
    )
    report = refresh_manifold_strike_match(
        match,
        bundle,
        candidate,
        np.array([2.1, -0.5]),
        maximum_hit_displacement_m=0.0,
    )

    assert match.distance_mode == "rz"
    np.testing.assert_allclose(match.distance_coordinates_m, [2.1, -0.5])
    assert report.accepted
    assert report.hit_displacement_m == pytest.approx(0.0)


def test_strike_refresh_tracks_seed_order_across_candidate_reordering():
    branch = _branch()
    bundle = manifold_branch_strike_seed_bundle(branch)
    previous = select_manifold_strike_match(
        bundle,
        _strikes(bundle),
        _xyz(2.1, -0.5, 0.4),
    )
    candidate_branch = _branch(seed_orders=(5, 2), shift=2.0e-4)
    candidate_bundle = manifold_branch_strike_seed_bundle(candidate_branch)
    candidate_strikes = _strikes(candidate_bundle, seed_index=(0,), r_shift=1.0e-4)

    report = refresh_manifold_strike_match(
        previous,
        candidate_bundle,
        candidate_strikes,
        _xyz(2.1, -0.5, 0.4),
        maximum_hit_displacement_m=2.0e-4,
    )

    assert report.accepted
    assert report.candidate_match is not None
    assert report.candidate_match.label == previous.label
    assert report.candidate_match.bundle_seed_index == 0
    assert report.hit_displacement_m == pytest.approx(1.0e-4)


def test_strike_refresh_rejects_unresolved_exact_seed_and_large_motion():
    bundle = manifold_branch_strike_seed_bundle(_branch())
    previous = select_manifold_strike_match(
        bundle,
        _strikes(bundle),
        _xyz(2.1, -0.5, 0.4),
    )
    unresolved = _strikes(bundle, seed_index=(0,))

    missing = refresh_manifold_strike_match(
        previous,
        bundle,
        unresolved,
        _xyz(2.1, -0.5, 0.4),
        maximum_hit_displacement_m=1.0,
    )
    moved = refresh_manifold_strike_match(
        previous,
        bundle,
        _strikes(bundle, r_shift=0.05),
        _xyz(2.1, -0.5, 0.4),
        maximum_hit_displacement_m=0.01,
    )

    assert not missing.accepted
    assert missing.candidate_match is None
    assert missing.rejection_reason == "strike_unresolved"
    assert not moved.accepted
    assert moved.candidate_match is not None
    assert moved.rejection_reason == "strike_displacement_exceeds_limit"


def test_local_wall_plane_rotates_periodic_projection_to_unwrapped_hit():
    nfp = 4
    period = 2.0 * np.pi / nfp
    phi = np.linspace(0.0, period, 6, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    wall_R = np.broadcast_to(2.0 + 0.4 * np.cos(theta), (phi.size, theta.size))
    wall_Z = np.broadcast_to(0.4 * np.sin(theta), (phi.size, theta.size))
    wall = ToroidalWall(phi, wall_R, wall_Z, nfp=nfp)
    bundle = manifold_branch_strike_seed_bundle(_branch())
    hit_phi = period + 0.2
    hit = WallStrikeSamples(
        label=bundle.label,
        mode="chaotic_manifold",
        R=np.array([2.4]),
        Z=np.array([0.0]),
        phi=np.array([hit_phi]),
        weights=np.array([1.0]),
        connection_length=np.array([10.0]),
        seed_index=np.array([0]),
        direction=bundle.direction,
        unresolved_weight=1.0,
        metadata=bundle.metadata,
    )
    match = select_manifold_strike_match(
        bundle,
        hit,
        _xyz(2.4, 0.0, hit_phi),
    )

    plane = local_wall_plane_from_strike(
        match,
        wall,
        maximum_projection_distance_m=1.0e-10,
    )

    np.testing.assert_allclose(
        plane.point_xyz_m,
        _xyz(2.4, 0.0, hit_phi),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(np.linalg.norm(plane.normal_xyz), 1.0)
    assert plane.wall_phi_rad == pytest.approx(hit_phi)
    assert plane.wall_s == pytest.approx(0.0, abs=1.0e-14)
    assert plane.level_set_params.shape == (6,)
