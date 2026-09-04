"""Fixed-label production wall strikes for manifold-aware optimization.

The Cyna wall tracer resolves which local manifold seeds reach the first wall.
This module turns that production result into an optimization-safe identity:
branch, tracing direction, and sparse seed order.  Candidate refreshes carry
that exact identity and never replace a disappeared strike with a nearby seed.

There is intentionally no JAX dependency here.  A differentiable inner-loop
wall intersection can consume the frozen label, while Cyna remains the
authoritative first-wall-hit and accepted-step backend.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_strike_contracts import (
    LocalWallPlane,
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
    ManifoldStrikeRefreshReport,
)
from pyna.toroidal.control.strike_heat import StrikeSeedBundle, WallStrikeSamples
from pyna.toroidal.geometry import project_points_to_toroidal_surface


def _finite_vector(value: object, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).ravel()
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-{size} vector")
    return vector


def _cylindrical_to_cartesian(point_RZPhi: object) -> np.ndarray:
    R, Z, phi = _finite_vector(point_RZPhi, 3, "point_RZPhi_m_rad")
    if R <= 0.0:
        raise ValueError("strike R must be positive")
    return np.asarray([R * np.cos(phi), R * np.sin(phi), Z], dtype=float)


def _seed_orders(bundle: StrikeSeedBundle) -> np.ndarray:
    raw = np.asarray(bundle.metadata.get("seed_order", []))
    if raw.ndim != 1:
        raw = raw.ravel()
    if raw.size != bundle.R.size:
        raise ValueError("manifold strike bundle requires one seed_order per seed")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed_order metadata must contain integers") from exc
    if (
        not np.all(np.isfinite(numeric))
        or not np.all(numeric == np.rint(numeric))
    ):
        raise ValueError("seed_order metadata must contain integers")
    orders = np.asarray(np.rint(numeric), dtype=int)
    if np.any(orders < 0) or np.unique(orders).size != orders.size:
        raise ValueError("seed_order metadata must be unique non-negative integers")
    return orders


def _validate_pair(
    bundle: StrikeSeedBundle,
    strikes: WallStrikeSamples,
) -> np.ndarray:
    if not isinstance(bundle, StrikeSeedBundle):
        raise TypeError("bundle must be StrikeSeedBundle")
    if not isinstance(strikes, WallStrikeSamples):
        raise TypeError("strikes must be WallStrikeSamples")
    if bundle.mode != "chaotic_manifold" or strikes.mode != "chaotic_manifold":
        raise ValueError("fixed-label strike matching requires a manifold bundle")
    if strikes.label != bundle.label or strikes.direction != bundle.direction:
        raise ValueError("strike samples do not belong to the supplied bundle")
    if np.any(strikes.seed_index >= bundle.R.size):
        raise ValueError("strike seed_index lies outside the supplied bundle")
    return _seed_orders(bundle)


def manifold_branch_strike_seed_bundle(
    branch: ManifoldBranchReference,
    *,
    weights: object | None = None,
    weight_kind: Literal["relative", "power"] = "relative",
    quantitative: bool = False,
) -> StrikeSeedBundle:
    """Convert one frozen branch reference into an ordered Cyna seed bundle."""

    if not isinstance(branch, ManifoldBranchReference):
        raise TypeError("branch must be ManifoldBranchReference")
    if weights is None:
        resolved_weights = np.ones(branch.n_seeds, dtype=float)
        weight_provenance = "uniform_relative_seed_proxy"
    else:
        resolved_weights = np.asarray(weights, dtype=float).ravel()
        if resolved_weights.size != branch.n_seeds:
            raise ValueError("weights must have one value per branch seed")
        weight_provenance = "caller_supplied"
    seed_points = (
        branch.anchor_RZ_m[None, :]
        + branch.seed_side
        * branch.seed_distances_m[:, None]
        * branch.direction_RZ[None, :]
    )
    physical_sign = np.sign(branch.map_span) * (
        1.0 if branch.stability == "unstable" else -1.0
    )
    direction: Literal["+", "-"] = "+" if physical_sign > 0.0 else "-"
    return StrikeSeedBundle(
        label=branch.label,
        mode="chaotic_manifold",
        R=seed_points[:, 0],
        Z=seed_points[:, 1],
        phi=np.full(branch.n_seeds, branch.section_phi, dtype=float),
        direction=direction,
        weights=resolved_weights,
        weight_kind=weight_kind,
        source_coordinate=branch.seed_distances_m,
        metadata={
            "quantitative": bool(quantitative),
            "topology_provenance": "ManifoldBranchReference",
            "weight_provenance": weight_provenance,
            "origin_label": branch.origin_label,
            "stability": branch.stability,
            "side": branch.seed_side,
            "signed_map_span": branch.map_span,
            "seed_order": branch.seed_orders.copy(),
            "jax_seed_index": np.arange(branch.n_seeds, dtype=int),
        },
    )


def local_wall_plane_from_strike(
    strike_match: ManifoldStrikeMatch,
    wall: object,
    *,
    maximum_projection_distance_m: float,
) -> LocalWallPlane:
    """Freeze the continuously projected wall tangent at one Cyna hit.

    Toroidal wall geometry may cover only one field period.  Projection is
    performed in that periodic sector, then its point and normal are rotated
    back to the unwrapped physical hit angle returned by Cyna.
    """

    if not isinstance(strike_match, ManifoldStrikeMatch):
        raise TypeError("strike_match must be ManifoldStrikeMatch")
    maximum = float(maximum_projection_distance_m)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError(
            "maximum_projection_distance_m must be non-negative and finite"
        )
    hit_R, hit_Z, hit_phi = strike_match.point_RZPhi_m_rad
    projection = project_points_to_toroidal_surface(
        [hit_R],
        [hit_Z],
        [hit_phi],
        wall,
    )
    distance = float(projection.distance[0])
    if distance > maximum:
        raise ValueError(
            "production strike is farther from the wall projection than "
            "maximum_projection_distance_m"
        )

    wrapped_phi = float(projection.phi[0])
    rotation_angle = float(hit_phi - wrapped_phi)
    cos_rotation = np.cos(rotation_angle)
    sin_rotation = np.sin(rotation_angle)
    rotation = np.asarray(
        [
            [cos_rotation, -sin_rotation, 0.0],
            [sin_rotation, cos_rotation, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    projected_R = float(projection.R[0])
    projected_Z = float(projection.Z[0])
    wrapped_point = np.asarray(
        [
            projected_R * np.cos(wrapped_phi),
            projected_R * np.sin(wrapped_phi),
            projected_Z,
        ]
    )
    return LocalWallPlane(
        strike_label=strike_match.label,
        point_xyz_m=rotation @ wrapped_point,
        normal_xyz=rotation @ projection.normal[0],
        wall_phi_rad=float(hit_phi),
        wall_s=float(projection.s[0]),
        projection_distance_m=distance,
    )


def select_manifold_strike_match(
    bundle: StrikeSeedBundle,
    strikes: WallStrikeSamples,
    target_xyz_m: object,
) -> ManifoldStrikeMatch:
    """Select the resolved first-wall hit nearest a physical target in 3-D."""

    seed_orders = _validate_pair(bundle, strikes)
    target = _finite_vector(target_xyz_m, 3, "target_xyz_m")
    if strikes.R.size == 0:
        raise ValueError("manifold strike bundle has no resolved first-wall hits")
    points_RZPhi = np.column_stack((strikes.R, strikes.Z, strikes.phi))
    points_xyz = np.asarray(
        [_cylindrical_to_cartesian(point) for point in points_RZPhi],
        dtype=float,
    )
    distances = np.linalg.norm(points_xyz - target[None, :], axis=1)
    strike_index = int(np.argmin(distances))
    bundle_seed_index = int(strikes.seed_index[strike_index])
    return ManifoldStrikeMatch(
        label=ManifoldStrikeLabel(
            branch_label=bundle.label,
            direction=bundle.direction,
            seed_order=int(seed_orders[bundle_seed_index]),
        ),
        point_RZPhi_m_rad=points_RZPhi[strike_index],
        target_distance_m=float(distances[strike_index]),
        connection_length_m=float(strikes.connection_length[strike_index]),
        bundle_seed_index=bundle_seed_index,
    )


def refresh_manifold_strike_match(
    previous_match: ManifoldStrikeMatch,
    candidate_bundle: StrikeSeedBundle,
    candidate_strikes: WallStrikeSamples,
    target_xyz_m: object,
    *,
    maximum_hit_displacement_m: float,
) -> ManifoldStrikeRefreshReport:
    """Carry an exact strike label; reject disappearance instead of relabelling."""

    if not isinstance(previous_match, ManifoldStrikeMatch):
        raise TypeError("previous_match must be ManifoldStrikeMatch")
    seed_orders = _validate_pair(candidate_bundle, candidate_strikes)
    target = _finite_vector(target_xyz_m, 3, "target_xyz_m")
    maximum = float(maximum_hit_displacement_m)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError(
            "maximum_hit_displacement_m must be non-negative and finite"
        )
    label = previous_match.label
    if (
        candidate_bundle.label != label.branch_label
        or candidate_bundle.direction != label.direction
    ):
        return ManifoldStrikeRefreshReport(
            previous_match=previous_match,
            candidate_match=None,
            hit_displacement_m=np.inf,
            maximum_hit_displacement_m=maximum,
            accepted=False,
            rejection_reason="branch_identity_changed",
        )
    matching_bundle_indices = np.flatnonzero(seed_orders == label.seed_order)
    if matching_bundle_indices.size != 1:
        return ManifoldStrikeRefreshReport(
            previous_match=previous_match,
            candidate_match=None,
            hit_displacement_m=np.inf,
            maximum_hit_displacement_m=maximum,
            accepted=False,
            rejection_reason="seed_order_missing",
        )
    bundle_seed_index = int(matching_bundle_indices[0])
    matching_strikes = np.flatnonzero(
        candidate_strikes.seed_index == bundle_seed_index
    )
    if matching_strikes.size != 1:
        return ManifoldStrikeRefreshReport(
            previous_match=previous_match,
            candidate_match=None,
            hit_displacement_m=np.inf,
            maximum_hit_displacement_m=maximum,
            accepted=False,
            rejection_reason="strike_unresolved",
        )

    strike_index = int(matching_strikes[0])
    point = np.asarray(
        [
            candidate_strikes.R[strike_index],
            candidate_strikes.Z[strike_index],
            candidate_strikes.phi[strike_index],
        ],
        dtype=float,
    )
    point_xyz = _cylindrical_to_cartesian(point)
    target_distance = float(np.linalg.norm(point_xyz - target))
    displacement = float(np.linalg.norm(point_xyz - previous_match.point_xyz_m))
    candidate_match = ManifoldStrikeMatch(
        label=label,
        point_RZPhi_m_rad=point,
        target_distance_m=target_distance,
        connection_length_m=float(
            candidate_strikes.connection_length[strike_index]
        ),
        bundle_seed_index=bundle_seed_index,
    )
    accepted = displacement <= maximum
    return ManifoldStrikeRefreshReport(
        previous_match=previous_match,
        candidate_match=candidate_match,
        hit_displacement_m=displacement,
        maximum_hit_displacement_m=maximum,
        accepted=accepted,
        rejection_reason=(
            None if accepted else "strike_displacement_exceeds_limit"
        ),
    )


__all__ = [
    "LocalWallPlane",
    "ManifoldStrikeLabel",
    "ManifoldStrikeMatch",
    "ManifoldStrikeRefreshReport",
    "local_wall_plane_from_strike",
    "manifold_branch_strike_seed_bundle",
    "refresh_manifold_strike_match",
    "select_manifold_strike_match",
]
