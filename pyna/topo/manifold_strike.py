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

import operator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.toroidal.control.strike_heat import StrikeSeedBundle, WallStrikeSamples


def _nonempty_label(value: object, name: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{name} must not be empty")
    return label


def _direction(value: object) -> Literal["+", "-"]:
    direction = str(value).strip()
    if direction not in {"+", "-"}:
        raise ValueError("direction must be '+' or '-'")
    return direction


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


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


@dataclass(frozen=True)
class ManifoldStrikeLabel:
    """Stable identity of one production first-wall hit."""

    branch_label: str
    direction: Literal["+", "-"]
    seed_order: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "branch_label",
            _nonempty_label(self.branch_label, "branch_label"),
        )
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(
            self,
            "seed_order",
            _nonnegative_integer(self.seed_order, "seed_order"),
        )

    @property
    def key(self) -> str:
        direction_name = "phi_plus" if self.direction == "+" else "phi_minus"
        return (
            f"{self.branch_label}.{direction_name}."
            f"s{self.seed_order}.first_wall_hit"
        )


@dataclass(frozen=True)
class ManifoldStrikeMatch:
    """One exact manifold seed and its resolved Cyna first-wall hit."""

    label: ManifoldStrikeLabel
    point_RZPhi_m_rad: np.ndarray
    target_distance_m: float
    connection_length_m: float
    bundle_seed_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, ManifoldStrikeLabel):
            raise TypeError("label must be ManifoldStrikeLabel")
        point = _finite_vector(
            self.point_RZPhi_m_rad,
            3,
            "point_RZPhi_m_rad",
        )
        if point[0] <= 0.0:
            raise ValueError("strike R must be positive")
        target_distance = float(self.target_distance_m)
        connection_length = float(self.connection_length_m)
        if not np.isfinite(target_distance) or target_distance < 0.0:
            raise ValueError("target_distance_m must be non-negative and finite")
        if not np.isfinite(connection_length) or connection_length < 0.0:
            raise ValueError("connection_length_m must be non-negative and finite")
        object.__setattr__(self, "point_RZPhi_m_rad", point.copy())
        object.__setattr__(self, "target_distance_m", target_distance)
        object.__setattr__(self, "connection_length_m", connection_length)
        object.__setattr__(
            self,
            "bundle_seed_index",
            _nonnegative_integer(self.bundle_seed_index, "bundle_seed_index"),
        )

    @property
    def point_xyz_m(self) -> np.ndarray:
        return _cylindrical_to_cartesian(self.point_RZPhi_m_rad)


@dataclass(frozen=True)
class ManifoldStrikeRefreshReport:
    """Result of carrying one exact first-wall-hit label to a candidate."""

    previous_match: ManifoldStrikeMatch
    candidate_match: ManifoldStrikeMatch | None
    hit_displacement_m: float
    maximum_hit_displacement_m: float
    accepted: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.previous_match, ManifoldStrikeMatch):
            raise TypeError("previous_match must be ManifoldStrikeMatch")
        if self.candidate_match is not None and not isinstance(
            self.candidate_match,
            ManifoldStrikeMatch,
        ):
            raise TypeError("candidate_match must be ManifoldStrikeMatch or None")
        displacement = float(self.hit_displacement_m)
        maximum = float(self.maximum_hit_displacement_m)
        if np.isnan(displacement) or displacement < 0.0:
            raise ValueError("hit_displacement_m must be non-negative or infinity")
        if not np.isfinite(maximum) or maximum < 0.0:
            raise ValueError(
                "maximum_hit_displacement_m must be non-negative and finite"
            )
        reason = None if self.rejection_reason is None else str(
            self.rejection_reason
        )
        if bool(self.accepted):
            if self.candidate_match is None or reason is not None:
                raise ValueError("an accepted refresh requires a candidate and no reason")
            if self.candidate_match.label != self.previous_match.label:
                raise ValueError("an accepted refresh must preserve the strike label")
            if displacement > maximum:
                raise ValueError("accepted refresh exceeds its displacement limit")
        elif reason is None:
            raise ValueError("a rejected refresh requires rejection_reason")
        object.__setattr__(self, "hit_displacement_m", displacement)
        object.__setattr__(self, "maximum_hit_displacement_m", maximum)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "rejection_reason", reason)


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
    "ManifoldStrikeLabel",
    "ManifoldStrikeMatch",
    "ManifoldStrikeRefreshReport",
    "manifold_branch_strike_seed_bundle",
    "refresh_manifold_strike_match",
    "select_manifold_strike_match",
]
