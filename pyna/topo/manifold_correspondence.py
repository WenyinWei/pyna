"""Outer-loop correspondence between production and differentiable manifolds.

The production Cyna tracer emits plotting-oriented flat arrays.  Stage-2 coil
optimization needs a stricter representation: a stable X-line identity, an
orientation-locked eigendirection, an initial seed side, and the exact
``(generation, seed-order)`` label of every sample used by a JAX loss.

This module converts one public ``trace_fixed_point_manifolds_field`` payload
into that dense, validation-oriented record.  It has no JAX dependency.  PyNA
therefore owns branch discovery, relabelling, target selection, and parity
decisions, while downstream packages only consume the frozen result.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


def _nonempty_label(value: object, name: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{name} must not be empty")
    return label


def _stability(value: object) -> str:
    stability = str(value).lower()
    if stability not in {"stable", "unstable"}:
        raise ValueError("stability must be 'stable' or 'unstable'")
    return stability


def _side(value: object, name: str = "seed_side") -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be -1 or +1")
    try:
        side = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if side not in (-1, 1):
        raise ValueError(f"{name} must be -1 or +1")
    return side


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


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_vector(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).ravel()
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite physical (R, Z) vector")
    return vector


def _unit_vector(value: object, name: str) -> np.ndarray:
    vector = _finite_vector(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _integer_array(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raw = raw.ravel()
    if raw.size == 0:
        return np.empty(0, dtype=int)
    try:
        finite = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not np.all(np.isfinite(finite)) or not np.all(finite == np.rint(finite)):
        raise ValueError(f"{name} must contain integers")
    return np.asarray(np.rint(finite), dtype=int)


@dataclass(frozen=True)
class ManifoldSampleLabel:
    """Stable identity of one manifold point across an outer refresh."""

    origin_label: str
    stability: str
    seed_side: int
    generation_index: int
    seed_order: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "origin_label",
            _nonempty_label(self.origin_label, "origin_label"),
        )
        object.__setattr__(self, "stability", _stability(self.stability))
        object.__setattr__(self, "seed_side", _side(self.seed_side))
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_integer(self.generation_index, "generation_index"),
        )
        object.__setattr__(
            self,
            "seed_order",
            _nonnegative_integer(self.seed_order, "seed_order"),
        )

    @property
    def key(self) -> str:
        side_name = "plus" if self.seed_side > 0 else "minus"
        return (
            f"{self.origin_label}.{self.stability}.{side_name}."
            f"g{self.generation_index}.s{self.seed_order}"
        )


@dataclass(frozen=True)
class ManifoldBranchReference:
    """Dense production trace for one orientation-locked initial seed side."""

    origin_label: str
    stability: str
    seed_side: int
    section_phi: float
    map_span: float
    anchor_RZ_m: np.ndarray
    direction_RZ: np.ndarray
    expansion: float
    orientation_reversing: bool
    seed_orders: np.ndarray
    seed_distances_m: np.ndarray
    points_RZ_m: np.ndarray
    available: np.ndarray
    metadata: Mapping[str, object] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        origin_label = _nonempty_label(self.origin_label, "origin_label")
        stability = _stability(self.stability)
        seed_side = _side(self.seed_side)
        section_phi = float(self.section_phi)
        map_span = float(self.map_span)
        if not np.isfinite(section_phi):
            raise ValueError("section_phi must be finite")
        if not np.isfinite(map_span) or map_span == 0.0:
            raise ValueError("map_span must be finite and non-zero")
        anchor = _finite_vector(self.anchor_RZ_m, "anchor_RZ_m")
        direction = _unit_vector(self.direction_RZ, "direction_RZ")
        expansion = float(self.expansion)
        if not np.isfinite(expansion) or expansion <= 1.0:
            raise ValueError("expansion must be finite and greater than 1")

        orders = _integer_array(self.seed_orders, "seed_orders")
        distances = np.asarray(self.seed_distances_m, dtype=float).ravel()
        if orders.size == 0 or orders.size != distances.size:
            raise ValueError("seed orders and distances must be non-empty and aligned")
        if np.any(orders < 0) or np.unique(orders).size != orders.size:
            raise ValueError("seed_orders must be unique non-negative integers")
        if not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
            raise ValueError("seed_distances_m must be positive and finite")

        points = np.asarray(self.points_RZ_m, dtype=float)
        availability = np.asarray(self.available, dtype=bool)
        if points.ndim != 3 or points.shape[0] < 1 or points.shape[2] != 2:
            raise ValueError("points_RZ_m must have shape (generation, seed, 2)")
        expected_shape = (points.shape[0], orders.size)
        if points.shape[1] != orders.size or availability.shape != expected_shape:
            raise ValueError("available and seed arrays must align with points_RZ_m")
        if not np.all(np.isfinite(points[availability])):
            raise ValueError("available manifold points must be finite physical metres")
        points = points.copy()
        points[~availability] = np.nan

        object.__setattr__(self, "origin_label", origin_label)
        object.__setattr__(self, "stability", stability)
        object.__setattr__(self, "seed_side", seed_side)
        object.__setattr__(self, "section_phi", section_phi)
        object.__setattr__(self, "map_span", map_span)
        object.__setattr__(self, "anchor_RZ_m", anchor.copy())
        object.__setattr__(self, "direction_RZ", direction.copy())
        object.__setattr__(self, "expansion", expansion)
        object.__setattr__(self, "orientation_reversing", bool(self.orientation_reversing))
        object.__setattr__(self, "seed_orders", orders.copy())
        object.__setattr__(self, "seed_distances_m", distances.copy())
        object.__setattr__(self, "points_RZ_m", points)
        object.__setattr__(self, "available", availability.copy())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def label(self) -> str:
        side_name = "plus" if self.seed_side > 0 else "minus"
        return f"{self.origin_label}.{self.stability}.{side_name}"

    @property
    def n_generations(self) -> int:
        return int(self.points_RZ_m.shape[0] - 1)

    @property
    def n_seeds(self) -> int:
        return int(self.seed_orders.size)

    @property
    def complete(self) -> bool:
        return bool(np.all(self.available))

    def seed_index(self, seed_order: int) -> int:
        order = _nonnegative_integer(seed_order, "seed_order")
        matches = np.flatnonzero(self.seed_orders == order)
        if matches.size != 1:
            raise KeyError(f"seed order {order} is not present in {self.label}")
        return int(matches[0])

    def sample_label(
        self,
        generation_index: int,
        seed_index: int,
    ) -> ManifoldSampleLabel:
        generation = _nonnegative_integer(generation_index, "generation_index")
        index = _nonnegative_integer(seed_index, "seed_index")
        if generation > self.n_generations:
            raise IndexError("generation_index is outside the traced branch")
        if index >= self.n_seeds:
            raise IndexError("seed_index is outside the traced branch")
        return ManifoldSampleLabel(
            origin_label=self.origin_label,
            stability=self.stability,
            seed_side=self.seed_side,
            generation_index=generation,
            seed_order=int(self.seed_orders[index]),
        )

    def point(self, label: ManifoldSampleLabel) -> np.ndarray:
        if not isinstance(label, ManifoldSampleLabel):
            raise TypeError("label must be ManifoldSampleLabel")
        if (
            label.origin_label != self.origin_label
            or label.stability != self.stability
            or label.seed_side != self.seed_side
        ):
            raise KeyError(f"sample {label.key!r} does not belong to {self.label!r}")
        if label.generation_index > self.n_generations:
            raise KeyError(f"sample {label.key!r} is outside the traced generations")
        seed_index = self.seed_index(label.seed_order)
        if not self.available[label.generation_index, seed_index]:
            raise KeyError(f"sample {label.key!r} is unavailable in the production trace")
        return self.points_RZ_m[label.generation_index, seed_index].copy()


@dataclass(frozen=True)
class ManifoldSampleMatch:
    """One outer-selected production sample and its JAX seed-axis index."""

    label: ManifoldSampleLabel
    point_RZ_m: np.ndarray
    distance_m: float
    jax_seed_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, ManifoldSampleLabel):
            raise TypeError("label must be ManifoldSampleLabel")
        point = _finite_vector(self.point_RZ_m, "point_RZ_m")
        distance = float(self.distance_m)
        index = _nonnegative_integer(self.jax_seed_index, "jax_seed_index")
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("distance_m must be finite and non-negative")
        object.__setattr__(self, "point_RZ_m", point.copy())
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "jax_seed_index", index)


@dataclass(frozen=True)
class ManifoldCorrespondenceReport:
    """Outer acceptance gate for a JAX branch against production samples."""

    branch_label: str
    n_expected: int
    n_compared: int
    n_missing: int
    complete: bool
    finite: bool
    max_deviation_m: float
    rms_deviation_m: float
    absolute_tolerance_m: float
    accepted: bool


@dataclass(frozen=True)
class ManifoldSampleRefreshReport:
    """Result of carrying one fixed sample label to a production candidate."""

    previous_match: ManifoldSampleMatch
    candidate_match: ManifoldSampleMatch | None
    sample_displacement_m: float
    maximum_sample_displacement_m: float
    accepted: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.previous_match, ManifoldSampleMatch):
            raise TypeError("previous_match must be ManifoldSampleMatch")
        if self.candidate_match is not None and not isinstance(
            self.candidate_match,
            ManifoldSampleMatch,
        ):
            raise TypeError("candidate_match must be ManifoldSampleMatch or None")
        displacement = float(self.sample_displacement_m)
        maximum = float(self.maximum_sample_displacement_m)
        if np.isnan(displacement) or displacement < 0.0:
            raise ValueError("sample_displacement_m must be non-negative or infinity")
        if not np.isfinite(maximum) or maximum < 0.0:
            raise ValueError(
                "maximum_sample_displacement_m must be non-negative and finite"
            )
        reason = None if self.rejection_reason is None else str(self.rejection_reason)
        if bool(self.accepted):
            if self.candidate_match is None or reason is not None:
                raise ValueError("an accepted refresh requires a candidate and no reason")
            if displacement > maximum:
                raise ValueError("accepted refresh exceeds its displacement limit")
        elif reason is None:
            raise ValueError("a rejected refresh requires rejection_reason")
        object.__setattr__(self, "sample_displacement_m", displacement)
        object.__setattr__(self, "maximum_sample_displacement_m", maximum)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "rejection_reason", reason)


def _payload_origin_label(payload: Mapping[str, object]) -> str:
    explicit = payload.get("manifold_origin_label")
    if explicit is not None:
        return _nonempty_label(explicit, "manifold_origin_label")
    return (
        f"orbit{payload.get('orbit_id', 'unknown')}."
        f"point{payload.get('point_index', 'unknown')}"
    )


def _payload_branch_data(
    payload: Mapping[str, object],
    stability: str,
) -> tuple[str, np.ndarray, np.ndarray, float, bool]:
    if stability == "unstable":
        return (
            "u",
            _finite_vector(
                [payload.get("origin_R"), payload.get("origin_Z")],
                "unstable anchor",
            ),
            _unit_vector(payload.get("unstable_eigenvector"), "unstable_eigenvector"),
            float(payload.get("unstable_expansion", np.nan)),
            bool(payload.get("unstable_orientation_reversing", False)),
        )
    return (
        "s",
        _finite_vector(
            [
                payload.get("stable_origin_R", payload.get("origin_R")),
                payload.get("stable_origin_Z", payload.get("origin_Z")),
            ],
            "stable anchor",
        ),
        _unit_vector(payload.get("stable_eigenvector"), "stable_eigenvector"),
        float(payload.get("stable_backward_expansion", np.nan)),
        bool(payload.get("stable_orientation_reversing", False)),
    )


def manifold_branch_reference_from_trace(
    payload: Mapping[str, object],
    *,
    stability: str,
    seed_side: int,
    n_generations: int,
    direction_reference: object | None = None,
    minimum_direction_alignment: float = 0.0,
    seed_geometry_tolerance_m: float = 1.0e-10,
) -> ManifoldBranchReference:
    """Convert one production trace payload into fixed JAX correspondence.

    ``seed_side`` is defined relative to the returned, orientation-locked
    direction.  If ``direction_reference`` is supplied, the production
    eigenvector is flipped when necessary to retain that convention.  Point
    labels prefer the explicit ``*_point_seed_side`` payload and reconstruct it
    from physical side and eigenvalue parity for older payloads.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a public manifold mapping")
    branch = _stability(stability)
    requested_side = _side(seed_side)
    generations = _positive_integer(n_generations, "n_generations")
    alignment_floor = float(minimum_direction_alignment)
    if not np.isfinite(alignment_floor) or not 0.0 <= alignment_floor <= 1.0:
        raise ValueError("minimum_direction_alignment must lie in [0, 1]")
    geometry_tolerance = float(seed_geometry_tolerance_m)
    if not np.isfinite(geometry_tolerance) or geometry_tolerance < 0.0:
        raise ValueError("seed_geometry_tolerance_m must be non-negative and finite")

    code, anchor, raw_direction, expansion, orientation_reversing = (
        _payload_branch_data(payload, branch)
    )
    if not np.isfinite(expansion) or expansion <= 1.0:
        raise ValueError("production branch expansion must be finite and greater than 1")

    orientation_sign = 1
    alignment = 1.0
    if direction_reference is not None:
        reference = _unit_vector(direction_reference, "direction_reference")
        signed_alignment = float(np.dot(raw_direction, reference))
        alignment = abs(signed_alignment)
        if alignment < alignment_floor:
            raise ValueError(
                "production eigendirection is not aligned with direction_reference"
            )
        if signed_alignment < 0.0:
            orientation_sign = -1
    direction = orientation_sign * raw_direction
    raw_requested_side = requested_side * orientation_sign

    seed_distances_all = np.asarray(
        payload.get(f"{code}_seed_distance", []),
        dtype=float,
    ).ravel()
    seed_sides_all = np.asarray(
        payload.get(f"{code}_seed_side", []),
        dtype=float,
    ).ravel()
    seed_orders_all = _integer_array(
        payload.get(f"{code}_seed_order", []),
        f"{code}_seed_order",
    )
    seed_r_all = np.asarray(payload.get(f"{code}_seed_R", []), dtype=float).ravel()
    seed_z_all = np.asarray(payload.get(f"{code}_seed_Z", []), dtype=float).ravel()
    seed_size = seed_distances_all.size
    if not all(
        values.size == seed_size
        for values in (
            seed_sides_all,
            seed_orders_all,
            seed_r_all,
            seed_z_all,
        )
    ):
        raise ValueError("production seed arrays must have matching lengths")
    if seed_size == 0 or not np.all(np.isfinite(seed_sides_all)):
        raise ValueError("production branch has no finite seed labels")
    signed_seed_sides = np.sign(seed_sides_all).astype(int)
    if np.any(np.abs(signed_seed_sides) != 1):
        raise ValueError("production seed sides must be -1 or +1")

    seed_mask = signed_seed_sides == raw_requested_side
    if not np.any(seed_mask):
        raise ValueError("requested seed side is absent from the production trace")
    seed_orders = seed_orders_all[seed_mask]
    seed_distances = seed_distances_all[seed_mask]
    seed_points = np.column_stack((seed_r_all[seed_mask], seed_z_all[seed_mask]))
    order_index = np.argsort(seed_orders, kind="stable")
    seed_orders = seed_orders[order_index]
    seed_distances = seed_distances[order_index]
    seed_points = seed_points[order_index]
    if np.unique(seed_orders).size != seed_orders.size:
        raise ValueError("production seed orders are not unique on the requested side")
    if not np.all(np.isfinite(seed_distances)) or np.any(seed_distances <= 0.0):
        raise ValueError("production seed distances must be positive and finite")
    expected_seed_points = (
        anchor[None, :]
        + requested_side * seed_distances[:, None] * direction[None, :]
    )
    seed_errors = np.linalg.norm(seed_points - expected_seed_points, axis=1)
    if not np.all(np.isfinite(seed_errors)) or np.max(seed_errors) > geometry_tolerance:
        raise ValueError("production seeds disagree with anchor/direction/side labels")

    point_r = np.asarray(payload.get(f"{code}_R", []), dtype=float).ravel()
    point_z = np.asarray(payload.get(f"{code}_Z", []), dtype=float).ravel()
    point_generation = _integer_array(
        payload.get(f"{code}_generation", []),
        f"{code}_generation",
    )
    point_seed_order = _integer_array(
        payload.get(f"{code}_point_seed_order", []),
        f"{code}_point_seed_order",
    )
    point_physical_side = np.asarray(
        payload.get(f"{code}_point_side", []),
        dtype=float,
    ).ravel()
    point_size = point_r.size
    if not all(
        values.size == point_size
        for values in (
            point_z,
            point_generation,
            point_seed_order,
            point_physical_side,
        )
    ):
        raise ValueError("production point and label arrays must have matching lengths")
    if np.any(point_generation < 0) or np.any(point_seed_order < 0):
        raise ValueError("production generation and seed-order labels must be non-negative")
    physical_sides = np.sign(point_physical_side).astype(int)
    if np.any(np.abs(physical_sides) != 1):
        raise ValueError("production physical sides must be -1 or +1")

    explicit_point_seed_side = payload.get(f"{code}_point_seed_side")
    parity = np.where(
        orientation_reversing & (point_generation % 2 == 1),
        -1,
        1,
    )
    reconstructed_seed_sides = physical_sides * parity
    if explicit_point_seed_side is None:
        raw_point_seed_sides = reconstructed_seed_sides
        point_seed_side_source = "reconstructed_from_physical_side"
    else:
        explicit_sides = np.asarray(explicit_point_seed_side, dtype=float).ravel()
        if explicit_sides.size != point_size or not np.all(np.isfinite(explicit_sides)):
            raise ValueError(f"{code}_point_seed_side must match production points")
        raw_point_seed_sides = np.sign(explicit_sides).astype(int)
        if np.any(np.abs(raw_point_seed_sides) != 1):
            raise ValueError("production point seed sides must be -1 or +1")
        if not np.array_equal(raw_point_seed_sides, reconstructed_seed_sides):
            raise ValueError("point seed-side labels disagree with physical-side parity")
        point_seed_side_source = "explicit"

    points = np.full((generations + 1, seed_orders.size, 2), np.nan, dtype=float)
    available = np.zeros((generations + 1, seed_orders.size), dtype=bool)
    seed_index_by_order = {
        int(seed_order): index for index, seed_order in enumerate(seed_orders)
    }
    selected_points = raw_point_seed_sides == raw_requested_side
    for point_index in np.flatnonzero(selected_points):
        generation = int(point_generation[point_index])
        if generation > generations:
            continue
        order = int(point_seed_order[point_index])
        if order not in seed_index_by_order:
            raise ValueError("point seed order is absent from the production seed segment")
        seed_index = seed_index_by_order[order]
        if available[generation, seed_index]:
            raise ValueError("production trace contains a duplicate sample label")
        point = np.asarray([point_r[point_index], point_z[point_index]], dtype=float)
        if not np.all(np.isfinite(point)):
            raise ValueError("production trace contains a non-finite labelled point")
        points[generation, seed_index] = point
        available[generation, seed_index] = True

    origin_label = _payload_origin_label(payload)
    section_phi = float(payload.get("origin_phi", np.nan))
    map_span = float(payload.get("manifold_field_period", np.nan))
    metadata = {
        "chain_id": payload.get("chain_id"),
        "orbit_id": payload.get("orbit_id"),
        "point_index": payload.get("point_index"),
        "orbit_point_index": payload.get("orbit_point_index"),
        "map_order_index": payload.get("map_order_index"),
        "same_orbit_key": payload.get("same_orbit_key"),
        "map_span_source": payload.get("manifold_field_period_source"),
        "seed_spacing": payload.get("seed_spacing"),
        "orientation_sign": orientation_sign,
        "direction_alignment": alignment,
        "raw_seed_side": raw_requested_side,
        "point_seed_side_source": point_seed_side_source,
        "stable_inverse_anchor_refined": payload.get(
            "stable_inverse_anchor_refined"
        ),
    }
    return ManifoldBranchReference(
        origin_label=origin_label,
        stability=branch,
        seed_side=requested_side,
        section_phi=section_phi,
        map_span=map_span,
        anchor_RZ_m=anchor,
        direction_RZ=direction,
        expansion=expansion,
        orientation_reversing=orientation_reversing,
        seed_orders=seed_orders,
        seed_distances_m=seed_distances,
        points_RZ_m=points,
        available=available,
        metadata=metadata,
    )


def nearest_available_manifold_sample(
    branch: ManifoldBranchReference,
    target_RZ_m: object,
    *,
    minimum_generation: int = 1,
    maximum_generation: int | None = None,
    seed_orders: Sequence[int] | None = None,
) -> ManifoldSampleMatch:
    """Select the nearest available production label to a physical target."""

    if not isinstance(branch, ManifoldBranchReference):
        raise TypeError("branch must be ManifoldBranchReference")
    target = _finite_vector(target_RZ_m, "target_RZ_m")
    minimum = _nonnegative_integer(minimum_generation, "minimum_generation")
    maximum = (
        branch.n_generations
        if maximum_generation is None
        else _nonnegative_integer(maximum_generation, "maximum_generation")
    )
    if minimum > maximum or maximum > branch.n_generations:
        raise ValueError("generation bounds must lie inside the traced branch")

    allowed_seeds = np.ones(branch.n_seeds, dtype=bool)
    if seed_orders is not None:
        allowed_seeds[:] = False
        for seed_order in seed_orders:
            allowed_seeds[branch.seed_index(seed_order)] = True
        if not np.any(allowed_seeds):
            raise ValueError("seed_orders must select at least one production seed")

    candidates = branch.available.copy()
    candidates[:minimum] = False
    candidates[maximum + 1 :] = False
    candidates[:, ~allowed_seeds] = False
    if not np.any(candidates):
        raise ValueError("no production manifold samples satisfy the selection")

    distances = np.linalg.norm(branch.points_RZ_m - target[None, None, :], axis=2)
    distances[~candidates] = np.inf
    generation_index, seed_index = np.unravel_index(
        int(np.argmin(distances)),
        distances.shape,
    )
    label = branch.sample_label(generation_index, seed_index)
    return ManifoldSampleMatch(
        label=label,
        point_RZ_m=branch.points_RZ_m[generation_index, seed_index],
        distance_m=float(distances[generation_index, seed_index]),
        jax_seed_index=seed_index,
    )


def refresh_manifold_sample_match(
    previous_match: ManifoldSampleMatch,
    candidate_branch: ManifoldBranchReference,
    target_RZ_m: object,
    *,
    maximum_sample_displacement_m: float,
) -> ManifoldSampleRefreshReport:
    """Carry one exact outer label to a newly traced production branch.

    This function never chooses a nearest replacement.  Identity changes,
    missing seed orders or generations, wall-terminated points, and excessive
    displacement all reject the refresh.  A separate explicit outer decision
    may select a new label after such a rejection.
    """

    if not isinstance(previous_match, ManifoldSampleMatch):
        raise TypeError("previous_match must be ManifoldSampleMatch")
    if not isinstance(candidate_branch, ManifoldBranchReference):
        raise TypeError("candidate_branch must be ManifoldBranchReference")
    target = _finite_vector(target_RZ_m, "target_RZ_m")
    maximum = float(maximum_sample_displacement_m)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError(
            "maximum_sample_displacement_m must be non-negative and finite"
        )

    label = previous_match.label
    if (
        label.origin_label != candidate_branch.origin_label
        or label.stability != candidate_branch.stability
        or label.seed_side != candidate_branch.seed_side
    ):
        return ManifoldSampleRefreshReport(
            previous_match=previous_match,
            candidate_match=None,
            sample_displacement_m=float("inf"),
            maximum_sample_displacement_m=maximum,
            accepted=False,
            rejection_reason="branch_identity_mismatch",
        )

    try:
        candidate_point = candidate_branch.point(label)
        seed_index = candidate_branch.seed_index(label.seed_order)
    except KeyError:
        return ManifoldSampleRefreshReport(
            previous_match=previous_match,
            candidate_match=None,
            sample_displacement_m=float("inf"),
            maximum_sample_displacement_m=maximum,
            accepted=False,
            rejection_reason="sample_unavailable",
        )

    displacement = float(
        np.linalg.norm(candidate_point - previous_match.point_RZ_m)
    )
    candidate_match = ManifoldSampleMatch(
        label=label,
        point_RZ_m=candidate_point,
        distance_m=float(np.linalg.norm(candidate_point - target)),
        jax_seed_index=seed_index,
    )
    accepted = displacement <= maximum
    return ManifoldSampleRefreshReport(
        previous_match=previous_match,
        candidate_match=candidate_match,
        sample_displacement_m=displacement,
        maximum_sample_displacement_m=maximum,
        accepted=accepted,
        rejection_reason=None if accepted else "sample_displacement_exceeds_limit",
    )


def compare_jax_manifold_branch(
    branch: ManifoldBranchReference,
    jax_generations: object,
    *,
    absolute_tolerance_m: float,
    require_complete: bool = True,
) -> ManifoldCorrespondenceReport:
    """Validate fixed-shape JAX generations against production Cyna labels."""

    if not isinstance(branch, ManifoldBranchReference):
        raise TypeError("branch must be ManifoldBranchReference")
    jax_points = np.asarray(jax_generations, dtype=float)
    if jax_points.shape != branch.points_RZ_m.shape:
        raise ValueError("jax_generations must match the production branch shape")
    tolerance = float(absolute_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance_m must be non-negative and finite")

    availability = branch.available
    n_expected = int(availability.size)
    n_compared = int(np.count_nonzero(availability))
    n_missing = n_expected - n_compared
    finite = bool(np.all(np.isfinite(jax_points)))
    if n_compared:
        differences = np.linalg.norm(
            jax_points[availability] - branch.points_RZ_m[availability],
            axis=1,
        )
        finite = finite and bool(np.all(np.isfinite(differences)))
        maximum = float(np.max(differences))
        rms = float(np.sqrt(np.mean(differences * differences)))
    else:
        maximum = float("inf")
        rms = float("inf")
        finite = False
    complete = branch.complete
    accepted = (
        finite
        and n_compared > 0
        and maximum <= tolerance
        and (complete or not bool(require_complete))
    )
    return ManifoldCorrespondenceReport(
        branch_label=branch.label,
        n_expected=n_expected,
        n_compared=n_compared,
        n_missing=n_missing,
        complete=complete,
        finite=finite,
        max_deviation_m=maximum,
        rms_deviation_m=rms,
        absolute_tolerance_m=tolerance,
        accepted=accepted,
    )


__all__ = [
    "ManifoldBranchReference",
    "ManifoldCorrespondenceReport",
    "ManifoldSampleLabel",
    "ManifoldSampleMatch",
    "ManifoldSampleRefreshReport",
    "compare_jax_manifold_branch",
    "manifold_branch_reference_from_trace",
    "nearest_available_manifold_sample",
    "refresh_manifold_sample_match",
]
