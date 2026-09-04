"""Dependency-light data contracts shared by Cyna and JAX strike paths."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Literal

import numpy as np


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


@dataclass(frozen=True)
class LocalWallPlane:
    """Outer-frozen tangent plane at one production wall strike."""

    strike_label: ManifoldStrikeLabel
    point_xyz_m: np.ndarray
    normal_xyz: np.ndarray
    wall_phi_rad: float
    wall_s: float
    projection_distance_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.strike_label, ManifoldStrikeLabel):
            raise TypeError("strike_label must be ManifoldStrikeLabel")
        point = _finite_vector(self.point_xyz_m, 3, "point_xyz_m")
        normal = _finite_vector(self.normal_xyz, 3, "normal_xyz")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("normal_xyz must be non-zero")
        wall_phi = float(self.wall_phi_rad)
        wall_s = float(self.wall_s)
        distance = float(self.projection_distance_m)
        if not np.isfinite(wall_phi):
            raise ValueError("wall_phi_rad must be finite")
        if not np.isfinite(wall_s) or not 0.0 <= wall_s <= 1.0:
            raise ValueError("wall_s must lie in [0, 1]")
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("projection_distance_m must be non-negative and finite")
        object.__setattr__(self, "point_xyz_m", point.copy())
        object.__setattr__(self, "normal_xyz", normal / normal_norm)
        object.__setattr__(self, "wall_phi_rad", wall_phi)
        object.__setattr__(self, "wall_s", wall_s)
        object.__setattr__(self, "projection_distance_m", distance)

    @property
    def level_set_params(self) -> np.ndarray:
        """Return ``(point_xyz, unit_normal_xyz)`` for the JAX level set."""

        return np.concatenate((self.point_xyz_m, self.normal_xyz))


__all__ = [
    "LocalWallPlane",
    "ManifoldStrikeLabel",
    "ManifoldStrikeMatch",
    "ManifoldStrikeRefreshReport",
]
