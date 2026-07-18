"""Toroidal component geometry containers.

These classes provide a small, explicit object layer over the legacy
``(wall_phi, wall_R_all, wall_Z_all)`` arrays used by field-line tracing.  They
are intentionally generic: a first wall, target plate, diagnostic window, or
other toroidal component can share the same data model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pyna.fields.cylindrical import validate_phi_grid
from pyna.fields.periodicity import ToroidalPeriodicity, normalize_nfp


@dataclass(frozen=True)
class ToroidalComponentSurface:
    """R/Z surface sections sampled on a toroidal ``phi`` grid.

    Parameters
    ----------
    phi:
        One-dimensional toroidal grid.  For ``nfp > 1`` this grid describes one
        field period, not the full torus.
    R, Z:
        Arrays with shape ``(n_phi, n_poloidal)``.
    nfp:
        Number of field periods represented by symmetry.
    kind:
        Semantic component type such as ``"wall"``, ``"target_plate"``, or
        ``"glass_window"``.
    """

    phi: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    nfp: int = 1
    kind: str = "component"
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nfp = normalize_nfp(self.nfp)
        phi = validate_phi_grid(self.phi, nfp=nfp, name="surface phi")
        R = np.asarray(self.R, dtype=np.float64)
        Z = np.asarray(self.Z, dtype=np.float64)
        if R.shape != Z.shape:
            raise ValueError(f"R and Z surface arrays must have identical shapes; got {R.shape}, {Z.shape}")
        if R.ndim != 2:
            raise ValueError("R and Z surface arrays must have shape (n_phi, n_poloidal)")
        if R.shape[0] != phi.size:
            raise ValueError(f"surface phi length {phi.size} does not match R/Z n_phi={R.shape[0]}")
        if R.shape[1] < 2:
            raise ValueError("surface must contain at least two poloidal points per toroidal section")
        if not (np.all(np.isfinite(R)) and np.all(np.isfinite(Z))):
            raise ValueError("surface R/Z arrays must contain only finite values")
        object.__setattr__(self, "nfp", nfp)
        object.__setattr__(self, "phi", np.ascontiguousarray(phi, dtype=np.float64))
        object.__setattr__(self, "R", np.ascontiguousarray(R, dtype=np.float64))
        object.__setattr__(self, "Z", np.ascontiguousarray(Z, dtype=np.float64))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def field_period(self) -> float:
        return self.periodicity.field_period

    @property
    def periodicity(self) -> ToroidalPeriodicity:
        return ToroidalPeriodicity(nfp=self.nfp, origin=float(self.phi[0]))

    @property
    def n_phi(self) -> int:
        return int(self.R.shape[0])

    @property
    def n_poloidal(self) -> int:
        return int(self.R.shape[1])

    @property
    def wall_phi(self) -> np.ndarray:
        """Compatibility alias for legacy wall-tracing code."""

        return self.phi

    @property
    def wall_R_all(self) -> np.ndarray:
        """Compatibility alias for legacy wall-tracing code."""

        return self.R

    @property
    def wall_Z_all(self) -> np.ndarray:
        """Compatibility alias for legacy wall-tracing code."""

        return self.Z

    def as_twall_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(wall_phi, wall_R_all, wall_Z_all)`` arrays."""

        return self.phi, self.R, self.Z

    def section(self, index: int | None = None, *, phi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return one poloidal R/Z section by index or nearest toroidal angle."""

        if index is None:
            if phi is None:
                raise ValueError("section requires either index or phi")
            phi_mod = float(self.periodicity.wrap(phi))
            index = min(range(self.n_phi), key=lambda i: abs(float(self.phi[i]) - phi_mod))
        idx = int(index) % self.n_phi
        return self.R[idx].copy(), self.Z[idx].copy()

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable geometry dictionary."""

        return {
            "phi": self.phi.copy(),
            "R": self.R.copy(),
            "Z": self.Z.copy(),
            "nfp": int(self.nfp),
            "kind": self.kind,
            "name": self.name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_arrays(
        cls,
        phi,
        R,
        Z,
        *,
        nfp: int = 1,
        kind: str = "component",
        name: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToroidalComponentSurface":
        return cls(phi=phi, R=R, Z=Z, nfp=nfp, kind=kind, name=name, metadata=dict(metadata or {}))


class ToroidalWall(ToroidalComponentSurface):
    """Toroidal first-wall surface."""

    def __init__(self, phi, R, Z, *, nfp: int = 1, name: str = "wall", metadata: Mapping[str, Any] | None = None):
        super().__init__(phi=phi, R=R, Z=Z, nfp=nfp, kind="wall", name=name, metadata=dict(metadata or {}))


class TargetPlate(ToroidalComponentSurface):
    """Toroidal target-plate surface."""

    def __init__(
        self,
        phi,
        R,
        Z,
        *,
        nfp: int = 1,
        name: str = "target_plate",
        metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            phi=phi,
            R=R,
            Z=Z,
            nfp=nfp,
            kind="target_plate",
            name=name,
            metadata=dict(metadata or {}),
        )


class GlassWindow(ToroidalComponentSurface):
    """Toroidal diagnostic-window surface."""

    def __init__(
        self,
        phi,
        R,
        Z,
        *,
        nfp: int = 1,
        name: str = "glass_window",
        metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            phi=phi,
            R=R,
            Z=Z,
            nfp=nfp,
            kind="glass_window",
            name=name,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class ToroidalSurfaceProjection:
    """Continuous closest-segment projection onto a sampled toroidal surface.

    ``s`` is normalized closed poloidal arclength on the wall section
    interpolated at each query ``phi``.  The projected cylindrical and
    Cartesian points therefore do not snap to either toroidal sections or
    poloidal vertices.
    """

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    s: np.ndarray
    xyz: np.ndarray
    distance: np.ndarray
    normal: np.ndarray
    segment_index: np.ndarray
    phi_lower_index: np.ndarray
    phi_upper_index: np.ndarray
    phi_fraction: np.ndarray


def coerce_toroidal_surface_arrays(
    surface_or_phi,
    R_all=None,
    Z_all=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(phi, R, Z)`` arrays from a surface object or legacy arrays."""

    if R_all is None and Z_all is None:
        obj = surface_or_phi
        if isinstance(obj, ToroidalComponentSurface):
            return obj.as_twall_arrays()
        if hasattr(obj, "as_twall_arrays"):
            phi, R, Z = obj.as_twall_arrays()
            surface = ToroidalComponentSurface(phi=phi, R=R, Z=Z, nfp=getattr(obj, "nfp", 1))
            return surface.as_twall_arrays()
        if all(hasattr(obj, name) for name in ("wall_phi", "wall_R_all", "wall_Z_all")):
            surface = ToroidalComponentSurface(
                phi=getattr(obj, "wall_phi"),
                R=getattr(obj, "wall_R_all"),
                Z=getattr(obj, "wall_Z_all"),
                nfp=getattr(obj, "nfp", 1),
                kind=getattr(obj, "kind", "component"),
                name=getattr(obj, "name", ""),
            )
            return surface.as_twall_arrays()
        if isinstance(obj, (tuple, list)) and len(obj) == 3:
            surface = ToroidalComponentSurface(phi=obj[0], R=obj[1], Z=obj[2])
            return surface.as_twall_arrays()
        raise TypeError("expected a ToroidalComponentSurface or legacy (phi, R, Z) arrays")
    if R_all is None or Z_all is None:
        raise ValueError("surface phi, R_all, and Z_all must be supplied together")
    surface = ToroidalComponentSurface(phi=surface_or_phi, R=R_all, Z=Z_all)
    return surface.as_twall_arrays()


def project_points_to_toroidal_surface(
    R,
    Z,
    phi,
    surface_or_phi,
    R_all=None,
    Z_all=None,
    *,
    field_period: float | None = None,
) -> ToroidalSurfaceProjection:
    """Project cylindrical points onto continuously interpolated wall segments.

    The two toroidal sections bracketing each query angle are interpolated
    before the closest point on every closed poloidal segment is evaluated.
    This avoids the grid-dependent nearest-vertex strike coordinate used by
    plotting-only footprint helpers.

    Parameters
    ----------
    R, Z, phi:
        Broadcast-compatible cylindrical query coordinates.
    surface_or_phi, R_all, Z_all:
        A :class:`ToroidalComponentSurface`, a compatible wall object, or the
        legacy ``(wall_phi, wall_R_all, wall_Z_all)`` arrays.
    field_period:
        Toroidal period represented by the wall.  It is inferred from a wall
        object's ``field_period``/``toroidal_period`` when possible and
        otherwise defaults to ``2*pi``.
    """

    try:
        query_R, query_Z, query_phi = np.broadcast_arrays(
            np.asarray(R, dtype=float),
            np.asarray(Z, dtype=float),
            np.asarray(phi, dtype=float),
        )
    except ValueError as exc:
        raise ValueError("R, Z, and phi must be broadcast-compatible") from exc
    query_R = query_R.ravel()
    query_Z = query_Z.ravel()
    query_phi = query_phi.ravel()
    if not (
        np.all(np.isfinite(query_R))
        and np.all(np.isfinite(query_Z))
        and np.all(np.isfinite(query_phi))
    ):
        raise ValueError("projection query coordinates must be finite")
    if np.any(query_R <= 0.0):
        raise ValueError("projection query R must be positive")

    if R_all is None and Z_all is None and all(
        hasattr(surface_or_phi, name) for name in ("phi_values", "R", "Z")
    ):
        wall_phi = np.asarray(getattr(surface_or_phi, "phi_values"), dtype=float)
        wall_R = np.asarray(getattr(surface_or_phi, "R"), dtype=float)
        wall_Z = np.asarray(getattr(surface_or_phi, "Z"), dtype=float)
    else:
        wall_phi, wall_R, wall_Z = coerce_toroidal_surface_arrays(
            surface_or_phi,
            R_all,
            Z_all,
        )
    wall_phi = np.asarray(wall_phi, dtype=float).ravel()
    wall_R = np.asarray(wall_R, dtype=float)
    wall_Z = np.asarray(wall_Z, dtype=float)
    if wall_phi.size < 2:
        raise ValueError("continuous toroidal projection requires at least two wall sections")
    if wall_R.ndim != 2 or wall_R.shape != wall_Z.shape or wall_R.shape[0] != wall_phi.size:
        raise ValueError("wall R/Z must share shape (n_phi, n_poloidal)")
    if wall_R.shape[1] < 2:
        raise ValueError("wall sections require at least two poloidal vertices")
    if not (
        np.all(np.isfinite(wall_phi))
        and np.all(np.isfinite(wall_R))
        and np.all(np.isfinite(wall_Z))
    ):
        raise ValueError("wall coordinates must be finite")
    if np.any(np.diff(wall_phi) <= 0.0):
        raise ValueError("wall phi must be strictly increasing")

    if field_period is None:
        if hasattr(surface_or_phi, "field_period"):
            period = float(getattr(surface_or_phi, "field_period"))
        elif hasattr(surface_or_phi, "toroidal_period"):
            period = float(getattr(surface_or_phi, "toroidal_period"))
        else:
            period = 2.0 * np.pi
    else:
        period = float(field_period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("field_period must be positive and finite")
    tolerance = max(1.0e-12, 1.0e-10 * period)
    span = float(wall_phi[-1] - wall_phi[0])
    if span > period + tolerance:
        raise ValueError("wall phi span exceeds field_period")
    if abs(span - period) <= tolerance:
        if not (
            np.allclose(wall_R[-1], wall_R[0], rtol=1.0e-9, atol=1.0e-11)
            and np.allclose(wall_Z[-1], wall_Z[0], rtol=1.0e-9, atol=1.0e-11)
        ):
            raise ValueError("repeated periodic wall endpoint must match the first section")
        wall_phi = wall_phi[:-1]
        wall_R = wall_R[:-1]
        wall_Z = wall_Z[:-1]
        if wall_phi.size < 2:
            raise ValueError("wall needs two distinct toroidal sections after endpoint removal")

    phi0 = float(wall_phi[0])
    projected_phi = phi0 + np.mod(query_phi - phi0, period)
    phi_extended = np.concatenate((wall_phi, [phi0 + period]))
    upper = np.searchsorted(phi_extended, projected_phi, side="right")
    upper = np.clip(upper, 1, wall_phi.size)
    lower = upper - 1
    upper_wrapped = upper % wall_phi.size
    phi_span = phi_extended[upper] - phi_extended[lower]
    if np.any(phi_span <= 0.0):
        raise ValueError("wall toroidal sections do not define positive interpolation spans")
    alpha = (projected_phi - phi_extended[lower]) / phi_span

    lower_R = wall_R[lower]
    lower_Z = wall_Z[lower]
    upper_R = wall_R[upper_wrapped]
    upper_Z = wall_Z[upper_wrapped]
    section_R = lower_R * (1.0 - alpha[:, None]) + upper_R * alpha[:, None]
    section_Z = lower_Z * (1.0 - alpha[:, None]) + upper_Z * alpha[:, None]
    next_R = np.roll(section_R, -1, axis=1)
    next_Z = np.roll(section_Z, -1, axis=1)
    delta_R = next_R - section_R
    delta_Z = next_Z - section_Z
    segment_length_sq = delta_R**2 + delta_Z**2
    if np.any(segment_length_sq <= 0.0):
        raise ValueError("interpolated wall contains a zero-length poloidal segment")

    segment_fraction = (
        (query_R[:, None] - section_R) * delta_R
        + (query_Z[:, None] - section_Z) * delta_Z
    ) / segment_length_sq
    segment_fraction = np.clip(segment_fraction, 0.0, 1.0)
    candidate_R = section_R + segment_fraction * delta_R
    candidate_Z = section_Z + segment_fraction * delta_Z
    distance_sq = (candidate_R - query_R[:, None]) ** 2 + (candidate_Z - query_Z[:, None]) ** 2
    segment_index = np.argmin(distance_sq, axis=1).astype(int)
    row_index = np.arange(query_R.size, dtype=int)
    local_fraction = segment_fraction[row_index, segment_index]
    projected_R = candidate_R[row_index, segment_index]
    projected_Z = candidate_Z[row_index, segment_index]
    distance = np.sqrt(distance_sq[row_index, segment_index])

    segment_length = np.sqrt(segment_length_sq)
    cumulative = np.cumsum(segment_length, axis=1) - segment_length
    perimeter = np.sum(segment_length, axis=1)
    if np.any(perimeter <= 0.0) or not np.all(np.isfinite(perimeter)):
        raise ValueError("interpolated wall section has non-positive perimeter")
    s = (
        cumulative[row_index, segment_index]
        + local_fraction * segment_length[row_index, segment_index]
    ) / perimeter

    cos_phi = np.cos(projected_phi)
    sin_phi = np.sin(projected_phi)
    xyz = np.column_stack(
        (projected_R * cos_phi, projected_R * sin_phi, projected_Z)
    )

    # Bilinear wall tangents give a continuous local normal.  Normalize the
    # orientation from each section polygon's winding, including concave walls.
    j_next = (segment_index + 1) % wall_R.shape[1]
    dR_pol = delta_R[row_index, segment_index]
    dZ_pol = delta_Z[row_index, segment_index]
    tangent_pol = np.column_stack((dR_pol * cos_phi, dR_pol * sin_phi, dZ_pol))
    dR_dphi_vertices = (upper_R - lower_R) / phi_span[:, None]
    dZ_dphi_vertices = (upper_Z - lower_Z) / phi_span[:, None]
    dR_dphi = (
        (1.0 - local_fraction) * dR_dphi_vertices[row_index, segment_index]
        + local_fraction * dR_dphi_vertices[row_index, j_next]
    )
    dZ_dphi = (
        (1.0 - local_fraction) * dZ_dphi_vertices[row_index, segment_index]
        + local_fraction * dZ_dphi_vertices[row_index, j_next]
    )
    tangent_phi = np.column_stack(
        (
            dR_dphi * cos_phi - projected_R * sin_phi,
            dR_dphi * sin_phi + projected_R * cos_phi,
            dZ_dphi,
        )
    )
    normal = np.cross(tangent_phi, tangent_pol)
    normal_norm = np.linalg.norm(normal, axis=1)
    if np.any(normal_norm <= 0.0) or not np.all(np.isfinite(normal_norm)):
        raise ValueError("interpolated wall has a degenerate surface normal")
    normal = normal / normal_norm[:, None]
    signed_section_area_twice = np.sum(
        section_R * next_Z - next_R * section_Z,
        axis=1,
    )
    if np.any(np.isclose(signed_section_area_twice, 0.0, rtol=0.0, atol=1.0e-14)):
        raise ValueError("interpolated wall section has undefined polygon orientation")
    flip = signed_section_area_twice < 0.0
    normal[flip] *= -1.0

    return ToroidalSurfaceProjection(
        R=projected_R,
        Z=projected_Z,
        phi=projected_phi,
        s=s,
        xyz=xyz,
        distance=distance,
        normal=normal,
        segment_index=segment_index,
        phi_lower_index=lower.astype(int),
        phi_upper_index=upper_wrapped.astype(int),
        phi_fraction=alpha,
    )


__all__ = [
    "GlassWindow",
    "TargetPlate",
    "ToroidalComponentSurface",
    "ToroidalSurfaceProjection",
    "ToroidalWall",
    "coerce_toroidal_surface_arrays",
    "project_points_to_toroidal_surface",
]
