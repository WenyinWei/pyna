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


__all__ = [
    "GlassWindow",
    "TargetPlate",
    "ToroidalComponentSurface",
    "ToroidalWall",
    "coerce_toroidal_surface_arrays",
]
