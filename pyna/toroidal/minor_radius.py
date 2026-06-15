"""Minor-radius labels for toroidal ``(R, Z, Phi)`` grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _as_nonempty_1d(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def minor_radius_label(
    R: np.ndarray,
    Z: np.ndarray,
    Phi: np.ndarray,
    axis_R: np.ndarray | float,
    axis_Z: np.ndarray | float,
    a_eff: float,
    *,
    clip: bool = True,
    upper: float = 1.0,
) -> np.ndarray:
    """Return normalized geometric minor radius on an ``(R, Z, Phi)`` grid.

    The label is ``sqrt((R - R_axis(phi))^2 + (Z - Z_axis(phi))^2) / a_eff``.
    It is deliberately a rho-only construction: no poloidal angle, toroidal
    angle, field-line phase, or flux-surface topology is inferred.
    """

    R_arr = _as_nonempty_1d("R", R)
    Z_arr = _as_nonempty_1d("Z", Z)
    Phi_arr = _as_nonempty_1d("Phi", Phi)
    scale = float(a_eff)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("a_eff must be a positive finite scalar")

    axis_R_arr = np.broadcast_to(np.asarray(axis_R, dtype=np.float64), Phi_arr.shape)
    axis_Z_arr = np.broadcast_to(np.asarray(axis_Z, dtype=np.float64), Phi_arr.shape)
    if not np.all(np.isfinite(axis_R_arr)) or not np.all(np.isfinite(axis_Z_arr)):
        raise ValueError("axis_R and axis_Z must contain finite values")

    rho = np.sqrt(
        (R_arr[:, np.newaxis, np.newaxis] - axis_R_arr[np.newaxis, np.newaxis, :]) ** 2
        + (Z_arr[np.newaxis, :, np.newaxis] - axis_Z_arr[np.newaxis, np.newaxis, :]) ** 2
    ) / scale
    if clip:
        return np.clip(rho, 0.0, float(upper))
    return rho


@dataclass(frozen=True)
class GeometricMinorRadiusProvider:
    """Callable provider for the axis-distance normalized minor-radius label.

    This is a geometry-only fallback for axisymmetric or synthetic workflows.
    It intentionally does not construct poloidal/toroidal angles or infer
    magnetic-surface topology.
    """

    a_eff: float | None = None
    clip: bool = True
    upper: float = 1.0
    source_name: str = "pyna_geometric_minor_radius"

    def __call__(
        self,
        eqd: Mapping[str, np.ndarray],
        axis_R: np.ndarray | float,
        axis_Z: np.ndarray | float,
        config: Any = None,
    ) -> np.ndarray:
        a_eff = self.a_eff
        if a_eff is None and config is not None:
            a_eff = getattr(config, "a_eff", None)
        if a_eff is None:
            raise ValueError("a_eff must be provided on the provider or config")
        return minor_radius_label(
            eqd["R"],
            eqd["Z"],
            eqd["Phi"],
            axis_R,
            axis_Z,
            float(a_eff),
            clip=self.clip,
            upper=self.upper,
        )


__all__ = [
    "GeometricMinorRadiusProvider",
    "minor_radius_label",
]
