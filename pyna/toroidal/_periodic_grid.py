"""Small helpers for periodic toroidal-surface grids."""

from __future__ import annotations

import numpy as np


TWOPI = 2.0 * np.pi


def strip_periodic_endpoint(axis_values: np.ndarray, period: float, name: str) -> tuple[np.ndarray, bool]:
    """Return ``axis_values`` without a duplicated periodic endpoint."""

    axis = np.asarray(axis_values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 3:
        raise ValueError(f"{name} must be one-dimensional with at least three points")
    has_endpoint = np.isclose(axis[-1] - axis[0], period, rtol=1.0e-10, atol=1.0e-12)
    return (axis[:-1], True) if has_endpoint else (axis, False)


def drop_endpoint(arr: np.ndarray, axis: int, has_endpoint: bool) -> np.ndarray:
    """Drop the last sample along ``axis`` when it duplicates a periodic endpoint."""

    if not has_endpoint:
        return arr
    return np.take(arr, np.arange(arr.shape[axis] - 1), axis=axis)


def periodic_derivative(values: np.ndarray, period: float, axis: int) -> np.ndarray:
    """Centered finite difference on a periodic uniform grid."""

    vals = np.asarray(values, dtype=np.float64)
    n = vals.shape[axis]
    if n < 3:
        raise ValueError("periodic derivative requires at least three points")
    step = float(period) / float(n)
    return (np.roll(vals, -1, axis=axis) - np.roll(vals, 1, axis=axis)) / (2.0 * step)


def periodic_interp(
    theta_src: np.ndarray,
    values: np.ndarray,
    theta_dst: np.ndarray,
    period: float,
) -> np.ndarray:
    """Interpolate periodic scalar samples to another periodic grid."""

    src = np.mod(np.asarray(theta_src, dtype=np.float64), float(period))
    vals = np.asarray(values, dtype=np.float64)
    dst = np.mod(np.asarray(theta_dst, dtype=np.float64), float(period))
    keep = np.isfinite(src) & np.isfinite(vals)
    if np.count_nonzero(keep) < 2:
        return np.full(dst.shape, np.nan, dtype=np.float64)
    src = src[keep]
    vals = vals[keep]
    order = np.argsort(src)
    src = src[order]
    vals = vals[order]
    _, index = np.unique(np.round(src, decimals=12), return_index=True)
    src = src[index]
    vals = vals[index]
    if src.size < 2:
        return np.full(dst.shape, np.nan, dtype=np.float64)
    src_ext = np.concatenate([src[-1:] - period, src, src[:1] + period])
    vals_ext = np.concatenate([vals[-1:], vals, vals[:1]])
    return np.interp(dst, src_ext, vals_ext)


def strip_field_grid(values: np.ndarray, theta_vals: np.ndarray, phi_vals: np.ndarray) -> np.ndarray:
    """Strip duplicated theta/phi endpoints from a grid whose last axis is theta."""

    _, theta_has_endpoint = strip_periodic_endpoint(theta_vals, TWOPI, "theta_vals")
    _, phi_has_endpoint = strip_periodic_endpoint(phi_vals, TWOPI, "phi_vals")
    arr = np.asarray(values)
    arr = drop_endpoint(arr, axis=-1, has_endpoint=theta_has_endpoint)
    arr = drop_endpoint(arr, axis=0, has_endpoint=phi_has_endpoint)
    return arr


def prepare_surface_arrays(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and strip duplicated endpoints from ``(n_phi, n_r, n_theta)`` surfaces."""

    theta, theta_has_endpoint = strip_periodic_endpoint(theta_vals, TWOPI, "theta_vals")
    phi, phi_has_endpoint = strip_periodic_endpoint(phi_vals, TWOPI, "phi_vals")
    R = np.asarray(R_surf, dtype=np.float64)
    Z = np.asarray(Z_surf, dtype=np.float64)
    if R.ndim != 3 or Z.shape != R.shape:
        raise ValueError("R_surf and Z_surf must have matching shape (n_phi, n_r, n_theta)")
    R = drop_endpoint(R, axis=2, has_endpoint=theta_has_endpoint)
    Z = drop_endpoint(Z, axis=2, has_endpoint=theta_has_endpoint)
    R = drop_endpoint(R, axis=0, has_endpoint=phi_has_endpoint)
    Z = drop_endpoint(Z, axis=0, has_endpoint=phi_has_endpoint)
    if R.shape[0] != phi.size or R.shape[2] != theta.size:
        raise ValueError(
            "surface array shape must match phi_vals and theta_vals after removing periodic endpoints"
        )
    return R, Z, phi, theta


__all__ = [
    "TWOPI",
    "drop_endpoint",
    "prepare_surface_arrays",
    "periodic_derivative",
    "periodic_interp",
    "strip_field_grid",
    "strip_periodic_endpoint",
]
