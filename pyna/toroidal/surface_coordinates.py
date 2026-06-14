"""Utilities for building toroidal flux-surface coordinate meshes.

This module contains data-format-independent pieces used when converting
ordered Poincare traces into straight-field-line theta grids.  Field-line
tracing remains in :mod:`pyna.toroidal.flt`/cyna; these helpers operate on
already collected crossings and axis samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class AxisCoreInsertion:
    """Result of inserting axis-to-first-reliable-surface core rows."""

    R_surf: np.ndarray
    Z_surf: np.ndarray
    radial_labels: np.ndarray
    fractions: np.ndarray


def stitch_periodic(
    theta: np.ndarray,
    values: np.ndarray,
    target_theta: np.ndarray,
    *,
    period: float = TWOPI,
    min_points: int = 8,
    duplicate_decimals: int = 10,
) -> np.ndarray:
    """Interpolate scattered periodic samples onto a target theta grid.

    Duplicate theta bins, after rounding, are averaged before interpolation.
    The interpolation is periodic by appending one wrapped sample at each end.
    """

    per = float(period)
    theta_arr = np.mod(np.asarray(theta, dtype=np.float64), per)
    val_arr = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_theta, dtype=np.float64)
    keep = np.isfinite(theta_arr) & np.isfinite(val_arr)
    if np.count_nonzero(keep) < int(min_points):
        return np.full(target.shape, np.nan, dtype=np.float64)

    theta_arr = theta_arr[keep]
    val_arr = val_arr[keep]
    order = np.argsort(theta_arr)
    theta_arr = theta_arr[order]
    val_arr = val_arr[order]

    key = np.round(theta_arr, decimals=int(duplicate_decimals))
    unique, inverse = np.unique(key, return_inverse=True)
    accum = np.zeros(unique.size, dtype=np.float64)
    count = np.zeros(unique.size, dtype=np.float64)
    theta_u = np.zeros(unique.size, dtype=np.float64)
    for idx in range(unique.size):
        sel = inverse == idx
        theta_u[idx] = float(theta_arr[sel][0])
    for idx, val in zip(inverse, val_arr):
        accum[idx] += float(val)
        count[idx] += 1.0
    vals_u = accum / np.maximum(count, 1.0)

    if theta_u.size < 4:
        return np.full(target.shape, np.nan, dtype=np.float64)
    theta_ext = np.concatenate([theta_u[-1:] - per, theta_u, theta_u[:1] + per])
    vals_ext = np.concatenate([vals_u[-1:], vals_u, vals_u[:1]])
    return np.interp(np.mod(target, per), theta_ext, vals_ext)


def circle_map_lift_iota(
    phase: np.ndarray,
    *,
    max_iota: float = 0.5,
    min_points: int = 16,
    grid_size: int = 1001,
    chunk: int = 128,
    xatol: float = 1.0e-10,
) -> tuple[float, float]:
    """Fit ``phase_k = phase0 + 2*pi*iota*k`` directly on the circle.

    The input phases remain modulo ``2*pi``; the turn index ``k`` supplies the
    covering-space information.  Both phase orientations are tried and the
    concentration with the larger circular score is retained.

    Returns
    -------
    (iota, circular_rms)
        ``nan`` values are returned when the phase sequence is insufficient.
    """

    ph = np.mod(np.asarray(phase, dtype=np.float64), TWOPI)
    keep = np.isfinite(ph)
    if np.count_nonzero(keep) < int(min_points):
        return float("nan"), float("nan")
    ph = ph[keep]
    turn = np.arange(ph.size, dtype=np.float64)

    def concentration(iota: float, signed_phase: np.ndarray) -> float:
        z = np.exp(1j * (signed_phase - TWOPI * float(iota) * turn))
        return float(np.abs(np.nanmean(z)))

    best_iota = float("nan")
    best_score = -np.inf
    grid = np.linspace(0.0, float(max_iota), int(grid_size), dtype=np.float64)
    for sign in (1.0, -1.0):
        signed = np.mod(sign * ph, TWOPI)
        scores = np.empty(grid.size, dtype=np.float64)
        for start in range(0, grid.size, int(chunk)):
            stop = min(start + int(chunk), grid.size)
            arg = signed[np.newaxis, :] - TWOPI * grid[start:stop, np.newaxis] * turn[np.newaxis, :]
            scores[start:stop] = np.abs(np.mean(np.exp(1j * arg), axis=1))
        idx = int(np.nanargmax(scores))
        lo = max(0.0, float(grid[max(idx - 2, 0)]))
        hi = min(float(max_iota), float(grid[min(idx + 2, grid.size - 1)]))
        if hi <= lo:
            cand = float(grid[idx])
            score = float(scores[idx])
        else:
            opt = minimize_scalar(
                lambda x: -concentration(float(x), signed),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": float(xatol)},
            )
            cand = float(opt.x) if opt.success else float(grid[idx])
            score = concentration(cand, signed)
        if score > best_score:
            best_iota = cand
            best_score = score

    if not np.isfinite(best_iota) or best_score <= 0.0:
        return float("nan"), float("nan")
    circular_rms = float(np.sqrt(max(0.0, -2.0 * np.log(min(best_score, 1.0)))))
    return float(best_iota), circular_rms


def rank_phase_from_axis(
    R: np.ndarray,
    Z: np.ndarray,
    axis_R: float,
    axis_Z: float,
) -> np.ndarray:
    """Convert one closed Poincare curve to rank phase around the axis."""

    rr = np.asarray(R, dtype=np.float64)
    zz = np.asarray(Z, dtype=np.float64)
    geom = np.mod(np.arctan2(zz - float(axis_Z), rr - float(axis_R)), TWOPI)
    order = np.argsort(geom)
    rank = np.empty(geom.size, dtype=np.int64)
    rank[order] = np.arange(geom.size, dtype=np.int64)
    return TWOPI * rank.astype(np.float64) / float(max(geom.size, 1))


def theta_coverage(theta: np.ndarray, bins: int, *, period: float = TWOPI) -> float:
    """Fraction of periodic theta bins occupied by finite samples."""

    th = np.asarray(theta, dtype=np.float64)
    keep = np.isfinite(th)
    if np.count_nonzero(keep) == 0:
        return 0.0
    hist, _ = np.histogram(np.mod(th[keep], float(period)), bins=max(int(bins), 4), range=(0.0, float(period)))
    return float(np.count_nonzero(hist) / hist.size)


def periodic_shift_theta(
    values: np.ndarray,
    theta_vals: np.ndarray,
    offset: float,
    *,
    period: float = TWOPI,
) -> np.ndarray:
    """Apply a surface-constant periodic theta offset to arrays ending in theta."""

    vals = np.asarray(values, dtype=np.float64)
    theta = np.asarray(theta_vals, dtype=np.float64)
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    theta_ext = np.concatenate([theta, [float(period)]])
    query = np.mod(theta + float(offset), float(period))
    for idx in np.ndindex(vals.shape[:-1]):
        row = vals[idx]
        row_ext = np.concatenate([row, row[:1]])
        keep = np.isfinite(theta_ext) & np.isfinite(row_ext)
        if np.count_nonzero(keep) >= 2:
            out[idx] = np.interp(query, theta_ext[keep], row_ext[keep])
    return out


def lfs_theta_offset_at_phi0(
    surf_R: np.ndarray,
    surf_Z: np.ndarray,
    theta_vals: np.ndarray,
    *,
    axis_R0: float,
    axis_Z0: float,
) -> float:
    """Theta value of the low-field-side point on the phi=0 section."""

    rr = np.asarray(surf_R[0], dtype=np.float64)
    zz = np.asarray(surf_Z[0], dtype=np.float64)
    theta = np.asarray(theta_vals, dtype=np.float64)
    keep = np.isfinite(rr) & np.isfinite(zz)
    if np.count_nonzero(keep) < 3:
        return 0.0
    score = (rr - float(axis_R0)) - 0.05 * np.abs(zz - float(axis_Z0))
    idx = int(np.nanargmax(np.where(keep, score, np.nan)))
    return float(theta[idx])


def sanitized_axis_core_fractions(fractions: Iterable[float] | np.ndarray) -> np.ndarray:
    """Return unique fractions strictly between axis and first surface."""

    arr = np.asarray(list(fractions) if not isinstance(fractions, np.ndarray) else fractions, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0) & (arr < 1.0)]
    return np.unique(np.sort(arr))


def insert_axis_core_surfaces(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    radial_labels: np.ndarray,
    axis_R: np.ndarray,
    axis_Z: np.ndarray,
    fractions: Iterable[float] | np.ndarray,
) -> AxisCoreInsertion:
    """Insert linearly interpolated core surfaces from axis to first surface.

    ``R_surf`` and ``Z_surf`` must have shape ``(n_phi, n_r, n_theta)``.
    ``axis_R`` and ``axis_Z`` must have shape ``(n_phi,)``.  The returned
    labels are ``fraction * radial_labels[0]`` followed by the input labels.
    """

    R = np.asarray(R_surf, dtype=np.float64)
    Z = np.asarray(Z_surf, dtype=np.float64)
    labels = np.asarray(radial_labels, dtype=np.float64)
    ax_R = np.asarray(axis_R, dtype=np.float64)
    ax_Z = np.asarray(axis_Z, dtype=np.float64)
    frac = sanitized_axis_core_fractions(fractions)
    if R.ndim != 3 or Z.shape != R.shape:
        raise ValueError("R_surf and Z_surf must have matching shape (n_phi, n_r, n_theta)")
    if labels.ndim != 1 or labels.size != R.shape[1]:
        raise ValueError("radial_labels must be one-dimensional with one label per surface")
    if ax_R.shape != (R.shape[0],) or ax_Z.shape != (R.shape[0],):
        raise ValueError("axis_R and axis_Z must have shape (n_phi,)")
    if R.shape[1] == 0 or frac.size == 0:
        return AxisCoreInsertion(R.copy(), Z.copy(), labels.copy(), frac)

    first_R = R[:, :1, :]
    first_Z = Z[:, :1, :]
    axis_R3 = ax_R[:, np.newaxis, np.newaxis]
    axis_Z3 = ax_Z[:, np.newaxis, np.newaxis]
    f = frac[np.newaxis, :, np.newaxis]
    core_R = axis_R3 + f * (first_R - axis_R3)
    core_Z = axis_Z3 + f * (first_Z - axis_Z3)
    return AxisCoreInsertion(
        R_surf=np.concatenate([core_R, R], axis=1),
        Z_surf=np.concatenate([core_Z, Z], axis=1),
        radial_labels=np.concatenate([frac * labels[0], labels]),
        fractions=frac,
    )


__all__ = [
    "AxisCoreInsertion",
    "TWOPI",
    "circle_map_lift_iota",
    "insert_axis_core_surfaces",
    "lfs_theta_offset_at_phi0",
    "periodic_shift_theta",
    "rank_phase_from_axis",
    "sanitized_axis_core_fractions",
    "stitch_periodic",
    "theta_coverage",
]
