"""Utilities for building toroidal flux-surface coordinate meshes.

This module contains data-format-independent pieces used when converting
ordered Poincare traces into straight-field-line theta grids.  Field-line
tracing remains in :mod:`pyna.toroidal.flt`/cyna; these helpers operate on
already collected crossings and axis samples.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import LinearOperator, lsmr

from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class AxisCoreInsertion:
    """Result of inserting axis-to-first-reliable-surface core rows."""

    R_surf: np.ndarray
    Z_surf: np.ndarray
    radial_labels: np.ndarray
    fractions: np.ndarray


@dataclass(frozen=True)
class PestSurfaceStraightFieldLineDiagnostic:
    """Numerical evidence for one candidate-field straight-angle solve.

    The radial leakage quantities are based on ``abs(d rho / d phi)``.  The
    normal leakage is the coordinate-independent ``abs(B . n) / abs(B)`` on
    the unchanged material surface.  Neither quantity is altered or projected
    away by the coordinate construction.
    """

    surface_index: int
    rho: float
    iota: float
    mde_residual_rms: float
    mde_residual_relative: float
    mde_residual_max: float
    theta_correction_rms: float
    theta_correction_max_abs: float
    diffeomorphism_min_jacobian: float
    diffeomorphism_max_jacobian: float
    radial_leakage_rms: float
    radial_leakage_p95: float
    radial_leakage_max: float
    normal_leakage_rms: float
    normal_leakage_p95: float
    normal_leakage_max: float
    bphi_min_abs: float
    lsmr_istop: int
    lsmr_iterations: int
    lsmr_condition_estimate: float

    def as_dict(self) -> dict[str, int | float]:
        """Return JSON-serialisable evidence for reports and generation logs."""

        return {
            name: (int(value) if name in {"surface_index", "lsmr_istop", "lsmr_iterations"} else float(value))
            for name, value in vars(self).items()
        }


@dataclass(frozen=True)
class CandidateFieldPestReparameterization:
    """PEST angle rebuilt on a candidate field without changing ``B`` or surfaces."""

    coordinates: SmoothPestCoordinates
    surface_diagnostics: tuple[PestSurfaceStraightFieldLineDiagnostic, ...]
    theta_correction: np.ndarray
    theta_slope: np.ndarray
    radial_slope: np.ndarray
    normal_leakage: np.ndarray
    selected_surface_indices: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def iota(self) -> np.ndarray:
        """Iota values aligned with ``coordinates.rho_vals`` (NaN if unselected)."""

        values = np.full(np.asarray(self.coordinates.rho_vals).shape, np.nan, dtype=np.float64)
        for diagnostic in self.surface_diagnostics:
            values[int(diagnostic.surface_index)] = float(diagnostic.iota)
        return values

    def as_summary_dict(self) -> dict[str, Any]:
        """Return compact JSON-serialisable construction evidence."""

        return {
            "schema": "pyna_candidate_field_pest_reparameterization_v1",
            "selected_surface_indices": [int(i) for i in self.selected_surface_indices],
            "surface_diagnostics": [item.as_dict() for item in self.surface_diagnostics],
            "metadata": dict(self.metadata),
        }


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


def _require_uniform_periodic_axis(
    values: np.ndarray,
    size: int,
    *,
    period: float,
    name: str,
    minimum_size: int = 8,
) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size != int(size):
        raise ValueError(f"{name} must be one-dimensional and match the surface array")
    if axis.size < int(minimum_size):
        raise ValueError(f"{name} requires at least {int(minimum_size)} endpoint-excluded samples")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be finite")
    expected = axis[0] + float(period) * np.arange(axis.size, dtype=np.float64) / float(axis.size)
    if not np.allclose(axis, expected, rtol=0.0, atol=2.0e-10):
        raise ValueError(f"{name} must be uniform, increasing, periodic, and endpoint-excluded")
    return axis


def _periodic_fft_derivative(values: np.ndarray, *, axis: int, period: float) -> np.ndarray:
    """Spectral derivative with a real skew-adjoint periodic discretisation."""

    array = np.asarray(values, dtype=np.float64)
    n = int(array.shape[axis])
    wave_number = 2.0 * np.pi * np.fft.fftfreq(n, d=float(period) / float(n))
    # A real even grid has one unpaired Nyquist coefficient.  Giving that mode
    # zero derivative preserves D.T == -D exactly on the represented real grid.
    if n % 2 == 0:
        wave_number[n // 2] = 0.0
    shape = [1] * array.ndim
    shape[axis] = n
    transformed = np.fft.fft(array, axis=axis)
    derivative = np.fft.ifft(1j * wave_number.reshape(shape) * transformed, axis=axis)
    return np.asarray(derivative.real, dtype=np.float64)


def _straight_field_line_linear_operator(
    theta_slope: np.ndarray,
    *,
    phi_period: float,
    theta_period: float,
    anchor_flat_index: int,
) -> LinearOperator:
    """Return ``[dphi + a*dtheta, -1; anchor, 0]`` and its exact adjoint."""

    a = np.asarray(theta_slope, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("theta_slope must have shape (phi, theta)")
    n_grid = int(a.size)
    anchor = int(anchor_flat_index)
    if anchor < 0 or anchor >= n_grid:
        raise ValueError("anchor_flat_index is outside the surface grid")
    anchor_weight = float(np.sqrt(max(n_grid, 1)))

    def matvec(vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=np.float64)
        u = x[:n_grid].reshape(a.shape)
        iota = float(x[n_grid])
        mde = (
            _periodic_fft_derivative(u, axis=0, period=phi_period)
            + a * _periodic_fft_derivative(u, axis=1, period=theta_period)
            - iota
        )
        return np.concatenate([mde.ravel(), [anchor_weight * u.ravel()[anchor]]])

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        y = np.asarray(vector, dtype=np.float64)
        mde_test = y[:n_grid].reshape(a.shape)
        # (a Dtheta).T v = Dtheta.T(a v) = -Dtheta(a v).
        u_adjoint = (
            -_periodic_fft_derivative(mde_test, axis=0, period=phi_period)
            - _periodic_fft_derivative(a * mde_test, axis=1, period=theta_period)
        ).ravel()
        u_adjoint[anchor] += anchor_weight * float(y[n_grid])
        return np.concatenate([u_adjoint, [-float(np.sum(mde_test))]])

    return LinearOperator(
        shape=(n_grid + 1, n_grid + 1),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.dtype(np.float64),
    )


def straight_field_line_mde_adjoint_error(
    theta_slope: np.ndarray,
    *,
    phi_period: float = TWOPI,
    theta_period: float = TWOPI,
    seed: int = 1741,
) -> float:
    """Numerically audit the exact adjoint used by the matrix-free MDE solve."""

    a = np.asarray(theta_slope, dtype=np.float64)
    operator = _straight_field_line_linear_operator(
        a,
        phi_period=float(phi_period),
        theta_period=float(theta_period),
        anchor_flat_index=0,
    )
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal(operator.shape[1])
    y = rng.standard_normal(operator.shape[0])
    left = float(np.dot(operator.matvec(x), y))
    right = float(np.dot(x, operator.rmatvec(y)))
    return float(abs(left - right) / max(abs(left), abs(right), 1.0))


def _coerce_candidate_field_values(values: Any, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    if isinstance(values, (tuple, list)) and len(values) == 3:
        components = tuple(np.asarray(component, dtype=np.float64) for component in values)
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != shape + (3,):
            raise ValueError(
                "candidate field evaluator must return (BR, BZ, BPhi) or an array with trailing size 3"
            )
        components = (array[..., 0], array[..., 1], array[..., 2])
    try:
        return tuple(np.broadcast_to(component, shape).copy() for component in components)
    except ValueError as exc:
        raise ValueError("candidate field components do not broadcast to the PEST surface grid") from exc


def _evaluate_candidate_field_on_surfaces(
    candidate_field: object,
    R: np.ndarray,
    Z: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    surface_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = R.shape
    if hasattr(candidate_field, "evaluate_pest_surface"):
        components = [np.full(shape, np.nan, dtype=np.float64) for _ in range(3)]
        for surface_index in surface_indices:
            ir = int(surface_index)
            values = candidate_field.evaluate_pest_surface(
                ir,
                theta[:, ir, :],
                phi[:, ir, :],
                R=R[:, ir, :],
                Z=Z[:, ir, :],
            )
            surface_components = _coerce_candidate_field_values(values, R[:, ir, :].shape)
            for target, source in zip(components, surface_components):
                target[:, ir, :] = source
        return tuple(components)

    selected = np.asarray(tuple(int(i) for i in surface_indices), dtype=np.int64)
    R_selected = R[:, selected, :]
    Z_selected = Z[:, selected, :]
    phi_selected = phi[:, selected, :]
    if hasattr(candidate_field, "interpolate_at"):
        values = candidate_field.interpolate_at(R_selected, Z_selected, phi_selected)
    elif callable(candidate_field):
        try:
            values = candidate_field(R_selected, Z_selected, phi_selected)
        except TypeError as three_argument_error:
            points = np.stack([R_selected, Z_selected, phi_selected], axis=-1)
            try:
                values = candidate_field(points)
            except TypeError:
                raise three_argument_error
    else:
        raise TypeError("candidate_field must be callable or expose interpolate_at(R, Z, phi)")
    selected_components = _coerce_candidate_field_values(values, R_selected.shape)
    components = [np.full(shape, np.nan, dtype=np.float64) for _ in range(3)]
    for target, source in zip(components, selected_components):
        target[:, selected, :] = source
    return tuple(components)


def _normalise_surface_indices(
    surface_indices: int | Sequence[int] | None,
    size: int,
) -> tuple[int, ...]:
    if surface_indices is None:
        return tuple(range(int(size)))
    raw = [int(surface_indices)] if np.isscalar(surface_indices) else [int(i) for i in surface_indices]
    if not raw:
        raise ValueError("surface_indices must select at least one surface")
    normalised: list[int] = []
    for value in raw:
        index = value + int(size) if value < 0 else value
        if index < 0 or index >= int(size):
            raise IndexError(f"surface index {value} is outside 0..{int(size) - 1}")
        if index not in normalised:
            normalised.append(index)
    return tuple(normalised)


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _solve_surface_straight_angle(
    surface_index: int,
    rho_value: float,
    theta_slope: np.ndarray,
    radial_slope: np.ndarray,
    normal_leakage: np.ndarray,
    bphi: np.ndarray,
    *,
    phi_period: float,
    theta_period: float,
    anchor_flat_index: int,
    lsmr_atol: float,
    lsmr_btol: float,
    lsmr_conlim: float,
    lsmr_maxiter: int | None,
    min_theta_jacobian: float,
    max_mde_relative_residual: float | None,
) -> tuple[int, np.ndarray, PestSurfaceStraightFieldLineDiagnostic]:
    a = np.asarray(theta_slope, dtype=np.float64)
    operator = _straight_field_line_linear_operator(
        a,
        phi_period=phi_period,
        theta_period=theta_period,
        anchor_flat_index=anchor_flat_index,
    )
    rhs = np.concatenate([-a.ravel(), [0.0]])
    solution = lsmr(
        operator,
        rhs,
        atol=float(lsmr_atol),
        btol=float(lsmr_btol),
        conlim=float(lsmr_conlim),
        maxiter=lsmr_maxiter,
    )
    u = np.asarray(solution[0][:-1], dtype=np.float64).reshape(a.shape)
    iota = float(solution[0][-1])
    # Make the requested phi=0/theta=0 gauge exact.  A constant shift is in the
    # nullspace of the MDE, so this does not change its residual.
    u -= float(u.ravel()[int(anchor_flat_index)])
    residual = (
        _periodic_fft_derivative(u, axis=0, period=phi_period)
        + a * _periodic_fft_derivative(u, axis=1, period=theta_period)
        - iota
        + a
    )
    residual_rms = _rms(residual)
    residual_relative = residual_rms / max(_rms(a), abs(iota), 1.0e-30)
    theta_jacobian = 1.0 + _periodic_fft_derivative(u, axis=1, period=theta_period)
    minimum_jacobian = float(np.min(theta_jacobian))
    maximum_jacobian = float(np.max(theta_jacobian))
    if not np.all(np.isfinite(u)) or not np.isfinite(iota):
        raise RuntimeError(f"surface {surface_index} straight-angle LSMR returned non-finite values")
    if minimum_jacobian <= float(min_theta_jacobian):
        raise RuntimeError(
            f"surface {surface_index} candidate PEST map is not a permitted diffeomorphism: "
            f"min dtheta_new/dtheta={minimum_jacobian:.6e} <= {float(min_theta_jacobian):.6e}"
        )
    if max_mde_relative_residual is not None and residual_relative > float(max_mde_relative_residual):
        raise RuntimeError(
            f"surface {surface_index} straight-angle MDE residual {residual_relative:.6e} exceeds "
            f"{float(max_mde_relative_residual):.6e}"
        )
    abs_radial = np.abs(np.asarray(radial_slope, dtype=np.float64))
    abs_normal = np.abs(np.asarray(normal_leakage, dtype=np.float64))
    diagnostic = PestSurfaceStraightFieldLineDiagnostic(
        surface_index=int(surface_index),
        rho=float(rho_value),
        iota=iota,
        mde_residual_rms=residual_rms,
        mde_residual_relative=float(residual_relative),
        mde_residual_max=float(np.max(np.abs(residual))),
        theta_correction_rms=_rms(u),
        theta_correction_max_abs=float(np.max(np.abs(u))),
        diffeomorphism_min_jacobian=minimum_jacobian,
        diffeomorphism_max_jacobian=maximum_jacobian,
        radial_leakage_rms=_rms(abs_radial),
        radial_leakage_p95=float(np.percentile(abs_radial, 95.0)),
        radial_leakage_max=float(np.max(abs_radial)),
        normal_leakage_rms=_rms(abs_normal),
        normal_leakage_p95=float(np.percentile(abs_normal, 95.0)),
        normal_leakage_max=float(np.max(abs_normal)),
        bphi_min_abs=float(np.min(np.abs(bphi))),
        lsmr_istop=int(solution[1]),
        lsmr_iterations=int(solution[2]),
        lsmr_condition_estimate=float(solution[6]),
    )
    return int(surface_index), u, diagnostic


def reparameterize_pest_on_candidate_field(
    base_coordinates: SmoothPestCoordinates,
    candidate_field: object,
    *,
    surface_indices: int | Sequence[int] | None = None,
    workers: int = 1,
    min_abs_bphi: float = 1.0e-10,
    min_relative_bphi: float = 1.0e-8,
    min_coordinate_jacobian: float = 1.0e-12,
    min_theta_jacobian: float = 1.0e-3,
    max_mde_relative_residual: float | None = 1.0e-7,
    lsmr_atol: float = 1.0e-11,
    lsmr_btol: float = 1.0e-11,
    lsmr_conlim: float = 1.0e12,
    lsmr_maxiter: int | None = None,
) -> CandidateFieldPestReparameterization:
    """Rebuild the straight field-line angle on an unchanged material mesh.

    ``base_coordinates`` supplies fixed pressure/material surfaces.  The
    candidate field is sampled in physical cylindrical components
    ``(B_R, B_Z, B_phi)`` through ``interpolate_at(R, Z, phi)`` or a callable
    with the same arguments (a callable accepting stacked ``[..., 3]`` points
    is also supported).  For each selected surface this routine computes the
    actual contravariant slopes and solves

    ``(d_phi + a d_theta) u - iota = -a``,  ``a = d_theta/d_phi``,

    with a matrix-free periodic Fourier discretisation and an exact discrete
    adjoint in LSMR.  The returned geometry is only re-sampled at uniform
    ``theta_new = theta + u``.  No magnetic-field healing, surface displacement,
    or radial projection is performed; measured radial leakage is retained as
    evidence.

    The default selects every surface.  Use ``surface_indices`` to exclude a
    coordinate-singular magnetic axis or to build plotting surfaces only.
    Independent surface solves can run concurrently with ``workers > 1``.
    """

    R = np.asarray(base_coordinates.R_surf, dtype=np.float64)
    Z = np.asarray(base_coordinates.Z_surf, dtype=np.float64)
    rho_values = np.asarray(base_coordinates.rho_vals, dtype=np.float64)
    theta_values = np.asarray(base_coordinates.theta_vals, dtype=np.float64)
    phi_values = np.asarray(base_coordinates.phi_vals, dtype=np.float64)
    if R.ndim != 3 or Z.shape != R.shape:
        raise ValueError("base PEST surfaces must have matching shape (phi, rho, theta)")
    n_phi, n_rho, n_theta = R.shape
    if rho_values.ndim != 1 or rho_values.size != n_rho or not np.all(np.isfinite(rho_values)):
        raise ValueError("rho_vals must be finite with one value per surface")
    if n_rho < 2 or np.any(np.diff(rho_values) <= 0.0):
        raise ValueError("at least two strictly increasing rho surfaces are required")
    period = float(getattr(base_coordinates, "period", TWOPI) or TWOPI)
    theta_values = _require_uniform_periodic_axis(
        theta_values, n_theta, period=TWOPI, name="theta_vals"
    )
    phi_values = _require_uniform_periodic_axis(
        phi_values, n_phi, period=period, name="phi_vals"
    )
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)) or np.any(R <= 0.0):
        raise ValueError("base PEST R/Z surfaces must be finite and R must be positive")
    selected = _normalise_surface_indices(surface_indices, n_rho)
    if int(workers) < 1:
        raise ValueError("workers must be at least one")
    for name, value in {
        "min_abs_bphi": min_abs_bphi,
        "min_relative_bphi": min_relative_bphi,
        "min_coordinate_jacobian": min_coordinate_jacobian,
        "min_theta_jacobian": min_theta_jacobian,
        "lsmr_atol": lsmr_atol,
        "lsmr_btol": lsmr_btol,
        "lsmr_conlim": lsmr_conlim,
    }.items():
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if max_mde_relative_residual is not None and (
        not np.isfinite(max_mde_relative_residual) or float(max_mde_relative_residual) <= 0.0
    ):
        raise ValueError("max_mde_relative_residual must be finite and positive or None")
    if lsmr_maxiter is not None and int(lsmr_maxiter) < 1:
        raise ValueError("lsmr_maxiter must be positive or None")

    R_rho = np.gradient(R, rho_values, axis=1, edge_order=2 if n_rho >= 3 else 1)
    Z_rho = np.gradient(Z, rho_values, axis=1, edge_order=2 if n_rho >= 3 else 1)
    R_theta = _periodic_fft_derivative(R, axis=2, period=TWOPI)
    Z_theta = _periodic_fft_derivative(Z, axis=2, period=TWOPI)
    R_phi = _periodic_fft_derivative(R, axis=0, period=period)
    Z_phi = _periodic_fft_derivative(Z, axis=0, period=period)
    phi_grid = np.broadcast_to(phi_values[:, None, None], R.shape)
    theta_grid = np.broadcast_to(theta_values[None, None, :], R.shape)
    BR, BZ, BPhi = _evaluate_candidate_field_on_surfaces(
        candidate_field, R, Z, phi_grid, theta_grid, selected
    )

    theta_slope = np.full(R.shape, np.nan, dtype=np.float64)
    radial_slope = np.full(R.shape, np.nan, dtype=np.float64)
    normal_leakage = np.full(R.shape, np.nan, dtype=np.float64)
    for surface_index in selected:
        ir = int(surface_index)
        field_components = (BR[:, ir, :], BZ[:, ir, :], BPhi[:, ir, :])
        if not all(np.all(np.isfinite(component)) for component in field_components):
            raise ValueError(f"candidate field is non-finite on surface {ir}")
        bphi = BPhi[:, ir, :]
        maximum_bphi = float(np.max(np.abs(bphi)))
        minimum_bphi = float(np.min(np.abs(bphi)))
        if maximum_bphi <= 0.0 or minimum_bphi <= float(min_abs_bphi):
            raise ValueError(
                f"candidate field Bphi is zero/too small on surface {ir}: min abs(Bphi)={minimum_bphi:.6e}"
            )
        if minimum_bphi / maximum_bphi <= float(min_relative_bphi):
            raise ValueError(
                f"candidate field Bphi is ill-conditioned on surface {ir}: min/max={minimum_bphi / maximum_bphi:.6e}"
            )
        if float(np.min(bphi)) < 0.0 < float(np.max(bphi)):
            raise ValueError(f"candidate field Bphi changes sign on surface {ir}")
        coordinate_determinant = (
            R_rho[:, ir, :] * Z_theta[:, ir, :]
            - R_theta[:, ir, :] * Z_rho[:, ir, :]
        )
        determinant_scale = np.sqrt(
            (R_rho[:, ir, :] ** 2 + Z_rho[:, ir, :] ** 2)
            * (R_theta[:, ir, :] ** 2 + Z_theta[:, ir, :] ** 2)
        )
        relative_determinant = np.abs(coordinate_determinant) / np.maximum(determinant_scale, 1.0e-300)
        if not np.all(np.isfinite(relative_determinant)) or float(np.min(relative_determinant)) <= float(min_coordinate_jacobian):
            raise ValueError(
                f"base material coordinates are singular on surface {ir}: "
                f"min relative RZ Jacobian={float(np.nanmin(relative_determinant)):.6e}"
            )
        phi_rate = bphi / R[:, ir, :]
        rhs_R = BR[:, ir, :] - phi_rate * R_phi[:, ir, :]
        rhs_Z = BZ[:, ir, :] - phi_rate * Z_phi[:, ir, :]
        rho_rate = (
            rhs_R * Z_theta[:, ir, :] - R_theta[:, ir, :] * rhs_Z
        ) / coordinate_determinant
        theta_rate = (
            R_rho[:, ir, :] * rhs_Z - rhs_R * Z_rho[:, ir, :]
        ) / coordinate_determinant
        radial_slope[:, ir, :] = rho_rate / phi_rate
        theta_slope[:, ir, :] = theta_rate / phi_rate

        normal_R = -Z_theta[:, ir, :] * R[:, ir, :]
        normal_Phi = (
            Z_theta[:, ir, :] * R_phi[:, ir, :]
            - R_theta[:, ir, :] * Z_phi[:, ir, :]
        )
        normal_Z = R_theta[:, ir, :] * R[:, ir, :]
        normal_norm = np.sqrt(normal_R**2 + normal_Phi**2 + normal_Z**2)
        field_norm = np.sqrt(BR[:, ir, :] ** 2 + bphi**2 + BZ[:, ir, :] ** 2)
        normal_leakage[:, ir, :] = np.abs(
            BR[:, ir, :] * normal_R + bphi * normal_Phi + BZ[:, ir, :] * normal_Z
        ) / (normal_norm * field_norm)
        if not (
            np.all(np.isfinite(theta_slope[:, ir, :]))
            and np.all(np.isfinite(radial_slope[:, ir, :]))
            and np.all(np.isfinite(normal_leakage[:, ir, :]))
        ):
            raise ValueError(f"candidate field coordinate slopes are non-finite on surface {ir}")

    phi_anchor = int(np.argmin(np.abs(np.angle(np.exp(1j * phi_values)))))
    theta_anchor = int(np.argmin(np.abs(np.angle(np.exp(1j * theta_values)))))
    anchor_flat_index = int(np.ravel_multi_index((phi_anchor, theta_anchor), (n_phi, n_theta)))
    solve_arguments = [
        (
            int(ir),
            float(rho_values[ir]),
            theta_slope[:, ir, :],
            radial_slope[:, ir, :],
            normal_leakage[:, ir, :],
            BPhi[:, ir, :],
        )
        for ir in selected
    ]

    def solve(arguments):
        return _solve_surface_straight_angle(
            *arguments,
            phi_period=period,
            theta_period=TWOPI,
            anchor_flat_index=anchor_flat_index,
            lsmr_atol=float(lsmr_atol),
            lsmr_btol=float(lsmr_btol),
            lsmr_conlim=float(lsmr_conlim),
            lsmr_maxiter=None if lsmr_maxiter is None else int(lsmr_maxiter),
            min_theta_jacobian=float(min_theta_jacobian),
            max_mde_relative_residual=max_mde_relative_residual,
        )

    if int(workers) == 1 or len(solve_arguments) == 1:
        solved = [solve(arguments) for arguments in solve_arguments]
    else:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(solve_arguments))) as pool:
            solved = list(pool.map(solve, solve_arguments))

    theta_correction = np.zeros(R.shape, dtype=np.float64)
    R_new = R.copy()
    Z_new = Z.copy()
    diagnostics: list[PestSurfaceStraightFieldLineDiagnostic] = []
    for surface_index, correction, diagnostic in solved:
        ir = int(surface_index)
        theta_correction[:, ir, :] = correction
        diagnostics.append(diagnostic)
        for iphi in range(n_phi):
            mapped_theta = theta_values + correction[iphi]
            R_new[iphi, ir] = stitch_periodic(mapped_theta, R[iphi, ir], theta_values)
            Z_new[iphi, ir] = stitch_periodic(mapped_theta, Z[iphi, ir], theta_values)
        if not np.all(np.isfinite(R_new[:, ir, :])) or not np.all(np.isfinite(Z_new[:, ir, :])):
            raise RuntimeError(f"surface {ir} periodic theta re-sampling produced non-finite geometry")

    original_source = getattr(base_coordinates, "source", None)
    source = (
        f"{original_source or 'in-memory SmoothPestCoordinates'} | "
        "candidate-field straight-angle reparameterization (B and material surfaces unchanged; no healing)"
    )
    rebuilt = SmoothPestCoordinates(
        R_surf=R_new,
        Z_surf=Z_new,
        rho_vals=rho_values.copy(),
        theta_vals=theta_values.copy(),
        phi_vals=phi_values.copy(),
        axis_R=None if base_coordinates.axis_R is None else np.asarray(base_coordinates.axis_R, dtype=np.float64).copy(),
        axis_Z=None if base_coordinates.axis_Z is None else np.asarray(base_coordinates.axis_Z, dtype=np.float64).copy(),
        source=source,
        nfp=int(getattr(base_coordinates, "nfp", 1)),
        toroidal_period=float(getattr(base_coordinates, "period", TWOPI) or TWOPI),
    )
    metadata = {
        "schema": "pyna_candidate_field_pest_reparameterization_v1",
        "construction": "fixed_material_surfaces_candidate_field_straight_angle",
        "field_modified": False,
        "surface_geometry_modified": False,
        "healing_used": False,
        "surface_layout": "phi_rho_theta",
        "nfp": int(getattr(base_coordinates, "nfp", 1)),
        "toroidal_domain_period_rad": float(period),
        "base_source": original_source,
        "workers": int(workers),
        "mde_discretization": "periodic_Fourier_matrix_free_LSMR_exact_adjoint",
        "theta_gauge": "u(phi_nearest_0,theta_nearest_0)=0",
    }
    return CandidateFieldPestReparameterization(
        coordinates=rebuilt,
        surface_diagnostics=tuple(sorted(diagnostics, key=lambda item: item.surface_index)),
        theta_correction=theta_correction,
        theta_slope=theta_slope,
        radial_slope=radial_slope,
        normal_leakage=normal_leakage,
        selected_surface_indices=selected,
        metadata=metadata,
    )


__all__ = [
    "AxisCoreInsertion",
    "CandidateFieldPestReparameterization",
    "PestSurfaceStraightFieldLineDiagnostic",
    "TWOPI",
    "circle_map_lift_iota",
    "insert_axis_core_surfaces",
    "lfs_theta_offset_at_phi0",
    "periodic_shift_theta",
    "reparameterize_pest_on_candidate_field",
    "rank_phase_from_axis",
    "sanitized_axis_core_fractions",
    "stitch_periodic",
    "straight_field_line_mde_adjoint_error",
    "theta_coverage",
]
