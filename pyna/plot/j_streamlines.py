"""PEST-seeded current streamline plotting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyna.fields import VectorFieldCylind
from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates, smooth_pest_derivatives


TWOPI = 2.0 * np.pi


def _cylindrical_to_cartesian(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    R_arr, Z_arr, phi_arr = np.broadcast_arrays(
        np.asarray(R, dtype=np.float64),
        np.asarray(Z, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
    )
    return np.stack([R_arr * np.cos(phi_arr), R_arr * np.sin(phi_arr), Z_arr], axis=-1)


@dataclass(frozen=True)
class PestSeededStreamlines:
    """Current streamlines seeded from a PEST surface mesh."""

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    seed_R: np.ndarray
    seed_Z: np.ndarray
    seed_phi: np.ndarray
    seed_rho: np.ndarray
    seed_theta: np.ndarray
    seed_surface_index: np.ndarray
    seed_phi_index: np.ndarray
    metadata: dict[str, object]

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z], axis=-1)

    @property
    def n_lines(self) -> int:
        return int(self.R.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.R.shape[1]) if self.R.ndim == 2 else 0


@dataclass(frozen=True)
class _PeriodicVectorFieldEvaluator:
    R: np.ndarray
    Z: np.ndarray
    Phi: np.ndarray
    nfp: int
    field_period_rad: float
    interp_R: RegularGridInterpolator
    interp_Z: RegularGridInterpolator
    interp_Phi: RegularGridInterpolator

    @classmethod
    def from_field(cls, field: VectorFieldCylind) -> "_PeriodicVectorFieldEvaluator":
        arrays = field.cyna_arrays(extend_phi=True)
        R = np.asarray(arrays.R_grid, dtype=np.float64)
        Z = np.asarray(arrays.Z_grid, dtype=np.float64)
        Phi = np.asarray(arrays.Phi_grid, dtype=np.float64)
        nfp = max(int(getattr(field, "nfp", arrays.nfp)), 1)
        field_period_rad = float(TWOPI / nfp)
        kw = dict(method="linear", bounds_error=False, fill_value=np.nan)
        axes = (R, Z, Phi)
        return cls(
            R=R,
            Z=Z,
            Phi=Phi,
            nfp=nfp,
            field_period_rad=field_period_rad,
            interp_R=RegularGridInterpolator(axes, np.asarray(arrays.BR, dtype=np.float64), **kw),
            interp_Z=RegularGridInterpolator(axes, np.asarray(arrays.BZ, dtype=np.float64), **kw),
            interp_Phi=RegularGridInterpolator(axes, np.asarray(arrays.BPhi, dtype=np.float64), **kw),
        )

    def __call__(self, R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
        phi0 = float(self.Phi[0]) if self.Phi.size else 0.0
        phi_eval = phi0 + np.mod(np.asarray(phi, dtype=np.float64) - phi0, self.field_period_rad)
        pts = np.stack([R, Z, phi_eval], axis=-1)
        return np.stack(
            [
                self.interp_R(pts),
                self.interp_Z(pts),
                self.interp_Phi(pts),
            ],
            axis=-1,
        )


def _pest_from_mapping(pest: Mapping[str, object], *, source: str | None = None) -> SmoothPestCoordinates:
    radial = pest.get("rho_vals", pest.get("radial_labels", pest.get("r_vals")))
    if radial is None:
        raise KeyError("PEST coordinates require rho_vals, radial_labels, or r_vals")
    axis_R = pest.get("axis_R", pest.get("R_AX"))
    axis_Z = pest.get("axis_Z", pest.get("Z_AX"))
    src = source if source is not None else str(pest.get("source", "")) or None
    return SmoothPestCoordinates(
        R_surf=np.asarray(pest["R_surf"], dtype=np.float64),
        Z_surf=np.asarray(pest["Z_surf"], dtype=np.float64),
        rho_vals=np.asarray(radial, dtype=np.float64),
        theta_vals=np.asarray(pest["theta_vals"], dtype=np.float64),
        phi_vals=np.asarray(pest["phi_vals"], dtype=np.float64),
        axis_R=np.asarray(axis_R, dtype=np.float64) if axis_R is not None else None,
        axis_Z=np.asarray(axis_Z, dtype=np.float64) if axis_Z is not None else None,
        source=src,
    )


def _as_pest_coordinates(pest: SmoothPestCoordinates | Mapping[str, object] | str | Path) -> SmoothPestCoordinates:
    if isinstance(pest, SmoothPestCoordinates):
        return pest
    if isinstance(pest, (str, Path)):
        path = Path(pest).expanduser()
        with np.load(path, allow_pickle=False) as data:
            payload = {name: data[name] for name in data.files}
        return _pest_from_mapping(payload, source=str(path))
    if isinstance(pest, Mapping):
        return _pest_from_mapping(pest)
    raise TypeError("pest must be SmoothPestCoordinates, mapping, or path")


def _validate_pest(pest: SmoothPestCoordinates) -> None:
    R = np.asarray(pest.R_surf, dtype=np.float64)
    Z = np.asarray(pest.Z_surf, dtype=np.float64)
    if R.shape != Z.shape or R.ndim != 3:
        raise ValueError("PEST R_surf and Z_surf must have shape (n_phi, n_rho, n_theta)")
    if np.asarray(pest.phi_vals).ndim != 1 or np.asarray(pest.phi_vals).size != R.shape[0]:
        raise ValueError("PEST phi_vals must be one-dimensional and match R_surf axis 0")
    if np.asarray(pest.rho_vals).ndim != 1 or np.asarray(pest.rho_vals).size != R.shape[1]:
        raise ValueError("PEST rho_vals must be one-dimensional and match R_surf axis 1")
    if np.asarray(pest.theta_vals).ndim != 1 or np.asarray(pest.theta_vals).size != R.shape[2]:
        raise ValueError("PEST theta_vals must be one-dimensional and match R_surf axis 2")


def _normalize_indices(index: int | Sequence[int] | None, size: int, *, default: Sequence[int]) -> np.ndarray:
    raw = np.asarray(default if index is None else ([index] if np.isscalar(index) else list(index)), dtype=np.int64)
    if raw.size == 0:
        raise ValueError("index selection must not be empty")
    return np.asarray([int(i) % int(size) for i in raw], dtype=np.int64)


def _wrap_angle_near_reference(angle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return reference + np.mod(angle - reference + np.pi, TWOPI) - np.pi


def _nan_stat(values: list[np.ndarray], *, percentile: float | None = None) -> float:
    if not values:
        return float("nan")
    arr = np.concatenate([np.ravel(np.asarray(v, dtype=np.float64)) for v in values])
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if percentile is None:
        return float(np.nanmedian(finite))
    return float(np.nanpercentile(finite, float(percentile)))


def _periodic_surface_bilinear(
    values: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    *,
    phi0: float,
    theta0: float,
    phi_period: float,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 2:
        raise ValueError("surface values must have shape (n_phi, n_theta)")
    n_phi, n_theta = vals.shape
    if n_phi < 2 or n_theta < 2:
        raise ValueError("surface interpolation requires at least two phi and theta points")
    phi_arr, theta_arr = np.broadcast_arrays(
        np.asarray(phi, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
    )
    u_phi = np.mod(phi_arr - float(phi0), float(phi_period)) * (float(n_phi) / float(phi_period))
    f_phi = np.floor(u_phi)
    i0 = np.asarray(f_phi, dtype=np.int64) % n_phi
    i1 = (i0 + 1) % n_phi
    a = u_phi - f_phi

    u_theta = np.mod(theta_arr - float(theta0), TWOPI) * (float(n_theta) / TWOPI)
    f_theta = np.floor(u_theta)
    j0 = np.asarray(f_theta, dtype=np.int64) % n_theta
    j1 = (j0 + 1) % n_theta
    b = u_theta - f_theta

    return (
        (1.0 - a) * (1.0 - b) * vals[i0, j0]
        + a * (1.0 - b) * vals[i1, j0]
        + (1.0 - a) * b * vals[i0, j1]
        + a * b * vals[i1, j1]
    )


@dataclass(frozen=True)
class _PestSurfaceEvaluator:
    R: np.ndarray
    Z: np.ndarray
    dR_dtheta: np.ndarray
    dZ_dtheta: np.ndarray
    dR_dphi: np.ndarray
    dZ_dphi: np.ndarray
    phi0: float
    theta0: float
    phi_period: float

    @classmethod
    def from_pest(cls, pest: SmoothPestCoordinates, surface_index: int) -> "_PestSurfaceEvaluator":
        deriv = smooth_pest_derivatives(pest)
        ir = int(surface_index) % pest.R_surf.shape[1]
        phi_vals = np.asarray(pest.phi_vals, dtype=np.float64)
        theta_vals = np.asarray(pest.theta_vals, dtype=np.float64)
        return cls(
            R=np.asarray(pest.R_surf[:, ir, :], dtype=np.float64),
            Z=np.asarray(pest.Z_surf[:, ir, :], dtype=np.float64),
            dR_dtheta=np.asarray(deriv[2][:, ir, :], dtype=np.float64),
            dZ_dtheta=np.asarray(deriv[3][:, ir, :], dtype=np.float64),
            dR_dphi=np.asarray(deriv[4][:, ir, :], dtype=np.float64),
            dZ_dphi=np.asarray(deriv[5][:, ir, :], dtype=np.float64),
            phi0=float(phi_vals[0]) if phi_vals.size else 0.0,
            theta0=float(theta_vals[0]) if theta_vals.size else 0.0,
            phi_period=float(getattr(pest, "period", TWOPI) or TWOPI),
        )

    def evaluate(self, phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, ...]:
        kwargs = dict(
            phi0=self.phi0,
            theta0=self.theta0,
            phi_period=self.phi_period,
        )
        return (
            _periodic_surface_bilinear(self.R, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.Z, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.dR_dtheta, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.dZ_dtheta, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.dR_dphi, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.dZ_dphi, phi, theta, **kwargs),
        )


def _cartesian_rhs(
    field_eval: _PeriodicVectorFieldEvaluator,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    min_field_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    values = np.asarray(field_eval(R, z, phi), dtype=np.float64)
    vR = values[..., 0]
    vZ = values[..., 1]
    vPhi = values[..., 2]
    norm = np.sqrt(vR * vR + vZ * vZ + vPhi * vPhi)
    safe = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(vR)
        & np.isfinite(vZ)
        & np.isfinite(vPhi)
        & (norm > float(min_field_norm))
        & (R > 1.0e-12)
    )
    dx = np.full_like(x, np.nan, dtype=np.float64)
    dy = np.full_like(y, np.nan, dtype=np.float64)
    dz = np.full_like(z, np.nan, dtype=np.float64)
    if np.any(safe):
        cp = np.cos(phi[safe])
        sp = np.sin(phi[safe])
        vx = vR[safe] * cp - vPhi[safe] * sp
        vy = vR[safe] * sp + vPhi[safe] * cp
        dx[safe] = vx / norm[safe]
        dy[safe] = vy / norm[safe]
        dz[safe] = vZ[safe] / norm[safe]
    return dx, dy, dz


def _rk4_step_cartesian(
    field_eval: _PeriodicVectorFieldEvaluator,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    phi_ref: np.ndarray,
    *,
    h: float,
    min_field_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1x, k1y, k1z = _cartesian_rhs(field_eval, x, y, z, min_field_norm=min_field_norm)
    k2x, k2y, k2z = _cartesian_rhs(
        field_eval,
        x + 0.5 * h * k1x,
        y + 0.5 * h * k1y,
        z + 0.5 * h * k1z,
        min_field_norm=min_field_norm,
    )
    k3x, k3y, k3z = _cartesian_rhs(
        field_eval,
        x + 0.5 * h * k2x,
        y + 0.5 * h * k2y,
        z + 0.5 * h * k2z,
        min_field_norm=min_field_norm,
    )
    k4x, k4y, k4z = _cartesian_rhs(
        field_eval,
        x + h * k3x,
        y + h * k3y,
        z + h * k3z,
        min_field_norm=min_field_norm,
    )
    out_x = x + h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    out_y = y + h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    out_z = z + h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    out_phi = _wrap_angle_near_reference(np.arctan2(out_y, out_x), phi_ref)
    return out_x, out_y, out_z, out_phi


def _pest_surface_rhs_single(
    surface_eval: _PestSurfaceEvaluator,
    field_eval: _PeriodicVectorFieldEvaluator,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    min_field_norm: float,
    min_tangent_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R, Z, dR_dtheta, dZ_dtheta, dR_dphi, dZ_dphi = surface_eval.evaluate(phi, theta)
    values = np.asarray(field_eval(R, Z, phi), dtype=np.float64)
    jR = values[..., 0]
    jZ = values[..., 1]
    jPhi = values[..., 2]

    cp = np.cos(phi)
    sp = np.sin(phi)
    jx = jR * cp - jPhi * sp
    jy = jR * sp + jPhi * cp
    jz = jZ

    e_theta = np.stack([dR_dtheta * cp, dR_dtheta * sp, dZ_dtheta], axis=-1)
    e_phi = np.stack([dR_dphi * cp - R * sp, dR_dphi * sp + R * cp, dZ_dphi], axis=-1)
    j_cart = np.stack([jx, jy, jz], axis=-1)

    gtt = np.sum(e_theta * e_theta, axis=-1)
    gtp = np.sum(e_theta * e_phi, axis=-1)
    gpp = np.sum(e_phi * e_phi, axis=-1)
    rhs_t = np.sum(j_cart * e_theta, axis=-1)
    rhs_p = np.sum(j_cart * e_phi, axis=-1)
    det = gtt * gpp - gtp * gtp

    j_norm = np.sqrt(np.sum(j_cart * j_cart, axis=-1))
    normal = np.cross(e_theta, e_phi)
    normal_norm = np.sqrt(np.sum(normal * normal, axis=-1))
    normal_unit = normal / np.where(normal_norm > 0.0, normal_norm, np.nan)[..., None]
    normal_leakage = np.abs(np.sum(j_cart * normal_unit, axis=-1)) / j_norm

    dtheta = np.full_like(theta, np.nan, dtype=np.float64)
    dphi = np.full_like(phi, np.nan, dtype=np.float64)
    tangent_fraction = np.full_like(theta, np.nan, dtype=np.float64)
    safe = (
        np.isfinite(theta)
        & np.isfinite(phi)
        & np.isfinite(j_norm)
        & np.isfinite(det)
        & (j_norm > float(min_field_norm))
        & (np.abs(det) > 1.0e-28)
    )
    if np.any(safe):
        a = np.full_like(theta, np.nan, dtype=np.float64)
        b = np.full_like(phi, np.nan, dtype=np.float64)
        a[safe] = (rhs_t[safe] * gpp[safe] - rhs_p[safe] * gtp[safe]) / det[safe]
        b[safe] = (gtt[safe] * rhs_p[safe] - gtp[safe] * rhs_t[safe]) / det[safe]
        tangent = a[..., None] * e_theta + b[..., None] * e_phi
        tangent_norm = np.sqrt(np.sum(tangent * tangent, axis=-1))
        good = safe & np.isfinite(tangent_norm) & (tangent_norm > float(min_tangent_norm))
        dtheta[good] = a[good] / tangent_norm[good]
        dphi[good] = b[good] / tangent_norm[good]
        tangent_fraction[good] = tangent_norm[good] / j_norm[good]
    normal_leakage[~np.isfinite(normal_leakage)] = np.nan
    return dtheta, dphi, normal_leakage, tangent_fraction


def _pest_surface_rhs(
    surface_evals: Mapping[int, _PestSurfaceEvaluator],
    surface_indices: np.ndarray,
    field_eval: _PeriodicVectorFieldEvaluator,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    min_field_norm: float,
    min_tangent_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtheta = np.full_like(theta, np.nan, dtype=np.float64)
    dphi = np.full_like(phi, np.nan, dtype=np.float64)
    leakage = np.full_like(theta, np.nan, dtype=np.float64)
    tangent_fraction = np.full_like(theta, np.nan, dtype=np.float64)
    for ir, surface_eval in surface_evals.items():
        selected = surface_indices == int(ir)
        if not np.any(selected):
            continue
        dth, dph, leak, tfrac = _pest_surface_rhs_single(
            surface_eval,
            field_eval,
            theta[selected],
            phi[selected],
            min_field_norm=min_field_norm,
            min_tangent_norm=min_tangent_norm,
        )
        dtheta[selected] = dth
        dphi[selected] = dph
        leakage[selected] = leak
        tangent_fraction[selected] = tfrac
    return dtheta, dphi, leakage, tangent_fraction


def _rk4_step_pest_surface(
    surface_evals: Mapping[int, _PestSurfaceEvaluator],
    surface_indices: np.ndarray,
    field_eval: _PeriodicVectorFieldEvaluator,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    h: float,
    min_field_norm: float,
    min_tangent_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1t, k1p, leakage, tangent_fraction = _pest_surface_rhs(
        surface_evals,
        surface_indices,
        field_eval,
        theta,
        phi,
        min_field_norm=min_field_norm,
        min_tangent_norm=min_tangent_norm,
    )
    k2t, k2p, _, _ = _pest_surface_rhs(
        surface_evals,
        surface_indices,
        field_eval,
        theta + 0.5 * h * k1t,
        phi + 0.5 * h * k1p,
        min_field_norm=min_field_norm,
        min_tangent_norm=min_tangent_norm,
    )
    k3t, k3p, _, _ = _pest_surface_rhs(
        surface_evals,
        surface_indices,
        field_eval,
        theta + 0.5 * h * k2t,
        phi + 0.5 * h * k2p,
        min_field_norm=min_field_norm,
        min_tangent_norm=min_tangent_norm,
    )
    k4t, k4p, _, _ = _pest_surface_rhs(
        surface_evals,
        surface_indices,
        field_eval,
        theta + h * k3t,
        phi + h * k3p,
        min_field_norm=min_field_norm,
        min_tangent_norm=min_tangent_norm,
    )
    out_theta = theta + h * (k1t + 2.0 * k2t + 2.0 * k3t + k4t) / 6.0
    out_phi = phi + h * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
    return out_theta, out_phi, leakage, tangent_fraction


def _surface_points_from_theta_phi(
    surface_evals: Mapping[int, _PestSurfaceEvaluator],
    surface_indices: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    R = np.full_like(theta, np.nan, dtype=np.float64)
    Z = np.full_like(theta, np.nan, dtype=np.float64)
    for ir, surface_eval in surface_evals.items():
        selected = surface_indices == int(ir)
        if not np.any(selected):
            continue
        R[selected], Z[selected], *_ = surface_eval.evaluate(phi[selected], theta[selected])
    return R, Z


def _seed_points(
    pest: SmoothPestCoordinates,
    *,
    surface_indices: np.ndarray,
    phi_indices: np.ndarray,
    seed_count: int,
    theta_offset: float,
) -> dict[str, np.ndarray]:
    n_theta = pest.R_surf.shape[2]
    if int(seed_count) <= 0:
        raise ValueError("seed_count must be positive")
    theta_base = np.linspace(0, n_theta, int(seed_count), endpoint=False, dtype=np.float64)
    theta_indices = np.mod(np.rint(theta_base + float(theta_offset) * n_theta / TWOPI).astype(np.int64), n_theta)
    rows = []
    for iphi in phi_indices:
        for irho in surface_indices:
            for itheta in theta_indices:
                R0 = float(pest.R_surf[int(iphi), int(irho), int(itheta)])
                Z0 = float(pest.Z_surf[int(iphi), int(irho), int(itheta)])
                if np.isfinite(R0) and np.isfinite(Z0):
                    rows.append(
                        (
                            R0,
                            Z0,
                            float(pest.phi_vals[int(iphi)]),
                            float(pest.rho_vals[int(irho)]),
                            float(pest.theta_vals[int(itheta)]),
                            int(irho),
                            int(iphi),
                        )
                    )
    if not rows:
        raise ValueError("no finite PEST seed points were found")
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "R": arr[:, 0],
        "Z": arr[:, 1],
        "phi": arr[:, 2],
        "rho": arr[:, 3],
        "theta": arr[:, 4],
        "surface_index": arr[:, 5].astype(np.int64),
        "phi_index": arr[:, 6].astype(np.int64),
    }


def trace_j_streamlines_on_pest(
    field: VectorFieldCylind,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    *,
    surface_index: int | Sequence[int] | None = -1,
    phi_indices: Sequence[int] | None = None,
    seed_count: int = 12,
    n_turns: float = 0.2,
    steps_per_turn: int = 512,
    bidirectional: bool = True,
    theta_offset: float = 0.0,
    min_field_norm: float = 1.0e-14,
    min_tangent_norm: float | None = None,
    constrain_to_surface: bool = True,
) -> PestSeededStreamlines:
    """Trace current streamlines from PEST-surface seeds.

    ``field`` is a :class:`pyna.fields.VectorFieldCylind`; it is treated as the
    full current-density vector in physical cylindrical components
    ``(J_R, J_Z, J_phi)``.  By default, streamlines are constrained to the
    selected PEST surfaces by projecting ``J`` onto the surface tangent plane
    before stepping in ``(theta, phi)``.  Set ``constrain_to_surface=False`` only
    for raw interpolation diagnostics.
    """

    if not isinstance(field, VectorFieldCylind):
        raise TypeError("field must be a VectorFieldCylind")
    coords = _as_pest_coordinates(pest)
    _validate_pest(coords)
    n_phi, n_rho, _n_theta = coords.R_surf.shape
    default_phi = [0] if n_phi < 4 else [0, n_phi // 4, n_phi // 2, (3 * n_phi) // 4]
    phi_idx = _normalize_indices(phi_indices, n_phi, default=default_phi)
    surf_idx = _normalize_indices(surface_index, n_rho, default=[n_rho - 1])
    seeds = _seed_points(
        coords,
        surface_indices=surf_idx,
        phi_indices=phi_idx,
        seed_count=int(seed_count),
        theta_offset=float(theta_offset),
    )

    seed_R = seeds["R"]
    seed_Z = seeds["Z"]
    seed_phi = seeds["phi"]
    n_seed = seed_R.size
    field_eval = _PeriodicVectorFieldEvaluator.from_field(field)
    n_steps = max(int(round(float(n_turns) * max(int(steps_per_turn), 1))), 1)
    n_points = 2 * n_steps + 1 if bidirectional else n_steps + 1
    h_base = TWOPI * max(float(np.nanmedian(seed_R)), 1.0e-12) / max(int(steps_per_turn), 1)
    tangent_floor = float(min_field_norm if min_tangent_norm is None else min_tangent_norm)
    trajectory_leakage_samples: list[np.ndarray] = []
    trajectory_tangent_fraction_samples: list[np.ndarray] = []
    surface_evals = {
        int(ir): _PestSurfaceEvaluator.from_pest(coords, int(ir))
        for ir in np.unique(seeds["surface_index"])
    }
    _, _, seed_leakage, seed_tangent_fraction = _pest_surface_rhs(
        surface_evals,
        seeds["surface_index"],
        field_eval,
        np.asarray(seeds["theta"], dtype=np.float64),
        seed_phi,
        min_field_norm=float(min_field_norm),
        min_tangent_norm=tangent_floor,
    )
    section_leakage_samples: list[np.ndarray] = []
    section_tangent_fraction_samples: list[np.ndarray] = []
    theta_grid = np.asarray(coords.theta_vals, dtype=np.float64)
    for ir in surf_idx:
        surface_eval = surface_evals[int(ir)]
        for ip in phi_idx:
            phi_grid = np.full_like(theta_grid, float(coords.phi_vals[int(ip)]), dtype=np.float64)
            _, _, leakage, tangent_fraction = _pest_surface_rhs_single(
                surface_eval,
                field_eval,
                theta_grid,
                phi_grid,
                min_field_norm=float(min_field_norm),
                min_tangent_norm=tangent_floor,
            )
            section_leakage_samples.append(leakage)
            section_tangent_fraction_samples.append(tangent_fraction)

    if constrain_to_surface:
        def integrate_surface(direction: float, count: int) -> tuple[np.ndarray, np.ndarray]:
            theta = np.asarray(seeds["theta"], dtype=np.float64).copy()
            phi = seed_phi.copy()
            thetas = [theta.copy()]
            phis = [phi.copy()]
            for _ in range(count):
                theta, phi, leakage, tangent_fraction = _rk4_step_pest_surface(
                    surface_evals,
                    seeds["surface_index"],
                    field_eval,
                    theta,
                    phi,
                    h=float(direction) * h_base,
                    min_field_norm=float(min_field_norm),
                    min_tangent_norm=tangent_floor,
                )
                trajectory_leakage_samples.append(leakage)
                trajectory_tangent_fraction_samples.append(tangent_fraction)
                thetas.append(theta.copy())
                phis.append(phi.copy())
            return np.stack(thetas, axis=1), np.stack(phis, axis=1)

        if bidirectional:
            btheta, bphi = integrate_surface(-1.0, n_steps)
            ftheta, fphi = integrate_surface(1.0, n_steps)
            theta = np.concatenate([btheta[:, :0:-1], ftheta], axis=1)
            phi = np.concatenate([bphi[:, :0:-1], fphi], axis=1)
        else:
            theta, phi = integrate_surface(1.0, n_steps)
        R, z = _surface_points_from_theta_phi(
            surface_evals,
            seeds["surface_index"],
            theta,
            phi,
        )
        xyz = _cylindrical_to_cartesian(R, z, phi)
        x = xyz[..., 0]
        y = xyz[..., 1]
        trace_backend = "pyna.plot.j_streamlines.python_rk4_pest_surface_arclength"
        trace_parameter = "normalized_pest_surface_arclength"
        trace_mode = "pest_surface_constrained"
    else:
        theta = np.full((n_seed, n_points), np.nan, dtype=np.float64)

        def integrate_cartesian(direction: float, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x0 = seed_R * np.cos(seed_phi)
            y0 = seed_R * np.sin(seed_phi)
            z0 = seed_Z.copy()
            phi0 = seed_phi.copy()
            xs = [x0.copy()]
            ys = [y0.copy()]
            zs = [z0.copy()]
            phis = [phi0.copy()]
            x_curr, y_curr, z_curr, phi_curr = x0, y0, z0, phi0
            for _ in range(count):
                x_curr, y_curr, z_curr, phi_curr = _rk4_step_cartesian(
                    field_eval,
                    x_curr,
                    y_curr,
                    z_curr,
                    phi_curr,
                    h=float(direction) * h_base,
                    min_field_norm=float(min_field_norm),
                )
                xs.append(x_curr.copy())
                ys.append(y_curr.copy())
                zs.append(z_curr.copy())
                phis.append(phi_curr.copy())
            return np.stack(xs, axis=1), np.stack(ys, axis=1), np.stack(zs, axis=1), np.stack(phis, axis=1)

        if bidirectional:
            bx, by, bz, bphi = integrate_cartesian(-1.0, n_steps)
            fx, fy, fz, fphi = integrate_cartesian(1.0, n_steps)
            x = np.concatenate([bx[:, :0:-1], fx], axis=1)
            y = np.concatenate([by[:, :0:-1], fy], axis=1)
            z = np.concatenate([bz[:, :0:-1], fz], axis=1)
            phi = np.concatenate([bphi[:, :0:-1], fphi], axis=1)
        else:
            x, y, z, phi = integrate_cartesian(1.0, n_steps)
        R = np.sqrt(x * x + y * y)
        trace_backend = "pyna.plot.j_streamlines.python_rk4_cartesian_arclength"
        trace_parameter = "normalized_cartesian_arclength"
        trace_mode = "raw_cartesian_unconstrained"

    metadata: dict[str, object] = {
        "schema": "pyna_pest_seeded_j_streamlines_v1",
        "trace_backend": trace_backend,
        "trace_mode": trace_mode,
        "trace_parameter": trace_parameter,
        "surface_constraint": bool(constrain_to_surface),
        "field_type": type(field).__name__,
        "nfp": int(field_eval.nfp),
        "field_period_rad": float(field_eval.field_period_rad),
        "pest_source": coords.source,
        "n_turns": float(n_turns),
        "steps_per_turn": int(steps_per_turn),
        "seed_count": int(seed_count),
        "n_seed_lines": int(n_seed),
        "n_points": int(n_points),
        "bidirectional": bool(bidirectional),
        "surface_indices": [int(i) for i in surf_idx],
        "phi_indices": [int(i) for i in phi_idx],
        "finite_fraction": float(np.count_nonzero(np.isfinite(R) & np.isfinite(z)) / max(R.size, 1)),
        "normal_leakage_abs_over_norm_median": _nan_stat(section_leakage_samples),
        "normal_leakage_abs_over_norm_p95": _nan_stat(section_leakage_samples, percentile=95.0),
        "surface_tangent_fraction_median": _nan_stat(section_tangent_fraction_samples),
        "surface_tangent_fraction_p05": _nan_stat(section_tangent_fraction_samples, percentile=5.0),
        "seed_normal_leakage_abs_over_norm_median": _nan_stat([seed_leakage]),
        "seed_normal_leakage_abs_over_norm_p95": _nan_stat([seed_leakage], percentile=95.0),
        "seed_surface_tangent_fraction_median": _nan_stat([seed_tangent_fraction]),
        "seed_surface_tangent_fraction_p05": _nan_stat([seed_tangent_fraction], percentile=5.0),
        "trajectory_normal_leakage_abs_over_norm_median": _nan_stat(trajectory_leakage_samples),
        "trajectory_normal_leakage_abs_over_norm_p95": _nan_stat(trajectory_leakage_samples, percentile=95.0),
        "trajectory_surface_tangent_fraction_median": _nan_stat(trajectory_tangent_fraction_samples),
        "trajectory_surface_tangent_fraction_p05": _nan_stat(trajectory_tangent_fraction_samples, percentile=5.0),
    }
    return PestSeededStreamlines(
        R=R,
        Z=z,
        phi=phi,
        theta=theta,
        x=x,
        y=y,
        z=z,
        seed_R=seed_R,
        seed_Z=seed_Z,
        seed_phi=seed_phi,
        seed_rho=seeds["rho"],
        seed_theta=seeds["theta"],
        seed_surface_index=seeds["surface_index"],
        seed_phi_index=seeds["phi_index"],
        metadata=metadata,
    )


def _surface_xyz(
    pest: SmoothPestCoordinates,
    surface_index: int,
    *,
    downsample: int = 1,
) -> np.ndarray:
    step = max(1, int(downsample))
    ir = int(surface_index) % pest.R_surf.shape[1]
    R = np.asarray(pest.R_surf[:, ir, :], dtype=np.float64)[::step, ::step]
    Z = np.asarray(pest.Z_surf[:, ir, :], dtype=np.float64)[::step, ::step]
    phi = np.asarray(pest.phi_vals, dtype=np.float64)[::step, np.newaxis]
    return _cylindrical_to_cartesian(R, Z, phi)


def plot_j_streamlines_on_pest_surface_plotly(
    streamlines: PestSeededStreamlines,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path | None = None,
    *,
    surface_index: int | None = None,
    html_path: str | Path | None = None,
    include_plotlyjs: str | bool = "cdn",
    show_surface: bool = True,
    surface_downsample: int = 1,
    surface_opacity: float = 0.24,
    line_width: float = 4.0,
    title: str | None = None,
    width: int = 1100,
    height: int = 850,
):
    """Return a Plotly 3-D view of J streamlines seeded on a PEST surface."""

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.colors import sample_colorscale
    except ImportError as exc:
        raise ImportError("plotly is required for plot_j_streamlines_on_pest_surface_plotly") from exc

    coords = _as_pest_coordinates(pest) if pest is not None else None
    fig = go.Figure()
    if show_surface and coords is not None:
        ir = (
            int(surface_index) % coords.R_surf.shape[1]
            if surface_index is not None
            else int(streamlines.seed_surface_index[0]) % coords.R_surf.shape[1]
        )
        xyz = _surface_xyz(coords, ir, downsample=surface_downsample)
        fig.add_trace(
            go.Surface(
                x=xyz[:, :, 0],
                y=xyz[:, :, 1],
                z=xyz[:, :, 2],
                showscale=False,
                opacity=float(surface_opacity),
                colorscale=[[0.0, "rgb(185,190,198)"], [1.0, "rgb(185,190,198)"]],
                hoverinfo="skip",
                name="PEST surface",
            )
        )

    finite_rho = streamlines.seed_rho[np.isfinite(streamlines.seed_rho)]
    rho_min = float(np.nanmin(finite_rho)) if finite_rho.size else 0.0
    rho_max = float(np.nanmax(finite_rho)) if finite_rho.size else 1.0
    if rho_max <= rho_min:
        rho_max = rho_min + 1.0
    for line_idx in range(streamlines.n_lines):
        keep = (
            np.isfinite(streamlines.x[line_idx])
            & np.isfinite(streamlines.y[line_idx])
            & np.isfinite(streamlines.z[line_idx])
        )
        if np.count_nonzero(keep) < 2:
            continue
        color_value = (float(streamlines.seed_rho[line_idx]) - rho_min) / (rho_max - rho_min)
        color = sample_colorscale("Viridis", color_value)[0]
        fig.add_trace(
            go.Scatter3d(
                x=streamlines.x[line_idx, keep],
                y=streamlines.y[line_idx, keep],
                z=streamlines.z[line_idx, keep],
                mode="lines",
                line=dict(color=color, width=float(line_width)),
                name=f"J line {line_idx}",
                hovertemplate=(
                    "seed rho=%{customdata[0]:.3f}<br>"
                    "seed theta=%{customdata[1]:.3f}<br>"
                    "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [
                        np.full(np.count_nonzero(keep), streamlines.seed_rho[line_idx]),
                        np.full(np.count_nonzero(keep), streamlines.seed_theta[line_idx]),
                    ]
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=-1.4, z=0.9), up=dict(x=0.0, y=0.0, z=1.0)),
        ),
        width=int(width),
        height=int(height),
        margin=dict(l=0, r=0, b=0, t=55),
    )
    if html_path is not None:
        pio.write_html(fig, str(Path(html_path)), include_plotlyjs=include_plotlyjs, auto_open=False)
    return fig


def plot_j_streamline_seed_sections(
    streamlines: PestSeededStreamlines,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    *,
    section_indices: Sequence[int] | None = None,
    title: str | None = None,
    line_width: float = 1.1,
    alpha: float = 0.86,
):
    """Plot R/Z projections of PEST-seeded J streamlines on seed sections."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from pyna.plot.mgrid import draw_smooth_pest_grid

    coords = _as_pest_coordinates(pest)
    _validate_pest(coords)
    if section_indices is None:
        section_indices = sorted({int(i) for i in streamlines.seed_phi_index})
    sec = np.asarray(list(section_indices), dtype=np.int64)
    if sec.size == 0:
        raise ValueError("section_indices must not be empty")
    ncols = min(int(sec.size), 4)
    nrows = int(np.ceil(sec.size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.0 * nrows), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    rho_min = float(np.nanmin(streamlines.seed_rho))
    rho_max = float(np.nanmax(streamlines.seed_rho))
    if rho_max <= rho_min:
        rho_max = rho_min + 1.0
    use_pest_projection = (
        streamlines.metadata.get("trace_mode") == "pest_surface_constrained"
        and np.shape(streamlines.theta) == np.shape(streamlines.R)
    )
    phi0 = float(coords.phi_vals[0]) if np.asarray(coords.phi_vals).size else 0.0
    theta0 = float(coords.theta_vals[0]) if np.asarray(coords.theta_vals).size else 0.0
    phi_period = float(getattr(coords, "period", TWOPI) or TWOPI)
    for ax in axes.ravel()[sec.size:]:
        ax.set_visible(False)
    for out_idx, iphi in enumerate(sec):
        ax = axes.ravel()[out_idx]
        ip = int(iphi) % coords.R_surf.shape[0]
        draw_smooth_pest_grid(ax, coords.R_surf[ip], coords.Z_surf[ip])
        selected = np.flatnonzero(streamlines.seed_phi_index == ip)
        for line_idx in selected:
            keep = np.isfinite(streamlines.R[line_idx]) & np.isfinite(streamlines.Z[line_idx])
            if use_pest_projection:
                keep &= np.isfinite(streamlines.theta[line_idx])
            if np.count_nonzero(keep) < 2:
                continue
            if use_pest_projection:
                ir = int(streamlines.seed_surface_index[line_idx]) % coords.R_surf.shape[1]
                theta_line = streamlines.theta[line_idx, keep]
                phi_line = np.full_like(theta_line, float(coords.phi_vals[ip]), dtype=np.float64)
                R_line = _periodic_surface_bilinear(
                    coords.R_surf[:, ir, :],
                    phi_line,
                    theta_line,
                    phi0=phi0,
                    theta0=theta0,
                    phi_period=phi_period,
                )
                Z_line = _periodic_surface_bilinear(
                    coords.Z_surf[:, ir, :],
                    phi_line,
                    theta_line,
                    phi0=phi0,
                    theta0=theta0,
                    phi_period=phi_period,
                )
            else:
                R_line = streamlines.R[line_idx, keep]
                Z_line = streamlines.Z[line_idx, keep]
            cval = (float(streamlines.seed_rho[line_idx]) - rho_min) / (rho_max - rho_min)
            ax.plot(R_line, Z_line, color=cmap(cval), lw=line_width, alpha=alpha)
        ax.set_title(f"phi={float(coords.phi_vals[ip]):.3f}")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    if title is not None:
        fig.suptitle(title)
    return fig, axes


__all__ = [
    "PestSeededStreamlines",
    "plot_j_streamline_seed_sections",
    "plot_j_streamlines_on_pest_surface_plotly",
    "trace_j_streamlines_on_pest",
]
