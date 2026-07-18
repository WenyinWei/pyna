"""Diagnostics for VMEC mgrid fields on smooth PEST surfaces."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from pyna.fields.periodicity import ToroidalPeriodicity
from pyna.io.mgrid import MGridCurrent, mgrid_toroidal_index, sample_plane_bilinear


@dataclass(frozen=True)
class SmoothPestCoordinates:
    """Smooth PEST-like surface coordinates stored as ``(phi, rho, theta)``."""

    R_surf: np.ndarray
    Z_surf: np.ndarray
    rho_vals: np.ndarray
    theta_vals: np.ndarray
    phi_vals: np.ndarray
    axis_R: Optional[np.ndarray] = None
    axis_Z: Optional[np.ndarray] = None
    source: Optional[str] = None
    nfp: int = 1
    toroidal_period: Optional[float] = None

    def __post_init__(self) -> None:
        periodicity = ToroidalPeriodicity(
            nfp=self.nfp,
            domain_period=self.toroidal_period,
        )
        object.__setattr__(self, "nfp", periodicity.nfp)
        object.__setattr__(self, "toroidal_period", periodicity.domain_period)

    @property
    def periodicity(self) -> ToroidalPeriodicity:
        return ToroidalPeriodicity(
            nfp=self.nfp,
            domain_period=self.toroidal_period,
        )

    @property
    def period(self) -> float:
        """Period of the stored toroidal mesh, not necessarily the full torus."""

        return float(self.periodicity.domain_period)

    @property
    def field_period_rad(self) -> float:
        """Physical stellarator field-period angle ``2*pi/nfp``."""

        return self.periodicity.field_period

    @property
    def field_period(self) -> float:
        return self.periodicity.field_period

    @property
    def stores_one_field_period(self) -> bool:
        return self.periodicity.stores_one_field_period

    def phi_slice(self, values: np.ndarray, phi: float) -> np.ndarray:
        """Interpolate a toroidal slice using this coordinate grid's domain."""

        return periodic_phi_slice(values, phi, period=self.period)


@dataclass(frozen=True)
class PestCurrentSection:
    """Contravariant current components on one toroidal PEST section."""

    section_deg: float
    section_phi: float
    R: np.ndarray
    Z: np.ndarray
    Jrho: np.ndarray
    Jtheta: np.ndarray
    Jphi: np.ndarray
    Jtheta_over_Jphi: np.ndarray


@dataclass(frozen=True)
class PestCurrentComponents:
    """PEST-section current-component diagnostic for one mgrid field."""

    label: str
    sections: tuple[PestCurrentSection, ...]

    def component_stats(self) -> dict[str, list[dict[str, object]]]:
        """Return per-section sign and percentile statistics."""

        out: dict[str, list[dict[str, object]]] = {
            "Jrho": [],
            "Jtheta": [],
            "Jphi": [],
            "Jtheta_over_Jphi": [],
        }
        for section in self.sections:
            arrays = {
                "Jrho": section.Jrho,
                "Jtheta": section.Jtheta,
                "Jphi": section.Jphi,
                "Jtheta_over_Jphi": section.Jtheta_over_Jphi,
            }
            for name, values in arrays.items():
                vals = np.asarray(values[1:], dtype=np.float64)
                finite = np.isfinite(vals)
                pct = np.nanpercentile(vals, [5.0, 50.0, 95.0]) if finite.any() else [np.nan] * 3
                out[name].append(
                    {
                        "section_deg": float(section.section_deg),
                        "negative_fraction": float(np.nanmean(vals < 0.0)),
                        "p05_p50_p95": [float(v) for v in pct],
                    }
                )
        return out


@dataclass(frozen=True)
class SurfaceShapeHarmonicSection:
    """Fourier diagnostics for one closed surface curve on one toroidal section."""

    radial_label: float
    radial_index: int
    section_phi: float
    modes: np.ndarray
    coefficients: np.ndarray
    high_modes: tuple[int, ...] = ()
    denominator_mode_max: int | None = None

    @property
    def section_deg(self) -> float:
        return float(np.rad2deg(self.section_phi))

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.coefficients)

    def abs_mode_amplitudes(self, mode_max: int | None = None) -> np.ndarray:
        """Return RMS-combined amplitudes grouped by ``abs(m)``."""

        modes = np.asarray(self.modes, dtype=int)
        coeff = np.asarray(self.coefficients, dtype=np.complex128)
        max_mode = int(np.max(np.abs(modes))) if mode_max is None else int(mode_max)
        out = np.zeros(max_mode + 1, dtype=np.float64)
        for mode in range(max_mode + 1):
            selected = np.abs(modes) == mode
            if np.any(selected):
                out[mode] = float(np.sqrt(np.sum(np.abs(coeff[selected]) ** 2)))
        return out

    def mode_energy(self, modes: Sequence[int]) -> float:
        """Return RMS coefficient energy for selected absolute poloidal modes."""

        wanted = np.asarray(tuple(int(abs(m)) for m in modes), dtype=int)
        if wanted.size == 0:
            return 0.0
        selected = np.isin(np.abs(np.asarray(self.modes, dtype=int)), wanted)
        return float(np.sqrt(np.sum(np.abs(np.asarray(self.coefficients)[selected]) ** 2)))

    @property
    def high_mode_fraction(self) -> float:
        """Fraction of selected high-mode energy relative to resolved shape energy."""

        high = self.mode_energy(self.high_modes)
        modes = np.asarray(self.modes, dtype=int)
        abs_modes = np.abs(modes)
        denom_mode_max = (
            int(np.max(abs_modes))
            if self.denominator_mode_max is None
            else int(self.denominator_mode_max)
        )
        selected = (abs_modes >= 1) & (abs_modes <= denom_mode_max)
        denom = float(np.sqrt(np.sum(np.abs(np.asarray(self.coefficients)[selected]) ** 2)))
        return high / denom if denom > 0.0 else np.nan


@dataclass(frozen=True)
class SurfaceShapeHarmonicLeakage:
    """Spectral support audit for a surface-shape correction."""

    radial_label: float
    radial_index: int
    section_phi: float
    total_delta_energy: float
    leaked_delta_energy: float
    allowed_delta_energy: float
    leakage_fraction: float
    leaking_modes: tuple[int, ...]

    @property
    def section_deg(self) -> float:
        return float(np.rad2deg(self.section_phi))


def load_smooth_pest_coordinates(path: Union[str, Path]) -> SmoothPestCoordinates:
    """Load a ``*_pest_coordinates_smooth.npz`` surface-coordinate bundle."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        if "nfp" in data.files:
            nfp_value = data["nfp"]
        elif "field_periods" in data.files:
            nfp_value = data["field_periods"]
        else:
            nfp_value = 1
        nfp = int(np.asarray(nfp_value).item())
        period_value = None
        for key in ("toroidal_period", "phi_period", "period"):
            if key in data.files:
                period_value = float(np.asarray(data[key]).item())
                break
        return SmoothPestCoordinates(
            R_surf=np.asarray(data["R_surf"], dtype=np.float64),
            Z_surf=np.asarray(data["Z_surf"], dtype=np.float64),
            rho_vals=np.asarray(data["rho_vals"], dtype=np.float64),
            theta_vals=np.asarray(data["theta_vals"], dtype=np.float64),
            phi_vals=np.asarray(data["phi_vals"], dtype=np.float64),
            axis_R=np.asarray(data["axis_R"], dtype=np.float64) if "axis_R" in data.files else None,
            axis_Z=np.asarray(data["axis_Z"], dtype=np.float64) if "axis_Z" in data.files else None,
            source=str(path),
            nfp=nfp,
            toroidal_period=period_value,
        )


def periodic_phi_slice(values: np.ndarray, phi: float, *, period: float = 2.0 * np.pi) -> np.ndarray:
    """Interpolate a periodic ``phi`` slice from an array with leading phi axis."""

    values = np.asarray(values, dtype=np.float64)
    nphi = values.shape[0]
    u = (np.mod(float(phi), period) / period) * nphi
    i0 = int(np.floor(u)) % nphi
    frac = u - np.floor(u)
    i1 = (i0 + 1) % nphi
    return values[i0] * (1.0 - frac) + values[i1] * frac


def smooth_pest_derivatives(coords: SmoothPestCoordinates) -> tuple[np.ndarray, ...]:
    """Return ``dR,dZ`` derivatives with respect to ``rho, theta, phi``."""

    R = np.asarray(coords.R_surf, dtype=np.float64)
    Z = np.asarray(coords.Z_surf, dtype=np.float64)
    rho = np.asarray(coords.rho_vals, dtype=np.float64)
    theta_step = 2.0 * np.pi / R.shape[2]
    phi_step = float(coords.period) / R.shape[0]
    dR_drho = np.gradient(R, rho, axis=1, edge_order=2)
    dZ_drho = np.gradient(Z, rho, axis=1, edge_order=2)
    dR_dtheta = (np.roll(R, -1, axis=2) - np.roll(R, 1, axis=2)) / (2.0 * theta_step)
    dZ_dtheta = (np.roll(Z, -1, axis=2) - np.roll(Z, 1, axis=2)) / (2.0 * theta_step)
    dR_dphi = (np.roll(R, -1, axis=0) - np.roll(R, 1, axis=0)) / (2.0 * phi_step)
    dZ_dphi = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2.0 * phi_step)
    return dR_drho, dZ_drho, dR_dtheta, dZ_dtheta, dR_dphi, dZ_dphi


def _require_uniform_periodic_theta(theta_vals: np.ndarray, n_theta: int) -> None:
    theta = np.asarray(theta_vals, dtype=np.float64)
    if theta.ndim != 1 or theta.size != int(n_theta):
        raise ValueError("theta_vals must match the last surface dimension")
    if theta.size < 4:
        raise ValueError("at least four theta samples are required")
    step = 2.0 * np.pi / float(theta.size)
    expected = theta[0] + step * np.arange(theta.size)
    wrapped_error = np.angle(np.exp(1j * (theta - expected)))
    if not np.allclose(wrapped_error, 0.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("surface harmonic diagnostics require a uniform periodic theta grid")


def _periodic_slice_at_phi(
    values: np.ndarray,
    phi_vals: np.ndarray,
    phi: float,
    *,
    period: float = 2.0 * np.pi,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    src = np.mod(np.asarray(phi_vals, dtype=np.float64), float(period))
    if vals.shape[0] != src.size:
        raise ValueError("phi_vals must match the leading values dimension")
    order = np.argsort(src)
    src = src[order]
    vals = np.take(vals, order, axis=0)
    _, unique_idx = np.unique(np.round(src, decimals=12), return_index=True)
    src = src[unique_idx]
    vals = np.take(vals, unique_idx, axis=0)
    if src.size < 2:
        return np.take(vals, 0, axis=0)
    src_ext = np.concatenate([src[-1:] - period, src, src[:1] + period])
    vals_ext = np.concatenate([np.take(vals, [-1], axis=0), vals, np.take(vals, [0], axis=0)], axis=0)
    phi_mod = float(np.mod(phi, period))
    lo = int(np.searchsorted(src_ext, phi_mod, side="right") - 1)
    lo = max(0, min(lo, src_ext.size - 2))
    hi = lo + 1
    denom = float(src_ext[hi] - src_ext[lo])
    frac = 0.0 if abs(denom) < 1.0e-15 else float((phi_mod - src_ext[lo]) / denom)
    return np.take(vals_ext, lo, axis=0) * (1.0 - frac) + np.take(vals_ext, hi, axis=0) * frac


def _axis_value(
    axis: np.ndarray | None,
    phi_vals: np.ndarray,
    phi: float,
    fallback: float,
    *,
    period: float = 2.0 * np.pi,
) -> float:
    if axis is None:
        return float(fallback)
    arr = np.asarray(axis, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("axis arrays must be one-dimensional")
    return float(
        _periodic_slice_at_phi(arr[:, None], phi_vals, phi, period=period).ravel()[0]
    )


def surface_shape_harmonic_spectrum(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    radial_labels: Sequence[float],
    theta_vals: Sequence[float],
    phi_vals: Sequence[float],
    *,
    radial_values: Sequence[float],
    sections_phi: Sequence[float],
    axis_R: Sequence[float] | None = None,
    axis_Z: Sequence[float] | None = None,
    mode_max: int = 20,
    high_modes: Sequence[int] = tuple(range(6, 11)),
    denominator_mode_max: int | None = None,
    toroidal_period: float = 2.0 * np.pi,
) -> list[SurfaceShapeHarmonicSection]:
    """Measure signed poloidal shape harmonics for selected surface sections.

    The curve is represented as ``w(theta) = R(theta) - R_axis
    + i (Z(theta) - Z_axis)``.  Signed Fourier modes are retained so a
    distorted surface cannot hide energy in the negative-``m`` branch.
    """

    R = np.asarray(R_surf, dtype=np.float64)
    Z = np.asarray(Z_surf, dtype=np.float64)
    if R.ndim != 3 or Z.shape != R.shape:
        raise ValueError("R_surf and Z_surf must have shape (n_phi, n_radial, n_theta)")
    radial = np.asarray(radial_labels, dtype=np.float64)
    phi_arr = np.asarray(phi_vals, dtype=np.float64)
    if radial.ndim != 1 or radial.size != R.shape[1]:
        raise ValueError("radial_labels must match the radial surface dimension")
    if phi_arr.ndim != 1 or phi_arr.size != R.shape[0]:
        raise ValueError("phi_vals must match the leading surface dimension")
    period = float(toroidal_period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("toroidal_period must be positive and finite")
    _require_uniform_periodic_theta(np.asarray(theta_vals, dtype=np.float64), R.shape[2])
    mode_max = int(mode_max)
    if mode_max < 0:
        raise ValueError("mode_max must be non-negative")
    denom_max = mode_max if denominator_mode_max is None else int(denominator_mode_max)
    modes_full = np.rint(np.fft.fftfreq(R.shape[2], d=1.0 / float(R.shape[2]))).astype(int)
    selected_modes = np.abs(modes_full) <= mode_max
    high = tuple(int(abs(m)) for m in high_modes)
    out: list[SurfaceShapeHarmonicSection] = []
    for radial_value in radial_values:
        ir = int(np.argmin(np.abs(radial - float(radial_value))))
        for phi in sections_phi:
            phi = float(phi)
            Rsec = np.asarray(
                _periodic_slice_at_phi(R, phi_arr, phi, period=period)[ir],
                dtype=np.float64,
            )
            Zsec = np.asarray(
                _periodic_slice_at_phi(Z, phi_arr, phi, period=period)[ir],
                dtype=np.float64,
            )
            axis_r = _axis_value(
                None if axis_R is None else np.asarray(axis_R),
                phi_arr,
                phi,
                float(np.nanmean(Rsec)),
                period=period,
            )
            axis_z = _axis_value(
                None if axis_Z is None else np.asarray(axis_Z),
                phi_arr,
                phi,
                float(np.nanmean(Zsec)),
                period=period,
            )
            w = (Rsec - axis_r) + 1j * (Zsec - axis_z)
            coeff = np.fft.fft(w) / float(w.size)
            out.append(
                SurfaceShapeHarmonicSection(
                    radial_label=float(radial[ir]),
                    radial_index=ir,
                    section_phi=phi,
                    modes=modes_full[selected_modes].copy(),
                    coefficients=coeff[selected_modes].copy(),
                    high_modes=high,
                    denominator_mode_max=denom_max,
                )
            )
    return out


def low_pass_surface_shape_harmonics(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    *,
    mode_cutoff: int,
    axis_R: Sequence[float] | None = None,
    axis_Z: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return surfaces with poloidal shape modes above ``mode_cutoff`` removed.

    This is intended as a conservative guard for island-healing experiments:
    smooth the surface representation itself, then compare harmonic diagnostics
    before using the result as an integrable reference.
    """

    R = np.asarray(R_surf, dtype=np.float64)
    Z = np.asarray(Z_surf, dtype=np.float64)
    if R.ndim != 3 or Z.shape != R.shape:
        raise ValueError("R_surf and Z_surf must have shape (n_phi, n_radial, n_theta)")
    cutoff = int(mode_cutoff)
    if cutoff < 0:
        raise ValueError("mode_cutoff must be non-negative")
    n_phi, n_radial, n_theta = R.shape
    modes = np.rint(np.fft.fftfreq(n_theta, d=1.0 / float(n_theta))).astype(int)
    keep = np.abs(modes) <= cutoff
    axis_r = None if axis_R is None else np.asarray(axis_R, dtype=np.float64)
    axis_z = None if axis_Z is None else np.asarray(axis_Z, dtype=np.float64)
    if axis_r is not None and axis_r.shape != (n_phi,):
        raise ValueError("axis_R must have shape (n_phi,)")
    if axis_z is not None and axis_z.shape != (n_phi,):
        raise ValueError("axis_Z must have shape (n_phi,)")
    filtered_R = np.empty_like(R)
    filtered_Z = np.empty_like(Z)
    for iphi in range(n_phi):
        for ir in range(n_radial):
            center_r = float(axis_r[iphi]) if axis_r is not None else float(np.nanmean(R[iphi, ir]))
            center_z = float(axis_z[iphi]) if axis_z is not None else float(np.nanmean(Z[iphi, ir]))
            w = (R[iphi, ir] - center_r) + 1j * (Z[iphi, ir] - center_z)
            coeff = np.fft.fft(w)
            coeff[~keep] = 0.0
            smooth = np.fft.ifft(coeff)
            filtered_R[iphi, ir] = center_r + smooth.real
            filtered_Z[iphi, ir] = center_z + smooth.imag
    return filtered_R, filtered_Z


def surface_shape_harmonic_leakage(
    before: Sequence[SurfaceShapeHarmonicSection],
    after: Sequence[SurfaceShapeHarmonicSection],
    *,
    allowed_modes: Sequence[int],
    mode_min: int = 1,
    amplitude_floor: float = 0.0,
) -> list[SurfaceShapeHarmonicLeakage]:
    """Measure correction energy outside an allowed absolute poloidal-mode set."""

    if len(before) != len(after):
        raise ValueError("before and after harmonic section lists must have the same length")
    allowed = {int(abs(m)) for m in allowed_modes}
    min_mode = int(mode_min)
    floor = float(amplitude_floor)
    rows: list[SurfaceShapeHarmonicLeakage] = []
    for lhs, rhs in zip(before, after):
        if lhs.radial_index != rhs.radial_index or not np.isclose(lhs.section_phi, rhs.section_phi):
            raise ValueError("before and after harmonic sections are not aligned")
        lhs_modes = np.asarray(lhs.modes, dtype=int)
        rhs_modes = np.asarray(rhs.modes, dtype=int)
        if lhs_modes.shape != rhs_modes.shape or not np.array_equal(lhs_modes, rhs_modes):
            raise ValueError("before and after harmonic sections must use the same mode basis")
        delta = np.asarray(rhs.coefficients, dtype=np.complex128) - np.asarray(lhs.coefficients, dtype=np.complex128)
        abs_modes = np.abs(lhs_modes)
        selected = abs_modes >= min_mode
        allowed_mask = selected & np.isin(abs_modes, list(allowed))
        leaked_mask = selected & ~allowed_mask
        total = float(np.sqrt(np.sum(np.abs(delta[selected]) ** 2)))
        allowed_energy = float(np.sqrt(np.sum(np.abs(delta[allowed_mask]) ** 2)))
        leaked_energy = float(np.sqrt(np.sum(np.abs(delta[leaked_mask]) ** 2)))
        leaking_modes = tuple(
            int(mode)
            for mode in sorted(set(abs_modes[leaked_mask & (np.abs(delta) > floor)]))
        )
        rows.append(
            SurfaceShapeHarmonicLeakage(
                radial_label=float(lhs.radial_label),
                radial_index=int(lhs.radial_index),
                section_phi=float(lhs.section_phi),
                total_delta_energy=total,
                leaked_delta_energy=leaked_energy,
                allowed_delta_energy=allowed_energy,
                leakage_fraction=leaked_energy / total if total > 0.0 else 0.0,
                leaking_modes=leaking_modes,
            )
        )
    return rows


def compute_pest_current_components(
    current: MGridCurrent,
    coords: SmoothPestCoordinates,
    sections_deg: Sequence[float],
    *,
    label: str = "",
) -> PestCurrentComponents:
    """Project cylindrical current density onto smooth PEST contravariant bases."""

    deriv = smooth_pest_derivatives(coords)
    sections: list[PestCurrentSection] = []
    for section_deg in sections_deg:
        phi = np.deg2rad(float(section_deg))
        iphi = mgrid_toroidal_index(current, phi)
        Rsec = coords.phi_slice(coords.R_surf, phi)
        Zsec = coords.phi_slice(coords.Z_surf, phi)
        dR_drho = coords.phi_slice(deriv[0], phi)
        dZ_drho = coords.phi_slice(deriv[1], phi)
        dR_dtheta = coords.phi_slice(deriv[2], phi)
        dZ_dtheta = coords.phi_slice(deriv[3], phi)
        dR_dphi = coords.phi_slice(deriv[4], phi)
        dZ_dphi = coords.phi_slice(deriv[5], phi)

        JR = sample_plane_bilinear(current.JR[iphi], current.R, current.Z, Rsec, Zsec)
        JPhi = sample_plane_bilinear(current.JPhi[iphi], current.R, current.Z, Rsec, Zsec)
        JZ = sample_plane_bilinear(current.JZ[iphi], current.R, current.Z, Rsec, Zsec)

        basis = np.empty(Rsec.shape + (3, 3), dtype=np.float64)
        rhs = np.empty(Rsec.shape + (3,), dtype=np.float64)
        basis[..., 0, 0] = dR_drho
        basis[..., 0, 1] = dR_dtheta
        basis[..., 0, 2] = dR_dphi
        basis[..., 1, 0] = 0.0
        basis[..., 1, 1] = 0.0
        basis[..., 1, 2] = Rsec
        basis[..., 2, 0] = dZ_drho
        basis[..., 2, 1] = dZ_dtheta
        basis[..., 2, 2] = dZ_dphi
        rhs[..., 0] = JR
        rhs[..., 1] = JPhi
        rhs[..., 2] = JZ

        components = np.full(Rsec.shape + (3,), np.nan, dtype=np.float64)
        valid = np.isfinite(rhs).all(axis=-1)
        valid[0, :] = False
        if np.any(valid):
            components[valid] = np.linalg.solve(basis[valid], rhs[valid][..., None])[..., 0]
        Jrho = components[..., 0]
        Jtheta = components[..., 1]
        Jphi = components[..., 2]
        ratio = Jtheta / Jphi
        ratio[~np.isfinite(ratio)] = np.nan
        sections.append(
            PestCurrentSection(
                section_deg=float(section_deg),
                section_phi=float(phi),
                R=Rsec,
                Z=Zsec,
                Jrho=Jrho,
                Jtheta=Jtheta,
                Jphi=Jphi,
                Jtheta_over_Jphi=ratio,
            )
        )
    return PestCurrentComponents(label=label, sections=tuple(sections))


def surface_fourier_spectrum(
    coords: SmoothPestCoordinates,
    *,
    rho_values: Sequence[float],
    sections_deg: Sequence[float],
    mode_max: int = 20,
    high_modes: Sequence[int] = tuple(range(6, 11)),
) -> list[dict[str, object]]:
    """Measure PEST-section surface-shape Fourier amplitudes."""

    sections = surface_shape_harmonic_spectrum(
        coords.R_surf,
        coords.Z_surf,
        coords.rho_vals,
        coords.theta_vals,
        coords.phi_vals,
        radial_values=rho_values,
        sections_phi=np.deg2rad(np.asarray(sections_deg, dtype=np.float64)),
        axis_R=coords.axis_R,
        axis_Z=coords.axis_Z,
        mode_max=mode_max,
        high_modes=high_modes,
        toroidal_period=coords.period,
    )
    high_modes_arr = np.asarray(tuple(high_modes), dtype=np.int64)
    results: list[dict[str, object]] = []
    for section in sections:
        amps = section.abs_mode_amplitudes(mode_max)
        high = float(np.sqrt(np.sum(amps[high_modes_arr] ** 2)))
        low = float(np.sqrt(np.sum(amps[1:6] ** 2)))
        results.append(
            {
                "rho": float(section.radial_label),
                "section_deg": float(section.section_deg),
                "amplitudes": [float(v) for v in amps],
                "high_modes": [int(v) for v in high_modes_arr],
                "high_rms_fraction": float(section.high_mode_fraction),
                "high_over_m1_5": high / low if low > 0.0 else np.nan,
                "dominant_high_m": int(high_modes_arr[int(np.argmax(amps[high_modes_arr]))]),
            }
        )
    return results


__all__ = [
    "SmoothPestCoordinates",
    "SurfaceShapeHarmonicLeakage",
    "PestCurrentSection",
    "PestCurrentComponents",
    "SurfaceShapeHarmonicSection",
    "load_smooth_pest_coordinates",
    "low_pass_surface_shape_harmonics",
    "periodic_phi_slice",
    "smooth_pest_derivatives",
    "compute_pest_current_components",
    "surface_fourier_spectrum",
    "surface_shape_harmonic_leakage",
    "surface_shape_harmonic_spectrum",
]
