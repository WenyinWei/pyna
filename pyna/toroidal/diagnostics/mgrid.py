"""Diagnostics for VMEC mgrid fields on smooth PEST surfaces."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

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
        periods = int(self.nfp)
        if periods <= 0:
            raise ValueError("nfp must be positive")
        object.__setattr__(self, "nfp", periods)
        if self.toroidal_period is not None:
            domain_period = float(self.toroidal_period)
            if not np.isfinite(domain_period) or domain_period <= 0.0:
                raise ValueError("toroidal_period must be positive and finite")
            object.__setattr__(self, "toroidal_period", domain_period)

    @property
    def period(self) -> float:
        """Period of the stored toroidal mesh, not necessarily the full torus."""

        return 2.0 * np.pi if self.toroidal_period is None else float(self.toroidal_period)

    @property
    def field_period_rad(self) -> float:
        """Physical stellarator field-period angle ``2*pi/nfp``."""

        return 2.0 * np.pi / float(self.nfp)

    @property
    def field_periods(self) -> int:
        """Alias matching :class:`pyna.fields.VectorFieldCylind`."""

        return int(self.nfp)

    @property
    def stores_one_field_period(self) -> bool:
        return bool(np.isclose(self.period, self.field_period_rad, rtol=1.0e-12, atol=1.0e-14))


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
        Rsec = periodic_phi_slice(coords.R_surf, phi, period=coords.period)
        Zsec = periodic_phi_slice(coords.Z_surf, phi, period=coords.period)
        dR_drho = periodic_phi_slice(deriv[0], phi, period=coords.period)
        dZ_drho = periodic_phi_slice(deriv[1], phi, period=coords.period)
        dR_dtheta = periodic_phi_slice(deriv[2], phi, period=coords.period)
        dZ_dtheta = periodic_phi_slice(deriv[3], phi, period=coords.period)
        dR_dphi = periodic_phi_slice(deriv[4], phi, period=coords.period)
        dZ_dphi = periodic_phi_slice(deriv[5], phi, period=coords.period)

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

    rho_values = np.asarray(rho_values, dtype=np.float64)
    high_modes_arr = np.asarray(tuple(high_modes), dtype=np.int64)
    results: list[dict[str, object]] = []
    for rho in rho_values:
        irho = int(np.argmin(np.abs(coords.rho_vals - rho)))
        for section_deg in sections_deg:
            phi = np.deg2rad(float(section_deg))
            Rsec = periodic_phi_slice(coords.R_surf, phi, period=coords.period)[irho]
            Zsec = periodic_phi_slice(coords.Z_surf, phi, period=coords.period)[irho]
            if coords.axis_R is not None and coords.axis_Z is not None:
                axis_R = float(
                    periodic_phi_slice(coords.axis_R[:, None], phi, period=coords.period).ravel()[0]
                )
                axis_Z = float(
                    periodic_phi_slice(coords.axis_Z[:, None], phi, period=coords.period).ravel()[0]
                )
            else:
                axis_R = float(
                    np.nanmean(periodic_phi_slice(coords.R_surf, phi, period=coords.period)[0])
                )
                axis_Z = float(
                    np.nanmean(periodic_phi_slice(coords.Z_surf, phi, period=coords.period)[0])
                )
            w = (Rsec - axis_R) + 1j * (Zsec - axis_Z)
            coeff = np.fft.fft(w) / w.size
            amps = np.abs(coeff[: mode_max + 1])
            denom = float(np.sqrt(np.sum(amps[1 : mode_max + 1] ** 2)))
            high = float(np.sqrt(np.sum(amps[high_modes_arr] ** 2)))
            low = float(np.sqrt(np.sum(amps[1:6] ** 2)))
            results.append(
                {
                    "rho": float(coords.rho_vals[irho]),
                    "section_deg": float(section_deg),
                    "amplitudes": [float(v) for v in amps],
                    "high_modes": [int(v) for v in high_modes_arr],
                    "high_rms_fraction": high / denom if denom > 0.0 else np.nan,
                    "high_over_m1_5": high / low if low > 0.0 else np.nan,
                    "dominant_high_m": int(high_modes_arr[int(np.argmax(amps[high_modes_arr]))]),
                }
            )
    return results


__all__ = [
    "SmoothPestCoordinates",
    "PestCurrentSection",
    "PestCurrentComponents",
    "load_smooth_pest_coordinates",
    "periodic_phi_slice",
    "smooth_pest_derivatives",
    "compute_pest_current_components",
    "surface_fourier_spectrum",
]
