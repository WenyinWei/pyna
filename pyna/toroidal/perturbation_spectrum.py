"""Radial magnetic-perturbation projection and Fourier spectra on flux surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyna.toroidal._periodic_grid import (
    TWOPI,
    drop_endpoint,
    prepare_surface_arrays,
    periodic_derivative,
    strip_field_grid,
    strip_periodic_endpoint,
)


@dataclass(frozen=True)
class RadialPerturbationFourierSpectrum:
    """Fourier spectrum of the magnetic perturbation normal to a flux surface."""

    m: np.ndarray
    n: np.ndarray
    dBr: np.ndarray
    dBr_grid: np.ndarray
    theta: np.ndarray
    phi: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        """Complex-mode amplitudes ``abs(dBr_mn)``."""

        return np.abs(self.dBr)

    @property
    def phase(self) -> np.ndarray:
        """Complex-mode phases ``arg(dBr_mn)`` in radians."""

        return np.angle(self.dBr)

    def split(self, iota: float, resonance_tol: float = 1.0e-9, radial_index: int | None = None):
        """Split this radial spectrum into resonant and non-resonant modes."""

        from pyna.toroidal.torus_deformation import split_radial_perturbation_spectrum

        dBr = self.dBr
        if dBr.ndim != 1:
            if radial_index is None:
                raise ValueError("radial_index is required when splitting a radial stack spectrum")
            dBr = dBr[int(radial_index)]
        return split_radial_perturbation_spectrum(
            self.m,
            self.n,
            dBr,
            iota=iota,
            resonance_tol=resonance_tol,
        )


def surface_unit_normal_cylindrical(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute outward surface normals in cylindrical components."""

    R_arr = np.asarray(R_surf, dtype=np.float64)
    Z_arr = np.asarray(Z_surf, dtype=np.float64)
    squeeze_r = False
    if R_arr.ndim == 2:
        R_arr = R_arr[:, np.newaxis, :]
        Z_arr = Z_arr[:, np.newaxis, :]
        squeeze_r = True
    R, Z, _, _ = prepare_surface_arrays(R_arr, Z_arr, phi_vals, theta_vals)
    dR_dtheta = periodic_derivative(R, TWOPI, axis=2)
    dZ_dtheta = periodic_derivative(Z, TWOPI, axis=2)
    dR_dphi = periodic_derivative(R, TWOPI, axis=0)
    dZ_dphi = periodic_derivative(Z, TWOPI, axis=0)

    n_R = R * dZ_dtheta
    n_phi = dZ_dphi * dR_dtheta - dR_dphi * dZ_dtheta
    n_Z = -R * dR_dtheta
    if normalize:
        norm = np.sqrt(n_R * n_R + n_Z * n_Z + n_phi * n_phi)
        norm = np.maximum(norm, 1.0e-300)
        n_R = n_R / norm
        n_Z = n_Z / norm
        n_phi = n_phi / norm
    if squeeze_r:
        return n_R[:, 0], n_Z[:, 0], n_phi[:, 0]
    return n_R, n_Z, n_phi


def radial_perturbation_component(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    delta_B_R: np.ndarray,
    delta_B_Z: np.ndarray,
    delta_B_phi: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Project an external magnetic perturbation onto the surface-normal direction."""

    n_R, n_Z, n_phi = surface_unit_normal_cylindrical(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        normalize=normalize,
    )
    dBR = strip_field_grid(np.asarray(delta_B_R, dtype=complex), theta_vals, phi_vals)
    dBZ = strip_field_grid(np.asarray(delta_B_Z, dtype=complex), theta_vals, phi_vals)
    if delta_B_phi is None:
        dBphi = np.zeros_like(dBR, dtype=complex)
    else:
        dBphi = strip_field_grid(np.asarray(delta_B_phi, dtype=complex), theta_vals, phi_vals)
    if dBR.shape != n_R.shape or dBZ.shape != n_R.shape or dBphi.shape != n_R.shape:
        raise ValueError("delta_B arrays must match the surface shape after removing endpoints")
    return dBR * n_R + dBZ * n_Z + dBphi * n_phi


def radial_perturbation_Fourier_spectrum(
    dBr_grid: np.ndarray,
    theta_vals: np.ndarray,
    phi_vals: np.ndarray,
    *,
    m_max: int | None = None,
    n_max: int | None = None,
    min_amplitude: float = 0.0,
) -> RadialPerturbationFourierSpectrum:
    """Compute ``dBr_mn`` for ``f(theta, phi)=sum dBr_mn exp(i(m theta+n phi))``.

    ``dBr_grid`` may be a single surface with shape ``(n_phi, n_theta)``, a
    radial-first stack ``(n_r, n_phi, n_theta)``, or the phi-first stack
    ``(n_phi, n_r, n_theta)`` returned by :func:`radial_perturbation_component`.
    For radial stacks, ``dBr`` has shape ``(n_r, n_modes)``.
    """

    grid = np.asarray(dBr_grid, dtype=complex)
    if grid.ndim not in (2, 3):
        raise ValueError("dBr_grid must have shape (n_phi, n_theta) or (n_r, n_phi, n_theta)")
    theta, theta_has_endpoint = strip_periodic_endpoint(theta_vals, TWOPI, "theta_vals")
    phi, phi_has_endpoint = strip_periodic_endpoint(phi_vals, TWOPI, "phi_vals")
    phi_input_size = np.asarray(phi_vals).size
    grid = drop_endpoint(grid, axis=-1, has_endpoint=theta_has_endpoint)

    single_surface = grid.ndim == 2
    if single_surface:
        grid = drop_endpoint(grid, axis=0, has_endpoint=phi_has_endpoint)
    elif grid.shape[1] in (phi_input_size, phi.size):
        grid = drop_endpoint(grid, axis=1, has_endpoint=phi_has_endpoint)
    elif grid.shape[0] in (phi_input_size, phi.size):
        grid = drop_endpoint(grid, axis=0, has_endpoint=phi_has_endpoint)
        grid = np.moveaxis(grid, 1, 0)
    else:
        raise ValueError(
            "3-D dBr_grid must be radial-first (n_r, n_phi, n_theta) or "
            "phi-first (n_phi, n_r, n_theta)"
        )

    if grid.shape[-2:] != (phi.size, theta.size):
        raise ValueError("dBr_grid shape must match phi_vals and theta_vals")

    fft = np.fft.fft2(np.swapaxes(grid, -2, -1), axes=(-2, -1)) / float(theta.size * phi.size)
    m_freq = np.fft.fftfreq(theta.size, 1.0 / theta.size).astype(int)
    n_freq = np.fft.fftfreq(phi.size, 1.0 / phi.size).astype(int)
    m_limit = int(np.max(np.abs(m_freq)) if m_max is None else m_max)
    n_limit = int(np.max(np.abs(n_freq)) if n_max is None else n_max)

    modes_m = []
    modes_n = []
    coeffs = []
    for m_val in range(-m_limit, m_limit + 1):
        m_idx = np.where(m_freq == m_val)[0]
        if m_idx.size == 0:
            continue
        for n_val in range(-n_limit, n_limit + 1):
            n_idx = np.where(n_freq == n_val)[0]
            if n_idx.size == 0:
                continue
            coeff = fft[..., int(m_idx[0]), int(n_idx[0])]
            if np.max(np.abs(coeff)) < float(min_amplitude):
                continue
            modes_m.append(m_val)
            modes_n.append(n_val)
            coeffs.append(coeff)

    if coeffs:
        dBr = np.stack(coeffs, axis=-1)
    else:
        dBr = np.empty(grid.shape[:-2] + (0,), dtype=complex)
    if single_surface:
        dBr = np.asarray(dBr, dtype=complex).reshape((-1,))
    return RadialPerturbationFourierSpectrum(
        m=np.asarray(modes_m, dtype=int),
        n=np.asarray(modes_n, dtype=int),
        dBr=dBr,
        dBr_grid=grid,
        theta=theta,
        phi=phi,
    )


__all__ = [
    "RadialPerturbationFourierSpectrum",
    "radial_perturbation_Fourier_spectrum",
    "radial_perturbation_component",
    "surface_unit_normal_cylindrical",
]
