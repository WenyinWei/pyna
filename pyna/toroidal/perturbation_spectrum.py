"""Radial magnetic-perturbation projection and Fourier spectra on flux surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    radial_labels: np.ndarray | None = None

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

    def mode_index(self, m: int, n: int) -> int | None:
        """Return the packed-mode index for ``(m, n)``, or ``None`` if absent."""

        idx = np.where((self.m == int(m)) & (self.n == int(n)))[0]
        return None if idx.size == 0 else int(idx[0])

    def mode_coefficient(self, m: int, n: int, radial_index: int | None = None) -> complex:
        """Return one Fourier coefficient from the packed spectrum."""

        idx = self.mode_index(m, n)
        if idx is None:
            return 0.0 + 0.0j
        if self.dBr.ndim == 1:
            return complex(self.dBr[idx])
        if radial_index is None:
            raise ValueError("radial_index is required for a radial stack spectrum")
        return complex(self.dBr[int(radial_index), idx])


@dataclass(frozen=True)
class ResonantIslandChain:
    """Nardon-style resonant island-chain estimate from ``tilde_b^1_{m,-n}``."""

    m: int
    n: int
    radial_label: float
    q: float
    q_prime: float
    coefficient: complex
    b_res: float
    half_width: float

    @property
    def phase(self) -> float:
        """Phase ``arg(tilde_b^1_{m,-n})`` in radians."""

        return float(np.angle(self.coefficient))

    def fixed_points(self, phi: float | np.ndarray, *, q_prime_sign: int | None = None) -> dict[str, np.ndarray]:
        """Return O/X poloidal angles for one or more toroidal sections.

        The convention is the Nardon expansion
        ``tilde_b^1 = sum b_mn exp(i(m theta* + n phi))``.  For the resonant
        coefficient ``b_{m,-n}``, fixed points satisfy
        ``m theta* - n phi + arg(b_{m,-n}) = +/- pi/2``.
        """

        sign = int(np.sign(self.q_prime)) if q_prime_sign is None else int(np.sign(q_prime_sign))
        if sign == 0:
            sign = 1
        return island_chain_fixed_points(self.m, self.n, self.coefficient, phi, q_prime_sign=sign)

    def with_phase_shift(self, phase_shift: float) -> "ResonantIslandChain":
        """Return a copy with ``arg(coefficient)`` advanced by ``phase_shift``."""

        return ResonantIslandChain(
            m=self.m,
            n=self.n,
            radial_label=self.radial_label,
            q=self.q,
            q_prime=self.q_prime,
            coefficient=self.coefficient * np.exp(1j * float(phase_shift)),
            b_res=self.b_res,
            half_width=self.half_width,
        )


@dataclass(frozen=True)
class ChirikovOverlap:
    """Chirikov overlap between two adjacent resonant island chains."""

    left: ResonantIslandChain
    right: ResonantIslandChain
    separation: float
    sigma: float

    @property
    def modes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ``((m_left, n_left), (m_right, n_right))``."""

        return (self.left.m, self.left.n), (self.right.m, self.right.n)


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


def _validate_radial_labels(radial_labels: np.ndarray, n_r: int) -> np.ndarray:
    labels = np.asarray(radial_labels, dtype=np.float64)
    if labels.ndim != 1 or labels.size != int(n_r):
        raise ValueError("radial_labels must be one-dimensional and match the radial surface count")
    if not np.all(np.isfinite(labels)) or np.any(np.diff(labels) <= 0.0):
        raise ValueError("radial_labels must be finite and strictly increasing")
    return labels


def contravariant_radial_component(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    B_R: np.ndarray,
    B_Z: np.ndarray,
    B_phi: np.ndarray | None,
    radial_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``B^1 = B dot grad(s)`` and ``B^3 = B dot grad(phi)``.

    ``R_surf`` and ``Z_surf`` must be a radial stack with shape
    ``(n_phi, n_r, n_theta)``.  The returned arrays have the same stripped
    shape, after any duplicated periodic endpoints have been removed.
    Cylindrical field components use the physical orthonormal basis
    ``(e_R, e_phi, e_Z)``.
    """

    R, Z, _, _ = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    labels = _validate_radial_labels(radial_labels, R.shape[1])
    BR = strip_field_grid(np.asarray(B_R, dtype=complex), theta_vals, phi_vals)
    BZ = strip_field_grid(np.asarray(B_Z, dtype=complex), theta_vals, phi_vals)
    if B_phi is None:
        Bphi = np.zeros_like(BR, dtype=complex)
    else:
        Bphi = strip_field_grid(np.asarray(B_phi, dtype=complex), theta_vals, phi_vals)
    if BR.shape != R.shape or BZ.shape != R.shape or Bphi.shape != R.shape:
        raise ValueError("field arrays must match the surface shape after removing endpoints")

    edge_order = 2 if labels.size >= 3 else 1
    dR_ds = np.gradient(R, labels, axis=1, edge_order=edge_order)
    dZ_ds = np.gradient(Z, labels, axis=1, edge_order=edge_order)
    dR_dtheta = periodic_derivative(R, TWOPI, axis=2)
    dZ_dtheta = periodic_derivative(Z, TWOPI, axis=2)
    dR_dphi = periodic_derivative(R, TWOPI, axis=0)
    dZ_dphi = periodic_derivative(Z, TWOPI, axis=0)

    # Reciprocal basis: grad(s) = (e_theta x e_phi) / J.
    # Components are in the local right-handed cylindrical basis
    # (e_R, e_phi, e_Z).
    cross_R = -R * dZ_dtheta
    cross_phi = dZ_dtheta * dR_dphi - dR_dtheta * dZ_dphi
    cross_Z = R * dR_dtheta
    jac = dR_ds * cross_R + dZ_ds * cross_Z
    jac = np.where(np.abs(jac) < 1.0e-300, np.nan, jac)
    grad_s_R = cross_R / jac
    grad_s_phi = cross_phi / jac
    grad_s_Z = cross_Z / jac

    B1 = BR * grad_s_R + Bphi * grad_s_phi + BZ * grad_s_Z
    B3 = Bphi / np.maximum(R, 1.0e-300)
    return B1, B3


def nardon_radial_perturbation(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    delta_B_R: np.ndarray,
    delta_B_Z: np.ndarray,
    delta_B_phi: np.ndarray | None,
    radial_labels: np.ndarray,
    *,
    denominator_B_phi: np.ndarray | None = None,
    denominator_B3: np.ndarray | None = None,
    eps: float = 1.0e-300,
) -> np.ndarray:
    """Compute Nardon's ``tilde_b^1 = delta B^1 / B_0^3`` on surfaces.

    Pass ``denominator_B_phi`` when the denominator should be the background
    toroidal contravariant field ``B_0 dot grad(phi)``.  Pass
    ``denominator_B3`` directly if it is already available on the same
    ``(phi, radial, theta)`` surface grid.
    """

    delta_B1, delta_B3 = contravariant_radial_component(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        delta_B_R,
        delta_B_Z,
        delta_B_phi,
        radial_labels,
    )
    if denominator_B3 is not None:
        denom = strip_field_grid(np.asarray(denominator_B3, dtype=complex), theta_vals, phi_vals)
    elif denominator_B_phi is not None:
        R, _, _, _ = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
        denom = strip_field_grid(np.asarray(denominator_B_phi, dtype=complex), theta_vals, phi_vals)
        denom = denom / np.maximum(R, 1.0e-300)
    else:
        denom = delta_B3
    if denom.shape != delta_B1.shape:
        raise ValueError("denominator field must match the surface shape after removing endpoints")
    denom = np.where(np.abs(denom) < float(eps), np.nan + 0.0j, denom)
    return delta_B1 / denom


def radial_perturbation_Fourier_spectrum(
    dBr_grid: np.ndarray,
    theta_vals: np.ndarray,
    phi_vals: np.ndarray,
    *,
    radial_labels: np.ndarray | None = None,
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
    labels = None
    if radial_labels is not None:
        if single_surface:
            raise ValueError("radial_labels are only valid for radial stack spectra")
        labels = _validate_radial_labels(radial_labels, grid.shape[0])

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
        radial_labels=labels,
    )


def island_chain_fixed_points(
    m: int,
    n: int,
    coefficient: complex,
    phi: float | np.ndarray,
    *,
    q_prime_sign: int = 1,
) -> dict[str, np.ndarray]:
    """Return O/X poloidal angles implied by ``tilde_b^1_{m,-n}``.

    The returned ``theta_O`` and ``theta_X`` arrays have shape ``(n_phi, m)``.
    A phase change ``coefficient *= exp(1j * alpha)`` rotates every branch by
    ``-alpha / m`` at fixed toroidal section.
    """

    m_int = int(m)
    n_int = int(n)
    if m_int <= 0 or n_int <= 0:
        raise ValueError("m and n must be positive resonant mode numbers")
    sign = 1 if int(np.sign(q_prime_sign)) >= 0 else -1
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    phase = float(np.angle(coefficient))
    if sign >= 0:
        base_O = n_int * phi_arr - 0.5 * np.pi - phase
        base_X = n_int * phi_arr + 0.5 * np.pi - phase
    else:
        base_O = n_int * phi_arr + 0.5 * np.pi - phase
        base_X = n_int * phi_arr - 0.5 * np.pi - phase
    branches = np.arange(m_int, dtype=np.float64)
    theta_O = (base_O[:, None] + TWOPI * branches[None, :]) / float(m_int)
    theta_X = (base_X[:, None] + TWOPI * branches[None, :]) / float(m_int)
    return {
        "phi": np.mod(phi_arr, TWOPI),
        "theta_O": np.mod(theta_O, TWOPI),
        "theta_X": np.mod(theta_X, TWOPI),
    }


def nardon_resonant_amplitude(coefficient: complex) -> float:
    """Return ``tilde_b_res^1 = 2 |tilde_b^1_{m,-n}|``."""

    return float(2.0 * abs(coefficient))


def nardon_island_half_width(q: float, q_prime: float, m: int, b_res: float) -> float:
    """Return Nardon's magnetic-island half-width in the radial coordinate.

    The thesis formula is ``sqrt(4 q^2 b_res / (q' m))``.  This implementation
    returns a positive geometric width and therefore uses ``abs(q' m)`` in the
    denominator.
    """

    m_int = int(m)
    if m_int <= 0:
        raise ValueError("m must be positive")
    denom = abs(float(q_prime) * float(m_int))
    if denom <= 0.0:
        return float("nan")
    value = 4.0 * float(q) * float(q) * max(float(b_res), 0.0) / denom
    return float(np.sqrt(value))


def _as_mode_values(m_values: Iterable[int] | None, q_profile: np.ndarray, n: int) -> list[int]:
    if m_values is not None:
        out = sorted({int(m) for m in m_values if int(m) > 0})
        return out
    q_min = float(np.nanmin(q_profile))
    q_max = float(np.nanmax(q_profile))
    lo = int(np.floor(min(q_min, q_max) * int(n))) - 1
    hi = int(np.ceil(max(q_min, q_max) * int(n))) + 1
    return [m for m in range(max(1, lo), max(1, hi) + 1)]


def _find_crossings(radial: np.ndarray, values: np.ndarray, target: float) -> list[float]:
    roots: list[float] = []
    diff = np.asarray(values, dtype=np.float64) - float(target)
    for i in range(radial.size - 1):
        f0 = diff[i]
        f1 = diff[i + 1]
        if not np.isfinite(f0) or not np.isfinite(f1):
            continue
        if f0 == 0.0:
            roots.append(float(radial[i]))
        if f0 * f1 < 0.0:
            t = -f0 / (f1 - f0)
            roots.append(float(radial[i] + t * (radial[i + 1] - radial[i])))
    if diff[-1] == 0.0:
        roots.append(float(radial[-1]))
    return roots


def _interp_complex(x: np.ndarray, y: np.ndarray, x0: float) -> complex:
    return complex(
        np.interp(float(x0), x, np.real(y)),
        np.interp(float(x0), x, np.imag(y)),
    )


def analyze_resonant_island_chains(
    spectrum: RadialPerturbationFourierSpectrum,
    q_profile: np.ndarray,
    *,
    n: int,
    radial_labels: np.ndarray | None = None,
    m_values: Iterable[int] | None = None,
    min_b_res: float = 0.0,
) -> list[ResonantIslandChain]:
    """Analyze resonant island chains from a radial Fourier spectrum.

    For each requested ``m`` this finds roots of ``q(s) = m/n``, interpolates
    the resonant coefficient ``tilde_b^1_{m,-n}``, and evaluates Nardon's
    island half-width formula in the same radial coordinate ``s``.
    """

    if spectrum.dBr.ndim != 2:
        raise ValueError("analyze_resonant_island_chains requires a radial stack spectrum")
    n_int = int(n)
    if n_int <= 0:
        raise ValueError("n must be positive")
    radial = spectrum.radial_labels if radial_labels is None else radial_labels
    if radial is None:
        raise ValueError("radial_labels are required")
    radial = _validate_radial_labels(radial, spectrum.dBr.shape[0])
    q_arr = np.asarray(q_profile, dtype=np.float64)
    if q_arr.shape != radial.shape:
        raise ValueError("q_profile must have the same shape as radial_labels")
    q_prime_profile = np.gradient(q_arr, radial, edge_order=2 if radial.size >= 3 else 1)

    chains: list[ResonantIslandChain] = []
    for m_int in _as_mode_values(m_values, q_arr, n_int):
        idx = spectrum.mode_index(m_int, -n_int)
        if idx is None:
            continue
        roots = _find_crossings(radial, q_arr, float(m_int) / float(n_int))
        coeff_profile = spectrum.dBr[:, idx]
        for s_res in roots:
            q_res = float(np.interp(s_res, radial, q_arr))
            q_prime = float(np.interp(s_res, radial, q_prime_profile))
            coeff = _interp_complex(radial, coeff_profile, s_res)
            b_res = nardon_resonant_amplitude(coeff)
            if b_res < float(min_b_res):
                continue
            chains.append(
                ResonantIslandChain(
                    m=m_int,
                    n=n_int,
                    radial_label=float(s_res),
                    q=q_res,
                    q_prime=q_prime,
                    coefficient=coeff,
                    b_res=b_res,
                    half_width=nardon_island_half_width(q_res, q_prime, m_int, b_res),
                )
            )
    chains.sort(key=lambda chain: (chain.radial_label, chain.m, chain.n))
    return chains


def chirikov_overlaps(chains: Iterable[ResonantIslandChain]) -> list[ChirikovOverlap]:
    """Compute Chirikov overlap for adjacent chains with the same toroidal ``n``."""

    grouped: dict[int, list[ResonantIslandChain]] = {}
    for chain in chains:
        grouped.setdefault(chain.n, []).append(chain)
    overlaps: list[ChirikovOverlap] = []
    for same_n in grouped.values():
        ordered = sorted(same_n, key=lambda chain: chain.radial_label)
        for left, right in zip(ordered[:-1], ordered[1:]):
            separation = abs(right.radial_label - left.radial_label)
            if separation <= 0.0:
                sigma = float("inf")
            else:
                sigma = float((left.half_width + right.half_width) / separation)
            overlaps.append(ChirikovOverlap(left=left, right=right, separation=separation, sigma=sigma))
    return overlaps


def sample_cylindrical_vector_grid_on_surfaces(
    grid_R: np.ndarray,
    grid_Z: np.ndarray,
    grid_phi: np.ndarray,
    field_R: np.ndarray,
    field_phi: np.ndarray,
    field_Z: np.ndarray,
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    *,
    bounds_error: bool = False,
    fill_value: float | None = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a rectilinear cylindrical vector grid on ``(phi, radial, theta)`` surfaces."""

    from scipy.interpolate import RegularGridInterpolator

    R, Z, phi, _ = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    axis_R = np.asarray(grid_R, dtype=np.float64)
    axis_Z = np.asarray(grid_Z, dtype=np.float64)
    axis_phi = np.asarray(grid_phi, dtype=np.float64)
    if axis_phi.ndim != 1 or axis_phi.size < 2:
        raise ValueError("grid_phi must be one-dimensional with at least two points")
    phi0 = float(axis_phi[0])
    phi_stripped, phi_has_endpoint = strip_periodic_endpoint(axis_phi, TWOPI, "grid_phi")

    def extend(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = np.asarray(values)
        vals = drop_endpoint(vals, axis=2, has_endpoint=phi_has_endpoint)
        if vals.shape != (axis_R.size, axis_Z.size, phi_stripped.size):
            raise ValueError("field arrays must have shape (n_R, n_Z, n_phi)")
        vals_ext = np.concatenate([vals, vals[:, :, :1]], axis=2)
        phi_ext = np.concatenate([phi_stripped, [phi0 + TWOPI]])
        return phi_ext, vals_ext

    phi_ext, vals_R = extend(field_R)
    _, vals_phi = extend(field_phi)
    _, vals_Z = extend(field_Z)
    pts = np.column_stack(
        [
            R.ravel(),
            Z.ravel(),
            (np.mod(np.repeat(phi[:, None], R.shape[1] * R.shape[2], axis=1).ravel() - phi0, TWOPI) + phi0),
        ]
    )
    kwargs = {"bounds_error": bounds_error, "fill_value": fill_value}
    interp_R = RegularGridInterpolator((axis_R, axis_Z, phi_ext), vals_R, **kwargs)
    interp_phi = RegularGridInterpolator((axis_R, axis_Z, phi_ext), vals_phi, **kwargs)
    interp_Z = RegularGridInterpolator((axis_R, axis_Z, phi_ext), vals_Z, **kwargs)
    out_shape = R.shape
    return (
        interp_R(pts).reshape(out_shape),
        interp_phi(pts).reshape(out_shape),
        interp_Z(pts).reshape(out_shape),
    )


__all__ = [
    "ChirikovOverlap",
    "RadialPerturbationFourierSpectrum",
    "ResonantIslandChain",
    "analyze_resonant_island_chains",
    "chirikov_overlaps",
    "contravariant_radial_component",
    "island_chain_fixed_points",
    "nardon_island_half_width",
    "nardon_radial_perturbation",
    "nardon_resonant_amplitude",
    "radial_perturbation_Fourier_spectrum",
    "radial_perturbation_component",
    "sample_cylindrical_vector_grid_on_surfaces",
    "surface_unit_normal_cylindrical",
]
