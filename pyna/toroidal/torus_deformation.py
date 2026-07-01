"""Non-resonant torus (flux-surface) deformation under external perturbation.

Canonical implementation for :mod:`pyna.toroidal`.

Implements the analytic spectral theory derived in:
    W. Wei, "Non-resonant perturbation effects on invariant tori (flux surfaces)",
    (internal document, 2025).

All formulas work in flux coordinates (r, θ, φ) with the covariant decomposition
convention of that document.  Specifically:

    B  = B_r g^r + B_θ g^θ + B_φ g^φ       (covariant components)

where  B_α = B · g_α  (dimension = field × length).

The perturbation is described by its three covariant Fourier spectra
  (δB_r)_mn,  (δB_θ)_mn,  (δB_φ)_mn

and the unperturbed equilibrium supplies the scalars
  B_φ(r),  B_θ(r),  ι(r),  ι'(r),
  and the contravariant metric components g^{rθ}(r), g^{rφ}(r).

Public API
----------
TorusDeformationSpectrum
    Container for the full (δr, δθ, δφ) Fourier spectra — Theorem 2 of the paper.

RadialPerturbationSplit
    Container that separates radial perturbation Fourier modes into resonant
    components (island-width drivers) and non-resonant components
    (surface-deformation drivers).

split_radial_perturbation_spectrum
    Classify a B^r Fourier spectrum against the local rotational transform.

non_resonant_deformation_spectrum
    Compute (δr)_mn, (δθ)_mn, (δφ)_mn for every (m,n) pair in the input.

poincare_section_deformation
    Project the 3-D torus deformation onto an arbitrary Poincaré section φ = φ_0
    — Theorem 3 (universal 1-D ring deformation).

iota_variation_pf
    First-order rotational-transform variation δι for an axisymmetric (n=0) PF
    coil perturbation — Eq. (4.1).

mean_radial_displacement
    <δr> = −δι / ι'  — core identity Eq. (3.1).

mean_radial_displacement_pf
    Full axisymmetric PF spectral formula — boxed Eq. (4.2).

mean_radial_displacement_dc
    DC (m=n=0) simplification — boxed Eq. (4.3).

mean_radial_displacement_second_order
    General non-axisymmetric second-order formula — boxed Eq. (5.1).

deformation_peak_valley
    Phase-based peak / valley coordinates of a single (m,n) harmonic — Section 6.

green_function_spectrum
    Fourier coefficients of the deformation Green's function G(Θ, Φ; r) — Eq. (2.8).

Notes
-----
*Sign convention*:  The Fourier convention throughout is
    f(θ, φ) = Σ_mn  f_mn  exp[i(mθ + nφ)].
Real fields satisfy  f_{-m,-n} = f_{mn}*.

*Units*:  All B_α carry units of  [T·m].  The radial label r is in [m].
  ι and ι' are dimensionless and [1/m] respectively.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy import ndarray


__all__ = [
    "TorusDeformationSpectrum",
    "RadialPerturbationSplit",
    "split_radial_perturbation_spectrum",
    "non_resonant_deformation_spectrum",
    "fieldline_deformation_spectrum",
    "poincare_section_deformation",
    "iota_variation_pf",
    "mean_radial_displacement",
    "mean_radial_displacement_pf",
    "mean_radial_displacement_dc",
    "mean_radial_displacement_second_order",
    "deformation_peak_valley",
    "green_function_spectrum",
    "iota_to_q",
    "q_to_iota",
    "iota_prime_from_q_prime",
]


def _check_resonant(m: ndarray, n: ndarray, iota: float,
                    tol: float = 1e-10) -> ndarray:
    """Return boolean mask: True where (m, n) is resonant with ι."""
    denom = m * iota + n
    return np.abs(denom) < tol


def _safe_denom(m: ndarray, n: ndarray, iota: float,
                eps: float = 0.0) -> ndarray:
    """m*ι + n, optionally regularised by eps (Eq. 2.9 of the paper)."""
    d = m.astype(complex) * iota + n.astype(complex)
    if eps > 0:
        d += 1j * eps * np.sign(d.real + 1e-300)
    return d


@dataclass
class TorusDeformationSpectrum:
    """Fourier spectra of the full 3-D torus deformation vector δχ.

    Attributes
    ----------
    m, n : 1-D int arrays of shape (K,)
        Poloidal and toroidal mode numbers.
    delta_r : complex 1-D array of shape (K,)
        Radial displacement Fourier coefficients (δr)_mn  [m].
    delta_theta : complex 1-D array of shape (K,)
        Poloidal-angle displacement (δθ)_mn  [rad].
    delta_phi : complex 1-D array of shape (K,)
        Toroidal-angle displacement (δφ)_mn  [rad].
    resonant_mask : bool 1-D array of shape (K,)
        True for (m,n) pairs that are resonant with ι — these are set to NaN.
    """
    m: ndarray
    n: ndarray
    delta_r: ndarray
    delta_theta: ndarray
    delta_phi: ndarray
    resonant_mask: ndarray

    def real_field_r(self, theta: ndarray, phi: ndarray) -> ndarray:
        """Evaluate the real-space radial displacement δr(θ, φ)."""
        return _eval_real_field(self.delta_r, self.m, self.n, theta, phi)

    def real_field_theta(self, theta: ndarray, phi: ndarray) -> ndarray:
        """Evaluate δθ(θ, φ)."""
        return _eval_real_field(self.delta_theta, self.m, self.n, theta, phi)

    def real_field_phi(self, theta: ndarray, phi: ndarray) -> ndarray:
        """Evaluate δφ(θ, φ)."""
        return _eval_real_field(self.delta_phi, self.m, self.n, theta, phi)

    def section_r(self, theta: ndarray, phi0: float) -> ndarray:
        """δr on a Poincaré section φ = φ0 (Theorem 3)."""
        coeffs = self.delta_r * np.exp(1j * self.n * phi0)
        return _eval_real_field_1d(coeffs, self.m, theta)

    def section_theta(self, theta: ndarray, phi0: float) -> ndarray:
        """δθ on a Poincaré section φ = φ0."""
        coeffs = self.delta_theta * np.exp(1j * self.n * phi0)
        return _eval_real_field_1d(coeffs, self.m, theta)


@dataclass
class RadialPerturbationSplit:
    """Resonant/non-resonant split of a radial magnetic perturbation spectrum.

    The resonant part ``m*ι+n≈0`` is the input for island-width estimates; the
    non-resonant part is the input for flux-surface deformation.
    """
    m: ndarray
    n: ndarray
    dBr: ndarray
    resonant_mask: ndarray

    @property
    def nonresonant_mask(self) -> ndarray:
        """Boolean mask selecting modes that deform nested flux surfaces."""
        return ~self.resonant_mask

    @property
    def resonant_m(self) -> ndarray:
        return self.m[self.resonant_mask]

    @property
    def resonant_n(self) -> ndarray:
        return self.n[self.resonant_mask]

    @property
    def resonant_dBr(self) -> ndarray:
        return self.dBr[self.resonant_mask]

    @property
    def nonresonant_m(self) -> ndarray:
        return self.m[self.nonresonant_mask]

    @property
    def nonresonant_n(self) -> ndarray:
        return self.n[self.nonresonant_mask]

    @property
    def nonresonant_dBr(self) -> ndarray:
        return self.dBr[self.nonresonant_mask]

    def nonresonant_deformation(
        self,
        iota: float,
        Bphi: float,
        Btheta: float,
        dBth_mn: Union[ndarray, list, None] = None,
        dBph_mn: Union[ndarray, list, None] = None,
        g_r_theta: float = 0.0,
        g_r_phi: float = 0.0,
        regularise_eps: float = 0.0,
    ) -> TorusDeformationSpectrum:
        """Compute torus deformation using only non-resonant modes."""
        keep = self.nonresonant_mask
        dBth = _optional_component(dBth_mn, self.m.shape, "dBth_mn")[keep]
        dBph = _optional_component(dBph_mn, self.m.shape, "dBph_mn")[keep]
        return non_resonant_deformation_spectrum(
            self.m[keep],
            self.n[keep],
            self.dBr[keep],
            dBth,
            dBph,
            iota=iota,
            Bphi=Bphi,
            Btheta=Btheta,
            g_r_theta=g_r_theta,
            g_r_phi=g_r_phi,
            resonance_tol=0.0,
            regularise_eps=regularise_eps,
        )


def _optional_component(values, shape: tuple[int, ...], name: str) -> ndarray:
    """Return an optional perturbation component as a validated complex array.

    Parameters
    ----------
    values
        Input component values, or ``None`` to use a zero array.
    shape
        Required shape, matching the radial spectrum arrays.
    name
        Component name used in validation errors.
    """
    if values is None:
        return np.zeros(shape, dtype=complex)
    arr = np.asarray(values, dtype=complex)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def split_radial_perturbation_spectrum(
    m: Union[ndarray, list],
    n: Union[ndarray, list],
    dBr_mn: Union[ndarray, list],
    iota: float,
    resonance_tol: float = 1e-9,
) -> RadialPerturbationSplit:
    """Split B^r Fourier modes into island-driving and deformation-driving parts.

    Modes satisfying ``abs(m*ι+n) < resonance_tol`` are resonant and should be
    passed to island-chain / island-width analysis.  All remaining modes can be
    sent to :func:`non_resonant_deformation_spectrum` to compute smooth flux
    surface deformation.
    """
    m = np.asarray(m, dtype=int)
    n = np.asarray(n, dtype=int)
    dBr = np.asarray(dBr_mn, dtype=complex)
    if m.shape != n.shape or m.shape != dBr.shape:
        raise ValueError(
            "m, n, and dBr_mn must have identical shapes; "
            f"got {m.shape}, {n.shape}, and {dBr.shape}"
        )
    return RadialPerturbationSplit(
        m=m,
        n=n,
        dBr=dBr,
        resonant_mask=_check_resonant(m, n, iota, tol=resonance_tol),
    )


def _eval_real_field(coeffs: ndarray, m: ndarray, n: ndarray,
                     theta: ndarray, phi: ndarray) -> ndarray:
    """Sum Re[Σ_mn c_mn exp(i(mθ+nφ))] over all (m,n) pairs."""
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    out = np.zeros(np.broadcast_shapes(theta.shape, phi.shape))
    for ck, mk, nk in zip(coeffs, m, n):
        if np.isnan(ck):
            continue
        phase = mk * theta + nk * phi
        out = out + np.real(ck * np.exp(1j * phase))
    return out


def _eval_real_field_1d(coeffs: ndarray, m: ndarray, theta: ndarray) -> ndarray:
    """Sum Re[Σ_m c_m exp(imθ)] for a section at fixed φ."""
    theta = np.asarray(theta)
    out = np.zeros(theta.shape)
    for ck, mk in zip(coeffs, m):
        if np.isnan(ck):
            continue
        out = out + np.real(ck * np.exp(1j * mk * theta))
    return out


def non_resonant_deformation_spectrum(
    m: Union[ndarray, list],
    n: Union[ndarray, list],
    dBr_mn: Union[ndarray, list],
    dBth_mn: Union[ndarray, list],
    dBph_mn: Union[ndarray, list],
    iota: float,
    Bphi: float,
    Btheta: float,
    g_r_theta: float = 0.0,
    g_r_phi: float = 0.0,
    resonance_tol: float = 1e-9,
    regularise_eps: float = 0.0,
) -> TorusDeformationSpectrum:
    """Compute the Fourier spectra of the full torus deformation vector δχ."""
    m = np.asarray(m, dtype=int)
    n = np.asarray(n, dtype=int)
    dBr = np.asarray(dBr_mn, dtype=complex)
    dBth = np.asarray(dBth_mn, dtype=complex)
    dBph = np.asarray(dBph_mn, dtype=complex)

    assert m.shape == n.shape == dBr.shape == dBth.shape == dBph.shape

    denom = _safe_denom(m, n, iota, eps=regularise_eps)
    res_mask = _check_resonant(m, n, iota, tol=resonance_tol)

    if res_mask.any():
        warnings.warn(
            f"{res_mask.sum()} resonant mode(s) found and set to NaN. "
            "Non-resonant deformation theory does not apply to these modes.",
            UserWarning,
            stacklevel=2,
        )

    dr_mn = np.where(
        res_mask,
        np.nan + 0j,
        dBr / (1j * Bphi * denom),
    )
    dth_mn = np.where(
        res_mask,
        np.nan + 0j,
        iota * (dBth / Btheta - dBr * g_r_theta / (Bphi * denom)) / (1j * denom),
    )
    dph_mn = np.where(
        res_mask,
        np.nan + 0j,
        (dBph / Bphi - dBr * g_r_phi / (Bphi * denom)) / (1j * denom),
    )

    return TorusDeformationSpectrum(
        m=m,
        n=n,
        delta_r=dr_mn,
        delta_theta=dth_mn,
        delta_phi=dph_mn,
        resonant_mask=res_mask,
    )


def fieldline_deformation_spectrum(
    m: Union[ndarray, list],
    n: Union[ndarray, list],
    radial_velocity_mn: Union[ndarray, list],
    poloidal_velocity_mn: Union[ndarray, list, None] = None,
    *,
    iota: float,
    iota_prime: float = 0.0,
    include_shear: bool = False,
    resonance_tol: float = 1e-9,
    regularise_eps: float = 0.0,
) -> TorusDeformationSpectrum:
    """Compute non-resonant torus deformation from field-line velocity spectra.

    This is the field-line ODE form of
    :func:`non_resonant_deformation_spectrum`.  It expects Fourier
    coefficients of the perturbation to

    ``dr/dphi = F_r(theta, phi)`` and ``dtheta/dphi = iota + F_theta(theta, phi)``,

    using the same convention ``exp(i*(m*theta+n*phi))``.  For each
    non-resonant mode ``alpha = m*iota+n`` it solves the homological equations

    ``i*alpha*delta_r = F_r`` and
    ``i*alpha*delta_theta = F_theta``.

    If ``include_shear`` is true, the angular equation uses
    ``F_theta + iota_prime*delta_r``.  This is the invariant-torus conjugacy
    form and is the one whose map residual should scale quadratically with the
    perturbation amplitude.  The default leaves the shear term out because some
    section-geometry comparisons only need the direct coordinate displacement.
    """

    m = np.asarray(m, dtype=int)
    n = np.asarray(n, dtype=int)
    Fr = np.asarray(radial_velocity_mn, dtype=complex)
    if poloidal_velocity_mn is None:
        Ft = np.zeros_like(Fr, dtype=complex)
    else:
        Ft = np.asarray(poloidal_velocity_mn, dtype=complex)
    if m.shape != n.shape or m.shape != Fr.shape or Ft.shape != Fr.shape:
        raise ValueError(
            "m, n, radial_velocity_mn, and poloidal_velocity_mn must have identical shapes"
        )

    denom = _safe_denom(m, n, float(iota), eps=regularise_eps)
    res_mask = _check_resonant(m, n, float(iota), tol=resonance_tol)
    if res_mask.any():
        warnings.warn(
            f"{res_mask.sum()} resonant mode(s) found and set to NaN. "
            "Non-resonant deformation theory does not apply to these modes.",
            UserWarning,
            stacklevel=2,
        )

    keep = ~res_mask
    delta_r = np.full(Fr.shape, np.nan + 0j, dtype=complex)
    delta_theta = np.full(Fr.shape, np.nan + 0j, dtype=complex)
    delta_r[keep] = Fr[keep] / (1j * denom[keep])
    if include_shear:
        theta_forcing = Ft[keep] + float(iota_prime) * delta_r[keep]
    else:
        theta_forcing = Ft[keep]
    delta_theta[keep] = theta_forcing / (1j * denom[keep])
    return TorusDeformationSpectrum(
        m=m,
        n=n,
        delta_r=delta_r,
        delta_theta=delta_theta,
        delta_phi=np.zeros_like(delta_r, dtype=complex),
        resonant_mask=res_mask,
    )


def poincare_section_deformation(
    spec: TorusDeformationSpectrum,
    phi0: float,
    theta: ndarray,
) -> tuple[ndarray, ndarray]:
    """Evaluate the 1-D ring deformation on the Poincaré section φ = φ0."""
    dr_sec = spec.section_r(theta, phi0)
    dth_sec = spec.section_theta(theta, phi0)
    return dr_sec, dth_sec


def iota_variation_pf(
    m: Union[ndarray, list],
    dBtheta_m0: Union[ndarray, list],
    iota: float,
    Btheta: float,
) -> complex:
    """First-order rotational-transform variation δι for an n=0 PF perturbation."""
    m = np.asarray(m, dtype=int)
    dBth = np.asarray(dBtheta_m0, dtype=complex)
    nonzero = m != 0

    m_nz = m[nonzero].astype(complex)
    dBth_nz = dBth[nonzero]

    phase = np.exp(1j * 2 * np.pi * m_nz * iota) - 1.0
    factor = phase / (1j * 2 * np.pi * m_nz * iota)

    delta_iota = iota * np.sum(dBth_nz / Btheta * factor)
    return float(np.real(delta_iota))


def mean_radial_displacement(delta_iota: float, iota_prime: float) -> float:
    """Mean radial displacement from the core identity <δr> = −δι / ι'."""
    if iota_prime == 0.0:
        raise ValueError(
            "ι' = 0 (zero magnetic shear): core identity is singular. "
            "The surface is at a shear reversal point."
        )
    return -delta_iota / iota_prime


def mean_radial_displacement_pf(
    m: Union[ndarray, list],
    dBtheta_m0: Union[ndarray, list],
    iota: float,
    iota_prime: float,
    Btheta: float,
) -> float:
    """Mean radial displacement for an axisymmetric (n=0) PF coil perturbation."""
    delta_iota = iota_variation_pf(m, dBtheta_m0, iota, Btheta)
    return mean_radial_displacement(delta_iota, iota_prime)


def mean_radial_displacement_dc(
    dBtheta_00: complex,
    iota: float,
    iota_prime: float,
    Btheta: float,
) -> float:
    """Mean radial displacement for a uniform (m=n=0) poloidal-field perturbation."""
    if iota_prime == 0.0:
        raise ValueError("ι' = 0: core identity singular at shear reversal.")
    return float(-iota * np.real(dBtheta_00) / (iota_prime * Btheta))


def mean_radial_displacement_second_order(
    m: Union[ndarray, list],
    n: Union[ndarray, list],
    dBr_mn: Union[ndarray, list],
    iota: float,
    iota_prime: float,
    Bphi: float = 1.0,
    resonance_tol: float = 1e-9,
) -> float:
    """Mean radial displacement to second order for a general non-axisymmetric perturbation."""
    m = np.asarray(m, dtype=int)
    n = np.asarray(n, dtype=int)
    dBr = np.asarray(dBr_mn, dtype=complex)

    res_mask = _check_resonant(m, n, iota, tol=resonance_tol)
    dc_mask = (m == 0) & (n == 0)
    keep = ~res_mask & ~dc_mask

    if not keep.any():
        return 0.0

    alpha = m[keep] * iota + n[keep]
    numer = m[keep] * np.abs(dBr[keep]) ** 2
    total = float(np.sum(numer / alpha ** 3))

    return -4.0 * iota_prime / Bphi ** 2 * total


def deformation_peak_valley(
    m: int,
    n: int,
    dBr_mn: complex,
    iota: float,
    Bphi: float,
    n_peaks: int = 1,
) -> dict:
    """Phase-based peak and valley coordinates of the (m,n) deformation harmonic."""
    denom = float(m) * iota + float(n)
    if abs(denom) < 1e-12:
        raise ValueError(
            f"Mode ({m},{n}) is resonant with ι={iota:.6f} (mι+n={denom:.2e}). "
            "Peak/valley formula requires non-resonant condition."
        )
    dr_mn = dBr_mn / (1j * Bphi * denom)
    phase_dr = float(np.angle(dr_mn))
    amplitude = 2.0 * abs(dr_mn)

    k_arr = np.arange(n_peaks)
    peak_phase = -phase_dr + 2 * np.pi * k_arr
    valley_phase = np.pi - phase_dr + 2 * np.pi * k_arr

    return {
        "dr_mn": dr_mn,
        "amplitude": amplitude,
        "phase_dr": phase_dr,
        "peak_phase": peak_phase,
        "valley_phase": valley_phase,
    }


def green_function_spectrum(
    m: Union[ndarray, list],
    n: Union[ndarray, list],
    iota: float,
    Bphi: float,
    regularise_eps: float = 0.0,
    resonance_tol: float = 1e-9,
) -> ndarray:
    """Fourier coefficients of the deformation Green's function G(Θ, Φ; r)."""
    m = np.asarray(m, dtype=int)
    n = np.asarray(n, dtype=int)
    denom = _safe_denom(m, n, iota, eps=regularise_eps)
    res = _check_resonant(m, n, iota, tol=resonance_tol)
    return np.where(res, np.nan + 0j, 1.0 / (1j * Bphi * denom))


def iota_to_q(iota: float) -> float:
    """Safety factor q = 1/ι."""
    return 1.0 / iota


def q_to_iota(q: float) -> float:
    """Rotational transform ι = 1/q."""
    return 1.0 / q


def iota_prime_from_q_prime(q: float, q_prime: float) -> float:
    """ι' = dι/dr = −q'/q²."""
    return -q_prime / (q * q)
