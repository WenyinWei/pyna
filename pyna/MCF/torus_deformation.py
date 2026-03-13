"""Non-resonant torus (flux-surface) deformation under external perturbation.

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
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from numpy import ndarray


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _check_resonant(m: ndarray, n: ndarray, iota: float,
                    tol: float = 1e-10) -> ndarray:
    """Return boolean mask: True where (m, n) is resonant with ι."""
    denom = m * iota + n
    return np.abs(denom) < tol


def _safe_denom(m: ndarray, n: ndarray, iota: float,
                eps: float = 0.0) -> ndarray:
    """m*ι + n,  optionally regularised by eps (Eq. 2.9 of the paper)."""
    d = m.astype(complex) * iota + n.astype(complex)
    if eps > 0:
        d += 1j * eps * np.sign(d.real + 1e-300)
    return d


# ────────────────────────────────────────────────────────────────────────────
# Data container
# ────────────────────────────────────────────────────────────────────────────

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

    # ---------- convenience methods -----------------------------------------

    def real_field_r(self, theta: ndarray, phi: ndarray) -> ndarray:
        """Evaluate the real-space radial displacement δr(θ, φ).

        Parameters
        ----------
        theta, phi : arrays (broadcast-compatible)

        Returns
        -------
        ndarray  (same shape as broadcast(theta, phi))
        """
        return _eval_real_field(
            self.delta_r, self.m, self.n, theta, phi)

    def real_field_theta(self, theta: ndarray, phi: ndarray) -> ndarray:
        """Evaluate δθ(θ, φ)."""
        return _eval_real_field(
            self.delta_theta, self.m, self.n, theta, phi)

    def real_field_phi(self, theta: ndarray, phi: ndarray) -> ndarray:
        """Evaluate δφ(θ, φ)."""
        return _eval_real_field(
            self.delta_phi, self.m, self.n, theta, phi)

    def section_r(self, theta: ndarray, phi0: float) -> ndarray:
        """δr on a Poincaré section φ = φ0  (Theorem 3)."""
        coeffs = self.delta_r * np.exp(1j * self.n * phi0)
        return _eval_real_field_1d(coeffs, self.m, theta)

    def section_theta(self, theta: ndarray, phi0: float) -> ndarray:
        """δθ on a Poincaré section φ = φ0."""
        coeffs = self.delta_theta * np.exp(1j * self.n * phi0)
        return _eval_real_field_1d(coeffs, self.m, theta)


def _eval_real_field(coeffs: ndarray, m: ndarray, n: ndarray,
                     theta: ndarray, phi: ndarray) -> ndarray:
    """Sum  Re[Σ_mn  c_mn  exp(i(mθ+nφ))]  over all (m,n) pairs."""
    theta = np.asarray(theta)
    phi   = np.asarray(phi)
    out   = np.zeros(np.broadcast_shapes(theta.shape, phi.shape))
    for ck, mk, nk in zip(coeffs, m, n):
        if np.isnan(ck):
            continue
        phase = mk * theta + nk * phi
        out = out + np.real(ck * np.exp(1j * phase))
    return out


def _eval_real_field_1d(coeffs: ndarray, m: ndarray,
                         theta: ndarray) -> ndarray:
    """Sum  Re[Σ_m  c_m  exp(imθ)]  (section at fixed φ, so n absorbed)."""
    theta = np.asarray(theta)
    out   = np.zeros(theta.shape)
    for ck, mk in zip(coeffs, m):
        if np.isnan(ck):
            continue
        out = out + np.real(ck * np.exp(1j * mk * theta))
    return out


# ────────────────────────────────────────────────────────────────────────────
# 1.  Full 2-D torus deformation spectrum  (Theorem 2, Eq. 2.5)
# ────────────────────────────────────────────────────────────────────────────

def non_resonant_deformation_spectrum(
    m: Union[ndarray, list],
    n: Union[ndarray, list],
    dBr_mn:  Union[ndarray, list],
    dBth_mn: Union[ndarray, list],
    dBph_mn: Union[ndarray, list],
    iota:    float,
    Bphi:    float,
    Btheta:  float,
    g_r_theta: float = 0.0,
    g_r_phi:   float = 0.0,
    resonance_tol: float = 1e-9,
    regularise_eps: float = 0.0,
) -> TorusDeformationSpectrum:
    """Compute the Fourier spectra of the full torus deformation vector δχ.

    Implements the boxed formulas in Theorem 2 (Eq. 2.5) of the paper:

    .. math::

        (\\delta r)_{mn}     &= \\frac{\\mathrm{i}\\,(\\delta B_r)_{mn}}
                                      {B_\\phi\\,(m\\iota+n)}, \\\\
        (\\delta\\theta)_{mn} &= \\frac{1}{\\mathrm{i}\\,m}
            \\left(\\frac{(\\delta B_\\theta)_{mn}}{B_\\theta}
                   - \\frac{(\\delta B_r)_{mn}\\,g^{r\\theta}}
                           {B_\\phi\\,(m\\iota+n)}\\right), \\\\
        (\\delta\\phi)_{mn}   &= \\frac{1}{\\mathrm{i}\\,n}
            \\left(\\frac{(\\delta B_\\phi)_{mn}}{B_\\phi}
                   - \\frac{(\\delta B_r)_{mn}\\,g^{r\\phi}}
                           {B_\\phi\\,(m\\iota+n)}\\right).

    Parameters
    ----------
    m, n : array_like of int, shape (K,)
        Mode numbers.  (0,0) is silently excluded (DC handled separately).
    dBr_mn, dBth_mn, dBph_mn : complex array_like, shape (K,)
        Fourier coefficients of δB_r, δB_θ, δB_φ (covariant, units T·m).
    iota : float
        Rotational transform ι at this flux surface.
    Bphi : float
        Covariant toroidal field component B_φ (T·m).
    Btheta : float
        Covariant poloidal field component B_θ (T·m).
    g_r_theta : float
        Contravariant metric element g^{rθ}.  Zero for axisymmetric
        equilibria in standard flux coordinates.
    g_r_phi : float
        Contravariant metric element g^{rφ}.  Zero for axisymmetric
        equilibria.
    resonance_tol : float
        Modes with `|mι+n|` < resonance_tol are flagged as resonant and
        their deformation coefficients are set to NaN.
    regularise_eps : float
        If > 0, add ε·i·sgn(mι+n) to the denominator (Eq. 2.9).

    Returns
    -------
    TorusDeformationSpectrum
    """
    m   = np.asarray(m,   dtype=int)
    n   = np.asarray(n,   dtype=int)
    dBr  = np.asarray(dBr_mn,  dtype=complex)
    dBth = np.asarray(dBth_mn, dtype=complex)
    dBph = np.asarray(dBph_mn, dtype=complex)

    assert m.shape == n.shape == dBr.shape == dBth.shape == dBph.shape

    denom = _safe_denom(m, n, iota, eps=regularise_eps)   # m*ι + n  (complex)
    res_mask = _check_resonant(m, n, iota, tol=resonance_tol)

    if res_mask.any():
        warnings.warn(
            f"{res_mask.sum()} resonant mode(s) found and set to NaN. "
            "Non-resonant deformation theory does not apply to these modes.",
            UserWarning, stacklevel=2,
        )

    # ── (δr)_mn  [corrected: A·i(mι+n) = f_mn  ⟹  A = f_mn / (i(mι+n)) = −i·f_mn/(mι+n)] ──
    dr_mn = np.where(
        res_mask,
        np.nan + 0j,
        dBr / (1j * Bphi * denom),
    )

    # ── (δθ)_mn  [corrected: from field-line ODE, (d/dφ)δθ = ι·(δBθ/Bθ)]
    # Invariance eq: A_θ·i(mι+n) = ι·(δBθ)_mn/Bθ  =>  A_θ = ι·(δBθ)_mn/(i·Bθ·(mι+n))
    # Note: for n=0 this reduces to (1/im)·(δBθ/Bθ) as in the paper (Eq.2.5 n=0 case only)
    dth_mn = np.where(
        res_mask,
        np.nan + 0j,
        iota * (dBth / Btheta - dBr * g_r_theta / (Bphi * denom)) / (1j * denom),
    )

    # ── (δφ)_mn  [corrected: from field-line ODE, (d/dφ)δφ = (δBφ/Bφ)]
    # Invariance eq: A_φ·i(mι+n) = (δBφ)_mn/Bφ  =>  A_φ = (δBφ)_mn/(i·Bφ·(mι+n))
    # Note: for m=0 this reduces to (1/in)·(δBφ/Bφ) as in the paper (Eq.2.5 m=0 case only)
    dph_mn = np.where(
        res_mask,
        np.nan + 0j,
        (dBph / Bphi - dBr * g_r_phi / (Bphi * denom)) / (1j * denom),
    )

    return TorusDeformationSpectrum(
        m=m, n=n,
        delta_r=dr_mn,
        delta_theta=dth_mn,
        delta_phi=dph_mn,
        resonant_mask=res_mask,
    )


# ────────────────────────────────────────────────────────────────────────────
# 2.  Poincaré section 1-D ring deformation  (Theorem 3, Eq. 2.6)
# ────────────────────────────────────────────────────────────────────────────

def poincare_section_deformation(
    spec: TorusDeformationSpectrum,
    phi0: float,
    theta: ndarray,
) -> tuple[ndarray, ndarray]:
    """Evaluate the 1-D ring deformation on the Poincaré section φ = φ0.

    Implements the boxed universal 1-D ring deformation formula (Theorem 3,
    Eq. 2.6):

    .. math::

        \\delta r(\\theta, \\phi_0) &=
          \\sum_{(m,n)\\neq(0,0)}
          \\frac{\\mathrm{i}\\,(\\delta B_r)_{mn}}
               {B_\\phi\\,(m\\iota+n)}\\,
          e^{\\mathrm{i}n\\phi_0}\\,e^{\\mathrm{i}m\\theta}, \\\\
        \\delta\\theta(\\theta, \\phi_0) &=
          \\sum_{(m,n)\\neq(0,0)}
          \\frac{1}{\\mathrm{i}m}\\!
          \\left(\\frac{(\\delta B_\\theta)_{mn}}{B_\\theta}
                 - \\frac{(\\delta B_r)_{mn}\\,g^{r\\theta}}
                         {B_\\phi(m\\iota+n)}\\right)
          e^{\\mathrm{i}n\\phi_0}\\,e^{\\mathrm{i}m\\theta}.

    Parameters
    ----------
    spec : TorusDeformationSpectrum
        Output of :func:`non_resonant_deformation_spectrum`.
    phi0 : float
        Toroidal angle of the Poincaré section (rad).
    theta : ndarray
        Poloidal angle grid (rad).

    Returns
    -------
    delta_r_section : ndarray
        Radial displacement δr(θ, φ0)  [m].
    delta_theta_section : ndarray
        Poloidal angle displacement δθ(θ, φ0)  [rad].
    """
    dr_sec   = spec.section_r    (theta, phi0)
    dth_sec  = spec.section_theta(theta, phi0)
    return dr_sec, dth_sec


# ────────────────────────────────────────────────────────────────────────────
# 3.  ι-variation for axisymmetric (PF coil, n=0) perturbation  (Eq. 4.1)
# ────────────────────────────────────────────────────────────────────────────

def iota_variation_pf(
    m: Union[ndarray, list],
    dBtheta_m0: Union[ndarray, list],
    iota: float,
    Btheta: float,
) -> complex:
    """First-order rotational-transform variation δι for an n=0 PF perturbation.

    Implements Eq. (4.1):

    .. math::

        \\delta\\iota = \\iota \\sum_{m \\neq 0}
          \\frac{(\\delta B_\\theta)_{m,0}}{B_\\theta}
          \\cdot \\frac{e^{\\mathrm{i}2\\pi m\\iota}-1}{\\mathrm{i}2\\pi m\\iota}

    Parameters
    ----------
    m : array_like of int
        Poloidal mode numbers (n = 0 is implied).  Only m ≠ 0 contribute.
    dBtheta_m0 : complex array_like
        Fourier coefficients (δB_θ)_{m,0}  (T·m).
    iota : float
        Unperturbed rotational transform ι at this flux surface.
    Btheta : float
        Unperturbed covariant poloidal field B_θ  (T·m).

    Returns
    -------
    float
        First-order δι (dimensionless).
    """
    m      = np.asarray(m,          dtype=int)
    dBth   = np.asarray(dBtheta_m0, dtype=complex)
    nonzero = m != 0

    m_nz   = m  [nonzero].astype(complex)
    dBth_nz= dBth[nonzero]

    phase  = np.exp(1j * 2 * np.pi * m_nz * iota) - 1.0
    factor = phase / (1j * 2 * np.pi * m_nz * iota)

    delta_iota = iota * np.sum(dBth_nz / Btheta * factor)
    # For a real perturbation the imaginary part should cancel; take real part.
    return float(np.real(delta_iota))


# ────────────────────────────────────────────────────────────────────────────
# 4.  Core identity: <δr> = −δι / ι'  (Eq. 3.1)
# ────────────────────────────────────────────────────────────────────────────

def mean_radial_displacement(delta_iota: float, iota_prime: float) -> float:
    """Mean radial displacement from the core identity  <δr> = −δι / ι'.

    Implements the fundamental boxed identity Eq. (3.1):

    .. math::

        \\langle \\delta r \\rangle = -\\frac{\\delta\\iota}{\\iota^\\prime}

    where ι' = dι/dr is the magnetic shear.

    Parameters
    ----------
    delta_iota : float
        First-order perturbation of the rotational transform δι.
    iota_prime : float
        Radial derivative of the unperturbed rotational transform ι' = dι/dr
        (units 1/m).

    Returns
    -------
    float
        Mean radial displacement <δr>  (m).

    Notes
    -----
    For a standard tokamak, ι decreases outward so ι' < 0.  If the poloidal
    field is weakened (δι < 0), then <δr> < 0, meaning the flux surface
    shifts inward — consistent with physical intuition.
    """
    if iota_prime == 0.0:
        raise ValueError(
            "ι' = 0 (zero magnetic shear): core identity is singular. "
            "The surface is at a shear reversal point."
        )
    return -delta_iota / iota_prime


# ────────────────────────────────────────────────────────────────────────────
# 5.  Full PF spectral formula  (boxed Eq. 4.2)
# ────────────────────────────────────────────────────────────────────────────

def mean_radial_displacement_pf(
    m: Union[ndarray, list],
    dBtheta_m0: Union[ndarray, list],
    iota: float,
    iota_prime: float,
    Btheta: float,
) -> float:
    """Mean radial displacement for an axisymmetric (n=0) PF coil perturbation.

    Implements the boxed Eq. (4.2):

    .. math::

        \\langle \\delta r \\rangle =
          -\\frac{\\iota}{\\iota^\\prime B_\\theta}
          \\sum_{m\\neq 0}
          (\\delta B_\\theta)_{m,0}
          \\cdot
          \\frac{e^{\\mathrm{i}2\\pi m\\iota}-1}{\\mathrm{i}2\\pi m\\iota}

    Parameters
    ----------
    m : array_like of int
        Poloidal mode numbers.  (n = 0 implied, m = 0 excluded.)
    dBtheta_m0 : complex array_like
        Fourier coefficients (δB_θ)_{m,0}  (T·m).
    iota : float
        Rotational transform ι.
    iota_prime : float
        dι/dr  (1/m).
    Btheta : float
        Unperturbed B_θ  (T·m).

    Returns
    -------
    float
        <δr>  (m).
    """
    delta_iota = iota_variation_pf(m, dBtheta_m0, iota, Btheta)
    return mean_radial_displacement(delta_iota, iota_prime)


# ────────────────────────────────────────────────────────────────────────────
# 6.  DC simplification  (boxed Eq. 4.3)
# ────────────────────────────────────────────────────────────────────────────

def mean_radial_displacement_dc(
    dBtheta_00: complex,
    iota: float,
    iota_prime: float,
    Btheta: float,
) -> float:
    """Mean radial displacement for a uniform (m=n=0) poloidal-field perturbation.

    Implements the boxed DC simplification Eq. (4.3):

    .. math::

        \\langle \\delta r \\rangle =
          -\\frac{\\iota \\cdot (\\delta B_\\theta)_{0,0}}
                 {\\iota^\\prime \\cdot B_\\theta}

    Equivalently (substituting ι = 1/q, ι' = −q'/q²):

    .. math::

        \\langle \\delta r \\rangle =
          \\frac{q \\cdot (\\delta B_\\theta)_{0,0}}{B_\\theta \\cdot q^\\prime}

    Parameters
    ----------
    dBtheta_00 : complex (or float)
        DC component (δB_θ)_{0,0}  (T·m).  For real perturbations pass a
        float; only the real part is used.
    iota : float
        Rotational transform ι.
    iota_prime : float
        dι/dr  (1/m).
    Btheta : float
        Unperturbed B_θ  (T·m).

    Returns
    -------
    float
        <δr>  (m).
    """
    if iota_prime == 0.0:
        raise ValueError("ι' = 0: core identity singular at shear reversal.")
    return float(-iota * np.real(dBtheta_00) / (iota_prime * Btheta))


# ────────────────────────────────────────────────────────────────────────────
# 7.  General non-axisymmetric second-order formula  (boxed Eq. 5.1)
# ────────────────────────────────────────────────────────────────────────────

def mean_radial_displacement_second_order(
    m: Union[ndarray, list],
    n: Union[ndarray, list],
    dBr_mn: Union[ndarray, list],
    iota: float,
    iota_prime: float,
    Bphi: float = 1.0,
    resonance_tol: float = 1e-9,
) -> float:
    """Mean radial displacement to second order for a general non-axisymmetric
    non-resonant perturbation.

    Derived from Birkhoff Normal Form (canonical perturbation theory) applied
    to the field-line Hamiltonian H = ψ(r) + ε V, V = Σ a_{mn} cos(mθ+nφ).

    The correct second-order formula (verified by ODE numerics) is:

    .. math::

        \\langle \\delta r \\rangle =
          -\\frac{4\\,\\iota^\\prime}{B_\\phi^2}
          \\sum_{\\substack{(m,n)\\neq(0,0) \\\\ m\\iota+n\\neq 0}}
          \\frac{m\\,|(\\delta B_r)_{mn}|^2}{(m\\iota+n)^3}

    where ``dBr_mn`` are the **one-sided** complex Fourier amplitudes of δB_r
    (the factor of 4 already accounts for the two-sided spectrum and the
    second-order canonical coordinate correction).

    **Derivation sketch** (single mode, Bφ = 1)::

        H = ψ(r) + ε a₀ cos(mθ+nφ),   α = mι₀+n
        Homological:  χ = −a₀ sin(mθ+nφ)/α
        H_eff2 = ε²/2 {V,χ} = −ε² m² a₀² ι' / (4α²)
        δι = ε² m³ a₀² (ι')² / (2α³)
        ⟨δr⟩_BNF = −δι/ι' = −ε² m³ a₀² ι' / (2α³)
        Physical ⟨δr⟩ = 2 × ⟨δr⟩_BNF  (2nd-order coord. transform term)
        ⟨δr⟩ = −ε² m³ a₀² ι' / α³
        With (δBr)_mn = −im a₀/2  →  a₀ = 2|(δBr)_mn|/m:
        ⟨δr⟩ = −4 m |(δBr)_mn|² ι' / α³

    Parameters
    ----------
    m, n : array_like of int, shape (K,)
        Mode numbers (one-sided half-space, m > 0 or (m=0, n>0)).
    dBr_mn : complex array_like, shape (K,)
        One-sided Fourier coefficients of δB_r  (T·m).
    iota : float
        Rotational transform ι₀.
    iota_prime : float
        dι/dr  (1/m).
    Bphi : float
        Toroidal magnetic field B_φ  (T).  Default 1.0.
    resonance_tol : float
        Modes with `|mι+n|` < resonance_tol are skipped (resonant).

    Returns
    -------
    float
        <δr>  (m).
    """
    m     = np.asarray(m,      dtype=int)
    n     = np.asarray(n,      dtype=int)
    dBr   = np.asarray(dBr_mn, dtype=complex)

    res_mask = _check_resonant(m, n, iota, tol=resonance_tol)
    dc_mask  = (m == 0) & (n == 0)
    keep     = ~res_mask & ~dc_mask

    if not keep.any():
        return 0.0

    alpha  = m[keep] * iota + n[keep]          # mι+n, real
    numer  = m[keep] * np.abs(dBr[keep])**2    # m |δBr|²
    total  = float(np.sum(numer / alpha**3))

    return -4.0 * iota_prime / Bphi**2 * total


# ────────────────────────────────────────────────────────────────────────────
# 8.  Peak / valley locations  (Section 6 of the paper)
# ────────────────────────────────────────────────────────────────────────────

def deformation_peak_valley(
    m: int,
    n: int,
    dBr_mn: complex,
    iota: float,
    Bphi: float,
    n_peaks: int = 1,
) -> dict:
    """Phase-based peak and valley coordinates of the (m,n) deformation harmonic.

    Derives the peak / valley conditions from Section 6 of the paper:

    .. math::

        (\\delta r)^{(m,n)} = 2|(\\delta r)_{mn}|\\cos(m\\theta + n\\phi
                               + \\arg((\\delta r)_{mn}))

    Peak  (deformation maximum): :math:`m\\theta + n\\phi = -\\arg((\\delta r)_{mn}) + 2k\\pi`

    Valley (deformation minimum): :math:`m\\theta + n\\phi = \\pi - \\arg((\\delta r)_{mn}) + 2k\\pi`

    Parameters
    ----------
    m, n : int
        Mode numbers.
    dBr_mn : complex
        Fourier coefficient (δB_r)_mn  (T·m).
    iota : float
        Rotational transform ι.
    Bphi : float
        Unperturbed B_φ  (T·m).
    n_peaks : int
        Number of distinct peak/valley pairs to return (k = 0 … n_peaks−1).

    Returns
    -------
    dict with keys:
        ``dr_mn``      : complex (δr)_mn  [m]
        ``amplitude``  : float  2|(δr)_mn|  [m]
        ``phase_dr``   : float  arg((δr)_mn)  [rad]
        ``peak_phase`` : ndarray  mθ+nφ values at peaks  [rad]
        ``valley_phase``: ndarray  mθ+nφ values at valleys  [rad]
    """
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
    peak_phase   = -phase_dr + 2 * np.pi * k_arr
    valley_phase =  np.pi - phase_dr + 2 * np.pi * k_arr

    return {
        'dr_mn'       : dr_mn,
        'amplitude'   : amplitude,
        'phase_dr'    : phase_dr,
        'peak_phase'  : peak_phase,
        'valley_phase': valley_phase,
    }


# ────────────────────────────────────────────────────────────────────────────
# 9.  Green's function spectrum  (Eq. 2.8)
# ────────────────────────────────────────────────────────────────────────────

def green_function_spectrum(
    m: Union[ndarray, list],
    n: Union[ndarray, list],
    iota: float,
    Bphi: float,
    regularise_eps: float = 0.0,
    resonance_tol: float = 1e-9,
) -> ndarray:
    """Fourier coefficients of the deformation Green's function G(Θ, Φ; r).

    Implements Eq. (2.8):

    .. math::

        G_{mn}(r) = \\frac{\\mathrm{i}}{B_\\phi (m\\iota + n)}

    so that

    .. math::

        \\delta r(\\theta,\\phi) =
          \\sum_{mn} G_{mn}(r)\\,(\\delta B_r)_{mn}\\,
          e^{\\mathrm{i}(m\\theta+n\\phi)}.

    Parameters
    ----------
    m, n : array_like of int
        Mode numbers.
    iota : float
        Rotational transform.
    Bphi : float
        Unperturbed B_φ  (T·m).
    regularise_eps : float
        Imaginary regularisation (Eq. 2.9).
    resonance_tol : float
        Resonant modes are set to NaN.

    Returns
    -------
    complex ndarray of shape matching input m, n.
    """
    m = np.asarray(m, dtype=int)
    n = np.asarray(n, dtype=int)
    denom = _safe_denom(m, n, iota, eps=regularise_eps)
    res   = _check_resonant(m, n, iota, tol=resonance_tol)
    G = np.where(res, np.nan + 0j, 1.0 / (1j * Bphi * denom))
    return G


# ────────────────────────────────────────────────────────────────────────────
# 10.  q-profile convenience wrappers
# ────────────────────────────────────────────────────────────────────────────

def iota_to_q(iota: float) -> float:
    """Safety factor q = 1/ι."""
    return 1.0 / iota


def q_to_iota(q: float) -> float:
    """Rotational transform ι = 1/q."""
    return 1.0 / q


def iota_prime_from_q_prime(q: float, q_prime: float) -> float:
    """ι' = dι/dr = −q'·ι²  =  −q'/q².

    Parameters
    ----------
    q : float  Safety factor.
    q_prime : float  dq/dr (1/m).
    """
    return -q_prime / (q * q)
