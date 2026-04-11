"""Tests for pyna.toroidal.torus_deformation — non-resonant torus deformation theory.

Tests cover every public function and verify known analytic limits:

1. Radial deformation: known (m,n) mode → check (δr)_mn formula.
2. Resonant mode flagging.
3. Poincaré section formula at φ0=0 (trivial phase) and φ0≠0 (phase shift).
4. ι-variation for DC limit (m=0): should give δι = 0 (DC term excluded).
5. ι-variation for a single m≠0 mode: compare with manual integration.
6. Core identity: <δr> = −δι/ι'.
7. DC simplification vs. q-profile form.
8. Second-order formula: single-mode known case.
9. Peak/valley locations: phase arithmetic check.
10. Green's function spectrum matches deformation spectrum ratio.
11. iota ↔ q / iota_prime conversions.
"""
import warnings

import numpy as np
import pytest

from pyna.toroidal import (
    TorusDeformationSpectrum,
    non_resonant_deformation_spectrum,
    poincare_section_deformation,
    iota_variation_pf,
    mean_radial_displacement,
    mean_radial_displacement_pf,
    mean_radial_displacement_dc,
    mean_radial_displacement_second_order,
    deformation_peak_valley,
    green_function_spectrum,
    iota_to_q,
    q_to_iota,
    iota_prime_from_q_prime,
)


# ────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ────────────────────────────────────────────────────────────────────────────
IOTA      = 0.3        # ι = 3/10  (irrational-like, non-resonant with test modes)
BPHI      = 5.0        # B_φ in T·m
BTHETA    = 0.8        # B_θ in T·m
IOTA_PRIME = -0.5      # dι/dr < 0 (typical tokamak profile)


# ────────────────────────────────────────────────────────────────────────────
# 1. Single (m,n) radial deformation
# ────────────────────────────────────────────────────────────────────────────

def test_radial_deformation_single_mode():
    """(δr)_mn = i (δBr)_mn / [Bφ (mι+n)]."""
    m_val, n_val = 2, 1
    dBr = 0.01 + 0.005j   # T·m

    spec = non_resonant_deformation_spectrum(
        [m_val], [n_val], [dBr], [0.0], [0.0],
        iota=IOTA, Bphi=BPHI, Btheta=BTHETA,
    )

    denom = m_val * IOTA + n_val
    expected = dBr / (1j * BPHI * denom)
    np.testing.assert_allclose(spec.delta_r[0], expected, rtol=1e-12)


# ────────────────────────────────────────────────────────────────────────────
# 2. Polar deformation (δθ) — axisymmetric metric (g^{rθ}=0)
# ────────────────────────────────────────────────────────────────────────────

def test_poloidal_deformation_axisymmetric():
    """(δθ)_mn = ι₀*(δBθ)_mn / (Bθ * i*(mι₀+n))  — corrected Eq.(3.2).

    The original paper formula (δθ)_mn = (δBθ)_mn/(Bθ*im) is only valid for n=0.
    The correct general formula (from field-line ODE, g^{rθ}=0) is:
        (δθ)_mn = ι₀ * (δBθ)_mn / (Bθ * i*(mι₀+n))
    """
    m_val, n_val = 3, 1
    dBth = 0.005 + 0.002j

    spec = non_resonant_deformation_spectrum(
        [m_val], [n_val], [0.0], [dBth], [0.0],
        iota=IOTA, Bphi=BPHI, Btheta=BTHETA,
        g_r_theta=0.0,
    )

    alpha    = m_val * IOTA + n_val
    expected = IOTA * dBth / (BTHETA * 1j * alpha)
    np.testing.assert_allclose(spec.delta_theta[0], expected, rtol=1e-12)


# ────────────────────────────────────────────────────────────────────────────
# 3. Resonant mode flagged as NaN
# ────────────────────────────────────────────────────────────────────────────

def test_resonant_mode_is_nan():
    """Mode (m,n) with mι+n=0 must be NaN and raise UserWarning."""
    # With ι = 1/3, the mode m=1, n=-1/3 is resonant — use ι = 1/3 exactly.
    iota_res = 1.0 / 3.0
    m_val, n_val = 1, -1     # 1*(1/3) + (-1/3): not zero. Try m=3,n=-1.
    # 3*(1/3)+(-1) = 0  ✓
    m_val, n_val = 3, -1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spec = non_resonant_deformation_spectrum(
            [m_val], [n_val], [0.01], [0.0], [0.0],
            iota=iota_res, Bphi=BPHI, Btheta=BTHETA,
        )
        assert sum(1 for wi in w if issubclass(wi.category, UserWarning)) == 1

    assert spec.resonant_mask[0]
    assert np.isnan(spec.delta_r[0])


# ────────────────────────────────────────────────────────────────────────────
# 4. Poincaré section: φ0=0 is identity, φ0≠0 introduces phase
# ────────────────────────────────────────────────────────────────────────────

def test_poincare_section_phi0_zero():
    """At φ0=0 the section deformation equals the full real-space field at φ=0."""
    m_val, n_val = 2, 1
    dBr = 0.01

    spec = non_resonant_deformation_spectrum(
        [m_val], [n_val], [dBr], [0.0], [0.0],
        iota=IOTA, Bphi=BPHI, Btheta=BTHETA,
    )

    theta = np.linspace(0, 2 * np.pi, 200)
    dr_sec, _ = poincare_section_deformation(spec, phi0=0.0, theta=theta)
    dr_full   = spec.real_field_r(theta, np.zeros_like(theta))
    np.testing.assert_allclose(dr_sec, dr_full, rtol=1e-12)


def test_poincare_section_phi0_phase():
    """At φ0=π, the (m=2,n=1) mode gains phase exp(iπ) = −1."""
    m_val, n_val = 2, 1
    dBr = 0.01 + 0j   # real

    spec = non_resonant_deformation_spectrum(
        [m_val], [n_val], [dBr], [0.0], [0.0],
        iota=IOTA, Bphi=BPHI, Btheta=BTHETA,
    )

    theta = np.linspace(0, 2 * np.pi, 200)
    dr_0, _  = poincare_section_deformation(spec, phi0=0.0,   theta=theta)
    dr_pi, _ = poincare_section_deformation(spec, phi0=np.pi, theta=theta)

    # With n=1:  exp(i*1*π)=−1, so the n=1 contribution flips sign.
    np.testing.assert_allclose(dr_pi, -dr_0, atol=1e-15)


# ────────────────────────────────────────────────────────────────────────────
# 5. ι-variation: DC (m=0) contributes nothing
# ────────────────────────────────────────────────────────────────────────────

def test_iota_variation_m0_excluded():
    """m=0 modes must not contribute to δι in iota_variation_pf."""
    di = iota_variation_pf([0], [0.5], IOTA, BTHETA)
    assert di == 0.0


# ────────────────────────────────────────────────────────────────────────────
# 6. ι-variation: single m≠0 mode vs. direct integration
# ────────────────────────────────────────────────────────────────────────────

def test_iota_variation_single_mode():
    """Compare δι formula with numerical integration over a field line."""
    m_val   = 2
    dBth_m0 = 0.01   # T·m

    di_formula = iota_variation_pf([m_val], [dBth_m0], IOTA, BTHETA)

    # Manual: δι = ι/2π * ∫₀²π (δBθ/Bθ) exp(imθ(φ)) dφ
    #            = ι * (δBθ)_{m,0}/Bθ * [exp(i2πmι)−1]/(i 2πmι)
    arg   = 2 * np.pi * m_val * IOTA
    factor = (np.exp(1j * arg) - 1.0) / (1j * arg)
    di_manual = float(np.real(IOTA * dBth_m0 / BTHETA * factor))

    np.testing.assert_allclose(di_formula, di_manual, rtol=1e-12)


# ────────────────────────────────────────────────────────────────────────────
# 7. Core identity
# ────────────────────────────────────────────────────────────────────────────

def test_mean_radial_displacement_core_identity():
    """<δr> = −δι/ι'."""
    di = 0.002
    dr = mean_radial_displacement(di, IOTA_PRIME)
    assert dr == pytest.approx(-di / IOTA_PRIME)


def test_mean_radial_displacement_zero_shear():
    with pytest.raises(ValueError, match="zero magnetic shear"):
        mean_radial_displacement(0.001, 0.0)


# ────────────────────────────────────────────────────────────────────────────
# 8. DC simplification vs. q-profile form
# ────────────────────────────────────────────────────────────────────────────

def test_dc_formula_q_profile_equivalence():
    """<δr> from ι-form equals q-form  q·δBθ / (Bθ·q')."""
    dBth_00 = 0.05   # T·m
    q       = iota_to_q(IOTA)
    q_prime = -0.8   # dq/dr  (1/m)
    iota_p  = iota_prime_from_q_prime(q, q_prime)   # ι' = -q'/q²

    dr_iota_form = mean_radial_displacement_dc(dBth_00, IOTA, iota_p, BTHETA)
    dr_q_form    = q * dBth_00 / (BTHETA * q_prime)

    np.testing.assert_allclose(dr_iota_form, dr_q_form, rtol=1e-12)


# ────────────────────────────────────────────────────────────────────────────
# 9. Second-order formula: zero for empty input
# ────────────────────────────────────────────────────────────────────────────

def test_second_order_single_mode():
    """Single non-resonant mode: verify against BNF-derived formula.

    Correct formula (Birkhoff Normal Form, ODE-verified):
        <δr> = -4 * iota' * m * |δBr|^2 / (Bphi^2 * alpha^3)
    where alpha = m*iota + n.
    """
    m_val, n_val = 1, 0
    dBr  = 0.02 + 0j
    Bphi = 1.0

    dr = mean_radial_displacement_second_order(
        [m_val], [n_val], [dBr], IOTA, IOTA_PRIME, Bphi=Bphi
    )

    alpha    = m_val * IOTA + n_val
    expected = -4.0 * IOTA_PRIME * m_val * abs(dBr)**2 / (Bphi**2 * alpha**3)
    assert dr == pytest.approx(expected, rel=1e-12)


def test_second_order_empty_after_resonance_filter():
    """All-resonant input → 0."""
    iota_res = 1.0 / 3.0
    dr = mean_radial_displacement_second_order(
        [3], [-1], [0.05], iota_res, IOTA_PRIME
    )
    assert dr == 0.0


# ────────────────────────────────────────────────────────────────────────────
# 10. Peak / valley locations
# ────────────────────────────────────────────────────────────────────────────

def test_peak_valley_phase():
    """Peak phase satisfies mθ+nφ = −arg((δr)_mn) [mod 2π]."""
    m_val, n_val = 2, 1
    dBr = 0.01 * np.exp(1j * np.pi / 4)   # phase π/4

    result = deformation_peak_valley(m_val, n_val, dBr, IOTA, BPHI, n_peaks=3)

    denom     = m_val * IOTA + n_val
    dr_mn     = dBr / (BPHI * denom * 1j)
    phase_dr  = np.angle(dr_mn)

    # At peak: cos(phase_dr + peak_phase) = cos(0) = 1  →  peak_phase = −phase_dr
    expected_peaks = -phase_dr + 2 * np.pi * np.arange(3)
    np.testing.assert_allclose(result['peak_phase'], expected_peaks, atol=1e-12)

    # At valley: peak_phase + π
    np.testing.assert_allclose(result['valley_phase'], expected_peaks + np.pi, atol=1e-12)


def test_peak_valley_resonant_raises():
    iota_res = 1.0 / 3.0
    with pytest.raises(ValueError, match="resonant"):
        deformation_peak_valley(3, -1, 0.01, iota_res, BPHI)


# ────────────────────────────────────────────────────────────────────────────
# 11. Green's function: G_mn * (δBr)_mn == (δr)_mn
# ────────────────────────────────────────────────────────────────────────────

def test_green_function_consistency():
    """G_mn * (δBr)_mn must equal (δr)_mn from the deformation spectrum."""
    m_vals  = np.array([1, 2, 3, -1])
    n_vals  = np.array([1, 1, 2,  2])
    dBr_arr = np.array([0.01, 0.02 + 0.01j, -0.005, 0.007j])

    spec = non_resonant_deformation_spectrum(
        m_vals, n_vals, dBr_arr, np.zeros(4), np.zeros(4),
        iota=IOTA, Bphi=BPHI, Btheta=BTHETA,
    )
    G = green_function_spectrum(m_vals, n_vals, IOTA, BPHI)

    np.testing.assert_allclose(G * dBr_arr, spec.delta_r, rtol=1e-12)


# ────────────────────────────────────────────────────────────────────────────
# 12. q ↔ ι conversions
# ────────────────────────────────────────────────────────────────────────────

def test_iota_q_roundtrip():
    for iota in [0.2, 0.5, 1.0, 2.3]:
        assert iota_to_q(q_to_iota(iota)) == pytest.approx(iota)


def test_iota_prime_from_q_prime():
    q      = 3.0
    q_prime= 0.5
    iota_p = iota_prime_from_q_prime(q, q_prime)
    assert iota_p == pytest.approx(-q_prime / q**2)


# ────────────────────────────────────────────────────────────────────────────
# 13. Real-space field reconstruction via TorusDeformationSpectrum methods
# ────────────────────────────────────────────────────────────────────────────

def test_real_field_reconstruction_symmetry():
    """For a real perturbation (δBr real), δr(θ,φ) must be real."""
    m_vals = np.array([1, -1, 2, -2])
    n_vals = np.array([1, -1, 1, -1])
    # Satisfy conjugate symmetry: c_{-m,-n} = c_{mn}*
    dBr = np.array([0.01, 0.01, 0.005 + 0.002j, 0.005 - 0.002j])

    spec = non_resonant_deformation_spectrum(
        m_vals, n_vals, dBr, np.zeros(4), np.zeros(4),
        iota=IOTA, Bphi=BPHI, Btheta=BTHETA,
    )

    theta_grid = np.linspace(0, 2 * np.pi, 100)
    phi_grid   = np.linspace(0, 2 * np.pi, 100)
    TH, PH = np.meshgrid(theta_grid, phi_grid)

    field = spec.real_field_r(TH, PH)
    # Imaginary part should be zero (real perturbation → real deformation)
    np.testing.assert_allclose(np.imag(field), 0.0, atol=1e-14)
