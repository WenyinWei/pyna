import numpy as np
import pytest

from pyna.fields.cylindrical import VectorFieldCylind
from pyna.toroidal.equilibrium.stellarator import simple_stellarator
from pyna.toroidal.perturbation_spectrum import radial_perturbation_Fourier_spectrum
from pyna.toroidal.torus_deformation import fieldline_deformation_spectrum
from pyna.toroidal.visual.RMP_spectrum import (
    ResonantComponent,
    compare_cyna_fixed_points_for_component,
    find_resonant_components_analytic,
    project_fixed_points_to_deformed_surface,
    rmp_closure_map_span,
)


def _cyna_available():
    try:
        import pyna._cyna as cyna
    except Exception:
        return False
    return bool(cyna.is_available())


def _sample_field(eq, delta_B_func, *, nR=128, nPhi=128):
    lim = 1.18 * eq.r0
    R_grid = np.linspace(eq.R0 - lim, eq.R0 + lim, nR)
    Z_grid = np.linspace(-lim, lim, nR)
    Phi_grid = np.linspace(0.0, 2.0 * np.pi, nPhi, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")

    theta = np.arctan2(ZZ, RR - eq.R0)
    psi = eq.psi_ax(RR, ZZ)
    q = eq.q_of_psi(psi)
    r_minor = np.hypot(RR - eq.R0, ZZ)
    Bphi = eq.B0 * eq.R0 / RR
    Bpol = Bphi * r_minor / (RR * np.maximum(np.abs(q), 1.0e-3))
    BR0 = np.where(r_minor > 1.0e-10, -Bpol * np.sin(theta), 0.0)
    BZ0 = np.where(r_minor > 1.0e-10, Bpol * np.cos(theta), 0.0)
    helical_BR = eq.epsilon_h * eq.B0 * psi * np.cos(eq.m_h * theta - eq.n_h * PP)

    dB = delta_B_func(RR, ZZ, PP)
    return VectorFieldCylind(
        R_grid,
        Z_grid,
        Phi_grid,
        BR=BR0 + helical_BR + dB[0],
        BZ=BZ0 + dB[1],
        BPhi=Bphi + dB[2],
        label="test_rmp_phase_field",
    )


def _helical_velocity_deformation(eq, psi_res, *, n_theta=256, n_phi=256):
    r_res = np.sqrt(psi_res) * eq.r0
    iota = 1.0 / float(eq.q_of_psi(psi_res))
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    TT, PP = np.meshgrid(theta, phi, indexing="xy")
    RR = eq.R0 + r_res * np.cos(TT)
    ZZ = r_res * np.sin(TT)
    Bphi = eq.B0 * eq.R0 / RR
    delta_BR = eq.epsilon_h * eq.B0 * eq.psi_ax(RR, ZZ) * np.cos(eq.m_h * TT - eq.n_h * PP)

    radial_velocity = RR * delta_BR * np.cos(TT) / Bphi
    poloidal_velocity = -RR * delta_BR * np.sin(TT) / (r_res * Bphi)
    radial_spec = radial_perturbation_Fourier_spectrum(
        radial_velocity,
        theta,
        phi,
        m_max=8,
        n_max=8,
        min_amplitude=1.0e-12,
    )
    poloidal_spec = radial_perturbation_Fourier_spectrum(
        poloidal_velocity,
        theta,
        phi,
        m_max=8,
        n_max=8,
        min_amplitude=1.0e-12,
    )
    poloidal_coeffs = np.array([
        poloidal_spec.mode_coefficient(int(m), int(n))
        for m, n in zip(radial_spec.m, radial_spec.n)
    ])
    nonresonant = np.abs(radial_spec.m * iota + radial_spec.n) > 1.0e-9
    return fieldline_deformation_spectrum(
        radial_spec.m[nonresonant],
        radial_spec.n[nonresonant],
        radial_spec.dBr[nonresonant],
        poloidal_coeffs[nonresonant],
        iota=iota,
    )


def test_rmp_closure_span_reduces_harmonics():
    comp = ResonantComponent(
        m=4,
        n=2,
        harmonic_order=2,
        b_mn=1.0 + 0.0j,
        psi_res=0.2,
        q_res=2.0,
        half_width_psi=0.01,
        half_width_r=0.001,
        opoint_theta=0.0,
        xpoint_theta=np.pi / 4.0,
    )

    assert rmp_closure_map_span(comp) == pytest.approx(4.0 * np.pi)


@pytest.mark.skipif(not _cyna_available(), reason="cyna extension is unavailable")
def test_cyna_fixed_points_match_pure_rmp_spectrum_phase():
    eq = simple_stellarator(
        R0=3.0,
        r0=0.3,
        B0=2.5,
        q0=1.5,
        q1=4.5,
        m_h=3,
        n_h=3,
        epsilon_h=0.0,
    )
    base_m, base_n = 2, 1
    B_rmp = 1.0e-3

    def delta_B_RMP(R, Z, phi):
        theta = np.arctan2(Z, R - eq.R0)
        phase = base_m * theta - base_n * phi
        return np.array([
            B_rmp * np.cos(phase) * np.cos(theta),
            B_rmp * np.cos(phase) * np.sin(theta),
            np.zeros_like(np.asarray(theta)),
        ])

    components = find_resonant_components_analytic(
        eq,
        delta_B_RMP,
        base_m=base_m,
        base_n=base_n,
        max_harmonic=1,
        n_theta=64,
        n_phi=32,
    )
    field = _sample_field(eq, delta_B_RMP)

    rows = compare_cyna_fixed_points_for_component(
        field,
        components[0],
        eq,
        DPhi=0.015,
        max_iter=80,
        tol=1.0e-11,
        n_threads=2,
    )

    assert len(rows) == 4
    assert all(row.converged for row in rows)
    assert [row.newton_kind for row in rows] == [row.predicted_kind for row in rows]
    assert max(abs(row.theta_error_deg) for row in rows) < 0.05
    assert max(abs(row.helical_phase_error_deg) for row in rows) < 0.1


@pytest.mark.skipif(not _cyna_available(), reason="cyna extension is unavailable")
def test_nonresonant_deformation_reduces_full_stellarator_phase_error():
    eq = simple_stellarator(
        R0=3.0,
        r0=0.3,
        B0=2.5,
        q0=1.5,
        q1=4.5,
        m_h=3,
        n_h=3,
        epsilon_h=0.03,
    )
    base_m, base_n = 2, 1
    B_rmp = 1.0e-3

    def delta_B_RMP(R, Z, phi):
        theta = np.arctan2(Z, R - eq.R0)
        phase = base_m * theta - base_n * phi
        return np.array([
            B_rmp * np.cos(phase) * np.cos(theta),
            B_rmp * np.cos(phase) * np.sin(theta),
            np.zeros_like(np.asarray(theta)),
        ])

    components = find_resonant_components_analytic(
        eq,
        delta_B_RMP,
        base_m=base_m,
        base_n=base_n,
        max_harmonic=1,
        n_theta=64,
        n_phi=32,
    )
    field = _sample_field(eq, delta_B_RMP)
    rows = compare_cyna_fixed_points_for_component(
        field,
        components[0],
        eq,
        DPhi=0.015,
        max_iter=80,
        tol=1.0e-11,
        n_threads=2,
    )

    raw_max = max(abs(row.theta_error_deg) for row in rows)
    deformation = _helical_velocity_deformation(eq, components[0].psi_res)
    projected = project_fixed_points_to_deformed_surface(
        rows,
        eq,
        deformation,
        r_minor=np.sqrt(components[0].psi_res) * eq.r0,
    )
    corrected_max = max(abs(row.theta_error_deg) for row in projected)

    assert raw_max > 2.0
    assert corrected_max < 1.3
    assert corrected_max < 0.4 * raw_max
