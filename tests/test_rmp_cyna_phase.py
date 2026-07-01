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
    radial_rmp_field_template,
    rmp_closure_map_span,
)


def _cyna_available():
    try:
        import pyna._cyna as cyna
    except Exception:
        return False
    return bool(cyna.is_available())


def _wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _periodic_derivative_theta(values, theta):
    dtheta = float(theta[1] - theta[0])
    return (np.roll(values, -1, axis=-1) - np.roll(values, 1, axis=-1)) / (2.0 * dtheta)


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


def test_resonant_rmp_spectrum_has_expected_amplitude_orders():
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
    amplitudes = np.array([2.5e-4, 5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3])
    b_abs = []
    widths = []
    phases = []

    for amplitude in amplitudes:
        component = find_resonant_components_analytic(
            eq,
            radial_rmp_field_template(base_m, base_n, amplitude=amplitude, axis_R=eq.R0),
            base_m=base_m,
            base_n=base_n,
            max_harmonic=1,
            n_theta=64,
            n_phi=32,
            min_amplitude=1.0e-14,
        )[0]
        b_abs.append(abs(component.b_mn))
        widths.append(component.half_width_r)
        phases.append(component.opoint_theta)

    b_slope = np.polyfit(np.log(amplitudes), np.log(b_abs), 1)[0]
    width_slope = np.polyfit(np.log(amplitudes), np.log(widths), 1)[0]

    assert b_slope == pytest.approx(1.0, abs=1.0e-12)
    assert width_slope == pytest.approx(0.5, abs=1.0e-12)
    assert np.ptp(np.unwrap(phases)) < 1.0e-12


def test_radial_rmp_template_is_divergence_free_on_circular_shell():
    axis_R = 3.0
    axis_Z = 0.0
    field = radial_rmp_field_template(
        2,
        1,
        amplitude=1.0e-3,
        phase=0.37,
        axis_R=axis_R,
        axis_Z=axis_Z,
    )

    r = np.linspace(0.08, 0.28, 9)
    theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    phi = 0.41
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    R = axis_R + rr * np.cos(tt)
    Z = axis_Z + rr * np.sin(tt)
    BR, BZ, Bphi = field(R, Z, phi)
    Br = BR * np.cos(tt) + BZ * np.sin(tt)
    Btheta = -BR * np.sin(tt) + BZ * np.cos(tt)

    radial_flux = rr * R * Br
    poloidal_flux = R * Btheta
    d_radial = np.gradient(radial_flux, r, axis=0, edge_order=2)
    d_poloidal = _periodic_derivative_theta(poloidal_flux, theta)
    divergence = (d_radial + d_poloidal) / (rr * R)

    assert getattr(field, "divergence_free") is True
    assert np.max(np.abs(Bphi)) == 0.0
    assert np.max(np.abs(divergence[1:-1])) < 3.0e-6


def test_resonant_phase_template_controls_xo_phase_order():
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

    def component_for_template(amplitude=1.0e-3, phase=0.0):
        return find_resonant_components_analytic(
            eq,
            radial_rmp_field_template(
                base_m,
                base_n,
                amplitude=amplitude,
                phase=phase,
                axis_R=eq.R0,
            ),
            base_m=base_m,
            base_n=base_n,
            max_harmonic=1,
            n_theta=128,
            n_phi=64,
            min_amplitude=1.0e-16,
        )[0]

    comp_pos = component_for_template(amplitude=1.0e-3)
    comp_neg = component_for_template(amplitude=-1.0e-3)

    phase_jump = np.angle(comp_neg.b_mn / comp_pos.b_mn)
    assert abs(abs(phase_jump) - np.pi) < 1.0e-12
    assert comp_neg.opoint_theta == pytest.approx(comp_pos.xpoint_theta, abs=1.0e-12)
    assert comp_neg.xpoint_theta == pytest.approx(comp_pos.opoint_theta, abs=1.0e-12)

    phase_controls = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
    eta = 0.4
    measured_b_phase = []
    exact_phase_residual = []
    first_order_theta_residual = []
    for k in phase_controls:
        template_phase = k + eta * k * k
        component = component_for_template(phase=template_phase)
        darg_b = float(_wrap_to_pi(np.angle(component.b_mn / comp_pos.b_mn)))
        dtheta_o = float(_wrap_to_pi(component.opoint_theta - comp_pos.opoint_theta))

        measured_b_phase.append(abs(darg_b))
        exact_phase_residual.append(abs(float(_wrap_to_pi(base_m * dtheta_o + darg_b))))
        first_order_theta_residual.append(abs(float(_wrap_to_pi(dtheta_o + k / base_m))))

        assert darg_b == pytest.approx(template_phase, abs=1.0e-12)

    measured_b_phase = np.asarray(measured_b_phase)
    exact_phase_residual = np.asarray(exact_phase_residual)
    first_order_theta_residual = np.asarray(first_order_theta_residual)
    phase_slope = np.polyfit(np.log(phase_controls), np.log(measured_b_phase), 1)[0]
    residual_slope = np.polyfit(np.log(phase_controls), np.log(first_order_theta_residual), 1)[0]

    assert phase_slope == pytest.approx(1.0, abs=0.08)
    assert np.max(exact_phase_residual) < 1.0e-12
    assert residual_slope == pytest.approx(2.0, abs=1.0e-9)


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
