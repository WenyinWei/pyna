import numpy as np
import pytest

from pyna.toroidal.equilibrium.stellarator import simple_stellarator
from pyna.toroidal.visual.RMP_spectrum import (
    ResonantComponent,
    circular_shell_divergence_diagnostic,
    compose_magnetic_perturbations,
    compare_cyna_fixed_points_for_component,
    fieldline_velocity_spectrum_on_circular_surface,
    find_resonant_components_analytic,
    NonResonantFieldlineResponse,
    project_fixed_points_to_deformed_surface,
    radial_rmp_field_template,
    rmp_nrmp_mode_rows,
    rmp_closure_map_span,
    sample_stellarator_cylindrical_field,
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


def _periodic_derivative_phi(values, phi):
    dphi = float(phi[1] - phi[0])
    return (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2.0 * dphi)


def _helical_ripple_delta_B(eq):
    def delta_B_helical(R, Z, phi):
        theta = np.arctan2(Z, R - eq.R0)
        psi = eq.psi_ax(R, Z)
        dBR = eq.epsilon_h * eq.B0 * psi * np.cos(eq.m_h * theta - eq.n_h * phi)
        return np.array([
            dBR,
            np.zeros_like(np.asarray(dBR), dtype=float),
            np.zeros_like(np.asarray(dBR), dtype=float),
        ])

    return delta_B_helical


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


@pytest.mark.parametrize("m,n", [(1, 1), (2, 1)])
def test_radial_rmp_template_is_divergence_free_on_circular_shell(m, n):
    axis_R = 3.0
    axis_Z = 0.0
    field = radial_rmp_field_template(
        m,
        n,
        amplitude=1.0e-3,
        phase=0.37,
        axis_R=axis_R,
        axis_Z=axis_Z,
    )

    r = np.linspace(0.08, 0.28, 9)
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    pp, rr, tt = np.meshgrid(phi, r, theta, indexing="ij")
    R = axis_R + rr * np.cos(tt)
    Z = axis_Z + rr * np.sin(tt)
    BR, BZ, Bphi = field(R, Z, pp)
    Br = BR * np.cos(tt) + BZ * np.sin(tt)
    Btheta = -BR * np.sin(tt) + BZ * np.cos(tt)

    radial_flux = rr * R * Br
    poloidal_flux = R * Btheta
    toroidal_flux = rr * Bphi
    d_radial = np.gradient(radial_flux, r, axis=1, edge_order=2)
    d_poloidal = _periodic_derivative_theta(poloidal_flux, theta)
    d_toroidal = _periodic_derivative_phi(toroidal_flux, phi)
    divergence = (d_radial + d_poloidal + d_toroidal) / (rr * R)

    assert getattr(field, "divergence_free") is True
    if m > 1:
        assert np.max(np.abs(Bphi)) == 0.0
    else:
        assert np.max(np.abs(Bphi)) > 0.0
    assert np.max(np.abs(divergence[:, 1:-1])) < 1.0e-5

    diagnostic = circular_shell_divergence_diagnostic(
        field,
        axis_R=axis_R,
        axis_Z=axis_Z,
        r_values=r,
        n_theta=256,
        n_phi=256,
    )
    assert diagnostic.max_abs < 1.0e-5
    assert diagnostic.relative_max < 1.0e-2


def test_m1_radial_rmp_template_controls_resonant_phase():
    eq = simple_stellarator(
        R0=3.0,
        r0=0.3,
        B0=2.5,
        q0=0.75,
        q1=1.25,
        m_h=3,
        n_h=3,
        epsilon_h=0.0,
    )
    base_m, base_n = 1, 1
    phase = 0.43
    component = find_resonant_components_analytic(
        eq,
        radial_rmp_field_template(
            base_m,
            base_n,
            amplitude=1.0e-3,
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

    assert np.angle(component.b_mn) == pytest.approx(phase, abs=1.0e-12)
    assert abs(component.b_mn) == pytest.approx(5.0e-4, rel=1.0e-12)


def test_mixed_rmp_nrmp_workflow_classifies_modes_and_deforms_surface():
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
    psi_res = eq.resonant_psi(2, 1)[0]
    mixed = compose_magnetic_perturbations(
        radial_rmp_field_template(2, 1, amplitude=1.0e-3, axis_R=eq.R0),
        radial_rmp_field_template(3, 1, amplitude=2.0e-4, phase=0.2, axis_R=eq.R0),
        radial_rmp_field_template(1, 1, amplitude=1.5e-4, phase=0.4, axis_R=eq.R0),
    )

    velocity = fieldline_velocity_spectrum_on_circular_surface(
        eq,
        mixed,
        psi_res,
        n_theta=128,
        n_phi=64,
        m_max=4,
        n_max=3,
        min_amplitude=1.0e-12,
    )
    rows = rmp_nrmp_mode_rows(
        velocity.radial_spectrum,
        velocity.iota,
        resonance_tol=1.0e-10,
        top=None,
        min_amplitude=1.0e-8,
    )
    by_mode = {(row.m, row.n): row for row in rows}

    assert by_mode[(2, -1)].kind == "RMP"
    assert by_mode[(2, -1)].detuning == pytest.approx(0.0, abs=1.0e-12)
    assert by_mode[(3, -1)].kind == "nRMP"
    assert by_mode[(1, -1)].kind == "nRMP"

    response = velocity.nonresonant_response(include_shear=True, resonance_tol=1.0e-10)
    assert isinstance(response, NonResonantFieldlineResponse)
    assert response.n_total_modes == velocity.radial_spectrum.m.size
    assert response.n_nonresonant_modes > 2
    assert response.n_resonant_modes >= 2
    assert response.n_nonresonant_modes == np.count_nonzero([
        abs(int(m) * velocity.iota + int(n)) > 1.0e-10
        for m, n in zip(velocity.radial_spectrum.m, velocity.radial_spectrum.n)
    ])

    contribution_rows = response.contribution_rows(top=None)
    assert len(contribution_rows) == response.n_nonresonant_modes
    assert contribution_rows[0].radial_response_weight >= contribution_rows[-1].radial_response_weight
    assert contribution_rows[-1].cumulative_fraction == pytest.approx(1.0, abs=1.0e-12)
    assert any((row.m, row.n) == (3, -1) for row in contribution_rows)
    assert any((row.m, row.n) == (1, -1) for row in contribution_rows)

    counts, cumulative = response.cumulative_contribution()
    assert counts[-1] == response.n_nonresonant_modes
    assert np.all(np.diff(cumulative) >= -1.0e-15)
    assert cumulative[-1] == pytest.approx(1.0, abs=1.0e-12)

    deformation = response.deformation
    legacy_deformation = velocity.nonresonant_deformation(include_shear=True, resonance_tol=1.0e-10)
    assert np.count_nonzero(~deformation.resonant_mask) > 0
    assert np.nanmax(np.abs(deformation.delta_r)) > 0.0
    np.testing.assert_allclose(deformation.delta_r, legacy_deformation.delta_r)
    np.testing.assert_allclose(deformation.delta_theta, legacy_deformation.delta_theta)

    sampled = sample_stellarator_cylindrical_field(
        eq,
        mixed,
        nR=32,
        nPhi=32,
        label="mixed_rmp_nrmp_test",
    )
    assert sampled.BR.shape == (32, 32, 32)
    assert sampled.label == "mixed_rmp_nrmp_test"


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
    delta_B_RMP = radial_rmp_field_template(base_m, base_n, amplitude=B_rmp, axis_R=eq.R0)

    components = find_resonant_components_analytic(
        eq,
        delta_B_RMP,
        base_m=base_m,
        base_n=base_n,
        max_harmonic=1,
        n_theta=64,
        n_phi=32,
    )
    field = sample_stellarator_cylindrical_field(
        eq,
        delta_B_RMP,
        label="test_rmp_phase_field",
    )

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
    assert max(abs(row.theta_error_deg) for row in rows) < 0.08
    assert max(abs(row.helical_phase_error_deg) for row in rows) < 0.16


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
    delta_B_RMP = radial_rmp_field_template(base_m, base_n, amplitude=B_rmp, axis_R=eq.R0)

    components = find_resonant_components_analytic(
        eq,
        delta_B_RMP,
        base_m=base_m,
        base_n=base_n,
        max_harmonic=1,
        n_theta=64,
        n_phi=32,
    )
    field = sample_stellarator_cylindrical_field(
        eq,
        delta_B_RMP,
        label="test_rmp_plus_helical_field",
    )
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
    velocity = fieldline_velocity_spectrum_on_circular_surface(
        eq,
        _helical_ripple_delta_B(eq),
        components[0].psi_res,
        n_theta=256,
        n_phi=256,
        m_max=8,
        n_max=8,
        min_amplitude=1.0e-12,
    )
    deformation = velocity.nonresonant_deformation(include_shear=False)
    projected = project_fixed_points_to_deformed_surface(
        rows,
        eq,
        deformation,
        r_minor=velocity.r_minor,
    )
    corrected_max = max(abs(row.theta_error_deg) for row in projected)

    assert raw_max > 2.0
    assert corrected_max < 1.6
    assert corrected_max < 0.45 * raw_max
