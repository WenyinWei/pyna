import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.toroidal.perturbation.beta_ramp import (
    BetaRampState,
    beta_ramp_states_from_fields,
    beta_scan_summary_rows,
    delta_beta_ramp_state,
    diagnose_beta_ramp_scan,
    diagnose_beta_ramp_state,
    radial_small_divisor_reports,
    sample_beta_ramp_delta_on_surfaces,
    scrub_beta_metadata,
)
from pyna.toroidal.perturbation_spectrum import RadialPerturbationFourierSpectrum


def _toy_surfaces():
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    radial = np.array([0.20, 0.25, 0.30, 0.40, 0.50])
    R0 = 3.0
    theta_grid = theta[None, None, :]
    phi_grid = phi[:, None, None]
    radial_grid = radial[None, :, None]
    R = R0 + radial_grid * np.cos(theta_grid) * np.ones_like(phi_grid)
    Z = radial_grid * np.sin(theta_grid) * np.ones_like(phi_grid)
    return R, Z, phi, theta, radial


def _toy_field_state(
    *,
    beta,
    label,
    perturbation_scale=0.0,
    q_shift=0.0,
    metadata=None,
):
    R_grid = np.linspace(2.2, 3.8, 49)
    Z_grid = np.linspace(-0.8, 0.8, 45)
    Phi_grid = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")
    R0 = 3.0
    minor = np.hypot(RR - R0, ZZ)
    theta = np.arctan2(ZZ, RR - R0)

    B0_phi = 2.0 * RR
    phase = 0.37
    resonant_tilde = 1.0e-3 * (1.0 + minor) * np.cos(5.0 * theta - 2.0 * PP + phase)
    nonres_tilde = 2.5e-4 * np.cos(4.0 * theta - 1.0 * PP - 0.2)
    tilde = perturbation_scale * (resonant_tilde + nonres_tilde)
    delta_B1 = 2.0 * tilde
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    R_surf, Z_surf, phi, theta_vals, radial = _toy_surfaces()
    q_profile = 2.0 + 2.0 * radial + float(q_shift)
    return BetaRampState(
        beta=beta,
        label=label,
        R_grid=R_grid,
        Z_grid=Z_grid,
        Phi_grid=Phi_grid,
        BR=delta_B1 * cos_theta,
        BZ=delta_B1 * sin_theta,
        BPhi=B0_phi,
        R_surf=R_surf,
        Z_surf=Z_surf,
        phi_vals=phi,
        theta_vals=theta_vals,
        radial_labels=radial,
        q_profile=q_profile,
        metadata={} if metadata is None else metadata,
    )


def test_beta_ramp_state_delta_and_vector_field_adapter():
    base = _toy_field_state(beta=0.0, label="toy base")
    state = _toy_field_state(beta=0.02, label="toy beta", perturbation_scale=1.0)

    delta = delta_beta_ramp_state(state, base)

    assert delta.beta == pytest.approx(0.02)
    assert delta.metadata["reference_label"] == "toy base"
    assert delta.metadata["source_label"] == "toy beta"
    assert delta.metadata["beta_delta"] == pytest.approx(0.02)
    np.testing.assert_allclose(delta.BPhi, 0.0)

    field = delta.as_vector_field()
    assert isinstance(field, VectorFieldCylind)
    assert field.BR.shape == state.BR.shape


def test_beta_ramp_state_from_field_mapping_and_native_field_period_sampling():
    nfp = 2
    field_period = 2.0 * np.pi / nfp
    R_grid = np.linspace(0.8, 1.2, 5)
    Z_grid = np.linspace(-0.2, 0.2, 5)
    Phi_grid = np.linspace(0.0, field_period, 32, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")
    base_payload = {
        "R": R_grid,
        "Z": Z_grid,
        "Phi": Phi_grid,
        "B0_R": np.zeros_like(RR),
        "B0_Z": np.zeros_like(RR),
        "B0_Phi": np.ones_like(RR),
        "label": "topoquest-style base",
    }
    state_payload = {
        **base_payload,
        "B0_R": np.cos(2.0 * PP),
        "B0_Z": np.sin(2.0 * PP),
        "label": "topoquest-style state",
    }

    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radial = np.array([0.05])
    R_surf = 1.0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z_surf = radial[None, :, None] * np.sin(theta)[None, None, :]
    R_surf = np.repeat(R_surf, phi.size, axis=0)
    Z_surf = np.repeat(Z_surf, phi.size, axis=0)

    base = BetaRampState.from_field_mapping(
        base_payload,
        nfp=nfp,
        R_surf=R_surf,
        Z_surf=Z_surf,
        phi_vals=phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=np.array([0.4]),
        metadata={"source_path": "/private/local/base"},
    )
    state = BetaRampState.from_field_mapping(
        state_payload,
        nfp=nfp,
        R_surf=R_surf,
        Z_surf=Z_surf,
        phi_vals=phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=np.array([0.4]),
        metadata={"source_path": "/private/local/state"},
    )

    samples = sample_beta_ramp_delta_on_surfaces(state, reference=base)

    expected_phi = np.broadcast_to(phi[:, None, None], R_surf.shape)
    np.testing.assert_allclose(samples.delta_BR, np.cos(2.0 * expected_phi), atol=1.0e-2)
    np.testing.assert_allclose(samples.delta_BZ, np.sin(2.0 * expected_phi), atol=1.0e-2)
    np.testing.assert_allclose(samples.denominator_BPhi, 1.0, atol=1.0e-12)
    assert state.nfp == nfp
    assert state.public_metadata()["source_path"] == "<redacted>"


def test_diagnose_beta_ramp_state_detects_rmp_nrmp_and_trust_report():
    base = _toy_field_state(beta=0.0, label="toy base")
    state = _toy_field_state(beta=0.02, label="toy beta", perturbation_scale=1.0)

    diag = diagnose_beta_ramp_state(
        state,
        reference=base,
        n_values=[1, 2],
        m_values={1: [4], 2: [5]},
        m_max=6,
        n_max=3,
        min_amplitude=1.0e-12,
        min_b_res=1.0e-8,
        small_divisor_tol=5.0e-2,
        min_mode_amplitude=1.0e-12,
    )

    mode_set = {(chain.m, chain.n) for chain in diag.chains}
    assert (5, 2) in mode_set
    chain = next(chain for chain in diag.chains if (chain.m, chain.n) == (5, 2))
    assert chain.radial_label == pytest.approx(0.25, abs=2.5e-3)
    assert chain.b_res == pytest.approx(1.25e-3, rel=6.0e-2)
    assert chain.half_width > 0.0

    report = min(diag.small_divisors, key=lambda item: item.min_abs_miota_plus_n)
    assert report.radial_label == pytest.approx(0.25)
    assert report.resonant_mode_count >= 2
    assert report.nonresonant_norm > 0.0
    assert diag.trust.status == "watch"
    assert "small_divisor_near_resonance" in diag.trust.reasons

    rows = beta_scan_summary_rows([diag])
    assert rows[0]["beta"] == pytest.approx(0.02)
    assert rows[0]["n_chains"] >= 1
    assert rows[0]["dominant_modes"]


def test_beta_ramp_scan_diagnoses_first_reference_and_summary_indices():
    base = _toy_field_state(beta=0.0, label="toy base")
    state1 = _toy_field_state(beta=0.01, label="toy beta 1", perturbation_scale=0.5)
    state2 = _toy_field_state(beta=0.02, label="toy beta 2", perturbation_scale=1.0)

    scan = diagnose_beta_ramp_scan(
        [base, state1, state2],
        reference="first",
        n_values=[1, 2],
        m_values={1: [4], 2: [5]},
        m_max=6,
        n_max=3,
        min_amplitude=1.0e-12,
        min_b_res=1.0e-8,
        small_divisor_tol=5.0e-2,
        min_mode_amplitude=1.0e-12,
    )

    assert scan.reference_mode == "first"
    assert scan.result_indices == (1, 2)
    assert scan.reference_indices == (0, 0)
    assert len(scan.results) == 2
    assert scan.results[0].reference is base
    assert scan.status_counts["watch"] >= 1
    assert scan.first_low_confidence is None

    rows = scan.summary_rows()
    assert rows[0]["scan_index"] == 1
    assert rows[0]["reference_index"] == 0
    assert rows[0]["beta_delta"] == pytest.approx(0.01)
    assert rows[1]["beta_delta"] == pytest.approx(0.02)


def test_beta_ramp_scan_can_use_previous_step_reference():
    base = _toy_field_state(beta=0.0, label="toy base")
    state1 = _toy_field_state(beta=0.01, label="toy beta 1", perturbation_scale=0.5)
    state2 = _toy_field_state(beta=0.02, label="toy beta 2", perturbation_scale=1.0)

    scan = diagnose_beta_ramp_scan(
        [base, state1, state2],
        reference="previous",
        n_values=[2],
        m_values={2: [5]},
        m_max=6,
        n_max=3,
        min_amplitude=1.0e-12,
        min_b_res=1.0e-8,
        small_divisor_tol=5.0e-2,
    )

    assert scan.result_indices == (1, 2)
    assert scan.reference_indices == (0, 1)
    assert scan.results[1].reference is state1
    rows = scan.summary_rows()
    assert rows[1]["reference_label"] == "toy beta 1"
    assert rows[1]["beta_delta"] == pytest.approx(0.01)


def test_beta_ramp_states_from_vector_fields_share_surface_payload():
    base = _toy_field_state(beta=0.0, label="toy base")
    state = _toy_field_state(beta=0.02, label="toy beta", perturbation_scale=1.0)

    built = beta_ramp_states_from_fields(
        [base.as_vector_field(), state.as_vector_field()],
        betas=[0.0, 0.02],
        labels=["built base", "built beta"],
        R_surf=base.R_surf,
        Z_surf=base.Z_surf,
        phi_vals=base.phi_vals,
        theta_vals=base.theta_vals,
        radial_labels=base.radial_labels,
        q_profiles=np.stack([base.q_profile, state.q_profile]),
        metadata=[{"case": "synthetic", "source_path": "/private/base"}, {"case": "synthetic"}],
    )

    assert len(built) == 2
    assert built[0].label == "built base"
    assert built[1].beta == pytest.approx(0.02)
    np.testing.assert_allclose(built[1].BR, state.BR)
    np.testing.assert_allclose(built[1].q_profile, state.q_profile)
    assert built[0].public_metadata()["source_path"] == "<redacted>"

    scan = diagnose_beta_ramp_scan(
        built,
        reference="first",
        n_values=[2],
        m_values={2: [5]},
        m_max=6,
        n_max=3,
        min_amplitude=1.0e-12,
        min_b_res=1.0e-8,
    )
    assert len(scan.results) == 1
    assert scan.summary_rows()[0]["scan_index"] == 1


def test_beta_ramp_small_divisor_near_resonance_and_metadata_gates():
    base = _toy_field_state(beta=0.0, label="toy base")
    state = _toy_field_state(
        beta=0.03,
        label="toy shifted beta",
        perturbation_scale=1.0,
        q_shift=0.04,
        metadata={"equilibrium_residual": 2.0e-3, "trace_exit_count": 2},
    )

    diag = diagnose_beta_ramp_state(
        state,
        reference=base,
        n_values=[2],
        m_values={2: [5]},
        m_max=6,
        n_max=3,
        min_amplitude=1.0e-12,
        small_divisor_tol=5.0e-2,
        trust_kwargs={"equilibrium_residual_low": 1.0e-3},
    )

    assert diag.trust.status == "low-confidence"
    assert "equilibrium_residual_above_threshold" in diag.trust.reasons
    assert "field_line_trace_exits_present" in diag.trust.reasons
    assert any(report.near_resonant_mode_count > 0 for report in diag.small_divisors)


def test_radial_small_divisor_reports_ignore_dc_mode():
    spectrum = RadialPerturbationFourierSpectrum(
        m=np.array([0, 1, 2]),
        n=np.array([0, -1, 1]),
        dBr=np.array([[10.0 + 0.0j, 1.0e-3 + 0.0j, 2.0e-3 + 0.0j]]),
        dBr_grid=np.ones((1, 4, 4), dtype=complex),
        theta=np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False),
        phi=np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False),
        radial_labels=np.array([0.5]),
    )

    report = radial_small_divisor_reports(
        spectrum,
        iota_profile=[1.0],
        small_divisor_tol=1.0e-3,
        min_mode_amplitude=0.0,
    )[0]

    assert (report.mode_m, report.mode_n) == (1, -1)
    assert report.resonant_mode_count == 1


def test_scrub_beta_metadata_redacts_path_like_keys():
    public = scrub_beta_metadata(
        {
            "case": "synthetic",
            "source_path": "local-state-data",
            "screenshot_file": "diagnostic.png",
        }
    )

    assert public["case"] == "synthetic"
    assert public["source_path"] == "<redacted>"
    assert public["screenshot_file"] == "<redacted>"
