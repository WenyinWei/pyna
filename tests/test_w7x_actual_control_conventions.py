from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "w7x_actual_control_coil_topology.py"
SPEC = spec_from_file_location("w7x_actual_control_convention_test", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_nardon_phase_shift_has_the_thesis_sign_and_wraps_by_island_branch():
    reference = 2.0e-3 * np.exp(0.4j)
    coefficient = abs(reference) * np.exp(1j * (np.angle(reference) + np.deg2rad(15.0)))

    predicted = MODULE._nardon_phase_shift_deg(coefficient, reference, m=5)

    assert predicted == pytest.approx(-3.0)
    assert MODULE._phase_closure_error_deg(-2.8, predicted, m=5) == pytest.approx(0.2)
    assert MODULE._phase_closure_error_deg(69.2, predicted, m=5) == pytest.approx(0.2)


def test_phase_b0_alignment_gate_requires_radial_and_iota_compatibility():
    compatible = SimpleNamespace(
        edge_radial_ratio_rms=0.02,
        iota_profile_error_rms=0.01,
    )
    rejected = SimpleNamespace(
        edge_radial_ratio_rms=0.30,
        iota_profile_error_rms=0.03,
    )

    accepted = MODULE._phase_b0_alignment_gate(
        compatible,
        radial_ratio_limit=0.05,
        iota_rms_limit=0.02,
    )
    failed = MODULE._phase_b0_alignment_gate(
        rejected,
        radial_ratio_limit=0.05,
        iota_rms_limit=0.02,
    )

    assert accepted["passed"] is True
    assert failed["passed"] is False
    assert failed["edge_radial_ratio_passed"] is False
    assert failed["iota_profile_passed"] is False


def test_newton_phase_projection_uses_the_spectrum_pest_surface():
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    radial = np.asarray([0.5, 0.8, 1.0])
    R = 5.5 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, 2, axis=0)
    Z = np.repeat(Z, 2, axis=0)
    case = SimpleNamespace(
        theta_vals=theta,
        radial_labels=radial,
        phi_vals=np.asarray([0.0, np.pi]),
        R_surf=R,
        Z_surf=Z,
        core_reference=SimpleNamespace(axis=np.asarray([5.5, 0.0])),
    )
    m = 5
    phase = float(theta[13])
    theta_O = (phase + 2.0 * np.pi * np.arange(m) / m) % (2.0 * np.pi)
    points = [
        {
            "kind": "O",
            "R": 5.5 + 0.8 * np.cos(value),
            "Z": 0.8 * np.sin(value),
        }
        for value in theta_O
    ]

    measured, projected, radial_shift, residual = MODULE._healed_phase(points, case, 0.8)

    assert measured == pytest.approx(phase)
    np.testing.assert_allclose(
        np.sort(np.mod(projected, 2.0 * np.pi)),
        np.sort(theta_O),
        atol=2.0 * np.pi / theta.size,
    )
    assert np.max(np.abs(radial_shift)) < 1.0e-6
    assert np.max(residual) < 1.0e-6

    outside_points = [
        {
            "kind": "O",
            "R": 5.5 + 1.2 * np.cos(value),
            "Z": 1.2 * np.sin(value),
        }
        for value in theta_O
    ]
    _phase, _theta, _shift, outside_residual, diagnostics = MODULE._healed_phase(
        outside_points,
        case,
        0.8,
        return_diagnostics=True,
    )
    assert diagnostics["all_radial_in_domain"] is False
    assert diagnostics["projection_extrapolates"] is True
    assert diagnostics["max_radial_extrapolation"] == pytest.approx(0.2, abs=1.0e-5)
    # An extrapolating spline can still fit the point accurately; domain
    # validity must therefore be gated independently from residual size.
    assert np.max(outside_residual) < 1.0e-5


def test_constant_spectral_amplitude_phase_path_preserves_width_proxy():
    reference = 2.0e-3 * np.exp(0.3j)
    response = np.asarray([[8.0e-4, -3.0e-4], [2.0e-4, 9.0e-4]])

    controls = MODULE._constant_spectral_amplitude_controls(12.0, reference, response)
    coefficient = reference + complex(*(response @ controls))

    assert abs(coefficient) == pytest.approx(abs(reference), rel=2.0e-14)
    assert np.degrees(np.angle(coefficient / reference)) == pytest.approx(12.0)


def test_newton_phase_calibration_uses_callback_as_authoritative_sensor():
    reference = 1.0 + 0.0j
    response = np.eye(2)

    def measured_newton_phase(controls):
        coefficient = reference + complex(*(response @ np.asarray(controls)))
        # Deliberately differs from the Nardon -delta(arg(b))/m prediction.
        return np.degrees(np.angle(coefficient / reference)) / 5.0

    result = MODULE._calibrate_newton_chain_phase_target(
        2.0,
        reference,
        response,
        ((-1.0, 1.0), (-1.0, 1.0)),
        measured_newton_phase,
        phase_tolerance_deg=1.0e-4,
    )

    assert result["accepted"] is True
    assert result["achieved_shift_deg"] == pytest.approx(2.0, abs=1.0e-4)
    assert result["coefficient_phase_shift_deg"] == pytest.approx(10.0, abs=5.0e-4)
    assert result["relative_spectral_amplitude_error"] < 1.0e-13
    assert "Newton O-chain" in result["sensor"]


def test_newton_phase_calibration_fails_closed_when_target_is_outside_bounds():
    with pytest.raises(RuntimeError, match="outside the bounded"):
        MODULE._calibrate_newton_chain_phase_target(
            5.0,
            1.0 + 0.0j,
            np.eye(2),
            ((-0.02, 0.02), (-0.02, 0.02)),
            lambda controls: np.degrees(np.angle(1.0 + complex(*np.asarray(controls)))) / 5.0,
            coefficient_phase_limit_deg=30.0,
        )
