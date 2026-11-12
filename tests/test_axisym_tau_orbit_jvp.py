import math

import numpy as np
import pytest

from pyna import _cyna as cyna
from pyna.fields import VectorFieldCylindAxisym
from pyna.toroidal.flt import (
    AXISYM_TAU_RETURN_BRANCH,
    trace_axisym_tau_closed_orbit,
    trace_axisym_tau_closed_orbit_jvp,
)


class _NoPythonFieldEvaluation(VectorFieldCylindAxisym):
    def __call__(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("tau tangent integration must execute inside cyna")


def _grid():
    R = np.linspace(1.0, 3.0, 129)
    Z = np.linspace(-1.0, 1.0, 129)
    return R, Z, np.meshgrid(R, Z, indexing="ij")


def _ellipse_field(alpha):
    R, Z, (RR, ZZ) = _grid()
    return _NoPythonFieldEvaluation(
        R,
        Z,
        BR=-(1.0 + alpha) * ZZ,
        BZ=(RR - 2.0) / (1.0 + alpha),
        BPhi=np.full_like(RR, 0.7 + 0.2 * alpha),
    )


def _ellipse_direction():
    R, Z, (RR, ZZ) = _grid()
    return _NoPythonFieldEvaluation(
        R,
        Z,
        BR=-ZZ,
        BZ=-(RR - 2.0),
        BPhi=np.full_like(RR, 0.2),
    )


def _zero_direction(R, Z):
    zeros = np.zeros((R.size, Z.size))
    return _NoPythonFieldEvaluation(
        R, Z, BR=zeros, BZ=zeros, BPhi=zeros
    )


def _scaled_poloidal_field(alpha):
    R, Z, (RR, ZZ) = _grid()
    scale = 1.0 + alpha
    return _NoPythonFieldEvaluation(
        R,
        Z,
        BR=-scale * ZZ,
        BZ=scale * (RR - 2.0),
        BPhi=np.full_like(RR, 0.7),
    )


def _nonuniform_magnitude_field(alpha):
    R, Z, (RR, ZZ) = _grid()
    return _NoPythonFieldEvaluation(
        R,
        Z,
        BR=-ZZ,
        BZ=RR - 2.0,
        BPhi=0.7 + 0.1 * ZZ + alpha * (0.05 + 0.02 * RR),
    )


def _nonuniform_magnitude_direction():
    R, Z, (RR, _) = _grid()
    zeros = np.zeros_like(RR)
    return _NoPythonFieldEvaluation(
        R, Z, BR=zeros, BZ=zeros, BPhi=0.05 + 0.02 * RR
    )


_SEAM_SHIFT_PHASE = 2.0 * math.pi - 1.5e-3
_SEAM_SHIFT_SPEED = 0.5
_SEAM_SHIFT_CUBIC = 1.0e4


def _last_interval_extremum_field(alpha):
    R, Z, (RR, ZZ) = _grid()
    phase = (
        _SEAM_SHIFT_PHASE
        + _SEAM_SHIFT_SPEED * alpha
        + _SEAM_SHIFT_CUBIC * alpha**3
    )
    scale = 0.1 / 0.3
    return _NoPythonFieldEvaluation(
        R,
        Z,
        BR=-ZZ,
        BZ=RR - 2.0,
        BPhi=(
            0.7
            + scale * math.cos(phase) * (RR - 2.0)
            + scale * math.sin(phase) * ZZ
        ),
    )


def _last_interval_extremum_direction():
    R, Z, (RR, ZZ) = _grid()
    scale = 0.1 * _SEAM_SHIFT_SPEED / 0.3
    return _NoPythonFieldEvaluation(
        R,
        Z,
        BR=np.zeros_like(RR),
        BZ=np.zeros_like(RR),
        BPhi=(
            -scale * math.sin(_SEAM_SHIFT_PHASE) * (RR - 2.0)
            + scale * math.cos(_SEAM_SHIFT_PHASE) * ZZ
        ),
    )


_TRACE = dict(
    seed_R_m=2.3,
    section_Z_m=0.0,
    step_tau_m_per_T=0.02,
    maximum_tau_m_per_T=8.0,
    closure_tolerance_m=2.0e-6,
    minimum_seed_poloidal_field_T=1.0e-10,
)


def _jvp(field, direction, **kwargs):
    return trace_axisym_tau_closed_orbit_jvp(
        field,
        direction,
        return_direction=1,
        return_branch=AXISYM_TAU_RETURN_BRANCH,
        **(_TRACE | kwargs),
    )


def test_compiled_ellipse_tangent_and_central_fd_convergence(monkeypatch):
    assert cyna.trace_axisym_tau_closed_orbit_jvp is not None
    import pyna.toroidal.flt.axisym_tau as axisym_tau_module

    original_primal = axisym_tau_module._cyna_trace_axisym_tau_closed_orbit
    monkeypatch.setattr(
        axisym_tau_module,
        "_cyna_trace_axisym_tau_closed_orbit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the tangent API must not retrace a baseline orbit")
        ),
    )
    tangent = _jvp(_ellipse_field(0.0), _ellipse_direction())
    monkeypatch.setattr(
        axisym_tau_module, "_cyna_trace_axisym_tau_closed_orbit", original_primal
    )
    primal = trace_axisym_tau_closed_orbit(_ellipse_field(0.0), **_TRACE)

    assert tangent.closed
    assert tangent.production_cyna_tangent_executed
    assert tangent.orbit.production_cyna_executed
    assert not tangent.orbit.B_minimum_event.certified
    assert not tangent.orbit.B_maximum_event.certified
    assert not tangent.B_minimum_event_jvp.certified
    assert not tangent.B_maximum_event_jvp.certified
    assert tangent.orbit.path_R_Z_phi.shape == (316, 3)
    assert tangent.return_direction == 1
    assert tangent.return_branch == AXISYM_TAU_RETURN_BRANCH
    np.testing.assert_array_equal(tangent.orbit.tau, primal.tau)
    np.testing.assert_array_equal(
        tangent.orbit.path_R_Z_phi, primal.path_R_Z_phi
    )
    np.testing.assert_array_equal(
        tangent.orbit.weights_B2_dtau, primal.weights_B2_dtau
    )
    phase = tangent.orbit.tau / tangent.orbit.period_tau_m_per_T
    exact_Z_jvp = -0.3 * np.sin(2.0 * np.pi * phase)
    np.testing.assert_allclose(
        tangent.path_R_Z_phi_jvp[:, 0], 0.0, rtol=0.0, atol=9.0e-16
    )
    np.testing.assert_allclose(
        tangent.path_R_Z_phi_jvp[:, 1], exact_Z_jvp, rtol=0.0, atol=5.0e-11
    )
    assert abs(tangent.period_tau_jvp_m_per_T) < 2.0e-14
    assert tangent.gauge_measure_B2_dtau_jvp == pytest.approx(
        2.0 * math.pi * (2.0 * 0.7 * 0.2 - 0.3**2), abs=5.0e-8
    )
    assert np.sum(tangent.weights_B2_dtau_jvp) == pytest.approx(
        tangent.gauge_measure_B2_dtau_jvp, rel=0.0, abs=3.0e-15
    )
    assert abs(float(np.sum(tangent.normalized_gauge_weights_jvp))) < 3.0e-15

    path_errors = []
    weight_errors = []
    measure_errors = []
    for epsilon in (0.2, 0.1, 0.05, 0.025):
        plus = trace_axisym_tau_closed_orbit(_ellipse_field(epsilon), **_TRACE)
        minus = trace_axisym_tau_closed_orbit(_ellipse_field(-epsilon), **_TRACE)
        assert plus.closed and minus.closed
        assert plus.tau.size == minus.tau.size == tangent.orbit.tau.size
        path_fd = (plus.path_R_Z_phi - minus.path_R_Z_phi) / (2.0 * epsilon)
        weights_fd = (
            plus.weights_B2_dtau - minus.weights_B2_dtau
        ) / (2.0 * epsilon)
        measure_fd = (
            plus.gauge_measure_B2_dtau_T_m
            - minus.gauge_measure_B2_dtau_T_m
        ) / (2.0 * epsilon)
        path_errors.append(
            float(
                np.linalg.norm(path_fd - tangent.path_R_Z_phi_jvp)
                / np.sqrt(path_fd.size)
            )
        )
        weight_errors.append(
            float(np.linalg.norm(weights_fd - tangent.weights_B2_dtau_jvp))
        )
        measure_errors.append(
            abs(measure_fd - tangent.gauge_measure_B2_dtau_jvp)
        )
    for errors in (path_errors, weight_errors, measure_errors):
        assert min(
            errors[index] / errors[index + 1]
            for index in range(len(errors) - 1)
        ) > 3.7


def test_moving_seed_and_section_follow_same_directed_return_event():
    R, Z, (RR, ZZ) = _grid()
    field = _NoPythonFieldEvaluation(
        R,
        Z,
        BR=-ZZ,
        BZ=RR - 2.0,
        BPhi=np.full_like(RR, 0.7),
    )
    seed_R_jvp = 0.04
    section_Z_jvp = -0.03
    tangent = _jvp(
        field,
        _zero_direction(R, Z),
        seed_R_jvp_m=seed_R_jvp,
        section_Z_jvp_m=section_Z_jvp,
        phi_start_jvp_rad=0.07,
    )

    tau = tangent.orbit.tau
    exact_R = seed_R_jvp * np.cos(tau) - section_Z_jvp * np.sin(tau)
    exact_Z = seed_R_jvp * np.sin(tau) + section_Z_jvp * np.cos(tau)
    np.testing.assert_allclose(
        tangent.path_R_Z_phi_jvp[:, 0], exact_R, rtol=0.0, atol=4.0e-10
    )
    np.testing.assert_allclose(
        tangent.path_R_Z_phi_jvp[:, 1], exact_Z, rtol=0.0, atol=4.0e-10
    )
    assert abs(tangent.period_tau_jvp_m_per_T) < 3.0e-10
    assert tangent.path_R_Z_phi_jvp[0, 1] == section_Z_jvp
    assert tangent.path_R_Z_phi_jvp[-1, 1] == section_Z_jvp
    np.testing.assert_allclose(
        tangent.return_displacement_R_Z_jvp_m,
        0.0,
        rtol=0.0,
        atol=1.1e-10,
    )


def test_return_event_time_and_phase_quadrature_jvp():
    R, Z, (RR, ZZ) = _grid()
    direction = _NoPythonFieldEvaluation(
        R,
        Z,
        BR=-ZZ,
        BZ=RR - 2.0,
        BPhi=np.zeros_like(RR),
    )
    tangent = _jvp(_scaled_poloidal_field(0.0), direction)

    assert tangent.period_tau_jvp_m_per_T == pytest.approx(
        -2.0 * math.pi, abs=3.0e-8
    )
    np.testing.assert_allclose(
        tangent.tau_jvp_m_per_T,
        tangent.period_tau_jvp_m_per_T
        * tangent.orbit.tau
        / tangent.orbit.period_tau_m_per_T,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert np.max(
        np.linalg.norm(tangent.path_R_Z_phi_jvp[:, :2], axis=1)
    ) < 1.2e-7
    assert tangent.gauge_measure_B2_dtau_jvp == pytest.approx(
        2.0 * math.pi * (0.3**2 - 0.7**2), abs=2.0e-8
    )

    period_errors = []
    measure_errors = []
    for epsilon in (4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5):
        plus = trace_axisym_tau_closed_orbit(
            _scaled_poloidal_field(epsilon), **_TRACE
        )
        minus = trace_axisym_tau_closed_orbit(
            _scaled_poloidal_field(-epsilon), **_TRACE
        )
        assert plus.accepted_steps == minus.accepted_steps == 315
        period_errors.append(
            abs(
                (plus.period_tau_m_per_T - minus.period_tau_m_per_T)
                / (2.0 * epsilon)
                - tangent.period_tau_jvp_m_per_T
            )
        )
        measure_errors.append(
            abs(
                (
                    plus.gauge_measure_B2_dtau_T_m
                    - minus.gauge_measure_B2_dtau_T_m
                )
                / (2.0 * epsilon)
                - tangent.gauge_measure_B2_dtau_jvp
            )
        )
    for errors in (period_errors, measure_errors):
        assert min(
            errors[index] / errors[index + 1]
            for index in range(len(errors) - 1)
        ) > 3.7


def test_directed_section_branch_and_grid_identity_fail_closed():
    field = _ellipse_field(0.0)
    direction = _ellipse_direction()
    mismatch = trace_axisym_tau_closed_orbit_jvp(
        field,
        direction,
        return_direction=-1,
        return_branch=AXISYM_TAU_RETURN_BRANCH,
        **_TRACE,
    )
    assert not mismatch.closed
    assert mismatch.orbit.status == "directed_section_mismatch"
    assert mismatch.return_direction == 1
    assert mismatch.orbit.tau.size == 0
    assert mismatch.path_R_Z_phi_jvp.shape == (0, 3)

    with pytest.raises(ValueError, match="return_branch"):
        trace_axisym_tau_closed_orbit_jvp(
            field,
            direction,
            return_direction=1,
            return_branch="longest_contour",
            **_TRACE,
        )

    R = np.linspace(1.0, 3.0, 127)
    Z = np.linspace(-1.0, 1.0, 129)
    with pytest.raises(ValueError, match="same grid"):
        trace_axisym_tau_closed_orbit_jvp(
            field,
            _zero_direction(R, Z),
            return_direction=1,
            return_branch=AXISYM_TAU_RETURN_BRANCH,
            **_TRACE,
        )


def test_continuous_non_degenerate_B_extrema_and_compiled_event_jvp():
    tangent = _jvp(
        _nonuniform_magnitude_field(0.0),
        _nonuniform_magnitude_direction(),
    )
    assert tangent.closed
    events = (
        (tangent.orbit.B_minimum_event, tangent.B_minimum_event_jvp, 1),
        (tangent.orbit.B_maximum_event, tangent.B_maximum_event_jvp, -1),
    )
    for event, event_jvp, curvature_sign in events:
        assert event.certified
        assert event_jvp.certified
        assert math.copysign(1.0, event.d2_B2_dtau2_T4_per_m2) == curvature_sign
        assert 0.0 < event.tau_m_per_T < tangent.orbit.period_tau_m_per_T
        assert np.all(np.isfinite(event.position_R_Z_phi))
        assert np.all(np.isfinite(event_jvp.position_R_Z_phi_jvp))

    errors = []
    for epsilon in (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4):
        plus = trace_axisym_tau_closed_orbit(
            _nonuniform_magnitude_field(epsilon), **_TRACE
        )
        minus = trace_axisym_tau_closed_orbit(
            _nonuniform_magnitude_field(-epsilon), **_TRACE
        )
        row = []
        for name, event_jvp in (
            ("B_minimum_event", tangent.B_minimum_event_jvp),
            ("B_maximum_event", tangent.B_maximum_event_jvp),
        ):
            plus_event = getattr(plus, name)
            minus_event = getattr(minus, name)
            assert plus_event.certified and minus_event.certified
            actual = np.concatenate(
                (
                    (event_jvp.tau_jvp_m_per_T,),
                    event_jvp.position_R_Z_phi_jvp,
                    (event_jvp.B_jvp_T,),
                )
            )
            centered = np.concatenate(
                (
                    ((plus_event.tau_m_per_T - minus_event.tau_m_per_T)
                     / (2.0 * epsilon),),
                    (plus_event.position_R_Z_phi - minus_event.position_R_Z_phi)
                    / (2.0 * epsilon),
                    ((plus_event.B_T - minus_event.B_T) / (2.0 * epsilon),),
                )
            )
            row.append(float(np.linalg.norm(centered - actual)))
        errors.append(row)
    errors = np.asarray(errors)
    assert np.max(errors[-1]) < 2.0e-7
    assert np.min(errors[0] / np.maximum(errors[-1], 1.0e-15)) > 20.0


def test_continuous_extremum_in_final_directed_return_interval_is_not_dropped():
    tangent = _jvp(
        _last_interval_extremum_field(0.0),
        _last_interval_extremum_direction(),
    )
    event = tangent.orbit.B_maximum_event
    event_jvp = tangent.B_maximum_event_jvp
    assert event.certified and event_jvp.certified
    assert event.interval_index == tangent.orbit.accepted_steps - 1
    assert tangent.orbit.tau[-2] < event.tau_m_per_T < tangent.orbit.tau[-1]
    assert event.tau_m_per_T == pytest.approx(_SEAM_SHIFT_PHASE, abs=1.0e-8)
    assert event_jvp.tau_jvp_m_per_T == pytest.approx(
        _SEAM_SHIFT_SPEED, abs=2.0e-7
    )

    errors = []
    for epsilon in (2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4):
        plus = trace_axisym_tau_closed_orbit(
            _last_interval_extremum_field(epsilon), **_TRACE
        ).B_maximum_event
        minus = trace_axisym_tau_closed_orbit(
            _last_interval_extremum_field(-epsilon), **_TRACE
        ).B_maximum_event
        assert plus.certified and minus.certified
        centered = np.concatenate(
            (
                ((plus.tau_m_per_T - minus.tau_m_per_T) / (2.0 * epsilon),),
                (plus.position_R_Z_phi - minus.position_R_Z_phi)
                / (2.0 * epsilon),
                ((plus.B_T - minus.B_T) / (2.0 * epsilon),),
            )
        )
        actual = np.concatenate(
            (
                (event_jvp.tau_jvp_m_per_T,),
                event_jvp.position_R_Z_phi_jvp,
                (event_jvp.B_jvp_T,),
            )
        )
        errors.append(float(np.linalg.norm(centered - actual)))
    assert errors[-1] < 7.0e-4
    assert min(errors[index] / errors[index + 1] for index in range(3)) > 3.7
