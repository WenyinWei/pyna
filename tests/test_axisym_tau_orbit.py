import math

import numpy as np
import pytest

from pyna import _cyna as cyna
from pyna.fields import VectorFieldCylindAxisym
from pyna.toroidal.flt import trace_axisym_tau_closed_orbit


class _NoPythonFieldEvaluation(VectorFieldCylindAxisym):
    def __call__(self, *args, **kwargs):  # pragma: no cover - must stay unused
        raise AssertionError("tau orbit integration must execute inside cyna")


def _circular_field(*, omega=1.0, toroidal_field=0.7, spiral_rate=0.0):
    R = np.linspace(1.0, 3.0, 81)
    Z = np.linspace(-1.0, 1.0, 81)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    radial = RR - 2.0
    BR = spiral_rate * radial - omega * ZZ
    BZ = omega * radial + spiral_rate * ZZ
    BPhi = np.full_like(BR, toroidal_field)
    return _NoPythonFieldEvaluation(R, Z, BR=BR, BZ=BZ, BPhi=BPhi)


def _trace(field, *, seed_R=2.35, step=0.04, maximum_tau=8.0, tolerance=2e-5):
    return trace_axisym_tau_closed_orbit(
        field,
        seed_R_m=seed_R,
        section_Z_m=0.0,
        step_tau_m_per_T=step,
        maximum_tau_m_per_T=maximum_tau,
        closure_tolerance_m=tolerance,
        minimum_seed_poloidal_field_T=1e-10,
    )


def test_axisym_tau_closed_orbit_uses_cyna_and_returns_physical_quadrature():
    assert cyna.trace_axisym_tau_closed_orbit is not None
    omega = 1.0
    toroidal_field = 0.7
    minor_radius = 0.35
    orbit = _trace(
        _circular_field(omega=omega, toroidal_field=toroidal_field),
        seed_R=2.0 + minor_radius,
        step=0.04,
    )

    exact_period = 2.0 * math.pi / omega
    exact_measure = (
        (omega * minor_radius) ** 2 + toroidal_field**2
    ) * exact_period
    exact_phi_advance = (
        toroidal_field
        * 2.0
        * math.pi
        / (omega * math.sqrt(2.0**2 - minor_radius**2))
    )

    assert orbit.closed
    assert orbit.status == "closed"
    assert orbit.production_cyna_executed
    assert orbit.opposite_section_crossed
    assert orbit.path_R_Z_phi.shape == (orbit.tau.size, 3)
    assert orbit.weights_B2_dtau.shape == orbit.tau.shape
    assert np.all(np.diff(orbit.tau) > 0.0)
    assert np.all(orbit.weights_B2_dtau >= 0.0)
    assert orbit.path_R_Z_phi[1, 1] > 0.0  # dZ/dtau = BZ > 0 at the seed
    assert np.min(orbit.path_R_Z_phi[:, 1]) < 0.0
    assert orbit.path_R_Z_phi[-1, 1] == 0.0
    assert orbit.period_tau_m_per_T == orbit.tau[-1]
    assert np.sum(orbit.weights_B2_dtau) == pytest.approx(
        orbit.gauge_measure_B2_dtau_T_m, rel=2e-14
    )
    assert np.sum(orbit.normalized_gauge_weights) == pytest.approx(1.0)
    assert orbit.period_tau_m_per_T == pytest.approx(exact_period, abs=3e-6)
    assert orbit.gauge_measure_B2_dtau_T_m == pytest.approx(
        exact_measure, rel=3e-6
    )
    assert orbit.phi_advance_rad == pytest.approx(exact_phi_advance, rel=3e-6)
    assert orbit.return_defect_m < 2e-7


def test_axisym_tau_orbit_converges_under_step_refinement():
    field = _circular_field()
    exact_period = 2.0 * math.pi
    exact_measure = (0.35**2 + 0.7**2) * exact_period
    coarse = _trace(field, step=0.16, tolerance=2e-3)
    fine = _trace(field, step=0.08, tolerance=2e-4)

    assert coarse.closed and fine.closed
    assert abs(fine.period_tau_m_per_T - exact_period) < abs(
        coarse.period_tau_m_per_T - exact_period
    )
    assert abs(fine.gauge_measure_B2_dtau_T_m - exact_measure) < abs(
        coarse.gauge_measure_B2_dtau_T_m - exact_measure
    )
    assert fine.return_defect_m < coarse.return_defect_m


@pytest.mark.parametrize(
    ("field", "seed_R", "maximum_tau", "tolerance", "expected_status"),
    [
        (_circular_field(), 3.2, 8.0, 1e-5, "seed_outside_domain"),
        (_circular_field(), 2.0, 8.0, 1e-5, "weak_poloidal_field"),
        (
            _circular_field(spiral_rate=0.01),
            2.35,
            8.0,
            1e-4,
            "return_defect_exceeded",
        ),
    ],
)
def test_axisym_tau_orbit_expected_failures_are_explicit(
    field, seed_R, maximum_tau, tolerance, expected_status
):
    orbit = _trace(
        field,
        seed_R=seed_R,
        maximum_tau=maximum_tau,
        tolerance=tolerance,
    )

    assert not orbit.closed
    assert orbit.status == expected_status
    assert orbit.production_cyna_executed
    assert orbit.tau.size == 0
    assert orbit.path_R_Z_phi.shape == (0, 3)
    assert orbit.weights_B2_dtau.size == 0


def test_axisym_tau_orbit_reports_missing_return_without_leaving_domain():
    R = np.linspace(1.0, 3.0, 17)
    Z = np.linspace(-1.0, 1.0, 17)
    shape = (R.size, Z.size)
    field = _NoPythonFieldEvaluation(
        R,
        Z,
        BR=np.zeros(shape),
        BZ=np.full(shape, 0.1),
        BPhi=np.ones(shape),
    )
    orbit = _trace(
        field,
        seed_R=2.0,
        step=0.01,
        maximum_tau=1.0,
        tolerance=1e-5,
    )

    assert not orbit.closed
    assert orbit.status == "return_not_found"
    assert orbit.accepted_steps == 100
    assert orbit.production_cyna_executed
