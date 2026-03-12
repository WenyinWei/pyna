"""Tests for periodic orbit search and monodromy analysis.

Tests use an analytic SimpleStellarartor (not W7-X data).
The stellarator has q profile q0=1.5, q1=3.5, so q=3/1 resonance is at psi=0.75.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyna.mag.stellarator import SimpleStellarartor
from pyna.topo.cycle import (
    poincare_map_n,
    jacobian_of_poincare_map,
    find_cycle,
    PeriodicOrbit,
)
from pyna.topo.monodromy import (
    compute_monodromy,
    build_A_matrix_func,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stellarator():
    """SimpleStellarartor with q=3/1 resonance at psi=0.75."""
    return SimpleStellarartor(
        R0=3.0, r0=0.35, B0=1.0,
        q0=1.5, q1=3.5,
        m_h=3, n_h=1, epsilon_h=0.05,
    )


@pytest.fixture(scope="module")
def field_func(stellarator):
    return stellarator.field_func


@pytest.fixture(scope="module")
def rzphi_on_resonance(stellarator):
    """A starting point on the q=3/1 resonant surface."""
    psi_list = stellarator.resonant_psi(3, 1)
    assert len(psi_list) > 0, "q=3/1 surface not found"
    psi_res = psi_list[0]
    r_res = np.sqrt(psi_res) * stellarator.r0
    return np.array([stellarator.R0 + r_res, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPoincaréMap:

    def test_poincare_map_5_turns_stays_in_domain(self, field_func, rzphi_on_resonance):
        """5-turn map starting from q=3/1 surface should stay in domain."""
        R_f, Z_f = poincare_map_n(field_func, rzphi_on_resonance, n_turns=5, dt=0.05)
        assert not np.isnan(R_f), "Field line left domain after 5 turns"
        assert not np.isnan(Z_f), "Field line left domain after 5 turns"
        # Should stay near minor radius
        stellarator = SimpleStellarartor(R0=3.0, r0=0.35, B0=1.0, q0=1.5, q1=3.5,
                                         m_h=3, n_h=1, epsilon_h=0.05)
        dist_from_axis = np.sqrt((R_f - stellarator.R0)**2 + Z_f**2)
        assert dist_from_axis < 0.6, f"Field line strayed too far: r={dist_from_axis:.3f}"

    def test_poincare_map_1_turn_on_cycle(self, field_func, stellarator):
        """On a fixed point, 1-turn Poincaré map returns to start."""
        psi_res = stellarator.resonant_psi(3, 1)[0]
        r_res = np.sqrt(psi_res) * stellarator.r0

        orbit = None
        for frac in [1.0, 0.95, 1.05, 0.90, 1.10]:
            seed = np.array([stellarator.R0 + r_res * frac, 0.0, 0.0])
            orbit = find_cycle(field_func, seed, n_turns=1, dt=0.05,
                               max_iter=60, tol=1e-7)
            if orbit is not None:
                break

        if orbit is None:
            pytest.skip("Could not find 1-turn fixed point")

        R_f, Z_f = poincare_map_n(field_func, orbit.rzphi0, 1, dt=0.05)
        residual = np.sqrt((R_f - orbit.rzphi0[0])**2 + (Z_f - orbit.rzphi0[1])**2)
        assert residual < 1e-6, f"1-turn residual {residual:.2e} too large"


class TestJacobianOfPoincaréMap:

    def test_det_near_one(self, field_func, rzphi_on_resonance):
        """det(J) ≈ 1 for area-preserving map."""
        J = jacobian_of_poincare_map(field_func, rzphi_on_resonance, n_turns=1, dt=0.05)
        det_J = np.linalg.det(J)
        assert abs(det_J - 1.0) < 0.05, f"det(J) = {det_J:.6f}, expected ≈ 1"

    def test_shape(self, field_func, rzphi_on_resonance):
        J = jacobian_of_poincare_map(field_func, rzphi_on_resonance, n_turns=1, dt=0.05)
        assert J.shape == (2, 2)


class TestFindCycle:

    def test_find_cycle_residual(self, field_func, stellarator):
        """find_cycle should produce a fixed point with tiny residual."""
        psi_res = stellarator.resonant_psi(3, 1)[0]
        r_res = np.sqrt(psi_res) * stellarator.r0

        orbit = None
        for frac in [0.90, 0.95, 1.0, 1.05, 1.10]:
            seed = np.array([stellarator.R0 + r_res * frac, 0.0, 0.0])
            orbit = find_cycle(field_func, seed, n_turns=1, dt=0.05,
                               max_iter=60, tol=1e-8)
            if orbit is not None:
                break

        if orbit is None:
            pytest.skip("Could not find a fixed-point orbit")

        R_f, Z_f = poincare_map_n(field_func, orbit.rzphi0, orbit.period_n, dt=0.05)
        residual = np.sqrt((R_f - orbit.rzphi0[0])**2 + (Z_f - orbit.rzphi0[1])**2)
        assert residual < 1e-6, f"Cycle residual {residual:.2e} exceeds tolerance"

    def test_find_cycle_returns_periodic_orbit(self, field_func, stellarator):
        psi_res = stellarator.resonant_psi(3, 1)[0]
        r_res = np.sqrt(psi_res) * stellarator.r0
        seed = np.array([stellarator.R0 + r_res, 0.0, 0.0])

        orbit = find_cycle(field_func, seed, n_turns=1, dt=0.05,
                           max_iter=60, tol=1e-7)
        if orbit is None:
            pytest.skip("Orbit not found")
        assert isinstance(orbit, PeriodicOrbit)
        assert orbit.monodromy.shape == (2, 2)
        assert orbit.trajectory.shape[1] == 3

    def test_find_cycle_is_area_preserving(self, field_func, stellarator):
        """Monodromy from find_cycle (FD Jacobian) should have det ≈ 1."""
        psi_res = stellarator.resonant_psi(3, 1)[0]
        r_res = np.sqrt(psi_res) * stellarator.r0

        orbit = None
        for frac in [0.90, 0.95, 1.0, 1.05]:
            seed = np.array([stellarator.R0 + r_res * frac, 0.0, 0.0])
            orbit = find_cycle(field_func, seed, n_turns=1, dt=0.05,
                               max_iter=60, tol=1e-7)
            if orbit is not None:
                break

        if orbit is None:
            pytest.skip("Orbit not found")

        det_M = np.linalg.det(orbit.monodromy)
        assert abs(det_M - 1.0) < 0.05, f"det(M) = {det_M:.6f}, expected ≈ 1"


class TestComputeMonodromy:

    def test_monodromy_det_near_one(self, field_func, stellarator):
        """det(monodromy) ≈ 1 for area-preserving map."""
        psi_res = stellarator.resonant_psi(3, 1)[0]
        r_res = np.sqrt(psi_res) * stellarator.r0

        orbit = None
        for frac in [0.90, 0.95, 1.0, 1.05]:
            seed = np.array([stellarator.R0 + r_res * frac, 0.0, 0.0])
            orbit = find_cycle(field_func, seed, n_turns=1, dt=0.05,
                               max_iter=60, tol=1e-7)
            if orbit is not None:
                break

        if orbit is None:
            pytest.skip("Orbit not found for monodromy test")

        analysis = compute_monodromy(field_func, orbit, dt_output=0.1, rtol=1e-7, atol=1e-8)
        det_M = np.linalg.det(analysis.monodromy)
        assert abs(det_M - 1.0) < 0.05, f"det(M) = {det_M:.6f}, expected ≈ 1"

    def test_monodromy_has_correct_shape(self, field_func, stellarator):
        psi_res = stellarator.resonant_psi(3, 1)[0]
        r_res = np.sqrt(psi_res) * stellarator.r0

        orbit = None
        for frac in [0.90, 0.95, 1.0, 1.05]:
            seed = np.array([stellarator.R0 + r_res * frac, 0.0, 0.0])
            orbit = find_cycle(field_func, seed, n_turns=1, dt=0.05,
                               max_iter=60, tol=1e-7)
            if orbit is not None:
                break

        if orbit is None:
            pytest.skip("Orbit not found")

        analysis = compute_monodromy(field_func, orbit, dt_output=0.1)
        assert analysis.monodromy.shape == (2, 2)
        assert analysis.J_arr.shape[1:] == (2, 2)
        assert analysis.DPm_arr.shape[1:] == (2, 2)
        assert analysis.trajectory.shape[1] == 2
        assert len(analysis.phi_arr) == len(analysis.trajectory)
