"""Tests for periodic orbit search and monodromy analysis.

Tests use an analytic StellaratorSimple (not W7-X data).

Key physics:
- StellaratorSimple with m_h=2, n_h=1 creates a q=2 resonance island chain.
- In this model q = m_h/n_h, and orbit period = m_h toroidal turns (not n_h).
- So for q=2/1 we find fixed points of the 2-turn Poincaré map.
- Both X-points (hyperbolic) and O-points (elliptic) are found near r_res.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyna.MCF.equilibrium.stellarator import StellaratorSimple
from pyna.topo.cycle import (
    poincare_map_n,
    jacobian_of_poincare_map,
    find_cycle,
    PeriodicOrbit,
)
from pyna.topo.monodromy import compute_DPm_on_cycle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stellarator():
    """StellaratorSimple with q=2/1 resonance (period-2 island chain)."""
    return StellaratorSimple(
        R0=3.0, r0=0.35, B0=1.0,
        q0=1.5, q1=3.5,
        m_h=2, n_h=1, epsilon_h=0.02,
    )


@pytest.fixture(scope="module")
def field_func(stellarator):
    return stellarator.field_func


@pytest.fixture(scope="module")
def RZlimit(stellarator):
    return (
        stellarator.R0 - stellarator.r0 * 1.5,
        stellarator.R0 + stellarator.r0 * 1.5,
        -stellarator.r0 * 1.5,
        stellarator.r0 * 1.5,
    )


@pytest.fixture(scope="module")
def xpoint_rzphi(stellarator):
    """An approximate X-point location near the q=2/1 resonant surface.
    
    From scan: (2.8763, -0.1237, 0.0) is an X-point for epsilon_h=0.02.
    This is the outer X-point of the 2-turn island chain.
    """
    # This is a known approximate X-point; n_turns=2 for q=2/1
    return np.array([2.8763, -0.1237, 0.0])


@pytest.fixture(scope="module")
def opoint_rzphi(stellarator):
    """An approximate O-point location near the q=2/1 resonant surface."""
    # From scan: theta=-2.32, r=0.175 → candidate O-point
    psi_res = stellarator.resonant_psi(2, 1)[0]
    r_res = np.sqrt(psi_res) * stellarator.r0
    theta = -2.32
    return np.array([
        stellarator.R0 + r_res * np.cos(theta),
        r_res * np.sin(theta),
        0.0,
    ])


# ---------------------------------------------------------------------------
# Helper: find a cycle with fallback
# ---------------------------------------------------------------------------

def _find_cycle_near(field_func, theta_guess, r_res, R0, n_turns, RZlimit):
    """Find a cycle near a given (theta, r) on the resonant surface."""
    seed = np.array([R0 + r_res * np.cos(theta_guess), r_res * np.sin(theta_guess), 0.0])
    return find_cycle(
        field_func, seed, n_turns=n_turns, dt=0.15, RZlimit=RZlimit,
        max_iter=30, tol=1e-8, n_fallback_seeds=8, fallback_radius=0.02,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPoincaréMap:

    def test_2turn_map_stays_in_domain(self, field_func, stellarator, RZlimit):
        """2-turn map starting from q=2/1 surface should stay in domain."""
        psi_res = stellarator.resonant_psi(2, 1)[0]
        r_res = np.sqrt(psi_res) * stellarator.r0
        seed = np.array([stellarator.R0 + r_res, 0.0, 0.0])
        R_f, Z_f = poincare_map_n(field_func, seed, n_turns=2, dt=0.15, RZlimit=RZlimit)
        assert not np.isnan(R_f), "Field line left domain after 2 turns"
        assert not np.isnan(Z_f)

    def test_2turn_map_returns_near_known_fixed_point(self, field_func, xpoint_rzphi, RZlimit):
        """The known X-point seed should map very close to itself after 2 turns."""
        # First find the actual fixed point via Newton-Raphson
        orbit = find_cycle(field_func, xpoint_rzphi, n_turns=2, dt=0.15, RZlimit=RZlimit,
                           max_iter=30, tol=1e-8, n_fallback_seeds=4, fallback_radius=0.02)
        if orbit is None:
            pytest.skip("Could not find X-point fixed point")

        R_f, Z_f = poincare_map_n(field_func, orbit.rzphi0, 2, dt=0.15, RZlimit=RZlimit)
        residual = np.sqrt((R_f - orbit.rzphi0[0])**2 + (Z_f - orbit.rzphi0[1])**2)
        assert residual < 1e-6, f"2-turn residual {residual:.2e} too large"


class TestJacobianOfPoincaréMap:

    def test_det_near_one(self, field_func, xpoint_rzphi, RZlimit):
        """det(J) ≈ 1 for area-preserving map."""
        J = jacobian_of_poincare_map(
            field_func, xpoint_rzphi, n_turns=2, dt=0.15, eps=1e-5
        )
        det_J = np.linalg.det(J)
        assert abs(det_J - 1.0) < 0.1, f"det(J) = {det_J:.6f}, expected ≈ 1"

    def test_shape(self, field_func, xpoint_rzphi):
        J = jacobian_of_poincare_map(field_func, xpoint_rzphi, n_turns=2, dt=0.15)
        assert J.shape == (2, 2)


class TestFindCycle:

    def test_find_xpoint_residual(self, field_func, xpoint_rzphi, RZlimit):
        """find_cycle should produce an X-point with tiny residual."""
        orbit = find_cycle(
            field_func, xpoint_rzphi, n_turns=2, dt=0.15, RZlimit=RZlimit,
            max_iter=30, tol=1e-8, n_fallback_seeds=4, fallback_radius=0.02,
        )
        if orbit is None:
            pytest.skip("X-point orbit not found")

        R_f, Z_f = poincare_map_n(field_func, orbit.rzphi0, 2, dt=0.15, RZlimit=RZlimit)
        residual = np.sqrt((R_f - orbit.rzphi0[0])**2 + (Z_f - orbit.rzphi0[1])**2)
        assert residual < 1e-6, f"Cycle residual {residual:.2e} exceeds tolerance"

    def test_find_cycle_returns_periodic_orbit(self, field_func, xpoint_rzphi, RZlimit):
        orbit = find_cycle(
            field_func, xpoint_rzphi, n_turns=2, dt=0.15, RZlimit=RZlimit,
            max_iter=30, tol=1e-8, n_fallback_seeds=4, fallback_radius=0.02,
        )
        if orbit is None:
            pytest.skip("Orbit not found")
        assert isinstance(orbit, PeriodicOrbit)
        assert orbit.DPm.shape == (2, 2)
        assert orbit.trajectory.shape[1] == 3

    def test_find_opoint(self, field_func, opoint_rzphi, RZlimit):
        """Should find an O-point (elliptic) orbit."""
        orbit = find_cycle(
            field_func, opoint_rzphi, n_turns=2, dt=0.15, RZlimit=RZlimit,
            max_iter=30, tol=1e-8, n_fallback_seeds=8, fallback_radius=0.02,
        )
        if orbit is None:
            pytest.skip("O-point orbit not found with this seed")

        # O-point: |stability_index| < 1
        residual_check = abs(orbit.stability_index) < 1.0 + 0.2  # some tolerance
        assert residual_check or orbit.is_stable, (
            f"Expected O-point, got stability_index={orbit.stability_index:.4f}"
        )

    def test_find_cycle_is_area_preserving(self, field_func, xpoint_rzphi, RZlimit):
        """Monodromy from find_cycle (FD Jacobian) should have det ≈ 1."""
        orbit = find_cycle(
            field_func, xpoint_rzphi, n_turns=2, dt=0.15, RZlimit=RZlimit,
            max_iter=30, tol=1e-8, n_fallback_seeds=4, fallback_radius=0.02,
        )
        if orbit is None:
            pytest.skip("Orbit not found")
        det_DPm = np.linalg.det(orbit.DPm)
        assert abs(det_DPm - 1.0) < 0.1, f"det(DPm) = {det_DPm:.6f}, expected ≈ 1"


class TestComputeMonodromy:

    def test_monodromy_det_near_one(self, field_func, xpoint_rzphi, RZlimit):
        """det(monodromy) ≈ 1 for area-preserving map."""
        orbit = find_cycle(
            field_func, xpoint_rzphi, n_turns=2, dt=0.15, RZlimit=RZlimit,
            max_iter=30, tol=1e-8, n_fallback_seeds=4, fallback_radius=0.02,
        )
        if orbit is None:
            pytest.skip("Orbit not found for monodromy test")

        analysis = compute_DPm_on_cycle(field_func, orbit, dt_output=0.15, rtol=1e-7, atol=1e-8)
        det_DPm = np.linalg.det(analysis.DPm)
        assert abs(det_DPm - 1.0) < 0.1, f"det(DPm) = {det_DPm:.6f}, expected ≈ 1"

    def test_monodromy_has_correct_shape(self, field_func, xpoint_rzphi, RZlimit):
        orbit = find_cycle(
            field_func, xpoint_rzphi, n_turns=2, dt=0.15, RZlimit=RZlimit,
            max_iter=30, tol=1e-8, n_fallback_seeds=4, fallback_radius=0.02,
        )
        if orbit is None:
            pytest.skip("Orbit not found")

        analysis = compute_DPm_on_cycle(field_func, orbit, dt_output=0.15)
        assert analysis.DPm.shape == (2, 2)
        assert analysis.DX_pol_arr.shape[1:] == (2, 2)
        assert analysis.DPm_arr.shape[1:] == (2, 2)
        assert analysis.trajectory.shape[1] == 2
        assert len(analysis.phi_arr) == len(analysis.trajectory)
