"""Tests for pyna.mag.equilibrium — synthetic tokamak equilibrium."""
import numpy as np
import pytest
from pyna.toroidal.equilibrium.axisymmetric import EquilibriumTokamakCircularSynthetic, time_linear_weighting


@pytest.fixture()
def eq():
    return EquilibriumTokamakCircularSynthetic()


# ---------------------------------------------------------------------------
# psi_norm
# ---------------------------------------------------------------------------

def test_psi_on_axis(eq):
    """psi_norm should be near 0 at the magnetic axis (grid interpolation accuracy)."""
    psi = eq.psi_norm(eq.R0, 0.0)
    assert abs(float(psi)) < 1e-3  # interpolation error on finite grid


def test_psi_at_lcfs_R(eq):
    """psi_norm at (R0+a, 0) matches the analytic formula."""
    import numpy as np
    R, Z = eq.R0 + eq.a, 0.0
    psi_analytic = (R**2 - eq.R0**2)**2 / (4 * eq.R0**2 * eq.a**2) + Z**2 / eq.a**2
    psi = eq.psi_norm(R, Z)
    assert abs(float(psi) - psi_analytic) < 1e-2  # within 1% of analytic on grid


def test_psi_at_lcfs_Z(eq):
    """psi_norm at (R0, a) matches the analytic formula."""
    import numpy as np
    R, Z = eq.R0, eq.a
    psi_analytic = (R**2 - eq.R0**2)**2 / (4 * eq.R0**2 * eq.a**2) + Z**2 / eq.a**2
    psi = eq.psi_norm(R, Z)
    assert abs(float(psi) - psi_analytic) < 1e-2


def test_psi_array(eq):
    R = np.linspace(eq.R0 - 0.4, eq.R0 + 0.4, 20)
    Z = np.zeros(20)
    psi = eq.psi_norm(R, Z)
    assert psi.shape == (20,)
    assert np.all(np.isfinite(psi))


# ---------------------------------------------------------------------------
# q profile
# ---------------------------------------------------------------------------

def test_q_at_axis(eq):
    assert abs(float(eq.q(0.0)) - eq.q0) < 1e-12


def test_q_at_lcfs(eq):
    assert abs(float(eq.q(1.0)) - eq.q1) < 1e-12


def test_q_monotone(eq):
    S = np.linspace(0, 1, 50)
    q = eq.q(S)
    assert np.all(np.diff(q) > 0)


# ---------------------------------------------------------------------------
# B field
# ---------------------------------------------------------------------------

def test_bphi_on_axis(eq):
    """B_phi = B0 * R0 / R; at R=R0 should equal B0."""
    _, _, BPhi = eq.B_field(eq.R0, 0.0)
    assert abs(float(BPhi) - eq.B0) < 1e-12


def test_bfield_array(eq):
    R = np.linspace(eq.R0 - 0.3, eq.R0 + 0.3, 10)
    Z = np.zeros(10)
    BR, BZ, BPhi = eq.B_field(R, Z)
    assert BR.shape == BZ.shape == BPhi.shape == (10,)


def test_bfield_finite(eq):
    R = np.array([eq.R0, eq.R0 + 0.2, eq.R0 - 0.2])
    Z = np.array([0.0, 0.1, -0.1])
    BR, BZ, BPhi = eq.B_field(R, Z)
    assert np.all(np.isfinite(BR))
    assert np.all(np.isfinite(BZ))
    assert np.all(np.isfinite(BPhi))


# ---------------------------------------------------------------------------
# time_linear_weighting
# ---------------------------------------------------------------------------

def test_time_weighting_exact():
    t = np.array([0.0, 1.0, 2.0])
    d = np.array([[0.0], [2.0], [4.0]])
    result = time_linear_weighting(t, d, 0.5)
    np.testing.assert_allclose(result, [1.0], atol=1e-14)


def test_time_weighting_endpoint():
    t = np.array([0.0, 1.0])
    d = np.array([[3.0], [7.0]])
    result = time_linear_weighting(t, d, 1.0)
    np.testing.assert_allclose(result, [7.0], atol=1e-14)
