"""Tests for SolovevEquilibrium."""
import numpy as np
import pytest
from pyna.mag.solovev import SolovevEquilibrium, solovev_iter_like


def test_factory_creates_equilibrium():
    eq = solovev_iter_like(scale=0.3)
    assert eq.R0 == pytest.approx(6.2 * 0.3, rel=1e-6)
    assert eq.a == pytest.approx(2.0 * 0.3, rel=1e-6)


def test_psi_zero_at_axis():
    eq = solovev_iter_like(scale=0.3)
    R_ax, Z_ax = eq.magnetic_axis
    psi_ax = float(eq.psi(np.array([R_ax]), np.array([Z_ax])))
    assert abs(psi_ax) < 0.02, f"psi at axis = {psi_ax}, expected ~0"


def test_psi_near_one_at_lcfs():
    eq = solovev_iter_like(scale=0.3)
    # Outer equatorial point of LCFS
    R_lcfs = eq.R0 + eq.a
    Z_lcfs = 0.0
    psi_lcfs = float(eq.psi(np.array([R_lcfs]), np.array([Z_lcfs])))
    assert abs(psi_lcfs - 1.0) < 0.05, f"psi at LCFS = {psi_lcfs}, expected ~1"


def test_q_profile_returns_array():
    eq = solovev_iter_like(scale=0.3)
    psi_vals = np.linspace(0.1, 0.8, 10)
    q_vals = eq.q_profile(psi_vals, n_theta=128)
    assert q_vals.shape == (10,)


def test_q_profile_positive():
    eq = solovev_iter_like(scale=0.3)
    psi_vals = np.linspace(0.1, 0.8, 8)
    q_vals = eq.q_profile(psi_vals, n_theta=128)
    finite = q_vals[~np.isnan(q_vals)]
    assert len(finite) > 0
    assert np.all(finite > 0)


def test_BR_BZ_shape():
    eq = solovev_iter_like(scale=0.3)
    R = np.array([eq.R0, eq.R0 + 0.1])
    Z = np.array([0.0, 0.0])
    BR, BZ = eq.BR_BZ(R, Z)
    assert BR.shape == (2,)
    assert BZ.shape == (2,)


def test_Bphi_magnitude():
    eq = solovev_iter_like(scale=0.3)
    R_ax, _ = eq.magnetic_axis
    Bphi_ax = float(eq.Bphi(np.array([R_ax])))
    assert abs(Bphi_ax - eq.B0) < 0.5 * eq.B0  # rough check
