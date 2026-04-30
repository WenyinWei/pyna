"""Tests for pyna.mag.coil analytic coil field formulas."""
import numpy as np
import pytest
from scipy.constants import mu_0, pi
from pyna.toroidal.coils.coil import BRBZ_induced_by_current_loop


# On-axis (R->0) check is singular; instead test the well-known on-axis formula
# B_Z(R=0, Z) = mu0 * I * a^2 / (2 * (a^2 + Z^2)^1.5)
# We test at small but nonzero R and verify BR ~= 0 (symmetry) and
# BZ matches the analytic formula.

def test_current_loop_on_axis_BZ():
    """B_Z at small R on axis should match the on-axis analytic value."""
    a = 1.0  # 1 m radius loop
    I = 1000.0  # 1 kA
    Z_o = 0.0
    R_small = 1e-4  # near axis
    Z_test = 0.5
    BR, BZ = BRBZ_induced_by_current_loop(a, Z_o, I, R_small, Z_test)
    BZ_analytic = mu_0 * I * a**2 / (2 * (a**2 + Z_test**2) ** 1.5)
    # Should agree to 0.1%
    assert abs(BZ - BZ_analytic) / abs(BZ_analytic) < 1e-3


def test_current_loop_on_axis_BR_small():
    """B_R near the axis should be much smaller than B_Z (axisymmetry)."""
    a = 1.0
    I = 1000.0
    Z_o = 0.0
    R_small = 1e-4
    Z_test = 0.5
    BR, BZ = BRBZ_induced_by_current_loop(a, Z_o, I, R_small, Z_test)
    # |BR| << |BZ|
    assert abs(BR) < abs(BZ) * 1e-2


def test_current_loop_Z_offset():
    """Loop at Z_o=1.0: the field at Z=1.0 (on-axis) equals the Z_o=0 case at Z=0."""
    a = 0.5
    I = 500.0
    _, BZ_at_loop = BRBZ_induced_by_current_loop(a, Z_o=1.0, I=I, R=1e-4, Z=1.0)
    _, BZ_origin = BRBZ_induced_by_current_loop(a, Z_o=0.0, I=I, R=1e-4, Z=0.0)
    assert abs(BZ_at_loop - BZ_origin) / abs(BZ_origin) < 1e-3


def test_current_loop_array_input():
    """BRBZ should accept numpy array R, Z arguments."""
    a = 1.0
    I = 1000.0
    R = np.linspace(0.01, 2.0, 10)
    Z = np.zeros(10)
    BR, BZ = BRBZ_induced_by_current_loop(a, 0.0, I, R, Z)
    assert BR.shape == (10,)
    assert BZ.shape == (10,)


def test_current_loop_midplane_BR_zero():
    """In the midplane Z=Z_o, B_R should be zero by symmetry."""
    a = 1.0
    I = 1000.0
    R = np.linspace(0.01, 0.9, 5)  # avoid R=a singularity
    Z_o = 0.3
    BR, _ = BRBZ_induced_by_current_loop(a, Z_o, I, R, Z_o * np.ones_like(R))
    np.testing.assert_allclose(BR, 0.0, atol=1e-10)

