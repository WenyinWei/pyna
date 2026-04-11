"""Tests for toroidal field-line measurement functions."""
import numpy as np
import pytest
from pyna.toroidal.diagnostics import (
    field_line_length,
    field_line_endpoints,
    field_line_min_psi,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic streamlines
# ---------------------------------------------------------------------------

def _straight_line_rzphi(n=100):
    """Straight line along Z: R=2, phi=0, Z from 0 to 1 m."""
    R = np.full(n, 2.0)
    Z = np.linspace(0.0, 1.0, n)
    Phi = np.zeros(n)
    return np.stack((R, Z, Phi), axis=-1)


def _circle_in_rz(n=200, R0=2.0, r=0.3):
    """Circle in (R, Z) plane at constant phi=0."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = R0 + r * np.cos(theta)
    Z = r * np.sin(theta)
    Phi = np.zeros(n)
    return np.stack((R, Z, Phi), axis=-1)


def _toroidal_helix(n=1000, R0=2.0, r=0.3, q_val=2.0):
    """Helical field line on a torus: makes q_val poloidal turns per toroidal turn."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta = q_val * phi  # q = dtheta/dphi
    R = R0 + r * np.cos(theta)
    Z = r * np.sin(theta)
    return np.stack((R, Z, phi), axis=-1)


# ---------------------------------------------------------------------------
# field_line_length
# ---------------------------------------------------------------------------

def test_length_straight_z():
    """Straight line along Z of length 1 m should return ≈1 m."""
    sl = _straight_line_rzphi()
    L = field_line_length(sl)
    assert abs(L - 1.0) < 1e-3


def test_length_circle():
    """Circle of radius r in (R, Z): circumference = 2π r."""
    r = 0.3
    sl = _circle_in_rz(n=500, r=r)
    L = field_line_length(sl)
    expected = 2 * np.pi * r
    assert abs(L - expected) / expected < 1e-2


def test_length_positive():
    sl = _toroidal_helix()
    assert field_line_length(sl) > 0.0


# ---------------------------------------------------------------------------
# field_line_endpoints
# ---------------------------------------------------------------------------

def test_endpoints_shape():
    sl = _straight_line_rzphi()
    start, end = field_line_endpoints(sl)
    assert start.shape == (3,)
    assert end.shape == (3,)


def test_endpoints_values():
    sl = _straight_line_rzphi()
    start, end = field_line_endpoints(sl)
    np.testing.assert_allclose(start, [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(end, [2.0, 1.0, 0.0], atol=1e-12)


def test_endpoints_independent_copies():
    sl = _straight_line_rzphi()
    start, end = field_line_endpoints(sl)
    start[0] = 99.0
    # Original streamline should be unmodified
    assert sl[0, 0] == 2.0


# ---------------------------------------------------------------------------
# field_line_min_psi
# ---------------------------------------------------------------------------

def test_min_psi_basic():
    sl = _toroidal_helix(n=200, R0=2.0, r=0.3)

    def psi_interp(R, Z):
        """Synthetic psi: 0 at (2.0, 0), increases with distance from axis."""
        return (R - 2.0)**2 / 0.3**2 + Z**2 / 0.3**2

    min_psi = field_line_min_psi(sl, psi_interp)
    assert 0.0 <= min_psi <= 2.0  # should be within the helix


def test_min_psi_straight():
    """For a straight line at R=2, Z=0..1, min psi should be at Z=0."""
    sl = _straight_line_rzphi()

    def psi_interp(R, Z):
        return Z  # psi = Z, minimum at Z=0

    min_psi = field_line_min_psi(sl, psi_interp)
    assert abs(min_psi) < 1e-10
