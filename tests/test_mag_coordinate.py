"""Tests for pyna.mag.coordinate — cylindrical/Cartesian transforms."""
import numpy as np
import pytest
from pyna.mag.coordinate import (
    rzphi_to_xyz,
    xyz_to_rzphi,
    coord_system_change,
    coord_mirror,
)

RNG = np.random.default_rng(42)


def _random_rzphi(n=20):
    R = RNG.uniform(0.5, 3.0, n)
    Z = RNG.uniform(-1.0, 1.0, n)
    Phi = RNG.uniform(0, 2 * np.pi, n)
    return np.stack((R, Z, Phi), axis=-1)


def _random_xyz(n=20):
    x = RNG.uniform(-2.0, 2.0, n)
    y = RNG.uniform(-2.0, 2.0, n)
    z = RNG.uniform(-1.0, 1.0, n)
    return np.stack((x, y, z), axis=-1)


# ---------------------------------------------------------------------------
# Round-trip: RZPhi -> XYZ -> RZPhi
# ---------------------------------------------------------------------------

def test_roundtrip_rzphi_xyz_rzphi():
    rzphi = _random_rzphi()
    xyz = rzphi_to_xyz(rzphi, category="coord")
    rzphi2 = xyz_to_rzphi(xyz, category="coord")
    # R and Z should round-trip exactly
    np.testing.assert_allclose(rzphi2[..., 0], rzphi[..., 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(rzphi2[..., 1], rzphi[..., 1], atol=1e-12, rtol=0)
    # phi: arctan2 gives consistent angle, compare trig
    np.testing.assert_allclose(
        np.cos(rzphi2[..., 2]), np.cos(rzphi[..., 2]), atol=1e-12
    )
    np.testing.assert_allclose(
        np.sin(rzphi2[..., 2]), np.sin(rzphi[..., 2]), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Round-trip: XYZ -> RZPhi -> XYZ
# ---------------------------------------------------------------------------

def test_roundtrip_xyz_rzphi_xyz():
    xyz = _random_xyz()
    rzphi = xyz_to_rzphi(xyz, category="coord")
    xyz2 = rzphi_to_xyz(rzphi, category="coord")
    np.testing.assert_allclose(xyz2, xyz, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Known values
# ---------------------------------------------------------------------------

def test_rzphi_to_xyz_known():
    # R=1, Z=0, phi=0 -> (1, 0, 0)
    pt = np.array([[1.0, 0.0, 0.0]])
    result = rzphi_to_xyz(pt)
    np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0], atol=1e-14)


def test_rzphi_to_xyz_phi_pi():
    # R=1, Z=0, phi=pi -> (-1, 0, 0)
    pt = np.array([[1.0, 0.0, np.pi]])
    result = rzphi_to_xyz(pt)
    np.testing.assert_allclose(result[0], [-1.0, 0.0, 0.0], atol=1e-14)


def test_xyz_to_rzphi_known():
    # (1, 0, 0) -> (1, 0, 0)
    pt = np.array([[1.0, 0.0, 0.0]])
    result = xyz_to_rzphi(pt)
    np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-14)  # R
    np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-14)  # Z
    np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-14)  # phi


# ---------------------------------------------------------------------------
# coord_system_change
# ---------------------------------------------------------------------------

def test_coord_system_change_identity():
    arr = _random_rzphi()
    result = coord_system_change("RZPhi", "RZPhi", arr)
    np.testing.assert_array_equal(result, arr)


def test_coord_system_change_xyz_rzphi():
    xyz = _random_xyz()
    r1 = coord_system_change("XYZ", "RZPhi", xyz)
    r2 = xyz_to_rzphi(xyz)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# coord_mirror
# ---------------------------------------------------------------------------

def test_coord_mirror_rzphi_xy():
    rzphi = _random_rzphi()
    mirrored = coord_mirror("RZPhi", rzphi, "xy")
    np.testing.assert_array_equal(mirrored[..., 0], rzphi[..., 0])   # R unchanged
    np.testing.assert_array_equal(mirrored[..., 2], rzphi[..., 2])   # phi unchanged
    np.testing.assert_array_equal(mirrored[..., 1], -rzphi[..., 1])  # Z negated


def test_coord_mirror_xyz_xy():
    xyz = _random_xyz()
    mirrored = coord_mirror("XYZ", xyz, "xy")
    np.testing.assert_array_equal(mirrored[..., 0], xyz[..., 0])
    np.testing.assert_array_equal(mirrored[..., 1], xyz[..., 1])
    np.testing.assert_array_equal(mirrored[..., 2], -xyz[..., 2])
