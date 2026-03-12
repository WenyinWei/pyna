"""Tests for pyna.field_data — CylindricalVectorField and CylindricalScalarField."""

import numpy as np
import pytest
import tempfile
import os

from pyna.field_data import CylindricalVectorField, CylindricalScalarField
from pyna.vector_calc import magnitude


# ---- Helpers ----------------------------------------------------------------

def make_simple_vfield(nR=12, nZ=10, nPhi=8):
    """Uniform toroidal field B_phi = 1/R, VR=VZ=0."""
    R = np.linspace(1.0, 2.0, nR)
    Z = np.linspace(-0.5, 0.5, nZ)
    Phi = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)
    VPhi = 1.0 / R[:, None, None] * np.ones((nR, nZ, nPhi))
    VR = np.zeros((nR, nZ, nPhi))
    VZ = np.zeros((nR, nZ, nPhi))
    return CylindricalVectorField(R=R, Z=Z, Phi=Phi, VR=VR, VZ=VZ, VPhi=VPhi)


def make_simple_sfield(nR=12, nZ=10, nPhi=8):
    """Scalar field s = R²."""
    R = np.linspace(1.0, 2.0, nR)
    Z = np.linspace(-0.5, 0.5, nZ)
    Phi = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)
    value = R[:, None, None]**2 * np.ones((nR, nZ, nPhi))
    return CylindricalScalarField(R=R, Z=Z, Phi=Phi, value=value)


# ---- Tests ------------------------------------------------------------------

def test_vfield_shape_assertion():
    """Wrong array shape raises AssertionError."""
    R = np.linspace(1, 2, 5)
    Z = np.linspace(-1, 1, 4)
    Phi = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    with pytest.raises(AssertionError):
        CylindricalVectorField(R=R, Z=Z, Phi=Phi,
                               VR=np.zeros((5, 4, 4)),  # wrong nPhi
                               VZ=np.zeros((5, 4, 3)),
                               VPhi=np.zeros((5, 4, 3)))


def test_vfield_call_returns_correct_shape():
    """Interpolation returns arrays with the same shape as input."""
    v = make_simple_vfield()
    R_q = np.array([1.3, 1.5, 1.7])
    Z_q = np.array([0.0, 0.1, -0.1])
    phi_q = np.array([0.5, 1.0, 2.0])
    vr, vz, vph = v(R_q, Z_q, phi_q)
    assert vr.shape == (3,)
    assert vz.shape == (3,)
    assert vph.shape == (3,)


def test_vfield_interpolation_close_to_grid():
    """Interpolated value at a grid point should equal the grid value."""
    v = make_simple_vfield()
    iR, iZ, iPhi = 5, 4, 3
    R_q = np.array([v.R[iR]])
    Z_q = np.array([v.Z[iZ]])
    phi_q = np.array([v.Phi[iPhi]])
    vr, vz, vph = v(R_q, Z_q, phi_q)
    assert np.isclose(vph[0], v.VPhi[iR, iZ, iPhi], rtol=1e-5)


def test_vfield_from_callable():
    """from_callable builds a field matching the source function."""
    def field_func(rzphi):
        # Simple: (BR=0, BZ=0, Bphi=1/R)
        R = rzphi[0]
        return np.array([0.0, 0.0, 1.0 / R])

    R = np.linspace(1.0, 2.0, 8)
    Z = np.linspace(-0.3, 0.3, 6)
    Phi = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    v = CylindricalVectorField.from_callable(field_func, R, Z, Phi, n_workers=2)

    # Check one interior point
    iR, iZ, iPhi = 3, 2, 1
    expected = 1.0 / R[iR]
    assert np.isclose(v.VPhi[iR, iZ, iPhi], expected, rtol=1e-10)


def test_vfield_to_from_npz_roundtrip():
    """to_npz / from_npz round-trip preserves all data."""
    v = make_simple_vfield()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_field")
        v.to_npz(path)
        v2 = CylindricalVectorField.from_npz(path + ".npz")
    assert np.allclose(v.VPhi, v2.VPhi)
    assert np.allclose(v.R, v2.R)
    assert v.field_periods == v2.field_periods


def test_sfield_to_from_npz_roundtrip():
    s = make_simple_sfield()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_scalar")
        s.to_npz(path)
        s2 = CylindricalScalarField.from_npz(path + ".npz")
    assert np.allclose(s.value, s2.value)


def test_magnitude_uniform_field():
    """Magnitude of (0, 0, 1/R) field equals 1/R."""
    v = make_simple_vfield()
    mag = magnitude(v)
    R3d = v.R[:, None, None]
    expected = 1.0 / R3d * np.ones_like(mag)
    assert np.allclose(mag, expected, rtol=1e-10)
