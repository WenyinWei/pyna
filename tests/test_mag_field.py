"""Tests for pyna.mag.field — VectorField3D hierarchy for cylindrical grids."""
import numpy as np
import pytest
from pyna.mag.field import (
    CylindricalGridVectorField3D,
    CylindricalGridAxiVectorField3D,
    RegualrCylindricalGridField,
)
from pyna.system import VectorField3D, AxiSymmetricVectorField3D, VectorField


# ---------------------------------------------------------------------------
# Fixtures: small synthetic grids
# ---------------------------------------------------------------------------

@pytest.fixture()
def vf3d():
    R = np.linspace(1.0, 3.0, 20)
    Z = np.linspace(-1.0, 1.0, 20)
    Phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    BR = 0.1 * np.sin(PP)
    BZ = 0.05 * ZZ
    BPhi = 2.0 * R[0] / RR
    return CylindricalGridVectorField3D(R, Z, Phi, BR, BZ, BPhi)


@pytest.fixture()
def axivf3d():
    R = np.linspace(1.0, 3.0, 30)
    Z = np.linspace(-1.0, 1.0, 30)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    BR = np.zeros_like(RR)
    BZ = 0.05 * ZZ
    BPhi = 2.0 / RR
    return CylindricalGridAxiVectorField3D(R, Z, BR, BZ, BPhi)


# ---------------------------------------------------------------------------
# MRO / isinstance
# ---------------------------------------------------------------------------

def test_cgvf3d_mro(vf3d):
    assert isinstance(vf3d, VectorField3D)
    assert isinstance(vf3d, VectorField)


def test_axiavf3d_mro(axivf3d):
    assert isinstance(axivf3d, AxiSymmetricVectorField3D)
    assert isinstance(axivf3d, VectorField3D)


# ---------------------------------------------------------------------------
# dim / state_dim
# ---------------------------------------------------------------------------

def test_cgvf3d_dim(vf3d):
    assert vf3d.dim == 3
    assert vf3d.state_dim == 3


def test_axivf3d_dim(axivf3d):
    assert axivf3d.dim == 3


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

def test_regualr_alias(vf3d):
    assert type(vf3d) is CylindricalGridVectorField3D
    # The alias should refer to the same class
    assert RegualrCylindricalGridField is CylindricalGridVectorField3D


# ---------------------------------------------------------------------------
# Grid properties
# ---------------------------------------------------------------------------

def test_grid_properties(vf3d):
    assert len(vf3d.R) == 20
    assert len(vf3d.Z) == 20
    assert len(vf3d.Phi) == 12
    assert vf3d.BR.shape == (20, 20, 12)


def test_axi_grid_properties(axivf3d):
    assert len(axivf3d.R) == 30
    assert len(axivf3d.Z) == 30
    assert axivf3d.BR.shape == (30, 30)


# ---------------------------------------------------------------------------
# __call__ / interpolation
# ---------------------------------------------------------------------------

def test_cgvf3d_call_single(vf3d):
    pt = np.array([[2.0, 0.0, 0.5]])
    result = vf3d(pt)
    assert result.shape == (1, 3)
    assert np.all(np.isfinite(result))


def test_cgvf3d_call_batch(vf3d):
    pts = np.array([[1.5, -0.3, 0.0], [2.0, 0.1, 1.0], [2.5, 0.5, 2.0]])
    result = vf3d(pts)
    assert result.shape == (3, 3)
    assert np.all(np.isfinite(result))


def test_axivf3d_call_ignores_phi(axivf3d):
    """Axisymmetric field: same result at different phi."""
    pt1 = np.array([[2.0, 0.0, 0.0]])
    pt2 = np.array([[2.0, 0.0, 1.57]])
    r1 = axivf3d(pt1)
    r2 = axivf3d(pt2)
    np.testing.assert_allclose(r1, r2, atol=1e-12)


def test_interpolate_at_cgvf3d(vf3d):
    BR, BZ, BPhi = vf3d.interpolate_at(2.0, 0.0, 0.3)
    assert np.isfinite(float(BR))
    assert np.isfinite(float(BZ))
    assert np.isfinite(float(BPhi))


def test_interpolate_at_axivf3d(axivf3d):
    BR, BZ, BPhi = axivf3d.interpolate_at(2.0, 0.0)
    assert np.isfinite(float(BR))
    assert np.isfinite(float(BPhi))
