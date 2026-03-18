"""Tests for pyna.system — core dynamical system class hierarchy."""
import pytest
from pyna.system import (
    DynamicalSystem,
    NonAutonomousDynamicalSystem,
    AutonomousDynamicalSystem,
    VectorField,
    VectorField1D,
    VectorField2D,
    VectorField3D,
    VectorField4D,
    VectorField3DAxiSymmetric,
)


# ---------------------------------------------------------------------------
# Helpers: minimal concrete subclasses
# ---------------------------------------------------------------------------

class _ConcreteVF1D(VectorField1D):
    def __call__(self, x, t=None):
        return -x


class _ConcreteVF2D(VectorField2D):
    def __call__(self, xy, t=None):
        return (-xy[1], xy[0])  # rotation


class _ConcreteVF3D(VectorField3D):
    def __call__(self, xyz, t=None):
        return (0.0, 0.0, 1.0)


class _ConcreteAxiVF3D(VectorField3DAxiSymmetric):
    """Minimal concrete subclass used only for MRO checks."""
    pass  # inherits __init__(R, Z, VR_2d, VZ_2d, VPhi_2d) from VectorField3DAxiSymmetric


# ---------------------------------------------------------------------------
# dim / state_dim
# ---------------------------------------------------------------------------

def test_vf1d_dim():
    vf = _ConcreteVF1D()
    assert vf.dim == 1
    assert vf.state_dim == 1


def test_vf2d_dim():
    vf = _ConcreteVF2D()
    assert vf.dim == 2
    assert vf.state_dim == 2


def test_vf3d_dim():
    vf = _ConcreteVF3D()
    assert vf.dim == 3
    assert vf.state_dim == 3


def test_vf4d_dim():
    class _VF4D(VectorField4D):
        def __call__(self, x, t=None):
            return x
    vf = _VF4D()
    assert vf.dim == 4
    assert vf.state_dim == 4


# ---------------------------------------------------------------------------
# MRO / isinstance
# ---------------------------------------------------------------------------

def test_vf1d_mro():
    vf = _ConcreteVF1D()
    assert isinstance(vf, VectorField)
    assert isinstance(vf, AutonomousDynamicalSystem)
    assert isinstance(vf, DynamicalSystem)
    assert not isinstance(vf, NonAutonomousDynamicalSystem)


def test_vf3d_mro():
    vf = _ConcreteVF3D()
    assert isinstance(vf, VectorField3D)
    assert isinstance(vf, VectorField)
    assert isinstance(vf, AutonomousDynamicalSystem)


def test_axisym_mro():
    import numpy as np
    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 5)
    zero = np.zeros((5, 5))
    vf = _ConcreteAxiVF3D(R, Z, zero, zero, np.ones((5, 5)))
    assert isinstance(vf, VectorField3DAxiSymmetric)
    assert isinstance(vf, VectorField3D)
    assert isinstance(vf, VectorField)
    assert isinstance(vf, AutonomousDynamicalSystem)
    assert isinstance(vf, DynamicalSystem)


# ---------------------------------------------------------------------------
# Abstract enforcement
# ---------------------------------------------------------------------------

def test_cannot_instantiate_vectorfield_directly():
    with pytest.raises(TypeError):
        VectorField()  # type: ignore[abstract]


def test_cannot_instantiate_dynamicalsystem_directly():
    with pytest.raises(TypeError):
        DynamicalSystem()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# poincare_map_2d raises NotImplementedError on base VectorField3D
# ---------------------------------------------------------------------------

def test_poincare_map_not_implemented():
    vf = _ConcreteVF3D()
    with pytest.raises(NotImplementedError):
        vf.poincare_map_2d(0.0)


# ---------------------------------------------------------------------------
# __call__ works on concrete subclasses
# ---------------------------------------------------------------------------

def test_vf1d_call():
    vf = _ConcreteVF1D()
    assert vf(2.0) == -2.0


def test_vf2d_call():
    vf = _ConcreteVF2D()
    result = vf((1.0, 0.0))
    assert result == (0.0, 1.0)
