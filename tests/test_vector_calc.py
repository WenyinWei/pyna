"""Tests for pyna.vector_calc — cylindrical coordinate vector calculus."""

import numpy as np
import pytest
from pyna.fields import VectorField3DCylindrical, ScalarField3DCylindrical
from pyna.vector_calc import (
    magnitude, cross, divergence,
    directional_derivative_of_scalar,
    directional_derivative_of_vector,
)


def make_grid(nR=16, nZ=14, nPhi=12):
    R = np.linspace(1.0, 2.0, nR)
    Z = np.linspace(-0.5, 0.5, nZ)
    Phi = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)
    return R, Z, Phi


def test_cross_self_is_zero():
    """v × v should be zero everywhere."""
    R, Z, Phi = make_grid()
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    VR = np.random.default_rng(0).normal(size=(nR, nZ, nPhi))
    VZ = np.random.default_rng(1).normal(size=(nR, nZ, nPhi))
    VPhi = np.random.default_rng(2).normal(size=(nR, nZ, nPhi))
    v = VectorField3DCylindrical(R=R, Z=Z, Phi=Phi, VR=VR, VZ=VZ, VPhi=VPhi)
    c = cross(v, v)
    assert np.allclose(c.VR, 0, atol=1e-14)
    assert np.allclose(c.VZ, 0, atol=1e-14)
    assert np.allclose(c.VPhi, 0, atol=1e-14)


def test_cross_anticommutative():
    """v1 × v2 = -(v2 × v1)."""
    R, Z, Phi = make_grid()
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    rng = np.random.default_rng(42)
    def rand_field():
        return VectorField3DCylindrical(R=R, Z=Z, Phi=Phi,
                                      VR=rng.normal(size=(nR, nZ, nPhi)),
                                      VZ=rng.normal(size=(nR, nZ, nPhi)),
                                      VPhi=rng.normal(size=(nR, nZ, nPhi)))
    v1, v2 = rand_field(), rand_field()
    c12 = cross(v1, v2)
    c21 = cross(v2, v1)
    assert np.allclose(c12.VR, -c21.VR, atol=1e-14)
    assert np.allclose(c12.VZ, -c21.VZ, atol=1e-14)
    assert np.allclose(c12.VPhi, -c21.VPhi, atol=1e-14)


def test_divergence_uniform_field():
    """Divergence of a uniform (constant) vector field is zero."""
    R, Z, Phi = make_grid()
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    # Constant VR = VZ = VPhi = 1 everywhere
    # ∇·V = ∂_R(1) + ∂_Z(1) + (1 + ∂_phi(1))/R = 0 + 0 + 1/R ≠ 0 for VR=1
    # So use only VZ=1, VR=VPhi=0 → div = 0
    VZ = np.ones((nR, nZ, nPhi))
    v = VectorField3DCylindrical(R=R, Z=Z, Phi=Phi,
                               VR=np.zeros_like(VZ), VZ=VZ, VPhi=np.zeros_like(VZ))
    div = divergence(v)
    assert np.allclose(div.value, 0, atol=1e-10)


def test_divergence_manufactured():
    """Test divergence on V_R = R (so ∂_R V_R = 1, ∇·V = 1 + V_R/R = 2)."""
    R, Z, Phi = make_grid(nR=30, nZ=20, nPhi=16)
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    # VR = R (a function of R only), VZ = VPhi = 0
    # ∇·V = ∂_R(R) + 0 + R/R = 1 + 1 = 2
    VR = R[:, None, None] * np.ones((nR, nZ, nPhi))
    v = VectorField3DCylindrical(R=R, Z=Z, Phi=Phi,
                               VR=VR, VZ=np.zeros_like(VR), VPhi=np.zeros_like(VR))
    div = divergence(v)
    # Expect ≈ 2 at interior points (boundary has one-sided diff errors)
    interior = div.value[2:-2, 2:-2, :]
    assert np.allclose(interior, 2.0, atol=5e-3)


def test_directional_derivative_of_scalar_constant():
    """For constant scalar field, directional derivative is zero."""
    R, Z, Phi = make_grid()
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    rng = np.random.default_rng(7)
    # Random vector field
    v = VectorField3DCylindrical(R=R, Z=Z, Phi=Phi,
                               VR=rng.normal(size=(nR, nZ, nPhi)),
                               VZ=rng.normal(size=(nR, nZ, nPhi)),
                               VPhi=rng.normal(size=(nR, nZ, nPhi)))
    # Constant scalar field
    s = ScalarField3DCylindrical(R=R, Z=Z, Phi=Phi,
                               value=5.0 * np.ones((nR, nZ, nPhi)))
    result = directional_derivative_of_scalar(v, s)
    assert np.allclose(result.value, 0, atol=1e-12)


def test_directional_derivative_of_vector_constant():
    """For constant vector field with VPhi=0, v1·∇v1 = 0."""
    R, Z, Phi = make_grid()
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    # v1 with only VZ component, constant everywhere
    v1 = VectorField3DCylindrical(R=R, Z=Z, Phi=Phi,
                                VR=np.zeros((nR, nZ, nPhi)),
                                VZ=np.ones((nR, nZ, nPhi)),
                                VPhi=np.zeros((nR, nZ, nPhi)))
    # v1·∇v1 for (0, 1, 0): all partial derivatives of constants are 0; VPhi=0 so Christoffel=0
    result = directional_derivative_of_vector(v1, v1)
    assert np.allclose(result.VR, 0, atol=1e-12)
    assert np.allclose(result.VZ, 0, atol=1e-12)
    assert np.allclose(result.VPhi, 0, atol=1e-12)
