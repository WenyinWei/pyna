"""Tests for VectorPotentialField: B = curl(A)."""
import numpy as np
import pytest
from pyna.MCF.coils.vector_potential import VectorPotentialField


def make_uniform_Bz_field(B0=1.0, nR=50, nZ=50, nPhi=32):
    """Uniform B = B0 ẑ  =>  A_φ = B0*R/2, A_R = A_Z = 0."""
    R = np.linspace(0.5, 2.5, nR)
    Z = np.linspace(-1.0, 1.0, nZ)
    Phi = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)

    R3, Z3, P3 = np.meshgrid(R, Z, Phi, indexing='ij')

    AR = np.zeros_like(R3)
    AZ = np.zeros_like(R3)
    APhi = B0 * R3 / 2.0  # A_φ = B0*R/2

    return VectorPotentialField(R, Z, Phi, AR, AZ, APhi), B0, R, Z, Phi


def test_uniform_Bz_recovery():
    B0 = 1.5
    field, _, R, Z, Phi = make_uniform_Bz_field(B0=B0)

    # Sample interior points
    R_test = np.array([1.0, 1.5, 2.0])
    Z_test = np.array([0.0, 0.1, -0.2])
    phi_test = np.array([0.0, np.pi / 4, np.pi])

    BR, BZ, BPhi = field.B_at(R_test, Z_test, phi_test)

    # BZ should equal B0 everywhere
    rel_err = np.abs(BZ - B0) / B0
    assert np.all(rel_err < 0.01), f"BZ relative error too large: {rel_err}"

    # BR and BPhi should be ~0
    assert np.all(np.abs(BR) < 0.01 * B0), f"BR not ~0: {BR}"
    assert np.all(np.abs(BPhi) < 0.01 * B0), f"BPhi not ~0: {BPhi}"


def test_divergence_free():
    field, _, R, Z, Phi = make_uniform_Bz_field()
    assert field.divergence_free() is True
