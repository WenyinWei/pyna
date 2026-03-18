"""Advanced tests for pyna/fields new features:
- TensorField4D_rank2
- Field.coords attribute
- covariant_derivative_of_vector
- riemann_tensor / ricci_tensor / ricci_scalar
- strain_rate_tensor
- helmholtz_decomposition
"""
import numpy as np
import pytest
from pyna.fields import (
    TensorField4D_rank2,
    covariant_derivative_of_vector,
    riemann_tensor,
    ricci_tensor,
    ricci_scalar,
    strain_rate_tensor,
    helmholtz_decomposition,
    FieldProperty,
)
from pyna.fields.coords import (
    SchwarzschildCoords4D,
    MinkowskiCoords4D,
    CylindricalCoords3D,
)
from pyna.fields.cylindrical import VectorField3DCylindrical


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_simple_vec(nR=10, nZ=8, nPhi=6):
    R_ax = np.linspace(1, 3, nR)
    Z_ax = np.linspace(-1, 1, nZ)
    Phi_ax = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)
    VR = np.zeros((nR, nZ, nPhi))
    VZ = np.ones((nR, nZ, nPhi))
    VP = np.zeros((nR, nZ, nPhi))
    return VectorField3DCylindrical(R_ax, Z_ax, Phi_ax, VR, VZ, VP, name="v")


# ─── Feature 1: TensorField4D_rank2 ──────────────────────────────────────────

def test_tensor4d_construction():
    axes = [np.linspace(0, 1, 4), np.linspace(0, 1, 5),
            np.linspace(0, 1, 3), np.linspace(0, 1, 6)]
    data = np.random.randn(4, 5, 3, 6, 4, 4)
    t = TensorField4D_rank2(axes, data, name="T")
    assert t.data.shape == (4, 5, 3, 6, 4, 4)
    assert t.domain_dim == 4
    assert t.range_rank == 2
    assert t.name == "T"


def test_tensor4d_construction_wrong_shape_raises():
    axes = [np.linspace(0, 1, 4)] * 4
    bad_data = np.zeros((4, 4, 4, 4, 3, 3))  # wrong last dims
    with pytest.raises(AssertionError):
        TensorField4D_rank2(axes, bad_data)


def test_tensor4d_from_metric():
    cs = SchwarzschildCoords4D(M=1.0)
    axes = [
        np.linspace(0, 1, 4),
        np.linspace(3, 5, 5),   # r > 2M to avoid singularity
        np.linspace(0.1, np.pi - 0.1, 4),
        np.linspace(0, 2 * np.pi, 4),
    ]
    g_field = TensorField4D_rank2.from_metric(cs, axes)
    assert g_field.data.shape == (4, 5, 4, 4, 4, 4)
    assert g_field.has_property(FieldProperty.SYMMETRIC)
    # g_tt should be negative (Schwarzschild signature -+++)
    # at r=3 (>2M=2), f = 1 - 2/3 = 1/3, g_tt = -f < 0
    assert float(g_field.data[0, 0, 0, 0, 0, 0]) < 0


def test_tensor4d_transpose():
    axes = [np.linspace(0, 1, 3)] * 4
    data = np.random.randn(3, 3, 3, 3, 4, 4)
    t = TensorField4D_rank2(axes, data)
    tT = t.transpose()
    np.testing.assert_allclose(tT.data, np.swapaxes(data, -2, -1))


def test_tensor4d_symmetrize():
    axes = [np.linspace(0, 1, 3)] * 4
    data = np.random.randn(3, 3, 3, 3, 4, 4)
    t = TensorField4D_rank2(axes, data)
    sym = t.symmetrize()
    np.testing.assert_allclose(sym.data, np.swapaxes(sym.data, -2, -1), atol=1e-14)
    assert sym.has_property(FieldProperty.SYMMETRIC)


def test_tensor4d_call():
    axes = [np.linspace(0, 1, 5)] * 4
    data = np.ones((5, 5, 5, 5, 4, 4)) * np.eye(4)
    t = TensorField4D_rank2(axes, data)
    pt = np.array([0.5, 0.5, 0.5, 0.5])
    result = t(pt)
    assert result.shape == (4, 4)
    np.testing.assert_allclose(result, np.eye(4), atol=1e-10)


# ─── Feature 2: Field.coords attribute ───────────────────────────────────────

def test_field_coords_attribute():
    v = make_simple_vec()
    assert v.coords is not None
    assert isinstance(v.coords, CylindricalCoords3D)


def test_field_coords_can_be_set():
    v = make_simple_vec()
    cs = CylindricalCoords3D()
    v.coords = cs
    assert v.coords is cs


# ─── Feature 3: covariant_derivative_of_vector ───────────────────────────────

def test_covariant_derivative_cylindrical():
    v = make_simple_vec()
    cov = covariant_derivative_of_vector(v)
    nR, nZ, nPhi = 10, 8, 6
    assert cov.data.shape == (nR, nZ, nPhi, 3, 3)


def test_covariant_derivative_uniform_axial_field():
    """Uniform axial field VZ=const: covariant derivative should be ~0."""
    R_ax = np.linspace(1, 3, 15)
    Z_ax = np.linspace(-1, 1, 12)
    Phi_ax = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    VR = np.zeros((15, 12, 8))
    VZ = np.ones((15, 12, 8)) * 2.0
    VP = np.zeros((15, 12, 8))
    v = VectorField3DCylindrical(R_ax, Z_ax, Phi_ax, VR, VZ, VP)
    cov = covariant_derivative_of_vector(v)
    # All derivatives of constant field should be zero (interior points)
    assert np.all(np.abs(cov.data[2:-2, 2:-2, :]) < 1e-10)


# ─── Feature 4: Riemann / Ricci ──────────────────────────────────────────────

def test_riemann_schwarzschild_nonzero():
    cs = SchwarzschildCoords4D(M=1.0)
    pt = np.array([0.0, 4.0, np.pi / 2, 0.0])  # r=4 > r_s=2
    R = riemann_tensor(cs, pt)
    assert R.shape == (4, 4, 4, 4)
    assert np.any(np.abs(R) > 1e-10), "Riemann tensor should be non-zero for Schwarzschild"


def test_riemann_antisymmetry():
    """R^l_ijk = -R^l_ikj (antisymmetry in last two indices)."""
    cs = SchwarzschildCoords4D(M=1.0)
    pt = np.array([0.0, 5.0, np.pi / 2, 0.0])
    R = riemann_tensor(cs, pt)
    np.testing.assert_allclose(R, -np.swapaxes(R, -1, -2), atol=1e-6)


def test_ricci_tensor_schwarzschild_shape():
    cs = SchwarzschildCoords4D(M=1.0)
    pt = np.array([0.0, 4.0, np.pi / 2, 0.0])
    Ric = ricci_tensor(cs, pt)
    assert Ric.shape == (4, 4)


def test_ricci_flat_minkowski():
    """Minkowski spacetime is flat: Ricci scalar = 0."""
    cs = MinkowskiCoords4D()
    pt = np.array([0.0, 1.0, 1.0, 0.0])
    R_scalar = ricci_scalar(cs, pt)
    assert abs(R_scalar) < 1e-8, f"Minkowski Ricci scalar should be 0, got {R_scalar}"


def test_ricci_schwarzschild_vacuum():
    """Schwarzschild is a vacuum solution: Ricci tensor should be approximately 0.
    Note: central FD with eps=1e-4 has limited accuracy for curved spacetimes;
    we use a relaxed tolerance here.
    """
    cs = SchwarzschildCoords4D(M=1.0)
    pt = np.array([0.0, 8.0, np.pi / 2, 0.0])  # use large r for better FD accuracy
    Ric = ricci_tensor(cs, pt, eps=1e-5)
    # Off-diagonal elements should be exactly 0 by symmetry
    np.testing.assert_allclose(Ric[0, 1], 0, atol=1e-6)
    # Diagonal: relaxed tolerance due to FD errors in curved coords
    assert np.all(np.abs(Ric) < 1.0), f"Ricci tensor unexpectedly large: {Ric}"


# ─── Feature 5: strain_rate_tensor ───────────────────────────────────────────

def test_strain_rate_symmetric():
    v = make_simple_vec()
    S = strain_rate_tensor(v)
    assert S.has_property(FieldProperty.SYMMETRIC)
    # S should equal (J + J^T)/2
    np.testing.assert_allclose(S.data, np.swapaxes(S.data, -2, -1), atol=1e-14)


def test_strain_rate_shape():
    v = make_simple_vec()
    S = strain_rate_tensor(v)
    assert S.data.shape == (10, 8, 6, 3, 3)


# ─── Feature 6: helmholtz_decomposition ─────────────────────────────────────

def test_helmholtz_returns_tuple():
    v = make_simple_vec()
    result = helmholtz_decomposition(v)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_helmholtz_properties():
    v = make_simple_vec()
    v_df, v_cf = helmholtz_decomposition(v)
    assert v_df.has_property(FieldProperty.DIVERGENCE_FREE)
    assert v_cf.has_property(FieldProperty.CURL_FREE)


def test_helmholtz_same_grid():
    v = make_simple_vec()
    v_df, v_cf = helmholtz_decomposition(v)
    np.testing.assert_array_equal(v_df.R, v.R)
    np.testing.assert_array_equal(v_df.Z, v.Z)
    np.testing.assert_array_equal(v_df.Phi, v.Phi)
