"""Tests for pyna.fields hierarchy."""
import numpy as np
import pytest
import tempfile
import os


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def grid_3d():
    R   = np.linspace(1.0, 3.0, 12)
    Z   = np.linspace(-1.0, 1.0, 10)
    Phi = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return R, Z, Phi


@pytest.fixture
def uniform_Bz_field(grid_3d):
    """B = (0, 0, B0) in cylindrical — constant BZ=1, BR=BPhi=0."""
    from pyna.fields.cylindrical import VectorField3DCylindrical
    R, Z, Phi = grid_3d
    shape = (len(R), len(Z), len(Phi))
    VR   = np.zeros(shape)
    VZ   = np.ones(shape)
    VPhi = np.zeros(shape)
    return VectorField3DCylindrical(R, Z, Phi, VR, VZ, VPhi, name="uniform_Bz")


@pytest.fixture
def linear_scalar_field(grid_3d):
    """f = R (linear in R)."""
    from pyna.fields.cylindrical import ScalarField3DCylindrical
    R, Z, Phi = grid_3d
    value = R[:, np.newaxis, np.newaxis] * np.ones((len(R), len(Z), len(Phi)))
    return ScalarField3DCylindrical(R, Z, Phi, value, name="linear_R")


# ── Group 1: FieldProperty ────────────────────────────────────────────────────

def test_property_flags():
    from pyna.fields.properties import FieldProperty
    combined = FieldProperty.DIVERGENCE_FREE | FieldProperty.CURL_FREE
    assert FieldProperty.DIVERGENCE_FREE in combined
    assert FieldProperty.CURL_FREE in combined


def test_property_none():
    from pyna.fields.properties import FieldProperty
    assert not FieldProperty.NONE


# ── Group 2: VectorField3DCylindrical ─────────────────────────────────────────

def test_vector_field_construction(uniform_Bz_field, grid_3d):
    R, Z, Phi = grid_3d
    f = uniform_Bz_field
    expected_shape = (len(R), len(Z), len(Phi))
    assert f.VR.shape == expected_shape
    assert f.VZ.shape == expected_shape
    assert f.VPhi.shape == expected_shape
    # BR/BZ/BPhi are aliases
    assert f.BR is f.VR
    assert f.BZ is f.VZ
    assert f.BPhi is f.VPhi


def test_vector_field_call(uniform_Bz_field, grid_3d):
    R, Z, Phi = grid_3d
    # Pick interior points
    R_pts   = np.array([1.5, 2.0, 2.5])
    Z_pts   = np.array([0.0, 0.0, 0.0])
    Phi_pts = np.array([0.0, 1.0, 2.0])
    coords = np.stack([R_pts, Z_pts, Phi_pts], axis=-1)  # shape (3, 3)
    result = uniform_Bz_field(coords)
    assert result.shape == (3, 3)
    # VZ component (index 1) should be ~1.0
    np.testing.assert_allclose(result[:, 1], 1.0, atol=1e-6)


def test_vector_field_interpolate_at(uniform_Bz_field):
    R_pts = np.array([1.5, 2.0])
    Z_pts = np.array([0.0, 0.0])
    result = uniform_Bz_field.interpolate_at(R_pts, Z_pts)
    assert isinstance(result, tuple)
    assert len(result) == 3  # (VR, VZ, VPhi)


def test_vector_field_npz_roundtrip(uniform_Bz_field):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_field.npz")
        uniform_Bz_field.to_npz(path)
        from pyna.fields.cylindrical import VectorField3DCylindrical
        loaded = VectorField3DCylindrical.from_npz(path)
        np.testing.assert_array_equal(loaded.VZ, uniform_Bz_field.VZ)


def test_vector_from_callable(grid_3d):
    from pyna.fields.cylindrical import VectorField3DCylindrical
    R, Z, Phi = grid_3d

    def const_field(r, z, phi):
        shape = np.broadcast(r, z, phi).shape
        VR   = np.zeros(shape)
        VZ   = np.ones(shape)
        VPhi = np.zeros(shape)
        return VR, VZ, VPhi

    f = VectorField3DCylindrical.from_callable(const_field, R, Z, Phi)
    assert f.VZ.shape == (len(R), len(Z), len(Phi))
    np.testing.assert_allclose(f.VZ, 1.0)


# ── Group 3: ScalarField3DCylindrical ─────────────────────────────────────────

def test_scalar_construction(linear_scalar_field, grid_3d):
    R, Z, Phi = grid_3d
    f = linear_scalar_field
    assert f.value.shape == (len(R), len(Z), len(Phi))


def test_scalar_call(linear_scalar_field):
    # At (R=2.0, Z=0, phi=0) the value should be ~2.0
    coords = np.array([[2.0, 0.0, 0.0]])
    result = linear_scalar_field(coords)
    assert np.abs(result.ravel()[0] - 2.0) < 0.01


def test_scalar_npz_roundtrip(linear_scalar_field):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scalar_field.npz")
        linear_scalar_field.to_npz(path)
        from pyna.fields.cylindrical import ScalarField3DCylindrical
        loaded = ScalarField3DCylindrical.from_npz(path)
        np.testing.assert_array_equal(loaded.value, linear_scalar_field.value)


# ── Group 4: AxiSymmetric ─────────────────────────────────────────────────────

def test_axi_vector_2d():
    from pyna.fields.cylindrical import VectorField3DAxiSymmetric
    R   = np.linspace(1.0, 3.0, 10)
    Z   = np.linspace(-1.0, 1.0, 8)
    shape2d = (len(R), len(Z))
    VR_2d   = np.zeros(shape2d)
    VZ_2d   = np.ones(shape2d)
    VPhi_2d = np.zeros(shape2d)
    f = VectorField3DAxiSymmetric(R, Z, VR_2d, VZ_2d, VPhi_2d)
    # Phi dimension should be length 1 (axisymmetric)
    assert f.VZ.shape[2] == 1
    # Calling should still return (N, 3)
    coords = np.array([[2.0, 0.0, 1.0]])
    result = f(coords)
    assert result.shape == (1, 3)
    np.testing.assert_allclose(result[0, 1], 1.0, atol=1e-6)


# ── Group 5: diff_ops ─────────────────────────────────────────────────────────

def test_divergence_uniform_Bz(uniform_Bz_field, grid_3d):
    from pyna.fields.diff_ops import divergence
    div_B = divergence(uniform_Bz_field)
    # For a uniform Bz field, divergence should be identically zero
    np.testing.assert_allclose(div_B.value, 0.0, atol=1e-8)


def test_curl_returns_div_free(uniform_Bz_field):
    from pyna.fields.diff_ops import curl
    from pyna.fields.properties import FieldProperty
    curl_B = curl(uniform_Bz_field)
    assert curl_B.has_property(FieldProperty.DIVERGENCE_FREE)


def test_gradient_of_linear_R(linear_scalar_field, grid_3d):
    from pyna.fields.diff_ops import gradient
    R, Z, Phi = grid_3d
    grad_f = gradient(linear_scalar_field)
    # ∂(R)/∂R = 1.0 everywhere (interior points, away from edges)
    # Check interior slice
    interior_VR = grad_f.VR[2:-2, 2:-2, :]
    np.testing.assert_allclose(interior_VR, 1.0, atol=0.05)


def test_laplacian_of_linear_R(linear_scalar_field):
    from pyna.fields.diff_ops import laplacian
    # In cylindrical: ∇²R = (1/R) ∂/∂R (R ∂R/∂R) = (1/R) ∂/∂R (R) = 1/R
    R   = linear_scalar_field.R
    lap = laplacian(linear_scalar_field)
    # Check interior R points only (Z, phi don't matter for f=R)
    interior_R = slice(2, -2)
    lap_R = lap.value[interior_R, 5, 2]  # single Z,phi point
    expected_R = 1.0 / R[interior_R]
    np.testing.assert_allclose(lap_R, expected_R, rtol=0.1)


def test_hessian_shape(linear_scalar_field, grid_3d):
    from pyna.fields.diff_ops import hessian
    R, Z, Phi = grid_3d
    H = hessian(linear_scalar_field)
    assert H.data.shape == (len(R), len(Z), len(Phi), 3, 3)


# ── Group 6: TensorField3DRank2 ──────────────────────────────────────────────

def test_tensor_construction(grid_3d):
    from pyna.fields.tensor import TensorField3DRank2
    R, Z, Phi = grid_3d
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    data = np.zeros((nR, nZ, nPhi, 3, 3))
    T = TensorField3DRank2(R, Z, Phi, data)
    assert T.data.shape == (nR, nZ, nPhi, 3, 3)


def test_tensor_component(grid_3d):
    from pyna.fields.tensor import TensorField3DRank2
    R, Z, Phi = grid_3d
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    data = np.random.rand(nR, nZ, nPhi, 3, 3)
    T = TensorField3DRank2(R, Z, Phi, data)
    comp = T.component(0, 1)
    assert comp.shape == (nR, nZ, nPhi)
    np.testing.assert_array_equal(comp, data[:, :, :, 0, 1])


def test_tensor_trace(grid_3d):
    from pyna.fields.tensor import TensorField3DRank2
    R, Z, Phi = grid_3d
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    # Identity tensor: trace = 3
    data = np.zeros((nR, nZ, nPhi, 3, 3))
    for i in range(3):
        data[:, :, :, i, i] = 1.0
    T = TensorField3DRank2(R, Z, Phi, data)
    tr = T.trace()
    assert tr.shape == (nR, nZ, nPhi)
    np.testing.assert_allclose(tr, 3.0)


def test_tensor_symmetrize(grid_3d):
    from pyna.fields.tensor import TensorField3DRank2
    from pyna.fields.properties import FieldProperty
    R, Z, Phi = grid_3d
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    data = np.random.rand(nR, nZ, nPhi, 3, 3)
    T = TensorField3DRank2(R, Z, Phi, data)
    T_sym = T.symmetrize()
    assert T_sym.has_property(FieldProperty.SYMMETRIC)


def test_tensor_call(grid_3d):
    from pyna.fields.tensor import TensorField3DRank2
    R, Z, Phi = grid_3d
    nR, nZ, nPhi = len(R), len(Z), len(Phi)
    data = np.zeros((nR, nZ, nPhi, 3, 3))
    for i in range(3):
        data[:, :, :, i, i] = 1.0
    T = TensorField3DRank2(R, Z, Phi, data)
    N = 5
    coords = np.column_stack([
        np.linspace(1.2, 2.8, N),
        np.zeros(N),
        np.linspace(0, np.pi, N),
    ])
    result = T(coords)
    assert result.shape == (N, 3, 3)


# ── Group 7: CoordinateSystem ─────────────────────────────────────────────────

def test_cylindrical_coords_metric():
    from pyna.fields.coords import Coords3DCylindrical
    coords = Coords3DCylindrical()
    # At (R=2, Z=0, phi=0): g_phiphi = R^2 = 4
    pts = np.array([[2.0, 0.0, 0.0]])
    metric = coords.metric_tensor(pts)
    assert metric.shape[0] == 1
    np.testing.assert_allclose(metric[0, 2, 2], 4.0, atol=1e-10)


def test_cylindrical_to_cartesian():
    from pyna.fields.coords import Coords3DCylindrical
    coords = Coords3DCylindrical()
    # (R=2, Z=1, phi=pi/2) → (x≈0, y=2, z=1)
    pts = np.array([[2.0, 1.0, np.pi / 2]])
    cart = coords.to_cartesian(pts)
    np.testing.assert_allclose(cart[0, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(cart[0, 1], 2.0, atol=1e-10)
    np.testing.assert_allclose(cart[0, 2], 1.0, atol=1e-10)


def test_schwarzschild_radius():
    from pyna.fields.coords import Coords4DSchwarzschild
    coords = Coords4DSchwarzschild(M=1.0)
    np.testing.assert_allclose(coords.schwarzschild_radius, 2.0)


def test_kerr_event_horizon():
    from pyna.fields.coords import Coords4DKerr
    coords = Coords4DKerr(M=1.0, a=0.5)
    expected = 1.0 + np.sqrt(0.75)
    np.testing.assert_allclose(coords.event_horizon_radius, expected, rtol=1e-10)


def test_minkowski_flat():
    from pyna.fields.coords import Coords4DMinkowski
    coords = Coords4DMinkowski()
    # Christoffel symbols should all be zero for flat spacetime
    pts = np.array([[0.0, 1.0, 0.0, 0.0]])
    christoffel = coords.christoffel_symbols(pts)
    np.testing.assert_allclose(christoffel, 0.0, atol=1e-15)


def test_cartesian_identity():
    from pyna.fields.coords import CoordsCartesian
    coords = CoordsCartesian(3)
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = coords.to_cartesian(pts)
    np.testing.assert_array_equal(result, pts)


# ── Group 8: backward compatibility ──────────────────────────────────────────

def test_compat_field_data():
    from pyna.fields.cylindrical import VectorField3DCylindrical, ScalarField3DCylindrical
    from pyna.fields import VectorField3DCylindrical as VF, ScalarField3DCylindrical as SF
    assert VF is VectorField3DCylindrical
    assert SF is ScalarField3DCylindrical


def test_compat_system(uniform_Bz_field):
    from pyna.system import VectorField3D, VectorField3DAxiSymmetric
    assert isinstance(uniform_Bz_field, VectorField3D)
