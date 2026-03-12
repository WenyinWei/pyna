"""Tests for pyna.topo.variational — Poincaré map variational equations."""

import numpy as np
import pytest
from pyna.topo.variational import PoincareMapVariationalEquations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_field(r, z, phi):
    """Field that produces circular orbits in (R, Z) plane.

    dR/dφ = −Z,   dZ/dφ = R − R0
    This is a rotation around (R0, 0).  The Poincaré map after one full
    turn (0 → 2π) is the identity, so the monodromy matrix is the identity.
    """
    R0 = 1.8
    return np.array([-(z - 0.0), r - R0])


def _linear_field_with_known_jacobian(r, z, phi):
    """Field with a known constant Jacobian A = [[0, -1], [1, 0]].

    The solution is X(φ) = exp(A φ) X₀.
    After one turn (φ = 2π), exp(2π A) = I  (rotation matrix).
    """
    return np.array([-z, r])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_jacobian_matrix_shape():
    """jacobian_matrix should return a 2×2 matrix."""
    vq = PoincareMapVariationalEquations(_circular_field)
    M = vq.jacobian_matrix([1.8, 0.1], [0.0, 2 * np.pi])
    assert M.shape == (2, 2)


def test_jacobian_det_area_preserving():
    """For an area-preserving (Hamiltonian) map, det(M) should be ≈ 1."""
    # Use a simple rotation field whose monodromy is well-defined
    vq = PoincareMapVariationalEquations(_linear_field_with_known_jacobian,
                                         fd_eps=1e-5)
    # Integrate over a small angle so the test is fast
    phi_half = np.pi / 4
    M = vq.jacobian_matrix([1.0, 0.0], [0.0, phi_half])
    det = np.linalg.det(M)
    assert abs(det - 1.0) < 1e-5, f"det(M) = {det:.6f}, expected 1.0"


def test_jacobian_matrix_rotation_field():
    """For a pure rotation f = (−Z, R), M(0→2π) should equal the identity."""
    vq = PoincareMapVariationalEquations(_linear_field_with_known_jacobian,
                                         fd_eps=1e-5)
    M = vq.jacobian_matrix([1.0, 0.0], [0.0, 2 * np.pi],
                            solve_ivp_kwargs={"method": "DOP853",
                                              "rtol": 1e-10, "atol": 1e-13})
    # exp(A * 2π) with A = [[0,-1],[1,0]] is the 2π rotation = identity
    assert np.allclose(M, np.eye(2), atol=1e-4), \
        f"Expected identity monodromy, got:\n{M}"


def test_tangent_map_order1_consistency():
    """tangent_map(order=1) should return the same M as jacobian_matrix."""
    vq = PoincareMapVariationalEquations(_linear_field_with_known_jacobian)
    phi_span = (0.0, np.pi / 3)
    M_direct = vq.jacobian_matrix([1.0, 0.5], phi_span)
    x_end, M_tangent = vq.tangent_map([1.0, 0.5], phi_span, order=1)
    assert np.allclose(M_direct, M_tangent, atol=1e-10)


def test_tangent_map_order_not_implemented():
    """tangent_map with order > 1 should raise NotImplementedError."""
    vq = PoincareMapVariationalEquations(_circular_field)
    with pytest.raises(NotImplementedError):
        vq.tangent_map([1.8, 0.0], [0.0, np.pi], order=2)


def test_jacobian_det_circular_field():
    """Circular-field Poincaré map is area-preserving: det(M) ≈ 1."""
    vq = PoincareMapVariationalEquations(_circular_field, fd_eps=1e-6)
    phi_span = (0.0, np.pi / 2)
    M = vq.jacobian_matrix([1.8, 0.05], phi_span)
    det = np.linalg.det(M)
    assert abs(det - 1.0) < 1e-4, f"det(M) = {det:.6f}"
