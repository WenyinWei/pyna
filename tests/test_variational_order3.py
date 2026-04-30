"""Tests for pyna.topo.variational — order-3 tangent map."""
from __future__ import annotations

import numpy as np
import pytest
from pyna.topo.variational import PoincareMapVariationalEquations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_field(r, z, phi):
    """Purely linear field (constant Jacobian, zero Hessian, zero 3rd deriv)."""
    return np.array([-z, r])


def _quadratic_field(r, z, phi):
    """Field with non-trivial Hessian: f = [-z + alpha*r^2, r - alpha*z^2]."""
    alpha = 0.1
    return np.array([-(z - alpha * r ** 2), r - alpha * z ** 2])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tangent_map_order3_shape():
    """tangent_map(order=3) should return (x_end, M, T, Q) with correct shapes."""
    vq = PoincareMapVariationalEquations(_linear_field, fd_eps=1e-5, fd_eps2=1e-4,
                                         fd_eps3=1e-3)
    phi_span = (0.0, np.pi / 8)
    result = vq.tangent_map([1.0, 0.0], phi_span, order=3)
    assert len(result) == 4, "order=3 should return a 4-tuple (x_end, M, T, Q)"
    x_end, M, T, Q = result
    assert x_end.shape == (2,)
    assert M.shape == (2, 2)
    assert T.shape == (2, 2, 2)
    assert Q.shape == (2, 2, 2, 2)


def test_tangent_map_order3_M_consistent_with_order1():
    """The monodromy matrix from order=3 must equal that from order=1."""
    vq = PoincareMapVariationalEquations(_quadratic_field,
                                         fd_eps=1e-5, fd_eps2=1e-4, fd_eps3=1e-3)
    phi_span = (0.0, np.pi / 6)
    x_end1, M1 = vq.tangent_map([1.0, 0.2], phi_span, order=1)
    x_end3, M3, T3, Q3 = vq.tangent_map([1.0, 0.2], phi_span, order=3)
    assert np.allclose(M1, M3, atol=1e-7), \
        f"Order-1 and order-3 should give same M; max diff={np.abs(M1-M3).max():.2e}"
    assert np.allclose(x_end1, x_end3, atol=1e-10)


def test_tangent_map_order3_T_consistent_with_order2():
    """The T tensor from order=3 must equal that from order=2."""
    vq = PoincareMapVariationalEquations(_quadratic_field,
                                         fd_eps=1e-5, fd_eps2=1e-4, fd_eps3=1e-3)
    phi_span = (0.0, np.pi / 6)
    _, _, T2 = vq.tangent_map([1.0, 0.2], phi_span, order=2)
    _, _, T3, _ = vq.tangent_map([1.0, 0.2], phi_span, order=3)
    assert np.allclose(T2, T3, atol=1e-5), \
        f"Order-2 and order-3 should give same T; max diff={np.abs(T2-T3).max():.2e}"


def test_tangent_map_order3_Q_symmetry():
    """Third-derivative tensor Q[i,j,k,l] must be symmetric in j,k,l."""
    vq = PoincareMapVariationalEquations(_quadratic_field,
                                         fd_eps=1e-5, fd_eps2=1e-4, fd_eps3=1e-3)
    phi_span = (0.0, np.pi / 8)
    _, _, _, Q = vq.tangent_map([1.0, 0.1], phi_span, order=3)
    # Q[i,j,k,l] should be symmetric in j,k,l
    # Check Q[i,j,k,l] == Q[i,j,l,k]
    assert np.allclose(Q, Q.transpose(0, 1, 3, 2), atol=1e-4), \
        "Q[i,j,k,l] should be symmetric under j↔k,l permutation"
    # Check Q[i,j,k,l] == Q[i,k,j,l]
    assert np.allclose(Q, Q.transpose(0, 2, 1, 3), atol=1e-4), \
        "Q[i,j,k,l] should be symmetric under j↔k permutation"


def test_tangent_map_order3_linear_field_Q_zero():
    """For a purely linear field, the third-derivative tensor Q must be zero."""
    vq = PoincareMapVariationalEquations(_linear_field,
                                         fd_eps=1e-5, fd_eps2=1e-4, fd_eps3=1e-3)
    phi_span = (0.0, np.pi / 4)
    _, _, _, Q = vq.tangent_map([1.0, 0.0], phi_span, order=3)
    assert np.allclose(Q, 0.0, atol=1e-5), \
        "Linear field has zero 3rd derivatives → Q must be zero"


def test_tangent_map_order4_raises():
    """tangent_map with order > 3 should raise NotImplementedError."""
    vq = PoincareMapVariationalEquations(_linear_field)
    with pytest.raises(NotImplementedError):
        vq.tangent_map([1.0, 0.0], [0.0, np.pi], order=4)
