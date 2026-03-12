"""Tests for the PEST coordinate system (pyna.coord.PEST)."""

import numpy as np
import pytest
from unittest.mock import patch


def _make_simple_equilibrium(nR=40, nZ=40):
    """Create a simple circular-cross-section Grad-Shafranov equilibrium
    for testing purposes.

    Returns (R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis).
    """
    R0, Z0 = 1.8, 0.0          # magnetic axis
    a = 0.4                     # minor radius
    B_tor = 2.0                 # toroidal field at R0

    R = np.linspace(R0 - a - 0.1, R0 + a + 0.1, nR)
    Z = np.linspace(Z0 - a - 0.1, Z0 + a + 0.1, nZ)
    Rg, Zg = np.meshgrid(R, Z, indexing='ij')

    rho2 = ((Rg - R0) ** 2 + (Zg - Z0) ** 2) / a ** 2
    psi_norm = np.clip(rho2, 0.0, 1.0)

    # Simple safety-factor approximation: q ≈ 1.5
    q = 1.5
    # Poloidal field components from ψ (circular geometry)
    dpsi_dR = 2.0 * (Rg - R0) / (a ** 2)
    dpsi_dZ = 2.0 * (Zg - Z0) / (a ** 2)
    # BZ ~ -1/(R) ∂ψ/∂R * const, BR ~ 1/(R) ∂ψ/∂Z * const
    pol_scale = 0.1
    BR0   = pol_scale * dpsi_dZ / Rg
    BZ0   = -pol_scale * dpsi_dR / Rg
    BPhi0 = B_tor / Rg

    return R, Z, BR0, BZ0, BPhi0, psi_norm, R0, Z0


# ---------------------------------------------------------------------------
# build_PEST_mesh — shape tests
# ---------------------------------------------------------------------------

def test_build_PEST_mesh_output_shapes():
    """build_PEST_mesh should return arrays of the expected shapes."""
    from pyna.coord.PEST import build_PEST_mesh

    R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis = _make_simple_equilibrium()
    ns, ntheta = 8, 13

    S, TET, R_mesh, Z_mesh, q_iS = build_PEST_mesh(
        R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis,
        ns=ns, ntheta=ntheta,
    )

    assert S.shape == (ns,), f"Expected S.shape={ns}, got {S.shape}"
    assert TET.shape == (ntheta,), f"Expected TET.shape={ntheta}, got {TET.shape}"
    assert R_mesh.shape == (ns, ntheta)
    assert Z_mesh.shape == (ns, ntheta)
    assert q_iS.shape == (ns,)


def test_build_PEST_mesh_S_monotone():
    """S should be monotonically increasing (or at least non-negative)."""
    from pyna.coord.PEST import build_PEST_mesh

    R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis = _make_simple_equilibrium()
    ns = 6

    S, *_ = build_PEST_mesh(
        R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis, ns=ns, ntheta=9)

    assert S[0] == 0.0, "S[0] should be 0 (magnetic axis)"
    assert np.all(S[1:] >= 0), "S values should be non-negative"


def test_RZmesh_isoSTET_deprecated():
    """RZmesh_isoSTET should emit a DeprecationWarning."""
    from pyna.coord.PEST import RZmesh_isoSTET

    R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis = _make_simple_equilibrium()

    with pytest.warns(DeprecationWarning):
        RZmesh_isoSTET(R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis, ns=5, ntheta=7)


# ---------------------------------------------------------------------------
# g_i_g__i_from_STET_mesh — metric tensor consistency
# ---------------------------------------------------------------------------

def test_metric_tensor_biorthogonality():
    """g^i · g_j = δᵢⱼ approximately on interior grid points."""
    from pyna.coord.PEST import build_PEST_mesh, g_i_g__i_from_STET_mesh

    R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis = _make_simple_equilibrium()
    ns, ntheta = 8, 13

    S, TET, R_mesh, Z_mesh, _ = build_PEST_mesh(
        R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis,
        ns=ns, ntheta=ntheta,
    )
    g_1, g_2, g_3, g__1, g__2, g__3 = g_i_g__i_from_STET_mesh(S, TET, R_mesh, Z_mesh)

    # Check on interior points only (skip NaN boundary rows)
    for is_ in range(2, ns - 1):
        for it in range(1, ntheta - 1):
            R_pt = R_mesh[is_, it]

            # g^i · g_j  (poloidal 2-D part only; g_3 and g^3 are scalars)
            g__1_pt = g__1[is_, it, :]
            g__2_pt = g__2[is_, it, :]
            g_1_pt  = g_1[is_, it, :]
            g_2_pt  = g_2[is_, it, :]

            # Off-diagonal should be near zero
            g__1_dot_g2 = np.dot(g__1_pt, g_2_pt)
            g__2_dot_g1 = np.dot(g__2_pt, g_1_pt)

            assert abs(g__1_dot_g2) < 0.15, \
                f"g^1·g_2 = {g__1_dot_g2:.3f} at ({is_},{it}), expected ≈0"
            assert abs(g__2_dot_g1) < 0.15, \
                f"g^2·g_1 = {g__2_dot_g1:.3f} at ({is_},{it}), expected ≈0"

            # Diagonal should be near 1
            g__1_dot_g1 = np.dot(g__1_pt, g_1_pt)
            g__2_dot_g2 = np.dot(g__2_pt, g_2_pt)

            assert abs(g__1_dot_g1 - 1.0) < 0.15, \
                f"g^1·g_1 = {g__1_dot_g1:.3f} at ({is_},{it}), expected ≈1"
            assert abs(g__2_dot_g2 - 1.0) < 0.15, \
                f"g^2·g_2 = {g__2_dot_g2:.3f} at ({is_},{it}), expected ≈1"

            # g^3 · g_3 = (1/R) · R = 1
            assert abs(g__3(R_pt) * g_3(R_pt) - 1.0) < 1e-12
