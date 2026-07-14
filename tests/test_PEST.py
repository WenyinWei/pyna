"""Tests for the PEST coordinate system (pyna.coord.PEST)."""

import numpy as np
import pytest
from unittest.mock import patch

# simple_eq_arrays fixture is provided by conftest.py (session-scoped, nR=nZ=40).


def _make_simple_equilibrium(nR=40, nZ=40):
    """Create a simple circular-cross-section Grad-Shafranov equilibrium
    for testing purposes.

    Returns (R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis).
    Prefer the ``simple_eq_arrays`` session fixture when nR/nZ=40 is fine.
    """
    R0, Z0 = 1.8, 0.0          # magnetic axis
    a = 0.4                     # minor radius
    B_tor = 2.0                 # toroidal field at R0

    R = np.linspace(R0 - a - 0.1, R0 + a + 0.1, nR)
    Z = np.linspace(Z0 - a - 0.1, Z0 + a + 0.1, nZ)
    Rg, Zg = np.meshgrid(R, Z, indexing='ij')

    rho2 = ((Rg - R0) ** 2 + (Zg - Z0) ** 2) / a ** 2
    psi_norm = np.clip(rho2, 0.0, 1.0)

    # Simple safety-factor approximation: q ~ 1.5
    dpsi_dR = 2.0 * (Rg - R0) / (a ** 2)
    dpsi_dZ = 2.0 * (Zg - Z0) / (a ** 2)
    pol_scale = 0.1
    BR0   = pol_scale * dpsi_dZ / Rg
    BZ0   = -pol_scale * dpsi_dR / Rg
    BPhi0 = B_tor / Rg

    return R, Z, BR0, BZ0, BPhi0, psi_norm, R0, Z0


def _make_3d_pest_coordinates():
    from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates

    rho = np.linspace(0.3, 0.8, 11)
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    phase = theta[None, None, :] + 0.12 * np.sin(5.0 * phi)[:, None, None]
    axis_R = 5.5 + 0.08 * np.cos(5.0 * phi)
    axis_Z = 0.06 * np.sin(5.0 * phi)
    R = axis_R[:, None, None] + rho[None, :, None] * np.cos(phase)
    Z = axis_Z[:, None, None] + 0.85 * rho[None, :, None] * np.sin(phase)
    return SmoothPestCoordinates(
        R_surf=R,
        Z_surf=Z,
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        axis_R=axis_R,
        axis_Z=axis_Z,
        source="synthetic five-period PEST surfaces",
    )


def test_project_cylindrical_points_to_3d_pest_coordinates():
    from pyna.toroidal.coords import project_cylindrical_points_to_pest

    pest = _make_3d_pest_coordinates()
    target_rho = np.array([0.437, 0.612])
    target_theta = np.array([0.71, 2.23])
    target_phi = np.asarray(pest.phi_vals)[[3, 19]]
    phase = target_theta + 0.12 * np.sin(5.0 * target_phi)
    R = 5.5 + 0.08 * np.cos(5.0 * target_phi) + target_rho * np.cos(phase)
    Z = 0.06 * np.sin(5.0 * target_phi) + 0.85 * target_rho * np.sin(phase)
    result = project_cylindrical_points_to_pest(
        pest, R, Z, target_phi, max_distance=2.0e-3
    )

    assert np.all(result.valid)
    np.testing.assert_allclose(result.rho, target_rho, atol=2.0e-3)
    angle_error = np.mod(result.theta - target_theta + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(angle_error, 0.0, atol=3.0e-3)
    assert np.max(result.residual_distance) < 2.0e-3
    assert result.as_summary_dict()["valid_count"] == 2


def test_project_cylindrical_points_to_pest_applies_distance_gate():
    from pyna.toroidal.coords import project_cylindrical_points_to_pest

    pest = _make_3d_pest_coordinates()
    result = project_cylindrical_points_to_pest(
        pest, np.array([8.0]), np.array([0.0]), np.array([0.0]), max_distance=0.02
    )
    assert not result.valid[0]


def test_project_cylindrical_points_wraps_native_field_period_geometry():
    from pyna.toroidal.coords import project_cylindrical_points_to_pest
    from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates

    nfp = 5
    period = 2.0 * np.pi / nfp
    rho = np.linspace(0.3, 0.8, 9)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    phi = np.linspace(0.0, period, 48, endpoint=False)
    phase = theta[None, None, :] + 0.08 * np.sin(nfp * phi)[:, None, None]
    axis_R = 5.5 + 0.04 * np.cos(nfp * phi)
    axis_Z = 0.03 * np.sin(nfp * phi)
    R_surf = axis_R[:, None, None] + rho[None, :, None] * np.cos(phase)
    Z_surf = axis_Z[:, None, None] + 0.88 * rho[None, :, None] * np.sin(phase)
    pest = SmoothPestCoordinates(
        R_surf=R_surf,
        Z_surf=Z_surf,
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        axis_R=axis_R,
        axis_Z=axis_Z,
        nfp=nfp,
        toroidal_period=period,
    )
    target_rho = np.array([0.47])
    target_theta = np.array([1.21])
    target_phi = np.array([phi[7] + 3.0 * period])
    wrapped_phi = np.mod(target_phi, period)
    target_phase = target_theta + 0.08 * np.sin(nfp * wrapped_phi)
    R = 5.5 + 0.04 * np.cos(nfp * wrapped_phi) + target_rho * np.cos(target_phase)
    Z = 0.03 * np.sin(nfp * wrapped_phi) + 0.88 * target_rho * np.sin(target_phase)

    result = project_cylindrical_points_to_pest(pest, R, Z, target_phi, max_distance=2.0e-3)

    assert result.valid[0]
    np.testing.assert_allclose(result.phi, wrapped_phi, atol=1.0e-14)
    np.testing.assert_allclose(result.rho, target_rho, atol=2.0e-3)
    theta_error = np.mod(result.theta - target_theta + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(theta_error, 0.0, atol=3.0e-3)


# ---------------------------------------------------------------------------
# build_PEST_mesh -- shape tests
# ---------------------------------------------------------------------------

def test_build_PEST_mesh_output_shapes(simple_eq_arrays):
    """build_PEST_mesh should return arrays of the expected shapes."""
    from pyna.toroidal.coords.PEST import build_PEST_mesh

    R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis = simple_eq_arrays
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


def test_build_PEST_mesh_S_monotone(simple_eq_arrays):
    """S should be monotonically increasing (or at least non-negative)."""
    from pyna.toroidal.coords.PEST import build_PEST_mesh

    R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis = simple_eq_arrays
    ns = 6

    S, *_ = build_PEST_mesh(
        R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis, ns=ns, ntheta=9)

    assert S[0] == 0.0, "S[0] should be 0 (magnetic axis)"
    assert np.all(S[1:] >= 0), "S values should be non-negative"


def test_RZmesh_isoSTET_deprecated(simple_eq_arrays):
    """RZmesh_isoSTET should emit a DeprecationWarning."""
    from pyna.toroidal.coords.PEST import RZmesh_isoSTET

    R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis = simple_eq_arrays

    with pytest.warns(DeprecationWarning):
        RZmesh_isoSTET(R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis, ns=5, ntheta=7)


# ---------------------------------------------------------------------------
# g_i_g__i_from_STET_mesh -- metric tensor consistency
# ---------------------------------------------------------------------------

def test_metric_tensor_biorthogonality(simple_eq_arrays):
    """g^i . g_j = delta_ij approximately on interior grid points."""
    from pyna.toroidal.coords.PEST import build_PEST_mesh, g_i_g__i_from_STET_mesh

    R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis = simple_eq_arrays
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

            g__1_pt = g__1[is_, it, :]
            g__2_pt = g__2[is_, it, :]
            g_1_pt  = g_1[is_, it, :]
            g_2_pt  = g_2[is_, it, :]

            # Off-diagonal should be near zero
            g__1_dot_g2 = np.dot(g__1_pt, g_2_pt)
            g__2_dot_g1 = np.dot(g__2_pt, g_1_pt)

            assert abs(g__1_dot_g2) < 0.15, \
                f"g^1.g_2 = {g__1_dot_g2:.3f} at ({is_},{it}), expected ~0"
            assert abs(g__2_dot_g1) < 0.15, \
                f"g^2.g_1 = {g__2_dot_g1:.3f} at ({is_},{it}), expected ~0"

            # Diagonal should be near 1
            g__1_dot_g1 = np.dot(g__1_pt, g_1_pt)
            g__2_dot_g2 = np.dot(g__2_pt, g_2_pt)

            assert abs(g__1_dot_g1 - 1.0) < 0.15, \
                f"g^1.g_1 = {g__1_dot_g1:.3f} at ({is_},{it}), expected ~1"
            assert abs(g__2_dot_g2 - 1.0) < 0.15, \
                f"g^2.g_2 = {g__2_dot_g2:.3f} at ({is_},{it}), expected ~1"

            # g^3 . g_3 = (1/R) . R = 1
            assert abs(g__3(R_pt) * g_3(R_pt) - 1.0) < 1e-12
