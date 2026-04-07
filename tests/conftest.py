"""
Shared pytest fixtures for pyna test suite.

Session-scoped fixtures are built once per test run and reused across all
modules, avoiding repeated expensive equilibrium construction.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Solov'ev equilibrium (ITER-like, scale=0.3)
# Used by: test_coordinates.py, test_solovev.py, test_fpt.py, ...
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def solovev_eq():
    """Session-scoped Solov'ev ITER-like equilibrium (scale=0.3).

    Built once for the entire test session.  All tests that need a realistic
    tokamak equilibrium should request this fixture instead of calling
    ``solovev_iter_like`` directly.
    """
    from pyna.MCF.equilibrium.Solovev import solovev_iter_like
    return solovev_iter_like(scale=0.3)


# ---------------------------------------------------------------------------
# PEST mesh on the Solov'ev equilibrium
# Used by: test_coordinates.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pest_mesh(solovev_eq):
    """Session-scoped PEST mesh built on *solovev_eq*.

    Returns ``(S, TET, R_mesh, Z_mesh, q_iS, eq)`` — the same tuple that
    test_coordinates.py used to build at module scope.  Moving to session
    scope avoids rebuilding the PEST mesh for every test module that imports
    it.
    """
    eq = solovev_eq
    nR, nZ = 120, 120
    R_grid = np.linspace(0.3 * eq.R0, 1.5 * eq.R0, nR)
    Z_grid = np.linspace(-eq.a * eq.kappa * 1.2, eq.a * eq.kappa * 1.2, nZ)
    Rg, Zg = np.meshgrid(R_grid, Z_grid, indexing='ij')
    BR_grid, BZ_grid = eq.BR_BZ(Rg, Zg)
    BPhi_grid = eq.Bphi(Rg)
    psi_norm_grid = eq.psi(Rg, Zg)
    Rmaxis, Zmaxis = eq.magnetic_axis

    from pyna.MCF.coords.PEST import build_PEST_mesh
    S, TET, R_mesh, Z_mesh, q_iS = build_PEST_mesh(
        R_grid, Z_grid, BR_grid, BZ_grid, BPhi_grid, psi_norm_grid,
        Rmaxis, Zmaxis, ns=20, ntheta=91,
    )
    return S, TET, R_mesh, Z_mesh, q_iS, eq


# ---------------------------------------------------------------------------
# Simple circular-cross-section equilibrium (lightweight, for PEST unit tests)
# Used by: test_PEST.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def simple_eq_arrays():
    """Session-scoped lightweight circular equilibrium arrays.

    Returns ``(R, Z, BR, BZ, BPhi, psi_norm, Rmaxis, Zmaxis)`` for a simple
    circular-cross-section tokamak.  Replaces the repeated
    ``_make_simple_equilibrium()`` calls in test_PEST.py.
    """
    R0, Z0 = 1.8, 0.0
    a = 0.4
    B_tor = 2.0
    nR, nZ = 40, 40

    R = np.linspace(R0 - a - 0.1, R0 + a + 0.1, nR)
    Z = np.linspace(Z0 - a - 0.1, Z0 + a + 0.1, nZ)
    Rg, Zg = np.meshgrid(R, Z, indexing='ij')

    rho2 = ((Rg - R0) ** 2 + (Zg - Z0) ** 2) / a ** 2
    psi_norm = np.clip(rho2, 0.0, 1.0)

    dpsi_dR = 2.0 * (Rg - R0) / (a ** 2)
    dpsi_dZ = 2.0 * (Zg - Z0) / (a ** 2)
    pol_scale = 0.1
    BR   = pol_scale * dpsi_dZ / Rg
    BZ   = -pol_scale * dpsi_dR / Rg
    BPhi = B_tor / Rg

    return R, Z, BR, BZ, BPhi, psi_norm, R0, Z0
