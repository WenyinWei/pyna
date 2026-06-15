import numpy as np
import pytest

from pyna._cyna import progress_DX_pol_along_orbit as _cyna_progress_DX_pol_along_orbit
from pyna._cyna import progress_delta_X_along_orbit as _cyna_progress_delta_X_along_orbit
from pyna._cyna import evolve_delta_X_cycle_along_orbit as _cyna_evolve_delta_X_cycle_along_orbit
from pyna._cyna import is_available as _cyna_available
from pyna.toroidal.flt import (
    evolve_delta_X_cycle_along_orbit,
    progress_DX_pol_along_orbit,
    progress_delta_X_along_orbit,
)


def test_progress_dx_pol_along_orbit_linear_radial_field():
    if not _cyna_available() or _cyna_progress_DX_pol_along_orbit is None:
        pytest.skip("cyna progress_DX_pol_along_orbit extension is unavailable")

    alpha = 0.17
    R0 = 1.4
    Z0 = 0.05
    phi = np.linspace(0.0, 1.25, 41)
    R_traj = R0 * np.exp(alpha * phi)
    Z_traj = np.full_like(phi, Z0)

    R_grid = np.linspace(1.0, 2.0, 17)
    Z_grid = np.linspace(-0.4, 0.4, 13)
    Phi_grid = np.linspace(0.0, 2.0 * np.pi, 9)
    shape = (R_grid.size, Z_grid.size, Phi_grid.size)
    BR = np.full(shape, alpha, dtype=np.float64)
    BZ = np.zeros(shape, dtype=np.float64)
    BPhi = np.ones(shape, dtype=np.float64)

    DX = progress_DX_pol_along_orbit(
        R_traj,
        Z_traj,
        phi,
        R_grid,
        Z_grid,
        Phi_grid,
        BR.ravel(),
        BZ.ravel(),
        BPhi.ravel(),
        max_step=0.002,
    )

    expected = np.zeros_like(DX)
    expected[:, 0, 0] = np.exp(alpha * (phi - phi[0]))
    expected[:, 1, 1] = 1.0

    assert DX.shape == (phi.size, 2, 2)
    np.testing.assert_allclose(DX[0], np.eye(2), atol=1e-14)
    np.testing.assert_allclose(DX, expected, rtol=1e-8, atol=1e-10)


def test_progress_delta_x_along_orbit_uses_delta_b_source():
    if not _cyna_available() or _cyna_progress_delta_X_along_orbit is None:
        pytest.skip("cyna progress_delta_X_along_orbit extension is unavailable")

    alpha = 0.17
    eps_br = 0.031
    eps_bz = -0.023
    R0 = 1.4
    Z0 = 0.05
    phi = np.linspace(0.0, 1.25, 41)
    R_traj = R0 * np.exp(alpha * phi)
    Z_traj = np.full_like(phi, Z0)

    R_grid = np.linspace(1.0, 2.0, 17)
    Z_grid = np.linspace(-0.4, 0.4, 13)
    Phi_grid = np.linspace(0.0, 2.0 * np.pi, 9)
    shape = (R_grid.size, Z_grid.size, Phi_grid.size)
    BR = np.full(shape, alpha, dtype=np.float64)
    BZ = np.zeros(shape, dtype=np.float64)
    BPhi = np.ones(shape, dtype=np.float64)
    dBR = np.full(shape, eps_br, dtype=np.float64)
    dBZ = np.full(shape, eps_bz, dtype=np.float64)
    dBPhi = np.zeros(shape, dtype=np.float64)

    dX = progress_delta_X_along_orbit(
        R_traj,
        Z_traj,
        phi,
        (0.0, 0.0),
        R_grid,
        Z_grid,
        Phi_grid,
        BR.ravel(),
        BZ.ravel(),
        BPhi.ravel(),
        dBR.ravel(),
        dBZ.ravel(),
        dBPhi.ravel(),
        max_step=0.002,
    )

    expected = np.empty_like(dX)
    expected[:, 0] = eps_br * R0 * phi * np.exp(alpha * phi)
    expected[:, 1] = eps_bz * R0 * (np.exp(alpha * phi) - 1.0) / alpha

    assert dX.shape == (phi.size, 2)
    np.testing.assert_allclose(dX[0], 0.0, atol=1e-14)
    # The cyna routine intentionally uses the supplied orbit samples as the
    # path and linearly interpolates between them, so this comparison allows the
    # small interpolation error relative to the continuous exponential orbit.
    np.testing.assert_allclose(dX, expected, rtol=5e-6, atol=1e-10)


def test_evolve_delta_x_cycle_alias_uses_same_response_ode():
    if not _cyna_available() or _cyna_evolve_delta_X_cycle_along_orbit is None:
        pytest.skip("cyna evolve_delta_X_cycle_along_orbit extension is unavailable")

    alpha = 0.11
    eps_br = 0.02
    R0 = 1.3
    phi = np.linspace(0.0, 0.8, 21)
    R_traj = R0 * np.exp(alpha * phi)
    Z_traj = np.zeros_like(phi)
    delta0 = np.asarray([0.004, -0.002])

    R_grid = np.linspace(1.0, 1.8, 17)
    Z_grid = np.linspace(-0.3, 0.3, 13)
    Phi_grid = np.linspace(0.0, 2.0 * np.pi, 9)
    shape = (R_grid.size, Z_grid.size, Phi_grid.size)
    BR = np.full(shape, alpha, dtype=np.float64)
    BZ = np.zeros(shape, dtype=np.float64)
    BPhi = np.ones(shape, dtype=np.float64)
    dBR = np.full(shape, eps_br, dtype=np.float64)
    dBZ = np.zeros(shape, dtype=np.float64)
    dBPhi = np.zeros(shape, dtype=np.float64)

    common_args = (
        R_traj,
        Z_traj,
        phi,
        delta0,
        R_grid,
        Z_grid,
        Phi_grid,
        BR.ravel(),
        BZ.ravel(),
        BPhi.ravel(),
        dBR.ravel(),
        dBZ.ravel(),
        dBPhi.ravel(),
    )
    progressed = progress_delta_X_along_orbit(*common_args, max_step=0.002)
    evolved = evolve_delta_X_cycle_along_orbit(*common_args, max_step=0.002)

    np.testing.assert_allclose(evolved, progressed, rtol=0.0, atol=0.0)
