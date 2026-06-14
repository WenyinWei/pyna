import numpy as np
import pytest

from pyna._cyna import progress_DX_pol_along_orbit as _cyna_progress_DX_pol_along_orbit
from pyna._cyna import is_available as _cyna_available
from pyna.toroidal.flt import progress_DX_pol_along_orbit


def test_evolve_dx_pol_along_orbit_linear_radial_field():
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
