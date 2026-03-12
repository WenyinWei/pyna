"""Equal-arc-length magnetic coordinate.

The equal-arc poloidal angle θ_ea is defined such that the arc length
along the flux surface is uniformly distributed in θ_ea.

This is the simplest modification of the geometric angle and is often
used as a starting point for more sophisticated transforms.
"""
import numpy as np
from scipy.interpolate import interp1d


def build_equal_arc_mesh(S, TET, R_mesh, Z_mesh, n_theta=181):
    """Build equal-arc-length coordinate mesh.

    For each flux surface (each S value), compute the cumulative arc length
    along the surface, then remap so arc length is uniform in theta_ea.

    Parameters
    ----------
    S : ndarray, shape (ns,)
        Radial PEST coordinate values.
    TET : ndarray, shape (ntheta,)
        PEST poloidal angles (0 to 2π inclusive).
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)
        PEST coordinate mesh in cylindrical (R, Z).
    n_theta : int
        Number of poloidal points in the output mesh.

    Returns
    -------
    S : ndarray, shape (ns,)
        Unchanged flux labels.
    TET_ea : ndarray, shape (n_theta,)
        Equal-arc poloidal angles, from 0 to 2π (inclusive).
    R_mesh_ea, Z_mesh_ea : ndarray, shape (ns, n_theta)
        Remapped coordinate mesh.
    """
    ns, ntheta = R_mesh.shape
    R_mesh_ea = np.empty((ns, n_theta))
    Z_mesh_ea = np.empty((ns, n_theta))
    TET_ea = np.linspace(0, 2 * np.pi, n_theta, endpoint=True)

    for i in range(ns):
        R_s = R_mesh[i, :]
        Z_s = Z_mesh[i, :]

        # Handle the axis (all points coincide → just copy)
        if np.allclose(R_s, R_s[0]) and np.allclose(Z_s, Z_s[0]):
            R_mesh_ea[i, :] = R_s[0]
            Z_mesh_ea[i, :] = Z_s[0]
            continue

        # Close the loop for arc-length computation
        R_closed = np.append(R_s, R_s[0])
        Z_closed = np.append(Z_s, Z_s[0])
        dR = np.diff(R_closed)
        dZ = np.diff(Z_closed)
        ds = np.sqrt(dR ** 2 + dZ ** 2)
        s_cumulative = np.concatenate([[0], np.cumsum(ds)])
        s_total = s_cumulative[-1]

        # Build uniform arc-length parametrisation
        # s_cumulative runs 0..s_total over ntheta+1 points (closed loop)
        # TET_ea runs 0..2π over n_theta points (last = 2π = same as first)
        # We map s_uniform[k] = k/n_theta * s_total for k=0..n_theta-1, then endpoint=2π → s_total
        s_uniform = np.linspace(0, s_total, n_theta, endpoint=True)

        R_mesh_ea[i] = interp1d(
            s_cumulative, R_closed, kind='linear')(s_uniform)
        Z_mesh_ea[i] = interp1d(
            s_cumulative, Z_closed, kind='linear')(s_uniform)

    return S, TET_ea, R_mesh_ea, Z_mesh_ea
