"""Hamada magnetic coordinates.

Hamada coordinates (ψ_H, θ_H, φ_H) satisfy:
    J·∇ψ_H = 0,  J·∇θ_H = 0,  J·∇φ_H = const

Physical meaning: current-density lines are straight in (θ_H, φ_H) space.
Used in MHD stability codes (e.g. TERPSICHORE).

For axisymmetric equilibria, Hamada theta is the equal-area angle:
the area swept from the magnetic axis to the flux-surface contour is
proportional to θ_H.  This is because the Hamada condition reduces (in
axisymmetry) to requiring a uniform Jacobian on each flux surface.

References
----------
Hamada, Nucl. Fusion 2, 23 (1962).
White & Chance, Phys. Fluids 27, 2455 (1984).
"""
import numpy as np
from scipy.interpolate import interp1d


def build_Hamada_mesh(S, TET, R_mesh, Z_mesh, q_iS=None, equilibrium=None,
                      n_theta=181):
    """Build Hamada coordinate mesh.

    Hamada theta is equal-area: the area of the triangle formed by the
    magnetic axis and consecutive contour segments is uniform in θ_H.

    For each flux surface, we compute the cumulative area using triangles
    between the magnetic axis (estimated as R_mesh[0, 0], Z_mesh[0, 0])
    and contour segments, then remap so this area is linearly proportional
    to θ_H.

    Parameters
    ----------
    S : ndarray, shape (ns,)
    TET : ndarray, shape (ntheta,)
        PEST poloidal angles (0 to 2π inclusive).
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)
    q_iS : ndarray, shape (ns,), optional
        Safety factor — not used currently.
    equilibrium : optional
        Not used currently.
    n_theta : int
        Number of poloidal points in the output.

    Returns
    -------
    S : ndarray
    TET_H : ndarray, shape (n_theta,)
    R_mesh_H, Z_mesh_H : ndarray, shape (ns, n_theta)
    """
    ns, ntheta = R_mesh.shape
    R_mesh_H = np.empty((ns, n_theta))
    Z_mesh_H = np.empty((ns, n_theta))
    TET_H = np.linspace(0, 2 * np.pi, n_theta, endpoint=True)

    # Estimate magnetic axis from axis row (all values should be the same)
    R_ax = R_mesh[0, 0]
    Z_ax = Z_mesh[0, 0]

    for i in range(ns):
        R_s = R_mesh[i, :]
        Z_s = Z_mesh[i, :]

        # Handle axis
        if np.allclose(R_s, R_s[0]) and np.allclose(Z_s, Z_s[0]):
            R_mesh_H[i, :] = R_s[0]
            Z_mesh_H[i, :] = Z_s[0]
            continue

        # TET may include endpoint (0 and 2π are the same physical point).
        # Use only unique points (drop the last if it duplicates the first).
        if np.allclose(R_s[0], R_s[-1]) and np.allclose(Z_s[0], Z_s[-1]):
            R_loop = R_s[:-1]
            Z_loop = Z_s[:-1]
        else:
            R_loop = R_s
            Z_loop = Z_s

        n_loop = len(R_loop)

        # Close the loop
        R_closed = np.append(R_loop, R_loop[0])
        Z_closed = np.append(Z_loop, Z_loop[0])

        # Area of triangle (axis, P_k, P_{k+1}) using cross product:
        # dA = 0.5 * [(R_k - R_ax)*(Z_{k+1} - Z_ax) - (R_{k+1} - R_ax)*(Z_k - Z_ax)]
        dA = 0.5 * (
            (R_closed[:-1] - R_ax) * (Z_closed[1:] - Z_ax)
            - (R_closed[1:] - R_ax) * (Z_closed[:-1] - Z_ax)
        )

        A_cumulative = np.concatenate([[0.0], np.cumsum(dA)])
        A_total = A_cumulative[-1]

        # Fallback to arc-length if area is degenerate
        if abs(A_total) < 1e-30:
            dR = np.diff(R_closed)
            dZ = np.diff(Z_closed)
            ds = np.sqrt(dR ** 2 + dZ ** 2)
            A_cumulative = np.concatenate([[0.0], np.cumsum(ds)])
            A_total = A_cumulative[-1]

        # Uniform area fraction over [0, A_total], n_theta points inclusive
        A_uniform = np.linspace(0, A_total, n_theta, endpoint=True)

        R_interp = interp1d(A_cumulative, R_closed, kind='linear')
        Z_interp = interp1d(A_cumulative, Z_closed, kind='linear')
        R_mesh_H[i] = R_interp(A_uniform)
        Z_mesh_H[i] = Z_interp(A_uniform)

    return S, TET_H, R_mesh_H, Z_mesh_H
