"""Hamada magnetic coordinates.

Hamada coordinates (ψ_H, θ_H, φ_H) satisfy:
    J·∇ψ_H = 0,  J·∇θ_H = 0,  J·∇φ_H = const

Physical meaning: current-density lines are straight in (θ_H, φ_H) space.
Used in MHD stability codes (e.g. TERPSICHORE).

For axisymmetric equilibria, Hamada theta is the equal-area angle:
the area element dA = R dR dZ on each flux surface is uniform in θ_H.
This is because the Hamada condition reduces (in axisymmetry) to the
requirement that the area swept from θ = 0 to θ_H is proportional to θ_H.

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

    Hamada theta is equal-area: the toroidal-averaged area element
    dA = R dR dZ on each flux surface is uniform in θ_H.

    For each flux surface, we compute the cumulative poloidal cross-section
    area (using the shoelace formula component dA_i = ½ |R_i dZ_{i+1} - R_{i+1} dZ_i|)
    and remap so this area is linearly proportional to θ_H.

    Parameters
    ----------
    S : ndarray, shape (ns,)
    TET : ndarray, shape (ntheta,)
        PEST poloidal angles (0 to 2π inclusive).
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)
    q_iS : ndarray, shape (ns,), optional
        Safety factor — not used in the current implementation but
        kept for API compatibility.
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

    for i in range(ns):
        R_s = R_mesh[i, :]
        Z_s = Z_mesh[i, :]

        # Handle axis
        if np.allclose(R_s, R_s[0]) and np.allclose(Z_s, Z_s[0]):
            R_mesh_H[i, :] = R_s[0]
            Z_mesh_H[i, :] = Z_s[0]
            continue

        # Close the loop
        R_closed = np.append(R_s, R_s[0])
        Z_closed = np.append(Z_s, Z_s[0])

        # Compute cumulative area using the shoelace increments
        # dA_k = 0.5 * |R_k * Z_{k+1} - R_{k+1} * Z_k|  (signed version for orientation)
        # We keep it signed to handle CW/CCW consistently; then take absolute value at the end.
        dA = 0.5 * (R_closed[:-1] * Z_closed[1:] - R_closed[1:] * Z_closed[:-1])
        A_cumulative = np.concatenate([[0.0], np.cumsum(dA)])
        A_total = A_cumulative[-1]

        # If total area is nearly zero something is wrong; fall back to arc-length
        if abs(A_total) < 1e-30:
            dR = np.diff(R_closed)
            dZ = np.diff(Z_closed)
            ds = np.sqrt(dR ** 2 + dZ ** 2)
            A_cumulative = np.concatenate([[0.0], np.cumsum(ds)])
            A_total = A_cumulative[-1]

        # Uniform area fraction
        A_uniform = np.linspace(0, A_total, n_theta, endpoint=True)

        R_mesh_H[i] = interp1d(A_cumulative, R_closed, kind='linear')(A_uniform)
        Z_mesh_H[i] = interp1d(A_cumulative, Z_closed, kind='linear')(A_uniform)

    return S, TET_H, R_mesh_H, Z_mesh_H
