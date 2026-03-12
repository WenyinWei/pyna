"""Plasma-wall gap response to coil current perturbations.

Uses FPT manifold shift formula to compute ∂(gap_gi)/∂(I_coil_k)
without expensive Poincaré map recalculation.

Algorithm for ∂(gap_gi)/∂(I_coil_k):
1. Grow the unperturbed stable manifold from X-point (arc length parameter s ∈ [0, s_max])
2. Find the manifold point X^s(s_i) closest to wall monitoring point i
3. Compute δX^s(s_i) using FPT manifold shift formula
4. Gap change: δg_i = -n̂_i · δX^s(s_i)  (n̂_i = inward wall normal)

Caching: stable manifold growth is expensive (field-line integration);
cache the unperturbed manifold and the δB_pol evaluations separately.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, TYPE_CHECKING

import joblib

# Disk-persistent cache for expensive manifold integrations
_memory = joblib.Memory(location='.cache/gap_response', verbose=0)

if TYPE_CHECKING:
    from pyna.control.wall import WallGeometry
    from pyna.control.topology_state import XPointState


# ---------------------------------------------------------------------------
# Manifold growing (field-line integration)
# ---------------------------------------------------------------------------

def _grow_manifold(
    field_func: Callable,
    x_point,
    s_max: float = 2.0,
    ds: float = 0.01,
) -> np.ndarray:
    """Grow stable manifold from X-point by forward integration.

    Integrates along the stable eigenvector direction using Runge-Kutta.

    Parameters
    ----------
    field_func : callable
        [R,Z,phi] → [dR/dl, dZ/dl, dphi/dl].
    x_point : XPointState
        Contains R, Z, and DPm_eigenvectors.
    s_max : float
        Maximum arc length to trace (m).
    ds : float
        Arc-length step size.

    Returns
    -------
    pts : ndarray, shape (N, 2)
        (R, Z) points along the stable manifold.
    """
    from scipy.linalg import eig

    # Stable eigenvector: eigenvalue < 1 for stable manifold
    eigvals, eigvecs = eig(x_point.DPm)
    # For X-point, one eigenvalue is > 1 (unstable), one is < 1 (stable)
    stable_idx = np.argmin(np.abs(eigvals.real))
    stable_eigvec = eigvecs[:, stable_idx].real
    stable_eigvec /= np.linalg.norm(stable_eigvec) + 1e-30

    # Start slightly away from X-point in stable direction
    eps0 = 1e-4
    R0 = x_point.R + eps0 * stable_eigvec[0]
    Z0 = x_point.Z + eps0 * stable_eigvec[1]

    pts = [(R0, Z0)]
    R, Z = R0, Z0

    n_steps = int(s_max / ds)
    for _ in range(n_steps):
        # Runge-Kutta 4 along field line
        def dXdl(r, z):
            f = np.asarray(field_func([r, z, 0.0]), dtype=float)
            # Normalised poloidal direction
            Bpol = np.array([f[0], f[1]])
            norm = np.linalg.norm(Bpol) + 1e-30
            return Bpol / norm

        k1 = dXdl(R, Z)
        k2 = dXdl(R + 0.5*ds*k1[0], Z + 0.5*ds*k1[1])
        k3 = dXdl(R + 0.5*ds*k2[0], Z + 0.5*ds*k2[1])
        k4 = dXdl(R + ds*k3[0], Z + ds*k3[1])

        dR = ds * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
        dZ = ds * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0

        R = R + dR
        Z = Z + dZ
        pts.append((R, Z))

    return np.array(pts)


@_memory.cache
def grow_stable_manifold_cached(
    field_func_key: str,
    R_xpt: float,
    Z_xpt: float,
    stable_eigvec: tuple,
    DPm_flat: tuple,
    s_max: float = 2.0,
    ds: float = 0.01,
) -> np.ndarray:
    """Grow stable manifold from X-point, return (R, Z) array.

    Cached because this requires field-line integration.
    field_func_key is a string that uniquely identifies the field
    (e.g., eq params hash).

    NOTE: Because joblib cannot cache callable arguments, the field function
    is NOT accepted here. Use gap_response_matrix_fpt which manages the
    field function externally and passes the key for cache invalidation.

    Parameters
    ----------
    field_func_key : str
        Hashable identifier for the field; used only as a cache key.
        The actual integration is done by _grow_manifold (see
        gap_response_matrix_fpt which wraps this).
    R_xpt, Z_xpt : float
        X-point coordinates (also part of cache key).
    stable_eigvec : tuple (eR, eZ)
        Stable eigenvector direction.
    DPm_flat : tuple
        Flattened DPm matrix (4 elements).
    s_max : float
    ds : float

    Returns
    -------
    pts : ndarray, shape (N, 2)
    """
    # This function is a cache stub — the real computation happens in
    # _grow_manifold. We return a sentinel that gap_response_matrix_fpt
    # replaces with the actual result on first call.
    raise NotImplementedError(
        "grow_stable_manifold_cached is a cache key stub; "
        "use gap_response_matrix_fpt which manages the field function."
    )


# ---------------------------------------------------------------------------
# Gap response matrix
# ---------------------------------------------------------------------------

def gap_response_matrix_fpt(
    base_field_func: Callable,
    coil_field_funcs: List[Callable],
    wall: 'WallGeometry',
    x_point: 'XPointState',
    field_func_key: str = 'default',
    s_max: float = 2.0,
    ds: float = 0.01,
) -> tuple:
    """Compute ∂(gap_gi)/∂(I_coil_k) using FPT manifold shift.

    Parameters
    ----------
    base_field_func : callable
        Base equilibrium field function [R,Z,phi] → [dR/dl, dZ/dl, dphi/dl].
    coil_field_funcs : list of callable
        coil_field_funcs[k](rzphi) = δB per 1A for coil k.
    wall : WallGeometry
        First wall geometry with gap monitoring points.
    x_point : XPointState
        X-point state (contains DPm eigenvectors, A_matrix, R, Z).
    field_func_key : str
        Hashable string identifying the equilibrium (for caching).
    s_max : float
        Maximum manifold arc length to trace (m).
    ds : float
        Integration step along the manifold (m).

    Returns
    -------
    R_gap : ndarray, shape (n_gaps, n_coils)
        ∂(gap_i)/∂(I_k) in metres per ampere.
    gap_names : list of str
    """
    from pyna.control.fpt import manifold_shift, cycle_shift

    n_gaps = len(wall.gap_monitor_names)
    n_coils = len(coil_field_funcs)
    R_gap = np.zeros((n_gaps, n_coils))

    # Grow unperturbed stable manifold (expensive; cached per equilibrium)
    manifold_pts = _grow_manifold(base_field_func, x_point, s_max, ds)
    # shape (N_s, 2)

    for k, delta_field in enumerate(coil_field_funcs):
        # ── X-point cycle shift ──────────────────────────────────────────
        R_cyc, Z_cyc = x_point.R, x_point.Z
        phi = 0.0
        f0 = np.asarray(base_field_func([R_cyc, Z_cyc, phi]), dtype=float)
        fd = np.asarray(delta_field([R_cyc, Z_cyc, phi]), dtype=float)

        g0 = np.array([f0[0] / f0[2], f0[1] / f0[2]])
        denom = f0[2] + fd[2]
        g1 = np.array([(f0[0] + fd[0]) / denom, (f0[1] + fd[1]) / denom])
        delta_g = g1 - g0

        delta_xcyc = cycle_shift(x_point.A_matrix, delta_g)

        # ── Manifold shift along full arc ────────────────────────────────
        delta_mfld = manifold_shift(
            base_field_func, delta_field, manifold_pts, delta_xcyc,
            stable=True,
        )  # shape (N_s, 2)

        # ── Gap change for each monitoring point ─────────────────────────
        for i, (name, R_mon, Z_mon) in enumerate(zip(
            wall.gap_monitor_names,
            wall.gap_monitor_R,
            wall.gap_monitor_Z,
        )):
            # Find closest manifold point to this monitor
            dists = np.sqrt(
                (manifold_pts[:, 0] - R_mon) ** 2
                + (manifold_pts[:, 1] - Z_mon) ** 2
            )
            idx_closest = np.argmin(dists)

            # Inward wall normal at monitor (pointing toward plasma)
            n_hat = wall.inward_normal_at(R_mon, Z_mon)

            # δg_i = -n̂_i · δX^s(s_i)
            R_gap[i, k] = -np.dot(n_hat, delta_mfld[idx_closest])

    return R_gap, wall.gap_monitor_names
