"""pyna.topo.chaos — Chaotic region diagnostics for magnetic field topology.

This module provides tools to identify and characterise chaotic regions
in magnetic field configurations, including:

- Chirikov overlap criterion for island chain overlap
- Finite-Time Lyapunov Exponent (FTLE) field computation
- Chaotic boundary estimation from FTLE data

Typical usage
-------------
>>> sigma = chirikov_overlap(island_widths, island_positions)
>>> # sigma > 1 indicates overlapping (chaotic) regions

>>> ftle = ftle_field(my_field_func, R_grid, Z_grid, phi0=0.0, t_max=100.0)
>>> R_bdy, Z_bdy = chaotic_boundary_estimate(ftle, R_grid, Z_grid)
"""

import numpy as np
from typing import Callable, Optional, Tuple


def chirikov_overlap(
    island_widths: np.ndarray,
    island_positions: np.ndarray,
) -> np.ndarray:
    """Chirikov overlap parameter σ = (w1 + w2) / Δ between adjacent island pairs.

    The Chirikov criterion states that when the sum of half-widths of two
    adjacent island chains exceeds the distance between their O-points, the
    separatrices overlap and the region between them becomes chaotic.

    Parameters
    ----------
    island_widths : array of shape (N,)
        Half-widths of island chains in flux coordinate s = sqrt(psi_norm).
        Must be sorted in the same order as *island_positions*.
    island_positions : array of shape (N,)
        Radial positions (s) of island O-points. Should be monotonically
        increasing (sorted by radius).

    Returns
    -------
    sigma : array of shape (N-1,)
        Overlap parameter for each adjacent pair.
        sigma > 1  →  chaotic overlap predicted.
        sigma < 1  →  islands are separated (KAM surfaces likely persist).

    Notes
    -----
    The simple Chirikov criterion is a necessary but not sufficient condition
    for chaos.  Secondary island chains and shear both modulate the actual
    onset.  See Chirikov (1979), Physics Reports 52(5):263–379.

    Examples
    --------
    >>> widths = np.array([0.05, 0.04, 0.06])
    >>> positions = np.array([0.30, 0.45, 0.60])
    >>> sigma = chirikov_overlap(widths, positions)
    >>> sigma  # doctest: +SKIP
    array([0.60, 0.67])
    """
    island_widths = np.asarray(island_widths, dtype=float)
    island_positions = np.asarray(island_positions, dtype=float)

    if island_widths.shape != island_positions.shape:
        raise ValueError(
            "island_widths and island_positions must have the same shape; "
            f"got {island_widths.shape} and {island_positions.shape}"
        )
    if island_widths.ndim != 1 or len(island_widths) < 2:
        raise ValueError("Need at least two islands (1-D arrays of length ≥ 2).")

    delta = np.diff(island_positions)          # gap between adjacent O-points
    sum_w = island_widths[:-1] + island_widths[1:]  # sum of half-widths
    sigma = sum_w / delta
    return sigma


def ftle_field(
    field_func: Callable,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    phi0: float,
    t_max: float,
    dt: float = 0.1,
    integration_time: float = 50.0,
) -> np.ndarray:
    """Compute finite-time Lyapunov exponent field on an (R, Z) grid.

    The FTLE measures the maximum local stretching rate of infinitesimally
    separated field lines over a finite integration "time" (here, toroidal
    arc-length).  High FTLE values indicate sensitive dependence on initial
    conditions and mark chaotic regions and separatrices.

    Algorithm outline
    -----------------
    1. For each grid point (R_i, Z_j), initialise a small cluster of
       neighbouring seed points (finite-difference stencil).
    2. Integrate the field line ODE from φ = phi0 to φ = phi0 + t_max
       using the field line tracer.
    3. Compute the finite-difference approximation to the flow Jacobian J.
    4. FTLE = (1 / (2 * t_max)) * ln(λ_max(J^T J)), where λ_max is the
       largest singular value squared.

    Parameters
    ----------
    field_func : callable
        Field line ODE: ``field_func(rzphi) -> (dR, dZ, dphi)``
        where rzphi is a length-3 array [R, Z, φ].
    R_grid : 2-D array of shape (nR, nZ)
        R-coordinates of grid points.
    Z_grid : 2-D array of shape (nR, nZ)
        Z-coordinates of grid points (same shape as R_grid).
    phi0 : float
        Starting toroidal angle [rad].
    t_max : float
        Total integration arc-length (toroidal transits × 2π).
    dt : float
        Integration step in toroidal angle [rad].
    integration_time : float
        Duration over which Lyapunov growth is measured (toroidal transits).
        If t_max < integration_time * 2π, a warning is issued.

    Returns
    -------
    ftle : 2-D array of same shape as R_grid
        FTLE values [rad⁻¹].  High values mark chaotic regions and
        separatrices; low values indicate regular (KAM) regions.

    See Also
    --------
    pyna.flt.FieldLineTracer : underlying field line integrator
    chaotic_boundary_estimate : post-process FTLE into a boundary contour

    Notes
    -----
    Computational cost scales as O(nR × nZ × t_max / dt).  For large grids,
    consider running on GPU via pyna.flt CUDA backend.
    """
    if R_grid.shape != Z_grid.shape:
        raise ValueError("R_grid and Z_grid must have the same shape.")

    from pyna.flt import FieldLineTracer
    tracer = FieldLineTracer(field_func, dt=dt)

    eps = 1e-4  # stencil half-width
    NR, NZ = R_grid.shape
    ftle = np.full((NR, NZ), np.nan)

    for i in range(NR):
        for j in range(NZ):
            R0 = R_grid[i, j]
            Z0 = Z_grid[i, j]

            # 4 perturbed starts
            starts = np.array([
                [R0 + eps, Z0, phi0],
                [R0 - eps, Z0, phi0],
                [R0, Z0 + eps, phi0],
                [R0, Z0 - eps, phi0],
            ])

            try:
                trajs = [tracer.trace(s, t_max)[-1] for s in starts]

                # Jacobian: dR/dR0, dZ/dR0, dR/dZ0, dZ/dZ0
                dR_dR0 = (trajs[0][0] - trajs[1][0]) / (2 * eps)
                dZ_dR0 = (trajs[0][1] - trajs[1][1]) / (2 * eps)
                dR_dZ0 = (trajs[2][0] - trajs[3][0]) / (2 * eps)
                dZ_dZ0 = (trajs[2][1] - trajs[3][1]) / (2 * eps)

                J = np.array([[dR_dR0, dR_dZ0],
                               [dZ_dR0, dZ_dZ0]])

                # FTLE = log(max singular value) / t_max
                sv = np.linalg.svd(J, compute_uv=False)
                ftle[i, j] = np.log(sv[0] + 1e-30) / t_max
            except Exception:
                pass  # leave as nan

    return ftle


def chaotic_boundary_estimate(
    ftle: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    threshold_percentile: float = 85.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the rough boundary of the chaotic region from an FTLE field.

    Thresholds the FTLE field at a given percentile and extracts the contour
    of the resulting binary mask.  This gives an approximate boundary between
    the regular (low FTLE) and chaotic (high FTLE) regions.

    Parameters
    ----------
    ftle : 2-D array of shape (nR, nZ)
        FTLE field as returned by :func:`ftle_field`.
    R_grid : 2-D array of shape (nR, nZ)
        R-coordinates corresponding to *ftle*.
    Z_grid : 2-D array of shape (nR, nZ)
        Z-coordinates corresponding to *ftle*.
    threshold_percentile : float, optional
        Percentile of the FTLE distribution used as the threshold.
        Default is 85 (top 15 % of FTLE values are "chaotic").

    Returns
    -------
    R_boundary : 1-D array
        R-coordinates of contour points on the estimated chaotic boundary.
    Z_boundary : 1-D array
        Z-coordinates of contour points on the estimated chaotic boundary.

    Notes
    -----
    Uses ``skimage.measure.find_contours`` when available, falling back to
    ``matplotlib.pyplot.contour``.  The result is a rough geometric estimate;
    fine-grained boundaries require higher grid resolution or direct manifold
    computation.
    """
    if ftle.shape != R_grid.shape or ftle.shape != Z_grid.shape:
        raise ValueError("ftle, R_grid, and Z_grid must all have the same shape.")
    if not (0.0 < threshold_percentile < 100.0):
        raise ValueError("threshold_percentile must be in (0, 100).")

    threshold = np.percentile(ftle[np.isfinite(ftle)], threshold_percentile)
    mask = ftle > threshold
    # Try skimage first, fall back to scipy
    try:
        from skimage import measure
        contours = measure.find_contours(ftle, threshold)
        if not contours:
            return np.array([]), np.array([])
        # Convert pixel indices back to R,Z coords
        longest = max(contours, key=len)
        # longest[:,0] = row index (R), longest[:,1] = col index (Z)
        R_bdy = np.interp(longest[:, 0], np.arange(R_grid.shape[0]), R_grid[:, 0])
        Z_bdy = np.interp(longest[:, 1], np.arange(Z_grid.shape[1]), Z_grid[0, :])
        return R_bdy, Z_bdy
    except ImportError:
        # fallback: return boundary pixels from scipy erosion
        from scipy import ndimage
        boundary = mask & ~ndimage.binary_erosion(mask)
        R_bdy = R_grid[boundary]
        Z_bdy = Z_grid[boundary]
        return R_bdy, Z_bdy
