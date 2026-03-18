"""Island O/X-point extraction from Poincar?? scatter data.

Provides
--------
* :class:`IslandChain` ???dataclass holding O/X points and widths.
* :func:`extract_island_width` ???infer island geometry from Poincar?? data.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class IslandChain:
    """Geometric description of an island chain.

    Attributes
    ----------
    O_points : ndarray of shape (n_islands, 2)
        O-point coordinates (R, Z) in metres.
    X_points : ndarray of shape (n_islands, 2)
        X-point coordinates (R, Z) in metres.
    half_width_r : float
        Average radial half-width in metres.
    half_width_psi : float
        Average half-width in normalised ?? coordinate.
    """

    O_points: np.ndarray
    X_points: np.ndarray
    half_width_r: float
    half_width_psi: float


def extract_island_width(
    poincare_pts: np.ndarray,
    R_axis: float,
    Z_axis: float,
    mode_m: int,
    psi_func,
    max_newton_iter: int = 50,
    n_fallback_seeds: int = 8,
) -> IslandChain:
    """Extract O/X points and island half-widths from Poincar?? scatter data.

    Algorithm
    ---------
    1. Compute angles from magnetic axis for each Poincar?? point.
    2. Bin into *mode_m* groups by angle.
    3. Cluster centroid ???initial O-point candidate.
    4. Refine each O-point using Nelder-Mead to minimise the radial variance
       of the points in that cluster.  Fallback seeds if optimisation diverges.
    5. X-points: midpoints between O-points in angle (same radial distance).
    6. Half-widths computed from point-cloud spread around each O-point.

    Parameters
    ----------
    poincare_pts : ndarray of shape (N, 2) or (N, 3)
        (R, Z) columns (additional columns ignored).
    R_axis, Z_axis : float
        Magnetic axis coordinates.
    mode_m : int
        Number of islands in the chain.
    psi_func : callable
        ``psi_func(R, Z) ???psi_norm``.
    max_newton_iter : int
        Not used directly; kept for API compatibility.
    n_fallback_seeds : int
        Number of fallback seeds if Nelder-Mead diverges.

    Returns
    -------
    IslandChain
    """
    pts = np.asarray(poincare_pts, dtype=float)
    R_pts = pts[:, 0]
    Z_pts = pts[:, 1]

    # Angles from magnetic axis
    angles = np.arctan2(Z_pts - Z_axis, R_pts - R_axis)  # in (-??, ??]
    # Normalise to [0, 2??)
    angles = angles % (2 * np.pi)
    # Radial distances from axis
    r_pts = np.sqrt((R_pts - R_axis) ** 2 + (Z_pts - Z_axis) ** 2)

    # Bin into mode_m sectors by angle
    bin_edges = np.linspace(0, 2 * np.pi, mode_m + 1)
    labels = np.digitize(angles, bin_edges) - 1
    labels = np.clip(labels, 0, mode_m - 1)

    O_points = []
    half_widths_R = []
    half_widths_psi = []

    for k in range(mode_m):
        mask = labels == k
        if mask.sum() < 3:
            continue

        R_k = R_pts[mask]
        Z_k = Z_pts[mask]
        r_k = r_pts[mask]

        # Initial centroid
        R0_k = float(np.mean(R_k))
        Z0_k = float(np.mean(Z_k))

        # Minimise radial variance in cluster ???proxy for O-point location
        def objective(rz):
            R_c, Z_c = rz[0], rz[1]
            r_c = np.sqrt((R_k - R_c) ** 2 + (Z_k - Z_c) ** 2)
            return float(np.var(r_c))

        res = minimize(objective, [R0_k, Z0_k], method='Nelder-Mead',
                       options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 500})
        R_O, Z_O = res.x[0], res.x[1]

        # Check if optimum is reasonable (inside domain)
        r_O = np.sqrt((R_O - R_axis) ** 2 + (Z_O - Z_axis) ** 2)
        r_mean = float(np.mean(r_k))
        if r_O > 2.0 * r_mean or r_O < 0.01 * r_mean:
            # Fallback: try seeds along radial direction
            theta_k = float(np.mean(np.arctan2(Z_k - Z_axis, R_k - R_axis)))
            best_obj = np.inf
            R_O, Z_O = R0_k, Z0_k
            for j in range(n_fallback_seeds):
                r_seed = r_mean * (0.5 + j / n_fallback_seeds)
                R_s = R_axis + r_seed * np.cos(theta_k)
                Z_s = Z_axis + r_seed * np.sin(theta_k)
                res2 = minimize(objective, [R_s, Z_s], method='Nelder-Mead',
                                options={'xatol': 1e-5, 'fatol': 1e-8})
                if res2.fun < best_obj:
                    best_obj = res2.fun
                    R_O, Z_O = res2.x[0], res2.x[1]

        O_points.append([R_O, Z_O])

        # Half-width: from point scatter around O-point
        r_from_O = np.sqrt((R_k - R_O) ** 2 + (Z_k - Z_O) ** 2)
        r_min = float(np.min(r_from_O))
        r_max = float(np.max(r_from_O))
        half_widths_R.append((r_max - r_min) / 2.0)

        # ?? half-width
        try:
            psi_O = float(psi_func(R_O, Z_O))
            angle_O = float(np.arctan2(Z_O - Z_axis, R_O - R_axis))
            dr = (r_max - r_min) / 2.0
            R_plus = R_O + dr * np.cos(angle_O)
            Z_plus = Z_O + dr * np.sin(angle_O)
            psi_plus = float(psi_func(R_plus, Z_plus))
            half_widths_psi.append(abs(psi_plus - psi_O))
        except Exception:
            half_widths_psi.append(np.nan)

    if not O_points:
        return IslandChain(
            O_points=np.empty((0, 2)),
            X_points=np.empty((0, 2)),
            half_width_r=np.nan,
            half_width_psi=np.nan,
        )

    O_points_arr = np.array(O_points)

    # X-points: midpoints between consecutive O-points (in angle)
    angles_O = np.arctan2(
        O_points_arr[:, 1] - Z_axis,
        O_points_arr[:, 0] - R_axis,
    )
    order = np.argsort(angles_O)
    O_sorted = O_points_arr[order]

    X_points_list = []
    r_O_sorted = np.sqrt(
        (O_sorted[:, 0] - R_axis) ** 2 + (O_sorted[:, 1] - Z_axis) ** 2
    )
    angles_O_sorted = np.arctan2(O_sorted[:, 1] - Z_axis, O_sorted[:, 0] - R_axis)
    n_O = len(O_sorted)
    for k in range(n_O):
        angle1 = angles_O_sorted[k]
        angle2 = angles_O_sorted[(k + 1) % n_O]
        angle_mid = (angle1 + angle2) / 2.0
        r_mid = (r_O_sorted[k] + r_O_sorted[(k + 1) % n_O]) / 2.0
        X_points_list.append([
            R_axis + r_mid * np.cos(angle_mid),
            Z_axis + r_mid * np.sin(angle_mid),
        ])

    X_points_arr = np.array(X_points_list) if X_points_list else np.empty((0, 2))

    avg_hw_R = float(np.nanmean(half_widths_R)) if half_widths_R else np.nan
    avg_hw_psi = float(np.nanmean(half_widths_psi)) if half_widths_psi else np.nan

    return IslandChain(
        O_points=O_sorted,
        X_points=X_points_arr,
        half_width_r=avg_hw_R,
        half_width_psi=avg_hw_psi,
    )


# ---------------------------------------------------------------------------
# Residual island detection in chaotic sea
# ---------------------------------------------------------------------------

def detect_residual_islands(
    poincare_pts: np.ndarray,
    R_axis: float,
    Z_axis: float,
    n_angle_bins: int = 72,
    min_cluster_fraction: float = 0.01,
    radial_bandwidth: float = 0.05,
    min_period: int = 1,
    max_period: int = 12,
) -> list:
    """Detect surviving (residual) magnetic islands embedded in a chaotic sea.

    In a chaotic Poincaré scatter plot, small regular islands appear as
    distinct, tightly-clustered arc-shaped structures amid the diffuse chaotic
    background.  This function identifies such clusters by:

    1. Partitioning the Poincaré points into annular radial shells.
    2. Within each shell, computing the angular density histogram.
    3. Looking for ``p``-fold periodic structure (for p in [min_period,
       max_period]) that exceeds the background by a statistically significant
       margin.
    4. Returning a list of candidate (period, radial_shell_index, angle_peaks)
       tuples from which callers can seed a fixed-point finder.

    Parameters
    ----------
    poincare_pts : ndarray, shape (N, 2) or (N, 3)
        Poincaré section points.  Only the first two columns (R, Z) are used.
    R_axis, Z_axis : float
        Magnetic axis coordinates.
    n_angle_bins : int
        Number of angular bins for the density histogram.  Higher values
        give finer angular resolution.  Default 72 (5° bins).
    min_cluster_fraction : float
        Minimum fraction of points in a radial shell that must fall in the
        candidate island peaks for it to be reported.  Default 0.01.
    radial_bandwidth : float
        Fractional width of radial shells (as a fraction of the max radius
        from axis).  Default 0.05 (5 % shells).
    min_period, max_period : int
        Range of periods (number of Poincaré-map iterations) to search.

    Returns
    -------
    candidates : list of dict
        Each dict contains:

        ``'period'`` : int
            Detected periodicity p (number of O-points expected in chain).
        ``'r_shell'`` : float
            Mean radial distance of the shell where the island was detected.
        ``'angle_peaks'`` : ndarray
            Approximate angular positions of the O-points.
        ``'seed_RZ'`` : ndarray, shape (p, 2)
            Rough seed points [R, Z] for fixed-point refinement.

    Notes
    -----
    This is a *rough* detection method intended to seed a Newton-refinement
    step (e.g. :func:`pyna.topo.fixed_points.find_periodic_orbit`).  The
    returned seed points are approximate and may not all converge to true
    fixed points.

    Examples
    --------
    >>> candidates = detect_residual_islands(poincare_pts, R_axis=1.0, Z_axis=0.0)
    >>> for c in candidates:
    ...     print(c['period'], c['r_shell'], c['seed_RZ'])
    """
    pts = np.asarray(poincare_pts, dtype=float)
    R_pts = pts[:, 0]
    Z_pts = pts[:, 1]

    r_pts = np.sqrt((R_pts - R_axis) ** 2 + (Z_pts - Z_axis) ** 2)
    angle_pts = np.arctan2(Z_pts - Z_axis, R_pts - R_axis) % (2 * np.pi)

    r_max = float(np.max(r_pts))
    if r_max < 1e-12:
        return []

    n_shells = max(1, int(1.0 / radial_bandwidth))
    shell_edges = np.linspace(0.0, r_max, n_shells + 1)

    candidates = []

    angle_bin_edges = np.linspace(0.0, 2.0 * np.pi, n_angle_bins + 1)

    for s in range(n_shells):
        r_lo, r_hi = shell_edges[s], shell_edges[s + 1]
        mask = (r_pts >= r_lo) & (r_pts < r_hi)
        if mask.sum() < max(10, int(len(pts) * min_cluster_fraction)):
            continue

        angles_in_shell = angle_pts[mask]
        hist, _ = np.histogram(angles_in_shell, bins=angle_bin_edges)
        hist = hist.astype(float)

        # Background level (robust median)
        bg = float(np.median(hist))
        hist_normalised = hist / (bg + 1e-10)

        for p in range(min_period, max_period + 1):
            # Look for p-fold peaks by folding the histogram
            n_per_segment = n_angle_bins // p
            if n_per_segment < 2:
                continue
            folded = np.zeros(n_per_segment)
            for k in range(p):
                segment = hist_normalised[k * n_per_segment: (k + 1) * n_per_segment]
                folded[:len(segment)] += segment
            # Detect peaks in the folded histogram
            folded_mean = float(np.mean(folded))
            folded_std = float(np.std(folded))
            threshold = folded_mean + 1.5 * folded_std
            peak_idx = np.where(folded > threshold)[0]
            if len(peak_idx) == 0:
                continue

            # The primary peak bin index in the folded array
            primary_bin = int(peak_idx[np.argmax(folded[peak_idx])])
            primary_angle = (primary_bin + 0.5) / n_per_segment * (2.0 * np.pi / p)

            # Replicate across p copies
            angle_peaks = np.array([
                (primary_angle + k * 2.0 * np.pi / p) % (2.0 * np.pi)
                for k in range(p)
            ])
            r_shell_mean = 0.5 * (r_lo + r_hi)

            # Seed points
            seed_RZ = np.column_stack([
                R_axis + r_shell_mean * np.cos(angle_peaks),
                Z_axis + r_shell_mean * np.sin(angle_peaks),
            ])

            candidates.append({
                'period': p,
                'r_shell': r_shell_mean,
                'angle_peaks': angle_peaks,
                'seed_RZ': seed_RZ,
            })

    return candidates
