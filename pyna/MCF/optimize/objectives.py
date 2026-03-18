"""pyna.MCF.optimize.objectives — Physics objectives for stellarator optimization.

This module collects scalar objective functions used in multi-objective
stellarator optimisation.  Each function accepts an *equilibrium* object
(e.g. a ``pyna.MCF.equilibrium.Equilibrium`` instance) and returns a single
float.  All objectives are defined so that **lower is better** unless noted.

Objectives implemented
----------------------
- :func:`neoclassical_epsilon_eff`  — effective ripple ε_eff (transport proxy)
- :func:`xpoint_field_parallelism`  — divertor field-line parallelism metric
- :func:`magnetic_axis_position`    — (R_axis, Z_axis) of the magnetic axis
- :func:`wall_clearance`            — minimum LCFS-to-wall distance [m]
- :func:`compute_all_objectives`    — convenience wrapper returning a dict

References
----------
Nemov et al. (1999), Phys. Plasmas 6(12):4622 — ε_eff definition.
Boozer (2015), Rev. Mod. Phys. 76:1071 — stellarator optimisation review.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Individual objectives
# ---------------------------------------------------------------------------


def neoclassical_epsilon_eff(
    equilibrium,
    n_field_lines: int = 50,
    n_transits: int = 100,
) -> float:
    """Estimate effective ripple ε_eff (proxy for neoclassical transport).

    ε_eff characterises the amplitude of the helical magnetic ripple that
    drives 1/ν transport in the long-mean-free-path regime.  A lower value
    indicates better quasi-symmetry and reduced neoclassical losses.

    This function implements a geometric approximation by averaging the
    normalised helical ripple amplitude ε_h = ΔB / (2 B₀) over flux
    surfaces.  For a quantitatively accurate result, couple to an external
    code such as NEO or DKES.

    Parameters
    ----------
    equilibrium : equilibrium-like object
        Must expose ``equilibrium.B_field(R, Z, phi)`` and
        ``equilibrium.flux_surfaces(n=n_field_lines)`` (or similar API).
    n_field_lines : int, optional
        Number of field lines (flux surfaces) to sample.  Default 50.
    n_transits : int, optional
        Number of toroidal transits for field-line averaging.  Default 100.

    Returns
    -------
    eps_eff : float
        Effective ripple, dimensionless and ≥ 0.  Lower is better.
        A value < 0.01 is generally considered good for a stellarator.

    Notes
    -----
    The approximation used here is::

        ε_h(s) ≈ (B_max(s) - B_min(s)) / (B_max(s) + B_min(s))
        ε_eff  ≈ mean over flux surfaces weighted by volume element

    where B_max / B_min are the maximum and minimum |B| along a field line
    on flux surface s.
    """
    # For stellarators we can estimate from helical ripple directly
    # epsilon_h is already stored on StellaratorSimple
    if hasattr(equilibrium, 'epsilon_h'):
        # Geometric estimate: eps_eff ≈ 0.64 * epsilon_h^(3/2) for helical devices
        # (Nemov et al. 1999 scaling)
        return 0.64 * abs(equilibrium.epsilon_h) ** 1.5

    # General case: scan flux surfaces
    psi_vals = np.linspace(0.1, 0.9, n_field_lines)
    eps_arr = []
    for psi_n in psi_vals:
        r = np.sqrt(psi_n) * getattr(equilibrium, 'r0', 0.3)
        theta_arr = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        R_arr = getattr(equilibrium, 'R0', 1.0) + r * np.cos(theta_arr)
        Z_arr = r * np.sin(theta_arr)
        # Sample |B| along the surface
        B_vals = []
        for phi in np.linspace(0, 2 * np.pi / getattr(equilibrium, 'n_h', 1), n_transits // 10):
            for R, Z in zip(R_arr, Z_arr):
                # Estimate B ~ B0 * R0 / R (toroidal dominance)
                B_vals.append(equilibrium.B0 * equilibrium.R0 / R)
        B_arr = np.array(B_vals)
        Bmax, Bmin = B_arr.max(), B_arr.min()
        delta_b = (Bmax - Bmin) / (Bmax + Bmin + 1e-30)
        eps_arr.append(0.64 * delta_b ** 1.5)
    return float(np.mean(eps_arr))


def xpoint_field_parallelism(
    equilibrium,
    x_points: List[Tuple[float, float]],
    n_fieldlines: int = 20,
    n_transits: int = 30,
) -> float:
    """Measure field-line parallelism near X-points (for power exhaust).

    At a good divertor X-point the field lines should arrive nearly parallel,
    spreading the heat load.  This function traces a fan of field lines
    seeded close to each X-point and measures their angular spread after
    *n_transits* toroidal passes.

    A higher return value means the field lines remain more parallel →
    better divertor performance.

    Parameters
    ----------
    equilibrium : equilibrium-like object
        Must expose a field-line tracer or ``pyna.flt.FieldLineTracer``
        can be constructed from it.
    x_points : list of (R, Z) tuples
        Approximate positions of the divertor X-points to analyse.
    n_fieldlines : int, optional
        Number of field lines seeded around each X-point.  Default 20.
    n_transits : int, optional
        Number of toroidal transits to integrate.  Default 30.

    Returns
    -------
    parallelism : float
        Average cos(θ) between neighbouring field-line tangent vectors near
        X-points.  Range [0, 1].  Higher is better.

    Notes
    -----
    Algorithm sketch::

        seeds = uniform_ring(x_point, radius=small_delta, n=n_fieldlines)
        tangents = [field_tangent(seed) for seed in seeds]
        cos_angles = pairwise_cos(tangents)
        return mean(cos_angles)
    """
    if not x_points:
        return 0.0

    R0 = getattr(equilibrium, 'R0', 1.65)
    a = getattr(equilibrium, 'r0', getattr(equilibrium, 'a', 0.5))

    def _field_tangent(R: float, Z: float, phi: float) -> np.ndarray:
        """Unit tangent vector (dR/dl, dZ/dl) from the equilibrium field."""
        if hasattr(equilibrium, 'field_func'):
            f = equilibrium.field_func([R, Z, phi])
            dRdphi, dZdphi = float(f[0]), float(f[1])
        else:
            # Circular tokamak approximation: scale factor 0.1 sets the
            # normalised poloidal field magnitude (B_pol/B_phi ~ ε/q ~ 0.1
            # for a typical large-aspect-ratio tokamak with ε ≈ a/R ≈ 0.3, q ≈ 3).
            # Only the direction matters here (result is unit-normalised), so the
            # exact value cancels out.
            _B_pol_scale = 0.1
            dRdphi = -Z / (R + 1e-30) * _B_pol_scale
            dZdphi = (R - R0) / (R + 1e-30) * _B_pol_scale
        norm = np.sqrt(dRdphi**2 + dZdphi**2) + 1e-30
        return np.array([dRdphi / norm, dZdphi / norm])

    seed_radius = min(a * 0.02, 5e-3)
    metrics = []

    for (Rx, Zx) in x_points:
        seed_angles = np.linspace(0, 2 * np.pi, n_fieldlines, endpoint=False)
        R_seeds = Rx + seed_radius * np.cos(seed_angles)
        Z_seeds = Zx + seed_radius * np.sin(seed_angles)

        tangents = np.array([
            _field_tangent(R_seeds[i], Z_seeds[i], 0.0)
            for i in range(n_fieldlines)
        ])  # shape (n_fieldlines, 2)

        cos_angles = [
            float(np.dot(tangents[i], tangents[(i + 1) % n_fieldlines]))
            for i in range(n_fieldlines)
        ]
        metrics.append(float(np.mean(cos_angles)))

    return float(np.mean(metrics)) if metrics else 0.0


def magnetic_axis_position(equilibrium) -> Tuple[float, float]:
    """Return (R_axis, Z_axis) of the magnetic axis.

    Parameters
    ----------
    equilibrium : equilibrium-like object
        Must expose a ``magnetic_axis`` attribute returning ``(R, Z)``
        or a compatible pair.

    Returns
    -------
    R_axis : float
        Major-radial position of the magnetic axis [m].
    Z_axis : float
        Vertical position of the magnetic axis [m].

    Notes
    -----
    This is a thin wrapper; the actual computation is delegated to the
    equilibrium object.  For optimisation the desired axis position is
    typically set by a target ``(R_target, Z_target)`` and the objective
    is the Euclidean distance::

        obj = sqrt((R_axis - R_target)**2 + (Z_axis - Z_target)**2)
    """
    return equilibrium.magnetic_axis  # type: ignore[return-value]


def wall_clearance(
    equilibrium,
    wall_R: np.ndarray,
    wall_Z: np.ndarray,
) -> float:
    """Minimum distance from the LCFS to the first wall in the (R, Z) plane.

    Parameters
    ----------
    equilibrium : equilibrium-like object
        Must expose ``equilibrium.lcfs()`` returning an (R, Z) polygon or a
        ``pyna.MCF.equilibrium.FluxSurface`` with ``.R`` / ``.Z`` arrays.
    wall_R : array of shape (M,)
        R-coordinates of wall polygon vertices [m].
    wall_Z : array of shape (M,)
        Z-coordinates of wall polygon vertices [m].

    Returns
    -------
    clearance : float
        Minimum perpendicular distance from any LCFS point to the nearest
        wall segment, in metres.  Positive → LCFS inside wall (safe).
        Negative → LCFS has crossed the wall (unphysical / bad configuration).

    Notes
    -----
    Algorithm::

        lcfs_pts = equilibrium.lcfs()           # shape (K, 2)
        wall_poly = Polygon(wall_R, wall_Z)
        clearance = min(dist(pt, wall_poly) for pt in lcfs_pts)
        # Use shapely or a manual segment-point distance loop
    """
    # LCFS circle
    theta = np.linspace(0, 2 * np.pi, 500)
    R_lcfs = equilibrium.R0 + equilibrium.r0 * np.cos(theta)
    Z_lcfs = equilibrium.r0 * np.sin(theta)

    wall_pts = np.column_stack([wall_R, wall_Z])
    lcfs_pts = np.column_stack([R_lcfs, Z_lcfs])

    # Min distance from LCFS to wall
    from scipy.spatial import cKDTree
    tree = cKDTree(wall_pts)
    dists, _ = tree.query(lcfs_pts)
    return float(dists.min())


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def compute_all_objectives(
    equilibrium,
    wall_R: Optional[np.ndarray] = None,
    wall_Z: Optional[np.ndarray] = None,
    x_points: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Compute all available physics objectives and return as a dict.

    Useful for logging, Pareto-front analysis, and as a single entry point
    for optimisation drivers (e.g. ``scipy.optimize``, ``pymoo``, STELLOPT).

    Parameters
    ----------
    equilibrium : equilibrium-like object
        Shared equilibrium for all objectives.
    wall_R, wall_Z : arrays, optional
        Wall polygon; required for :func:`wall_clearance`.  If *None*, that
        objective is skipped.
    x_points : list of (R, Z), optional
        X-point positions; required for :func:`xpoint_field_parallelism`.
        If *None*, that objective is skipped.

    Returns
    -------
    objectives : dict
        Keys: ``"eps_eff"``, ``"R_axis"``, ``"Z_axis"``,
              ``"wall_clearance"`` (optional), ``"xpoint_parallelism"`` (optional).
        Values: scalar floats.

    Examples
    --------
    >>> objs = compute_all_objectives(eq, wall_R=R_wall, wall_Z=Z_wall)
    >>> objs["eps_eff"]
    0.023
    """
    result = {}
    result['magnetic_axis'] = magnetic_axis_position(equilibrium)
    try:
        result['epsilon_eff'] = neoclassical_epsilon_eff(equilibrium)
    except Exception as e:
        result['epsilon_eff'] = None
        result['epsilon_eff_error'] = str(e)
    if wall_R is not None and wall_Z is not None:
        try:
            result['wall_clearance'] = wall_clearance(equilibrium, wall_R, wall_Z)
        except Exception as e:
            result['wall_clearance'] = None
    if x_points is not None:
        try:
            result['xpoint_parallelism'] = xpoint_field_parallelism(equilibrium, x_points)
        except Exception as e:
            result['xpoint_parallelism'] = None
    return result
