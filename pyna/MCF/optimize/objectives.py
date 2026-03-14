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
    # TODO: implement epsilon_h averaging over flux surfaces
    #
    # Pseudocode:
    #   surfaces = equilibrium.flux_surfaces(n=n_field_lines)
    #   eps_h_list = []
    #   for surf in surfaces:
    #       B_along = trace_field_line_B(equilibrium, surf, n_transits)
    #       B_max, B_min = B_along.max(), B_along.min()
    #       eps_h_list.append((B_max - B_min) / (B_max + B_min))
    #   return float(np.mean(eps_h_list))
    raise NotImplementedError("neoclassical_epsilon_eff: TODO")


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
        endpoints = trace_field_lines(seeds, n_transits)
        angles = pairwise_angle(endpoints - seeds)
        return mean(cos(angles))
    """
    # TODO: trace field lines near X-points, compute angular spread
    #
    # from pyna.flt import FieldLineTracer
    # tracer = FieldLineTracer.from_equilibrium(equilibrium)
    # metrics = []
    # for (Rx, Zx) in x_points:
    #     seeds = _seed_ring(Rx, Zx, n_fieldlines, radius=1e-3)
    #     tangents = tracer.tangent_vectors(seeds, n_transits)
    #     cos_angles = [
    #         np.dot(tangents[i], tangents[(i+1) % n_fieldlines])
    #         for i in range(n_fieldlines)
    #     ]
    #     metrics.append(np.mean(cos_angles))
    # return float(np.mean(metrics)) if metrics else 0.0
    raise NotImplementedError("xpoint_field_parallelism: TODO")


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
    wall_R = np.asarray(wall_R, dtype=float)
    wall_Z = np.asarray(wall_Z, dtype=float)
    if wall_R.shape != wall_Z.shape or wall_R.ndim != 1:
        raise ValueError("wall_R and wall_Z must be 1-D arrays of equal length.")

    # TODO: compute LCFS contour, find min distance to wall polygon
    #
    # Option A – shapely (clean):
    #   from shapely.geometry import LineString, Polygon
    #   lcfs = equilibrium.lcfs()
    #   lcfs_line = LineString(np.column_stack(lcfs))
    #   wall_poly = Polygon(np.column_stack([wall_R, wall_Z]))
    #   return lcfs_line.distance(wall_poly.exterior)
    #
    # Option B – manual (no extra deps):
    #   lcfs_R, lcfs_Z = equilibrium.lcfs()
    #   dists = _min_dist_to_polygon(lcfs_R, lcfs_Z, wall_R, wall_Z)
    #   return float(dists.min())
    raise NotImplementedError("wall_clearance: TODO")


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
    # TODO: call each objective, handle None gracefully
    #
    # objectives: Dict[str, float] = {}
    #
    # # Always computed
    # objectives["eps_eff"] = neoclassical_epsilon_eff(equilibrium)
    # R_ax, Z_ax = magnetic_axis_position(equilibrium)
    # objectives["R_axis"] = R_ax
    # objectives["Z_axis"] = Z_ax
    #
    # # Optional: wall clearance
    # if wall_R is not None and wall_Z is not None:
    #     objectives["wall_clearance"] = wall_clearance(equilibrium, wall_R, wall_Z)
    #
    # # Optional: X-point parallelism
    # if x_points is not None:
    #     objectives["xpoint_parallelism"] = xpoint_field_parallelism(
    #         equilibrium, x_points
    #     )
    #
    # return objectives
    raise NotImplementedError("compute_all_objectives: TODO")
