"""pyna.toroidal.optimize.objectives — toroidal / stellarator optimisation objectives.

Toroidal ownership for scalar objective functions used in
multi-objective stellarator optimisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Individual objectives
# ---------------------------------------------------------------------------


def neoclassical_epsilon_eff(
    equilibrium,
    n_field_lines: int = 50,
    n_transits: int = 100,
) -> float:
    """Estimate effective ripple ε_eff (proxy for neoclassical transport)."""
    if hasattr(equilibrium, "epsilon_h"):
        return 0.64 * abs(equilibrium.epsilon_h) ** 1.5

    psi_vals = np.linspace(0.1, 0.9, n_field_lines)
    eps_arr = []
    for psi_n in psi_vals:
        r = np.sqrt(psi_n) * getattr(equilibrium, "r0", 0.3)
        theta_arr = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        R_arr = getattr(equilibrium, "R0", 1.0) + r * np.cos(theta_arr)
        Z_arr = r * np.sin(theta_arr)
        B_vals = []
        for phi in np.linspace(0, 2 * np.pi / getattr(equilibrium, "n_h", 1), n_transits // 10):
            for R, Z in zip(R_arr, Z_arr):
                B_vals.append(equilibrium.B0 * equilibrium.R0 / R)
        B_arr = np.array(B_vals)
        Bmax, Bmin = B_arr.max(), B_arr.min()
        delta_b = (Bmax - Bmin) / (Bmax + Bmin + 1e-30)
        eps_arr.append(0.64 * delta_b**1.5)
    return float(np.mean(eps_arr))



def xpoint_field_parallelism(
    equilibrium,
    x_points: List[Tuple[float, float]],
    n_fieldlines: int = 20,
    n_transits: int = 30,
) -> float:
    """Measure field-line parallelism near X-points (for power exhaust)."""
    if not x_points:
        return 0.0

    R0 = getattr(equilibrium, "R0", 1.65)
    a = getattr(equilibrium, "r0", getattr(equilibrium, "a", 0.5))

    def _field_tangent(R: float, Z: float, phi: float) -> np.ndarray:
        if hasattr(equilibrium, "field_func"):
            f = equilibrium.field_func([R, Z, phi])
            dRdphi, dZdphi = float(f[0]), float(f[1])
        else:
            _B_pol_scale = 0.1
            dRdphi = -Z / (R + 1e-30) * _B_pol_scale
            dZdphi = (R - R0) / (R + 1e-30) * _B_pol_scale
        norm = np.sqrt(dRdphi**2 + dZdphi**2) + 1e-30
        return np.array([dRdphi / norm, dZdphi / norm])

    _ = n_transits
    seed_radius = min(a * 0.02, 5e-3)
    metrics = []

    for (Rx, Zx) in x_points:
        seed_angles = np.linspace(0, 2 * np.pi, n_fieldlines, endpoint=False)
        R_seeds = Rx + seed_radius * np.cos(seed_angles)
        Z_seeds = Zx + seed_radius * np.sin(seed_angles)

        tangents = np.array([
            _field_tangent(R_seeds[i], Z_seeds[i], 0.0)
            for i in range(n_fieldlines)
        ])

        cos_angles = [
            float(np.dot(tangents[i], tangents[(i + 1) % n_fieldlines]))
            for i in range(n_fieldlines)
        ]
        metrics.append(float(np.mean(cos_angles)))

    return float(np.mean(metrics)) if metrics else 0.0



def magnetic_axis_position(equilibrium) -> Tuple[float, float]:
    """Return ``(R_axis, Z_axis)`` of the magnetic axis."""
    return equilibrium.magnetic_axis



def wall_clearance(
    equilibrium,
    wall_R: np.ndarray,
    wall_Z: np.ndarray,
) -> float:
    """Minimum distance from the LCFS to the first wall in the ``(R, Z)`` plane."""
    theta = np.linspace(0, 2 * np.pi, 500)
    R_lcfs = equilibrium.R0 + equilibrium.r0 * np.cos(theta)
    Z_lcfs = equilibrium.r0 * np.sin(theta)

    wall_pts = np.column_stack([wall_R, wall_Z])
    lcfs_pts = np.column_stack([R_lcfs, Z_lcfs])

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
    """Compute all available physics objectives and return them as a dict."""
    result = {}
    result["magnetic_axis"] = magnetic_axis_position(equilibrium)
    try:
        result["epsilon_eff"] = neoclassical_epsilon_eff(equilibrium)
    except Exception as e:
        result["epsilon_eff"] = None
        result["epsilon_eff_error"] = str(e)
    if wall_R is not None and wall_Z is not None:
        try:
            result["wall_clearance"] = wall_clearance(equilibrium, wall_R, wall_Z)
        except Exception:
            result["wall_clearance"] = None
    if x_points is not None:
        try:
            result["xpoint_parallelism"] = xpoint_field_parallelism(equilibrium, x_points)
        except Exception:
            result["xpoint_parallelism"] = None
    return result


__all__ = [
    "neoclassical_epsilon_eff",
    "xpoint_field_parallelism",
    "magnetic_axis_position",
    "wall_clearance",
    "compute_all_objectives",
]
