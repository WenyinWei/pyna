"""First wall geometry for plasma-wall gap computation.

Defines wall contours (R-Z polygons) and gap monitoring points.
Provides methods to:
  - compute gap between LCFS and wall
  - find closest wall point for a given plasma boundary point
  - compute inward wall normal at monitoring points

Gap monitoring is IAEA-standard: define N discrete gap locations g_i
around the first wall, typically at positions where heat load matters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WallGeometry:
    """First wall contour in R-Z poloidal cross-section.

    Attributes
    ----------
    R_wall : ndarray, shape (N_wall,)
        R coordinates of wall polygon vertices (closed, last == first).
    Z_wall : ndarray, shape (N_wall,)
        Z coordinates of wall polygon vertices.
    gap_monitor_names : list of str
        Names of gap monitoring locations (e.g., 'inner_mid', 'outer_mid',
        'divertor_inner', 'divertor_outer').
    gap_monitor_R : ndarray
        R coordinates of gap monitoring points on the wall.
    gap_monitor_Z : ndarray
        Z coordinates of gap monitoring points on the wall.
    """
    R_wall: np.ndarray
    Z_wall: np.ndarray
    gap_monitor_names: List[str] = field(default_factory=list)
    gap_monitor_R: Optional[np.ndarray] = None
    gap_monitor_Z: Optional[np.ndarray] = None

    def _wall_segment_normal(self, seg_idx: int, inward_ref_R: float = None,
                              inward_ref_Z: float = None):
        """Return unit normal to wall segment seg_idx.

        Inward = pointing toward (inward_ref_R, inward_ref_Z).
        If ref not supplied, uses centroid of wall polygon.
        """
        R = self.R_wall
        Z = self.Z_wall
        N = len(R)

        # Segment vector
        i0 = seg_idx % N
        i1 = (seg_idx + 1) % N
        dR = R[i1] - R[i0]
        dZ = Z[i1] - Z[i0]

        # Two candidate normals (perpendicular to segment)
        n1 = np.array([-dZ,  dR])
        n2 = np.array([ dZ, -dR])

        # Normalise
        n1 /= np.linalg.norm(n1) + 1e-30
        n2 /= np.linalg.norm(n2) + 1e-30

        # Reference point (centroid of polygon or provided point)
        if inward_ref_R is None:
            inward_ref_R = np.mean(R)
        if inward_ref_Z is None:
            inward_ref_Z = np.mean(Z)

        # Midpoint of segment
        midR = 0.5 * (R[i0] + R[i1])
        midZ = 0.5 * (Z[i0] + Z[i1])

        # The inward normal should point from midpoint toward reference
        ref_vec = np.array([inward_ref_R - midR, inward_ref_Z - midZ])
        if np.dot(n1, ref_vec) >= 0:
            return n1
        return n2

    def inward_normal_at(self, R_pt: float, Z_pt: float) -> np.ndarray:
        """Compute inward wall normal at the wall segment nearest to (R_pt, Z_pt).

        The "inward" direction points from the wall toward the plasma interior
        (i.e., toward the polygon centroid).

        Parameters
        ----------
        R_pt, Z_pt : float
            A point (typically a wall monitoring point).

        Returns
        -------
        n_hat : ndarray, shape (2,)
            Unit inward normal vector [n_R, n_Z].
        """
        R = self.R_wall
        Z = self.Z_wall
        N = len(R)

        # Find closest segment by minimum distance from point to segment
        min_dist = np.inf
        best_seg = 0
        for i in range(N):
            i1 = (i + 1) % N
            # Project onto segment
            seg_R = R[i1] - R[i]
            seg_Z = Z[i1] - Z[i]
            seg_len2 = seg_R**2 + seg_Z**2
            if seg_len2 < 1e-20:
                continue
            t = ((R_pt - R[i]) * seg_R + (Z_pt - Z[i]) * seg_Z) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            proj_R = R[i] + t * seg_R
            proj_Z = Z[i] + t * seg_Z
            dist = np.sqrt((R_pt - proj_R)**2 + (Z_pt - proj_Z)**2)
            if dist < min_dist:
                min_dist = dist
                best_seg = i

        return self._wall_segment_normal(best_seg)

    def gap_to_LCFS(self, LCFS_R: np.ndarray, LCFS_Z: np.ndarray,
                    monitor_idx: int) -> float:
        """Compute gap from wall monitoring point to nearest LCFS point.

        Parameters
        ----------
        LCFS_R, LCFS_Z : ndarray
            Arrays of (R, Z) points on the LCFS.
        monitor_idx : int
            Index into gap_monitor_R / gap_monitor_Z.

        Returns
        -------
        gap : float
            Euclidean distance from monitoring point to nearest LCFS point (m).
        """
        R_mon = self.gap_monitor_R[monitor_idx]
        Z_mon = self.gap_monitor_Z[monitor_idx]
        dists = np.sqrt((LCFS_R - R_mon)**2 + (LCFS_Z - Z_mon)**2)
        return float(np.min(dists))

    def all_gaps(self, LCFS_R: np.ndarray, LCFS_Z: np.ndarray) -> dict:
        """Compute all g_i gaps dict {name: gap_m}.

        Parameters
        ----------
        LCFS_R, LCFS_Z : ndarray
            Arrays of (R, Z) points tracing the LCFS (stable manifold).

        Returns
        -------
        gaps : dict
            {monitor_name: gap_distance_in_meters}
        """
        if self.gap_monitor_R is None:
            return {}
        gaps = {}
        for i, name in enumerate(self.gap_monitor_names):
            gaps[name] = self.gap_to_LCFS(LCFS_R, LCFS_Z, i)
        return gaps


def make_east_like_wall(R0: float = 1.85, a: float = 0.45,
                        kappa: float = 1.6, delta: float = 0.4,
                        n_pts: int = 64) -> WallGeometry:
    """Create a simplified EAST-like D-shaped first wall.

    Not actual EAST data — analytic approximation for testing.
    Wall is offset outward from LCFS by ~8-12% all around.

    Parameters
    ----------
    R0 : float
        Major radius (m).
    a : float
        Minor radius (m).
    kappa : float
        Elongation.
    delta : float
        Triangularity.
    n_pts : int
        Number of polygon vertices.

    Returns
    -------
    wall : WallGeometry
    """
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    scale = 1.12  # 12% larger than plasma minor radius

    R_wall = R0 + scale * a * np.cos(theta + delta * np.sin(theta))
    Z_wall = scale * a * kappa * np.sin(theta)

    # Standard gap monitors: inner mid, outer mid, top, bottom,
    # inner divertor, outer divertor
    monitors = {
        'inner_mid': (R0 - scale * a - 0.01, 0.0),
        'outer_mid': (R0 + scale * a + 0.01, 0.0),
        'top':       (R0, scale * a * kappa + 0.01),
        'bottom':    (R0, -scale * a * kappa - 0.01),
        'div_inner': (R0 - 0.3 * a, -scale * a * kappa + 0.05),
        'div_outer': (R0 + 0.3 * a, -scale * a * kappa + 0.05),
    }
    names = list(monitors.keys())
    mon_R = np.array([v[0] for v in monitors.values()])
    mon_Z = np.array([v[1] for v in monitors.values()])

    return WallGeometry(R_wall, Z_wall, names, mon_R, mon_Z)
