"""Multi-section Poincaré map infrastructure.

Provides an extensible framework for collecting field-line crossings on
arbitrary 2-D surfaces in (R, Z, φ) space.

Classes
-------
Section          — abstract base for any crossing surface
ToroidalSection  — φ = φ₀ plane (most common case)
PoincareMap      — accumulate crossings on multiple sections

Functions
---------
poincare_from_fieldlines — trace field lines and collect crossings
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract section
# ---------------------------------------------------------------------------

class Section(ABC):
    """Abstract crossing surface for Poincaré maps.

    Concrete subclasses define any 2-D surface in 3-D (R, Z, φ) space.
    The :meth:`detect_crossing` method uses linear interpolation between
    consecutive trajectory points to find where the trajectory crosses
    this surface.

    Design note: sections are intentionally not limited to toroidal
    (φ=const) planes.  Subclasses can represent poloidal planes,
    oblique planes, or arbitrary smooth surfaces.
    """

    @abstractmethod
    def detect_crossing(
        self,
        pt_prev: np.ndarray,
        pt_curr: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return the interpolated crossing point (R, Z, φ), or None.

        Parameters
        ----------
        pt_prev, pt_curr : ndarray of shape (3,)
            Consecutive trajectory points ``(R, Z, φ)``.

        Returns
        -------
        ndarray of shape (3,) or None
        """

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable label for this section."""


# ---------------------------------------------------------------------------
# Toroidal section  (φ = φ₀)
# ---------------------------------------------------------------------------

class ToroidalSection(Section):
    """Toroidal section at φ = φ₀ (crossing surface perpendicular to φ).

    Detects when the toroidal angle φ crosses φ₀ (modulo 2π), using
    linear interpolation to find the precise crossing point.

    Parameters
    ----------
    phi0 : float
        The toroidal angle of the section (radians).  Default 0.0.
    """

    def __init__(self, phi0: float = 0.0) -> None:
        self.phi0 = float(phi0) % (2 * np.pi)

    @property
    def label(self) -> str:
        return f"ToroidalSection(φ={np.degrees(self.phi0):.1f}°)"

    def detect_crossing(
        self,
        pt_prev: np.ndarray,
        pt_curr: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Detect when φ crosses φ₀ (in the direction of increasing φ)."""
        phi_prev = pt_prev[2] % (2 * np.pi)
        phi_curr = pt_curr[2] % (2 * np.pi)
        phi0 = self.phi0

        # Signed phase difference
        dphi = phi_curr - phi_prev
        # Wrap to (-π, π]
        if dphi > np.pi:
            dphi -= 2 * np.pi
        elif dphi <= -np.pi:
            dphi += 2 * np.pi

        if abs(dphi) < 1e-12:
            return None

        # Distance from phi_prev to phi0 in the direction of travel
        if dphi > 0:
            d_prev_to_0 = (phi0 - phi_prev) % (2 * np.pi)
        else:
            d_prev_to_0 = -(phi_prev - phi0) % (2 * np.pi)

        # Check if crossing occurs in this step
        if abs(dphi) > 0 and 0 < abs(d_prev_to_0) <= abs(dphi):
            t = d_prev_to_0 / dphi  # fractional step [0, 1]
            if 0.0 <= t <= 1.0:
                return pt_prev + t * (pt_curr - pt_prev)
        return None


# ---------------------------------------------------------------------------
# Poincaré map
# ---------------------------------------------------------------------------

class PoincareMap:
    """Collect and store trajectory crossings on multiple sections.

    Parameters
    ----------
    sections : list of Section
        The crossing surfaces to monitor.
    """

    def __init__(self, sections: list) -> None:
        self.sections = list(sections)
        self._crossings: list[list[np.ndarray]] = [[] for _ in sections]

    def record_step(self, pt_prev: np.ndarray, pt_curr: np.ndarray) -> None:
        """Check all sections for a crossing between two consecutive points."""
        for i, sec in enumerate(self.sections):
            hit = sec.detect_crossing(pt_prev, pt_curr)
            if hit is not None:
                self._crossings[i].append(hit)

    def record_trajectory(self, traj: np.ndarray) -> None:
        """Record all crossings in a full trajectory array.

        Parameters
        ----------
        traj : ndarray of shape (N, 3)
            Trajectory with columns (R, Z, φ).
        """
        for k in range(len(traj) - 1):
            self.record_step(traj[k], traj[k + 1])

    def crossing_array(self, section_idx: int) -> np.ndarray:
        """Return all crossings for a given section as an (N, 3) array.

        Parameters
        ----------
        section_idx : int
            Index into the ``sections`` list.

        Returns
        -------
        ndarray of shape (N, 3)  — columns (R, Z, φ).
            Returns shape (0, 3) if no crossings.
        """
        pts = self._crossings[section_idx]
        if not pts:
            return np.empty((0, 3))
        return np.array(pts)


# ---------------------------------------------------------------------------
# Convenience driver
# ---------------------------------------------------------------------------

def poincare_from_fieldlines(
    field_func,
    start_pts: np.ndarray,
    sections: list,
    t_max: float,
    dt: float = 0.04,
    backend=None,
) -> PoincareMap:
    """Trace field lines and collect Poincaré crossings.

    Parameters
    ----------
    field_func : callable
        ``field_func(rzphi) → (dR, dZ, dphi)`` unit-tangent ODE rhs.
    start_pts : ndarray of shape (N, 3)
        Starting points (R, Z, φ).
    sections : list of Section
        Sections on which to collect crossings.
    t_max : float
        Maximum arc-length parameter for each field line.
    dt : float
        Step size for the RK4 integrator.
    backend : FieldLineTracer or None
        If None, a :class:`pyna.flt.FieldLineTracer` is created.

    Returns
    -------
    PoincareMap
    """
    if backend is None:
        from pyna.flt import FieldLineTracer
        backend = FieldLineTracer(field_func, dt=dt)

    pmap = PoincareMap(sections)

    if hasattr(backend, 'trace_many'):
        trajs = backend.trace_many(start_pts, t_max)
    else:
        trajs = [backend.trace(pt, t_max) for pt in start_pts]

    for traj in trajs:
        if len(traj) > 1:
            pmap.record_trajectory(traj)

    return pmap
