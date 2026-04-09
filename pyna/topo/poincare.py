"""Multi-section Poincaré map infrastructure.

Provides an extensible framework for collecting field-line crossings on
arbitrary 2-D surfaces in (R, Z, φ) space.

Rename history
--------------
* ``PoincareMap`` → ``PoincareAccumulator`` (conflicts with
  ``pyna.topo.dynamics.PoincareMap``)

Backward-compatible aliases at module level keep old import paths working::

    from pyna.topo.poincare import PoincareMap        # still works
    from pyna.topo.poincare import ToroidalSection    # still works

Classes
-------
PoincareSection         — abstract base for crossing surfaces (poincare-accumulator use)
PoincareToroidalSection — φ = φ₀ plane, implements detect_crossing
PoincareAccumulator     — accumulate crossings on multiple sections

Functions
---------
poincare_from_fieldlines — trace field lines and collect crossings
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Section protocol for PoincareAccumulator
# ---------------------------------------------------------------------------

class PoincareSection(ABC):
    """Abstract crossing surface for use with PoincareAccumulator.

    Concrete subclasses define any 2-D surface in 3-D (R, Z, φ) space.
    The :meth:`detect_crossing` method uses linear interpolation between
    consecutive trajectory points to find where the trajectory crosses
    this surface.
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


class PoincareToroidalSection(PoincareSection):
    """Toroidal section at φ = φ₀ for use with PoincareAccumulator.

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

        dphi = phi_curr - phi_prev
        if dphi > np.pi:
            dphi -= 2 * np.pi
        elif dphi <= -np.pi:
            dphi += 2 * np.pi

        if abs(dphi) < 1e-12:
            return None

        if dphi > 0:
            d_prev_to_0 = (phi0 - phi_prev) % (2 * np.pi)
        else:
            d_prev_to_0 = -(phi_prev - phi0) % (2 * np.pi)

        if abs(dphi) > 0 and 0 < abs(d_prev_to_0) <= abs(dphi):
            t = d_prev_to_0 / dphi
            if 0.0 <= t <= 1.0:
                return pt_prev + t * (pt_curr - pt_prev)
        return None


# ---------------------------------------------------------------------------
# Poincaré accumulator
# ---------------------------------------------------------------------------

class PoincareAccumulator:
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
# Backward-compatible aliases
# ---------------------------------------------------------------------------

#: Alias for ``PoincareAccumulator`` kept for backward compatibility.
PoincareMap = PoincareAccumulator

#: Backward-compatible alias: ``ToroidalSection`` from poincare module → PoincareToroidalSection
#: (has detect_crossing for use with PoincareAccumulator)
#: For section-based topology queries, use ``pyna.topo.section.ToroidalSection``.
ToroidalSection = PoincareToroidalSection

#: Backward-compatible alias: ``Section`` from poincare module → PoincareSection ABC
#: For the canonical Section ABC, use ``pyna.topo.section.Section``.
Section = PoincareSection


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
) -> PoincareAccumulator:
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
    PoincareAccumulator
    """
    if backend is None:
        from pyna.flt import FieldLineTracer
        backend = FieldLineTracer(field_func, dt=dt)

    pmap = PoincareAccumulator(sections)

    if hasattr(backend, 'trace_many'):
        trajs = backend.trace_many(start_pts, t_max)
    else:
        trajs = [backend.trace(pt, t_max) for pt in start_pts]

    for traj in trajs:
        if len(traj) > 1:
            pmap.record_trajectory(traj)

    return pmap


# ---------------------------------------------------------------------------
# Rotational transform estimation
# ---------------------------------------------------------------------------

def rotational_transform_from_trajectory(
    traj: np.ndarray,
    axis_RZ: Optional[np.ndarray] = None,
    n_turns: Optional[int] = None,
) -> float:
    r"""Estimate the rotational transform ι (or safety factor q = 1/ι) from
    a traced field-line trajectory.

    The rotational transform is the average poloidal angle advance per
    toroidal turn.  For a closed (periodic) orbit on a flux surface with
    safety factor q = m/n, we have ι = 1/q = n/m.

    Algorithm
    ---------
    1. Compute the cumulative *poloidal* angle Θ(s) of the trajectory
       around the magnetic axis ``axis_RZ``.
    2. Compute the cumulative *toroidal* angle Φ(s) from the φ column of
       the trajectory.
    3. Fit a linear relationship Θ ≈ ι · Φ by least squares.  The slope
       ι is the rotational transform.

    Parameters
    ----------
    traj : ndarray, shape (N, 3)
        Trajectory (R, Z, φ) as returned by
        :meth:`pyna.flt.FieldLineTracer.trace`.
    axis_RZ : array_like of shape (2,) or None
        Approximate position [R, Z] of the magnetic axis (or any reference
        point inside the flux surface).  If ``None``, the centroid of the
        trajectory projection onto the (R, Z) plane is used.
    n_turns : int or None
        If provided, restrict the computation to the first ``n_turns``
        toroidal traversals.  If ``None``, use the full trajectory.

    Returns
    -------
    iota : float
        Estimated rotational transform ι = dΘ/dΦ.
        The safety factor is q = 1 / ι.

    Notes
    -----
    For a regular (KAM) flux surface, the returned value converges as
    the trajectory length increases.  For a chaotic trajectory, the
    "effective" rotational transform still gives a useful indicator of the
    local winding rate, though it will not converge to a rational number.

    Examples
    --------
    >>> iota = rotational_transform_from_trajectory(traj, axis_RZ=[1.0, 0.0])
    >>> q = 1.0 / iota   # safety factor
    """
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("traj must be shape (N, ≥2) with columns [R, Z, ...]")

    R = traj[:, 0]
    Z = traj[:, 1]

    # Determine axis reference
    if axis_RZ is None:
        axis_R = float(np.mean(R))
        axis_Z = float(np.mean(Z))
    else:
        axis_R, axis_Z = float(axis_RZ[0]), float(axis_RZ[1])

    # Poloidal angle relative to axis
    poloidal_angle = np.arctan2(Z - axis_Z, R - axis_R)
    # Unwrap to get a monotonic cumulative angle
    Theta = np.unwrap(poloidal_angle)

    # Toroidal angle (phi column if available, else use index as proxy)
    if traj.shape[1] >= 3:
        Phi = np.unwrap(traj[:, 2])
    else:
        # No phi column: use step index as a proxy for toroidal arc length
        Phi = np.arange(len(traj), dtype=float)

    if n_turns is not None:
        # Estimate how many points correspond to n_turns toroidal traversals
        Phi_per_turn = 2.0 * np.pi
        max_Phi = n_turns * Phi_per_turn + Phi[0]
        mask = Phi <= max_Phi
        if np.sum(mask) < 2:
            mask = np.ones(len(Phi), dtype=bool)
        Theta = Theta[mask]
        Phi = Phi[mask]

    # Linear least-squares fit: Theta = iota * Phi + const
    if len(Phi) < 2 or (np.ptp(Phi) < 1e-15):
        return float("nan")

    A = np.column_stack([Phi, np.ones(len(Phi))])
    result, _, _, _ = np.linalg.lstsq(A, Theta, rcond=None)
    iota = float(result[0])
    return iota
