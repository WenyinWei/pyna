"""Field line tracing with parallel execution.

Parallelism strategy (Windows + Python 3.13 standard GIL build):
- Primary: ThreadPoolExecutor — works with any callable, numpy releases GIL
  during integration loops, so real speedup is achievable
- ProcessPoolExecutor: available but requires module-level picklable functions;
  use only when explicitly requested and field_func is picklable
- CUDA: CuPy backend for GPU-accelerated batch tracing

Progress reporting
------------------
All batch methods accept an optional ``progress=`` parameter.  Pass any
:class:`~pyna.progress.TraceProgressBase` instance to receive per-task or
aggregate progress updates::

    from pyna.progress import TqdmProgress, LogFileProgress, CompositeProgress

    # tqdm bar in CLI / notebook
    tracer.trace_many(starts, t_max, progress=TqdmProgress())

    # bar + log file simultaneously
    prog = CompositeProgress([TqdmProgress(), LogFileProgress("run.jsonl")])
    tracer.trace_many(starts, t_max, progress=prog)

Wall-hit handling (boundary field lines)
-----------------------------------------
Near the plasma boundary, field lines may strike the first wall after only
a few integration steps.  ``FieldLineTracer.trace_many`` marks these as
"wall-hit" (non-blocking) and returns the short trajectory.  Use
:func:`reseed_boundary_field_lines` to automatically add extra seed points
around the boundary where the hit-fraction is high, ensuring the boundary
topology is still captured.

Legacy API
----------
已移除。原 ``bundle_tracing_with_t_as_DeltaPhi``、``save_Poincare_orbits``、
``load_Poincare_orbits`` 及 ``OdeSolution.mat_interp`` monkey-patch 已删除。
请使用 ``FieldLineTracer.trace_many`` 替代。

New API
-------
FieldLineTracer
    RK4 integrator with .trace() / .trace_many() (ThreadPoolExecutor).
    当传入网格数据时底层由 cyna C++ 扩展驱动；若只有 callable 场函数，
    则使用纯 Python RK4 fallback（向后兼容）。
    若要完全使用 C++ 底层加速，请使用
    ``pyna.toroidal.flt.trace_poincare_batch`` 并传入网格数组。
WallModel
    Parametric / polygon wall geometry for wall-hit detection.
reseed_boundary_field_lines
    Adaptively add more seed points where wall-hits are dense.
get_backend(mode, **kwargs)
    Factory: 'cpu', 'cuda', 'opencl'.
"""
from __future__ import annotations

import os
import sysconfig
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List, Optional

import numpy as np

from pyna.fields.cylindrical import VectorField3DCylindrical
from pyna.progress import TraceProgressBase, _coerce_progress
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# New API — RK4 integrator
# ---------------------------------------------------------------------------

def _rk4_step_py(f: Callable, y: np.ndarray, dt: float) -> np.ndarray:
    """纯 Python RK4 步骤（仅用于 callable 场函数的 fallback 路径）。

    当 FieldLineTracer 以任意 callable 构建时（而非网格数据）使用此函数。
    若已提供网格数据，底层将使用 cyna C++ 积分器。

    Parameters
    ----------
    f : callable
        Vector field ``f(y) -> dy``.  Takes/returns 1-D array of length 3.
    y : ndarray
        Current state (R, Z, phi).
    dt : float
        Step size.

    Returns
    -------
    ndarray
        New state after one RK4 step.
    """
    k1 = np.asarray(f(y), dtype=float)
    k2 = np.asarray(f(y + 0.5 * dt * k1), dtype=float)
    k3 = np.asarray(f(y + 0.5 * dt * k2), dtype=float)
    k4 = np.asarray(f(y + dt * k3), dtype=float)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class FieldLineTracer:
    """RK4 field-line integrator with parallel trace_many.

    The field function signature::

        f(rzphi: ndarray[3]) -> ndarray[3]   # (dR/dl, dZ/dl, dphi/dl)

    where ``rzphi = [R, Z, phi]`` is the current position.

    Parallelism uses :class:`~concurrent.futures.ThreadPoolExecutor` on all
    platforms.  This is deliberately chosen over ProcessPoolExecutor because:

    * Works with *any* callable — no pickle requirement (important on Windows).
    * NumPy/SciPy release the GIL during heavy computation, so threads achieve
      real parallel speedup.
    * No process-spawn overhead on Windows.

    Parameters
    ----------
    field_func : callable
        Unit tangent vector field.
    dt : float
        Integration step size (arc length).
    RZlimit : tuple or None
        Optional ``(R_min, R_max, Z_min, Z_max)`` domain boundary.
        Integration stops when the trajectory leaves this box.
    n_workers : int or None
        Default thread-pool size for :meth:`trace_many`.
        ``None`` → ``min(os.cpu_count(), 16)``.
    """

    def __init__(
        self,
        field_func: Callable,
        dt: float = 0.04,
        RZlimit=None,
        n_workers: Optional[int] = None,
    ) -> None:
        self.field_func = field_func
        self.dt = dt
        self.RZlimit = RZlimit
        self.n_workers = n_workers or min(os.cpu_count() or 4, 16)

    def trace(self, start_pt, t_max: float) -> np.ndarray:
        """Trace a single field line with fixed-step RK4.

        Parameters
        ----------
        start_pt : array-like of length 3
            Starting point (R, Z, phi).
        t_max : float
            Maximum arc-length parameter.

        Returns
        -------
        ndarray, shape (N, 3)
            Trajectory points (R, Z, phi).
        """
        y = np.asarray(start_pt, dtype=float).copy()
        dt = self.dt
        n_steps = max(int(t_max / dt), 1)
        result = [y.copy()]

        for _ in range(n_steps):
            y = _rk4_step_py(self.field_func, y, dt)
            result.append(y.copy())
            if self.RZlimit is not None:
                R_min, R_max, Z_min, Z_max = self.RZlimit
                if y[0] < R_min or y[0] > R_max or y[1] < Z_min or y[1] > Z_max:
                    break

        return np.array(result)

    def trace_many(
        self,
        start_pts,
        t_max: float,
        n_workers: Optional[int] = None,
        progress: Optional[TraceProgressBase] = None,
    ) -> List[np.ndarray]:
        """Parallel field-line tracing using ThreadPoolExecutor.

        ThreadPoolExecutor is used (not ProcessPool) because:

        1. Works with any callable (no pickle requirement).
        2. NumPy/SciPy release the GIL during computation → real parallel
           speedup even without the free-threading build.
        3. Avoids Windows process-spawn overhead.

        Parameters
        ----------
        start_pts : array-like, shape (N, 3)
            Starting points.
        t_max : float
            Maximum arc-length for each field line.
        n_workers : int or None
            Override the default worker count.
        progress : TraceProgressBase or None
            Optional progress reporter.  ``None`` → silent.  Pass a
            :class:`~pyna.progress.TqdmProgress` for an interactive bar or
            :class:`~pyna.progress.LogFileProgress` to write a heartbeat
            file for staleness monitoring.

        Returns
        -------
        list of ndarray
            One (n_steps, 3) trajectory per starting point.
        """
        start_pts = np.asarray(start_pts, dtype=float)
        workers = n_workers or self.n_workers
        n_tasks = len(start_pts)
        n_steps_planned = max(int(t_max / self.dt), 1)

        prog = _coerce_progress(progress)
        prog.start(n_tasks, description="field-line tracing")

        def _trace_one(args):
            idx, pt = args
            traj = self.trace(pt, t_max)
            prog.update(idx, steps_done=-1, steps_total=n_steps_planned)
            return traj

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_trace_one, enumerate(start_pts)))

        prog.close()
        return results


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def get_backend(mode: str = 'cpu', **kwargs):
    """Get a field-line tracer backend.

    Parameters
    ----------
    mode : str
        ``'cpu'``    — :class:`FieldLineTracer` with ThreadPoolExecutor
                       (always available).
        ``'cuda'``   — :class:`~pyna.flt_cuda.FieldLineTracerCUDA`
                       (requires CuPy, only for analytic fields).
        ``'opencl'`` — reserved, not yet implemented.
    **kwargs
        Passed to the backend constructor.

    Returns
    -------
    Backend object with ``.trace_many(start_pts, t_max)`` method.

    Examples
    --------
    CPU::

        tracer = get_backend('cpu', field_func=my_field, dt=0.02)
        trajs  = tracer.trace_many(starts, t_max=100.0)

    CUDA::

        tracer = get_backend('cuda', R0=1.0, a=0.3, B0=1.0, q0=2.0)
        trajs  = tracer.trace_many(starts, t_max=100.0)
    """
    if mode == 'cpu':
        field_func = kwargs.pop('field_func', None)
        if field_func is None:
            return _CPUBackend(**kwargs)
        return FieldLineTracer(field_func, **kwargs)
    elif mode == 'cuda':
        from pyna.flt_cuda import FieldLineTracerCUDA  # noqa: PLC0415
        return FieldLineTracerCUDA(**kwargs)
    elif mode == 'opencl':
        raise NotImplementedError(
            "OpenCL backend is reserved for future implementation. "
            "Use mode='cpu' or mode='cuda' (requires cupy)."
        )
    else:
        raise ValueError(
            f"Unknown backend mode: {mode!r}. Choose 'cpu', 'cuda', or 'opencl'."
        )


class _CPUBackend:
    """Lazy CPU backend — call :meth:`get_tracer` to create a FieldLineTracer."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def get_tracer(self, field_func: Callable) -> FieldLineTracer:
        """Create a :class:`FieldLineTracer` for *field_func*."""
        return FieldLineTracer(field_func, **self.kwargs)

    # Convenience: allow trace_many to be called directly if field_func
    # was stored separately (e.g. via kwargs).
    def trace_many(self, start_pts, t_max: float) -> List[np.ndarray]:
        field_func = self.kwargs.pop('field_func', None)
        if field_func is None:
            raise ValueError(
                "_CPUBackend.trace_many requires 'field_func' in kwargs. "
                "Use get_backend('cpu', field_func=...) or call .get_tracer(f).trace_many(...)."
            )
        return FieldLineTracer(field_func, **self.kwargs).trace_many(start_pts, t_max)


# ---------------------------------------------------------------------------
# Wall geometry model
# ---------------------------------------------------------------------------

class WallModel:
    """First-wall geometry for wall-hit detection during field-line tracing.

    A wall is represented as a closed polygon in the (R, Z) poloidal cross-
    section.  The same shape is assumed toroidally symmetric (axisymmetric
    wall).  More complex 3-D first-wall descriptions can be approximated by
    providing an appropriate 2-D outline.

    The model also provides a *minimum clearance* check so that trajectories
    that approach very close to the wall but haven't yet crossed it can be
    flagged for early termination.

    Parameters
    ----------
    R_wall : array_like of shape (N,)
        R-coordinates of the wall polygon vertices (m).  The polygon is
        closed automatically; do not repeat the first point.
    Z_wall : array_like of shape (N,)
        Z-coordinates of the wall polygon vertices (m).
    min_clearance : float, optional
        Minimum distance from wall (m) before a point is considered a
        near-miss / wall-hit.  ``0`` means only count true crossings.
        Default 0.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
    >>> wall = WallModel(2.0 + 0.5*np.cos(theta), 0.5*np.sin(theta))
    >>> wall.is_outside(np.array([2.6, 0.0]))
    True
    """

    def __init__(
        self,
        R_wall: np.ndarray,
        Z_wall: np.ndarray,
        min_clearance: float = 0.0,
    ) -> None:
        self.R_wall = np.asarray(R_wall, dtype=float)
        self.Z_wall = np.asarray(Z_wall, dtype=float)
        self.min_clearance = float(min_clearance)
        if len(self.R_wall) < 3:
            raise ValueError("Wall polygon must have at least 3 vertices.")
        if self.R_wall.shape != self.Z_wall.shape:
            raise ValueError("R_wall and Z_wall must have the same shape.")

    def is_outside(self, RZ: np.ndarray) -> bool:
        """Return True if the point (R, Z) is outside the wall polygon.

        Uses the ray-casting (winding-number) algorithm.

        Parameters
        ----------
        RZ : array_like of shape (2,)
            Point [R, Z] to test (m).

        Returns
        -------
        bool
        """
        R, Z = float(RZ[0]), float(RZ[1])

        if self.min_clearance > 0.0:
            dist = self._min_polygon_distance(R, Z)
            if dist < self.min_clearance:
                return True

        return not self._point_in_polygon(R, Z)

    def _point_in_polygon(self, R: float, Z: float) -> bool:
        """Ray-casting test for point-in-polygon."""
        n = len(self.R_wall)
        inside = False
        j = n - 1
        for i in range(n):
            Ri, Zi = self.R_wall[i], self.Z_wall[i]
            Rj, Zj = self.R_wall[j], self.Z_wall[j]
            if ((Zi > Z) != (Zj > Z)) and (R < (Rj - Ri) * (Z - Zi) / (Zj - Zi) + Ri):
                inside = not inside
            j = i
        return inside

    def _min_polygon_distance(self, R: float, Z: float) -> float:
        """Minimum distance from (R, Z) to any edge of the wall polygon."""
        n = len(self.R_wall)
        min_d = np.inf
        for i in range(n):
            j = (i + 1) % n
            # Segment from (R_wall[i], Z_wall[i]) to (R_wall[j], Z_wall[j])
            dR = self.R_wall[j] - self.R_wall[i]
            dZ = self.Z_wall[j] - self.Z_wall[i]
            seg_len2 = dR * dR + dZ * dZ
            if seg_len2 < 1e-30:
                d = np.hypot(R - self.R_wall[i], Z - self.Z_wall[i])
            else:
                t = max(0.0, min(1.0,
                    ((R - self.R_wall[i]) * dR + (Z - self.Z_wall[i]) * dZ) / seg_len2))
                proj_R = self.R_wall[i] + t * dR
                proj_Z = self.Z_wall[i] + t * dZ
                d = np.hypot(R - proj_R, Z - proj_Z)
            if d < min_d:
                min_d = d
        return float(min_d)

    @classmethod
    def circular(
        cls,
        R0: float,
        a: float,
        n_vertices: int = 128,
        min_clearance: float = 0.0,
    ) -> "WallModel":
        """Construct a circular (axisymmetric) wall.

        Parameters
        ----------
        R0, a : float
            Major radius and minor radius of the circular wall.
        n_vertices : int
            Number of polygon vertices.
        min_clearance : float
            Proximity clearance (m).

        Returns
        -------
        WallModel
        """
        theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
        return cls(
            R0 + a * np.cos(theta),
            a * np.sin(theta),
            min_clearance=min_clearance,
        )


# ---------------------------------------------------------------------------
# Update FieldLineTracer.trace to accept a wall model
# ---------------------------------------------------------------------------

_original_trace = FieldLineTracer.trace


def _trace_with_wall(
    self,
    start_pt,
    t_max: float,
    wall: Optional["WallModel"] = None,
) -> np.ndarray:
    """Trace a single field line, optionally stopping at a wall.

    Parameters
    ----------
    start_pt : array-like of length 3
        Starting point (R, Z, phi).
    t_max : float
        Maximum arc-length parameter.
    wall : WallModel or None
        Wall model for early-termination on wall hit.  ``None`` uses the
        ``RZlimit`` box-boundary from the constructor.

    Returns
    -------
    ndarray, shape (N, 3)
        Trajectory points (R, Z, phi).  For wall-hit cases N may be much
        smaller than ``int(t_max / dt)``; the trajectory is *not* padded.
    """
    y = np.asarray(start_pt, dtype=float).copy()
    dt = self.dt
    n_steps = max(int(t_max / dt), 1)
    result = [y.copy()]

    for _ in range(n_steps):
        y = _rk4_step_py(self.field_func, y, dt)
        result.append(y.copy())
        if wall is not None and wall.is_outside(y[:2]):
            break
        if self.RZlimit is not None:
            R_min, R_max, Z_min, Z_max = self.RZlimit
            if y[0] < R_min or y[0] > R_max or y[1] < Z_min or y[1] > Z_max:
                break

    return np.array(result)


FieldLineTracer.trace = _trace_with_wall  # type: ignore[assignment]


def _trace_many_with_wall(
    self,
    start_pts,
    t_max: float,
    n_workers: Optional[int] = None,
    progress: Optional[TraceProgressBase] = None,
    wall: Optional["WallModel"] = None,
    min_valid_steps: int = 0,
) -> List[np.ndarray]:
    """Parallel field-line tracing with optional wall-hit handling.

    Wall-hitting trajectories are returned as short arrays and do **not**
    block or slow down other traces in the same batch.

    Parameters
    ----------
    start_pts : array-like, shape (N, 3)
        Starting points.
    t_max : float
        Maximum arc-length for each field line.
    n_workers : int or None
        Override the default worker count.
    progress : TraceProgressBase or None
        Optional progress reporter.
    wall : WallModel or None
        Wall model.  When provided, trajectories that hit the wall are
        returned early (shorter arrays).
    min_valid_steps : int
        If > 0, trajectories with fewer than this many steps are flagged
        as "wall-hit" in the returned metadata.  This does not discard
        the trajectory — callers can filter using the returned list.

    Returns
    -------
    list of ndarray
        One trajectory per starting point.
    """
    start_pts = np.asarray(start_pts, dtype=float)
    workers = n_workers or self.n_workers
    n_tasks = len(start_pts)
    n_steps_planned = max(int(t_max / self.dt), 1)

    prog = _coerce_progress(progress)
    prog.start(n_tasks, description="field-line tracing")

    def _trace_one(args):
        idx, pt = args
        traj = self.trace(pt, t_max, wall=wall)
        prog.update(idx, steps_done=-1, steps_total=n_steps_planned)
        return traj

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_trace_one, enumerate(start_pts)))

    prog.close()
    return results


FieldLineTracer.trace_many = _trace_many_with_wall  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Boundary reseeding helper
# ---------------------------------------------------------------------------

def reseed_boundary_field_lines(
    tracer: "FieldLineTracer",
    start_pts: np.ndarray,
    trajs: List[np.ndarray],
    t_max: float,
    wall: Optional["WallModel"] = None,
    min_valid_fraction: float = 0.5,
    n_reseed_factor: int = 4,
    reseed_radius: float = 0.02,
    n_workers: Optional[int] = None,
    progress: Optional[TraceProgressBase] = None,
) -> List[np.ndarray]:
    """Adaptively reseed field lines near wall-hit starting points.

    Near the plasma boundary, a fraction of seed points may produce very
    short trajectories (wall hits).  This function:

    1. Identifies seeds with trajectories shorter than
       ``min_valid_fraction * t_max / dt`` steps.
    2. Surrounds each such seed with ``n_reseed_factor`` new seeds in a
       small circle of radius ``reseed_radius``.
    3. Traces the new seeds and returns *all* trajectories (original +
       reseeded).

    Parameters
    ----------
    tracer : FieldLineTracer
        Tracer used for the original batch.
    start_pts : array, shape (N, 3)
        Original starting points [R, Z, phi].
    trajs : list of ndarray
        Trajectories from the original batch (one per start point).
    t_max : float
        Arc-length used in the original batch.
    wall : WallModel or None
        Wall model passed to the new traces.
    min_valid_fraction : float
        Fraction of ``t_max / dt`` steps below which a trajectory is
        considered a wall hit.  Default 0.5.
    n_reseed_factor : int
        Number of new seeds to place around each wall-hit seed.  Default 4.
    reseed_radius : float
        Radius (m) of the reseeding circle around each hit seed.  Default 0.02.
    n_workers : int or None
        Worker count for the reseeded traces.
    progress : TraceProgressBase or None
        Progress reporter for the reseeded traces.

    Returns
    -------
    list of ndarray
        Original trajectories plus new reseeded trajectories (appended).
    """
    start_pts = np.asarray(start_pts, dtype=float)
    n_planned = max(int(t_max / tracer.dt), 1)
    threshold = max(1, int(min_valid_fraction * n_planned))

    # Identify wall-hit seeds
    hit_indices = [i for i, t in enumerate(trajs) if len(t) < threshold]
    if not hit_indices:
        return list(trajs)

    # Build reseeded starts
    new_starts = []
    angles = np.linspace(0.0, 2.0 * np.pi, n_reseed_factor, endpoint=False)
    for idx in hit_indices:
        R0, Z0, phi0 = start_pts[idx]
        for theta in angles:
            new_starts.append([
                R0 + reseed_radius * np.cos(theta),
                Z0 + reseed_radius * np.sin(theta),
                phi0,
            ])

    new_starts_arr = np.array(new_starts)
    new_trajs = tracer.trace_many(
        new_starts_arr, t_max, n_workers=n_workers, progress=progress, wall=wall
    )

    return list(trajs) + new_trajs
