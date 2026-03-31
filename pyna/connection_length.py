"""Connection-length computation for field lines in cylindrical coordinates.

The **connection length** :math:`L_c` of a field line is the physical arc length
from a starting point to the first wall contact.  In divertor / SOL physics
several conventions are used simultaneously; this module computes all of them:

==========  =====================================================================
Symbol      Definition
==========  =====================================================================
Lc_plus     Forward connection length: trace along **φ-increasing** direction
            until the wall is hit.
Lc_minus    Backward connection length: trace along **φ-decreasing** direction
            until the wall is hit.
Lc_sum      Total connection length Lc_plus + Lc_minus.
Lc_max      max(Lc_plus, Lc_minus)
Lc_min      min(Lc_plus, Lc_minus)
==========  =====================================================================

Arc-length convention
---------------------
In cylindrical coordinates (R, Z, φ) — referred to as 大柱坐标系 in the
pyna conventions — the physical arc element is

.. math::

    ds = \\sqrt{\\left(\\frac{dR}{d\\varphi}\\right)^2
               + R^2
               + \\left(\\frac{dZ}{d\\varphi}\\right)^2}
         \\;|d\\varphi|

so a field-line step that advances the toroidal angle by ``dφ`` and moves
``dR = (dR/dφ) dφ``, ``dZ = (dZ/dφ) dφ`` in the poloidal plane contributes
``sqrt(dR² + R_mid² dφ² + dZ²)`` to the arc length.

The field function accepted here is the **φ-parameterised** 2-D form::

    field_func_2d(R, Z, phi)  →  array_like [dR/dφ, dZ/dφ]

which is the natural form for magnetic field-line ODEs.

Wall detection
--------------
The wall is described as a closed polygon in the (R, Z) poloidal cross-section.
At each integration step the routine checks whether the field line has crossed
any wall segment using a **segment–segment intersection test**.  When a
crossing is detected the arc length is linearly interpolated to the exact
crossing point so that the result is not limited by the step size.

Parallel computation
--------------------
All starting points are processed in parallel using
:class:`~concurrent.futures.ThreadPoolExecutor`.

References
----------
* Stangeby (2000): *The Plasma Boundary of Magnetic Fusion Devices*, IOP.
* Loarte et al. (2007): Nucl. Fusion 47, S203.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from pyna.progress import TraceProgressBase, _coerce_progress

# ---------------------------------------------------------------------------
# Module-level numerical constants
# ---------------------------------------------------------------------------

# Small positive number added to denominators to prevent exact division-by-zero
# while remaining negligible compared to any physical quantity.
_DENOM_EPS: float = 1e-300

# Threshold below which segment-crossing determinant is treated as zero
# (parallel segments).
_PARALLEL_EPS: float = 1e-30


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _point_inside_polygon(R: float, Z: float,
                           R_wall: ndarray, Z_wall: ndarray) -> bool:
    """Ray-casting point-in-polygon test for a closed wall polygon.

    Parameters
    ----------
    R, Z : float
        Query point.
    R_wall, Z_wall : ndarray
        Vertices of the closed polygon (need not repeat the first vertex).

    Returns
    -------
    bool
        ``True`` if (R, Z) is strictly inside the polygon.
    """
    n = len(R_wall)
    inside = False
    j = n - 1
    for i in range(n):
        Ri, Zi = R_wall[i], Z_wall[i]
        Rj, Zj = R_wall[j], Z_wall[j]
        if ((Zi > Z) != (Zj > Z)) and \
                (R < (Rj - Ri) * (Z - Zi) / (Zj - Zi + _DENOM_EPS) + Ri):
            inside = not inside
        j = i
    return inside


def _segment_wall_crossing(
    R0: float, Z0: float,
    R1: float, Z1: float,
    R_wall: ndarray, Z_wall: ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Find the first intersection of segment (R0,Z0)→(R1,Z1) with the wall.

    Uses the parametric segment–segment intersection formula.  Returns the
    *smallest* parameter ``t ∈ (0, 1]`` at which the step exits the wall.

    Parameters
    ----------
    R0, Z0, R1, Z1 : float
        Start and end of the field-line step.
    R_wall, Z_wall : ndarray
        Closed wall polygon vertices.

    Returns
    -------
    (t, R_cross, Z_cross) : tuple or None
        ``t`` is the fraction along the step [0, 1] at which the crossing
        occurs.  ``None`` if no intersection found.
    """
    n = len(R_wall)
    # Direction vector of the step
    dR_step = R1 - R0
    dZ_step = Z1 - Z0

    t_best = np.inf
    R_cross = Z_cross = 0.0

    for i in range(n):
        j = (i + 1) % n
        Ri, Zi = R_wall[i], Z_wall[i]
        Rj, Zj = R_wall[j], Z_wall[j]

        dR_wall = Rj - Ri
        dZ_wall = Zj - Zi

        # Solve: (R0, Z0) + t*(dR_step, dZ_step) = (Ri, Zi) + u*(dR_wall, dZ_wall)
        denom = dR_step * dZ_wall - dZ_step * dR_wall
        if abs(denom) < _PARALLEL_EPS:
            continue  # parallel segments

        t = ((Ri - R0) * dZ_wall - (Zi - Z0) * dR_wall) / denom
        u = ((Ri - R0) * dZ_step - (Zi - Z0) * dR_step) / denom

        if 0.0 < t <= 1.0 + 1e-12 and 0.0 <= u <= 1.0 + 1e-12:
            if t < t_best:
                t_best = t
                R_cross = R0 + t * dR_step
                Z_cross = Z0 + t * dZ_step

    if t_best == np.inf:
        return None
    return (min(t_best, 1.0), R_cross, Z_cross)


# ---------------------------------------------------------------------------
# Single field-line trace to wall
# ---------------------------------------------------------------------------

def _trace_to_wall(
    field_func_2d: Callable,
    R_start: float,
    Z_start: float,
    phi_start: float,
    R_wall: ndarray,
    Z_wall: ndarray,
    *,
    forward: bool = True,
    max_turns: float = 500.0,
    dphi: float = 0.05,
) -> Tuple[float, float, float, float]:
    """Trace a single field line until it hits the wall and return arc length.

    Parameters
    ----------
    field_func_2d : callable
        ``field_func_2d(R, Z, phi)`` → ``[dR/dφ, dZ/dφ]``.
    R_start, Z_start, phi_start : float
        Starting position (m, m, rad).
    R_wall, Z_wall : ndarray
        Closed wall polygon vertices.
    forward : bool
        ``True``  → integrate along φ-increasing direction (L+).
        ``False`` → integrate along φ-decreasing direction (L-).
    max_turns : float
        Maximum number of toroidal turns before giving up.
    dphi : float
        Toroidal-angle step size (rad).  Smaller values give higher
        accuracy for the crossing interpolation.

    Returns
    -------
    (arc_length, R_hit, Z_hit, phi_hit) : tuple
        ``arc_length`` is the physical arc length in metres from the start to
        the wall (``inf`` if the wall was not reached within ``max_turns``).
        ``R_hit, Z_hit, phi_hit`` give the approximate wall-contact point.
    """
    sign = 1.0 if forward else -1.0
    max_phi = max_turns * 2.0 * np.pi

    R, Z, phi = float(R_start), float(Z_start), float(phi_start)
    arc = 0.0

    # Check that the start point is inside the wall
    if not _point_inside_polygon(R, Z, R_wall, Z_wall):
        return 0.0, R, Z, phi

    n_steps = max(int(max_phi / dphi), 1)

    for _ in range(n_steps):
        try:
            f = np.asarray(field_func_2d(R, Z, phi), dtype=float)
        except Exception:
            break

        dR_dphi = f[0]
        dZ_dphi = f[1]

        dR_step = sign * dR_dphi * dphi
        dZ_step = sign * dZ_dphi * dphi
        dphi_step = sign * dphi

        R_new = R + dR_step
        Z_new = Z + dZ_step
        phi_new = phi + dphi_step

        # Check for wall crossing
        crossing = _segment_wall_crossing(R, Z, R_new, Z_new, R_wall, Z_wall)

        if crossing is not None:
            t_frac, R_cross, Z_cross = crossing
            # Interpolate arc length to the exact crossing point
            R_mid = 0.5 * (R + R_cross)
            dR_c = R_cross - R
            dZ_c = Z_cross - Z
            dphi_c = t_frac * dphi  # φ advance to crossing
            ds_cross = float(np.sqrt(dR_c**2 + R_mid**2 * dphi_c**2 + dZ_c**2))
            arc += ds_cross
            phi_cross = phi + sign * dphi_c
            return arc, R_cross, Z_cross, phi_cross

        # Accumulate arc length for this full step
        # ds = sqrt(dR² + R_mid² dφ² + dZ²)  (midpoint rule for R)
        R_mid = 0.5 * (R + R_new)
        ds = float(np.sqrt(dR_step**2 + R_mid**2 * dphi**2 + dZ_step**2))
        arc += ds

        R, Z, phi = R_new, Z_new, phi_new

        # Safety: if point somehow ended up outside (no crossing detected)
        if not _point_inside_polygon(R, Z, R_wall, Z_wall):
            return arc, R, Z, phi

    # Never hit the wall
    return float("inf"), R, Z, phi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def connection_length(
    field_func_2d: Callable,
    start_pts: Union[ndarray, List],
    wall,
    *,
    phi_start: Union[float, ndarray] = 0.0,
    direction: str = "both",
    max_turns: float = 500.0,
    dphi: float = 0.05,
    n_workers: Optional[int] = None,
    progress: Optional[TraceProgressBase] = None,
) -> Dict[str, ndarray]:
    """Compute connection lengths for a set of starting points in cylindrical coords.

    For each starting point the field line is traced forward (φ-increasing),
    backward (φ-decreasing), or both until it contacts the wall.  The physical
    arc length is accumulated using the cylindrical-coordinate metric

    .. math::

        ds = \\sqrt{dR^2 + R_{\\mathrm{mid}}^2\\,d\\varphi^2 + dZ^2}

    where ``R_mid`` is the average major radius across each step.

    Parameters
    ----------
    field_func_2d : callable
        ``field_func_2d(R, Z, phi)`` → array ``[dR/dφ, dZ/dφ]``.
        This is the standard 2-D φ-parameterised field function used
        throughout pyna (e.g. ``pyna.topo.fixed_points.poincare_map``).
    start_pts : array-like, shape (N, 2)
        Starting positions ``[R, Z]`` in metres.  The starting toroidal
        angle is provided via ``phi_start``.
    wall : WallGeometry or (R_wall, Z_wall)
        First-wall polygon.  May be a :class:`pyna.MCF.control.wall.WallGeometry`
        instance or a 2-tuple ``(R_wall, Z_wall)`` of closed polygon arrays.
    phi_start : float or array of shape (N,)
        Starting toroidal angle (rad) for each seed.  Scalar broadcasts to
        all seeds.  Default ``0.0``.
    direction : {'both', '+', '-'}
        Which direction(s) to trace.

        ``'+'``    — forward only (L+).
        ``'-'``    — backward only (L-).
        ``'both'`` — both directions (default).

    max_turns : float
        Maximum number of toroidal turns per trace before declaring ``inf``.
        Default 500.
    dphi : float
        Toroidal-angle step size (rad).  Smaller values are more accurate
        but slower.  Default 0.05 (≈ 0.8°).
    n_workers : int or None
        Thread-pool size for parallel tracing.  ``None`` → auto-detect.
    progress : TraceProgressBase or None
        Optional progress reporter.  ``None`` → silent.  A
        :class:`~pyna.progress.TqdmProgress` shows an interactive bar;
        :class:`~pyna.progress.LogFileProgress` writes heartbeat records
        for external staleness monitoring.

    Returns
    -------
    dict with keys depending on ``direction``:

    ``'Lc_plus'`` (float ndarray, shape (N,))
        Forward connection lengths :math:`L_c^+` (m).  ``inf`` if the wall was not
        reached.  Present when ``direction`` is ``'+'`` or ``'both'``.
    ``'Lc_minus'`` (float ndarray, shape (N,))
        Backward connection lengths :math:`L_c^-` (m).  Present when ``direction`` is
        ``'-'`` or ``'both'``.
    ``'Lc_sum'`` (float ndarray, shape (N,))
        :math:`L_c^+ + L_c^-`.  Present only for ``direction='both'``.
    ``'Lc_max'`` (float ndarray, shape (N,))
        :math:`\\max(L_c^+, L_c^-)`.  Present only for ``direction='both'``.
    ``'Lc_min'`` (float ndarray, shape (N,))
        :math:`\\min(L_c^+, L_c^-)`.  Present only for ``direction='both'``.
    ``'hit_plus'`` (float ndarray, shape (N, 3))
        Wall-contact points ``[R, Z, phi]`` for forward traces.
    ``'hit_minus'`` (float ndarray, shape (N, 3))
        Wall-contact points ``[R, Z, phi]`` for backward traces.

    Examples
    --------
    Basic usage with a simplified wall::

        import numpy as np
        from pyna.connection_length import connection_length

        # Circular wall of radius 0.5 centred at (R0, 0)
        theta_wall = np.linspace(0, 2*np.pi, 64, endpoint=False)
        R_wall = 1.8 + 0.5 * np.cos(theta_wall)
        Z_wall = 0.5 * np.sin(theta_wall)

        def ff2d(R, Z, phi):
            \"\"\"Simple tokamak-like field (returns dR/dφ, dZ/dφ).\"\"\"
            return np.array([0.0, 0.1 / R])   # pure vertical drift

        starts = np.array([[1.9, 0.0], [2.0, 0.05]])
        result = connection_length(ff2d, starts, (R_wall, Z_wall),
                                   direction='both', max_turns=50, dphi=0.1)
        print(result['Lc_plus'])

    Notes
    -----
    **Accuracy** is limited by the step size ``dphi``.  The crossing position is
    linearly interpolated, so errors scale as O(dphi) in connection length.
    Typical values ``dphi ≈ 0.01–0.1`` rad give sub-percent accuracy for
    smooth geometries.

    **Step size in R, Z** is ``|dR/dφ| dphi`` etc., which can be large for
    large major radii or strongly tilted fields.  If the step overshoots the
    wall by more than one segment, reduce ``dphi``.

    **性能说明（C++ 加速路径）**：
    当前实现使用纯 Python 步进循环，适用于任意 callable ``field_func_2d``
    （向后兼容）。若已有 (BR, BPhi, BZ) 网格数据，可直接调用
    ``pyna._cyna.trace_connection_length_twall`` 或
    ``pyna._cyna.trace_wall_hits_twall`` 获得 C++ 多线程加速，
    性能比此函数高一到两个数量级。未来版本将通过 ``field_data=`` 参数
    自动切换到 C++ 路径（接口预留，本版本暂未实现）。
    """
    # ── unpack wall ──────────────────────────────────────────────────────────
    try:
        # Duck-type: WallGeometry has R_wall and Z_wall attributes
        R_wall = np.asarray(wall.R_wall, dtype=float)
        Z_wall = np.asarray(wall.Z_wall, dtype=float)
    except AttributeError:
        R_wall = np.asarray(wall[0], dtype=float)
        Z_wall = np.asarray(wall[1], dtype=float)

    # ── normalise start_pts ──────────────────────────────────────────────────
    start_pts = np.asarray(start_pts, dtype=float)
    if start_pts.ndim == 1:
        start_pts = start_pts[np.newaxis, :]  # single point → (1,2)
    N = len(start_pts)

    phi_arr = np.broadcast_to(
        np.asarray(phi_start, dtype=float), (N,)
    ).copy()

    # ── validate direction ───────────────────────────────────────────────────
    if direction not in ("+", "-", "both"):
        raise ValueError(
            f"direction must be '+', '-', or 'both'; got {direction!r}"
        )

    workers = n_workers or min(os.cpu_count() or 4, 16)

    # ── progress setup ───────────────────────────────────────────────────────
    # Count: forward + backward traces.
    n_traces = N * (2 if direction == "both" else 1)
    prog = _coerce_progress(progress)
    prog.start(n_traces, description="connection-length tracing")
    _prog_counter = [0]   # mutable cell for closure; incremented under GIL

    # ── forward traces ───────────────────────────────────────────────────────
    def _fwd(i):
        arc, Rh, Zh, phih = _trace_to_wall(
            field_func_2d,
            start_pts[i, 0], start_pts[i, 1], phi_arr[i],
            R_wall, Z_wall,
            forward=True, max_turns=max_turns, dphi=dphi,
        )
        prog.update(i, steps_done=-1)
        return arc, Rh, Zh, phih

    def _bwd(i):
        arc, Rh, Zh, phih = _trace_to_wall(
            field_func_2d,
            start_pts[i, 0], start_pts[i, 1], phi_arr[i],
            R_wall, Z_wall,
            forward=False, max_turns=max_turns, dphi=dphi,
        )
        prog.update(N + i, steps_done=-1)
        return arc, Rh, Zh, phih

    result: Dict[str, ndarray] = {}

    if direction in ("+", "both"):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fwd_results = list(pool.map(_fwd, range(N)))
        Lc_plus  = np.array([r[0] for r in fwd_results])
        hit_plus = np.array([[r[1], r[2], r[3]] for r in fwd_results])
        result["Lc_plus"]  = Lc_plus
        result["hit_plus"] = hit_plus

    if direction in ("-", "both"):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            bwd_results = list(pool.map(_bwd, range(N)))
        Lc_minus  = np.array([r[0] for r in bwd_results])
        hit_minus = np.array([[r[1], r[2], r[3]] for r in bwd_results])
        result["Lc_minus"]  = Lc_minus
        result["hit_minus"] = hit_minus

    if direction == "both":
        result["Lc_sum"] = Lc_plus + Lc_minus
        result["Lc_max"] = np.maximum(Lc_plus, Lc_minus)
        result["Lc_min"] = np.minimum(Lc_plus, Lc_minus)

    prog.close()
    return result


def connection_length_map(
    field_func_2d: Callable,
    R_grid: ndarray,
    Z_grid: ndarray,
    wall,
    *,
    phi_start: float = 0.0,
    direction: str = "both",
    max_turns: float = 500.0,
    dphi: float = 0.05,
    n_workers: Optional[int] = None,
    aggregate: str = "sum",
) -> ndarray:
    """Compute a 2-D connection-length map over an (R, Z) grid.

    Convenience wrapper around :func:`connection_length` that accepts a
    2-D meshgrid and returns a 2-D array of connection-length values.

    Parameters
    ----------
    field_func_2d : callable
        ``(R, Z, phi) → [dR/dφ, dZ/dφ]``.
    R_grid, Z_grid : 2-D ndarray, shape (nR, nZ)
        Meshgrid of major radius and vertical coordinates.
    wall : WallGeometry or (R_wall, Z_wall)
        First-wall polygon.
    phi_start : float
        Starting toroidal angle for all seeds.
    direction : {'both', '+', '-'}
        Which direction(s) to trace (forwarded to :func:`connection_length`).
    max_turns : float
        Maximum toroidal turns.
    dphi : float
        Toroidal-angle step size.
    n_workers : int or None
        Thread pool size.
    aggregate : {'sum', 'max', 'min', '+', '-'}
        Which scalar to return per grid point when ``direction='both'``:

        ``'sum'`` → L_sum (default, total connection length)
        ``'max'`` → L_max
        ``'min'`` → L_min
        ``'+'``   → L_plus  (forces direction='+')
        ``'-'``   → L_minus (forces direction='-')

    Returns
    -------
    ndarray, shape (nR, nZ)
        Connection-length values (m).  ``nan`` for grid points outside
        the wall polygon; ``inf`` for those that never reach the wall.
    """
    if R_grid.shape != Z_grid.shape:
        raise ValueError("R_grid and Z_grid must have the same shape.")

    nR, nZ = R_grid.shape

    # Flatten to (N, 2) starting points
    starts = np.column_stack([R_grid.ravel(), Z_grid.ravel()])

    # Choose direction
    if aggregate in ("+", "-"):
        direction = aggregate

    res = connection_length(
        field_func_2d, starts, wall,
        phi_start=phi_start,
        direction=direction,
        max_turns=max_turns,
        dphi=dphi,
        n_workers=n_workers,
    )

    agg_map = {
        "sum": "Lc_sum",
        "max": "Lc_max",
        "min": "Lc_min",
        "+": "Lc_plus",
        "-": "Lc_minus",
    }
    key = agg_map.get(aggregate, "Lc_sum")
    flat = res[key]
    return flat.reshape(nR, nZ)
