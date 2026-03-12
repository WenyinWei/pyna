"""Field line tracing with parallel execution.

Parallelism strategy (Windows + Python 3.13 standard GIL build):
- Primary: ThreadPoolExecutor — works with any callable, numpy releases GIL
  during integration loops, so real speedup is achievable
- ProcessPoolExecutor: available but requires module-level picklable functions;
  use only when explicitly requested and field_func is picklable
- CUDA: CuPy backend for GPU-accelerated batch tracing

Legacy API
----------
bundle_tracing_with_t_as_DeltaPhi(...)
    Original interface — fully preserved.
save_Poincare_orbits / load_Poincare_orbits
    File I/O helpers — fully preserved.

New API
-------
FieldLineTracer
    RK4 integrator with .trace() / .trace_many() (ThreadPoolExecutor).
get_backend(mode, **kwargs)
    Factory: 'cpu', 'cuda', 'opencl'.
"""
from __future__ import annotations

import os
import sysconfig
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List, Optional

import numpy as np

from pyna.field import RegualrCylindricalGridField
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import OdeSolution, solve_ivp


# ---------------------------------------------------------------------------
# Monkey-patch OdeSolution for legacy callers
# ---------------------------------------------------------------------------

def _mat_interp(self, t):
    return self.__call__(t).reshape((self.pts_num, 3), order='F')

OdeSolution.mat_interp = _mat_interp


# ---------------------------------------------------------------------------
# Legacy API (fully backward-compatible)
# ---------------------------------------------------------------------------

def bundle_tracing_with_t_as_DeltaPhi(
    afield: RegualrCylindricalGridField,
    total_deltaPhi,
    initpts_RZPhi,
    phi_increasing: bool,
    *arg,
    **kwarg,
):
    R, Z, Phi = afield.R, afield.Z, afield.Phi
    BR, BZ, BPhi = afield.BR, afield.BZ, afield.BPhi

    RBRdBPhi = R[:, None, None] * BR / BPhi
    RBZdBPhi = R[:, None, None] * BZ / BPhi

    RBRdBPhi_interp = RegularGridInterpolator(
        (R, Z, Phi), RBRdBPhi, method="linear", bounds_error=True
    )
    RBZdBPhi_interp = RegularGridInterpolator(
        (R, Z, Phi), RBZdBPhi, method="linear", bounds_error=True
    )

    pts_num = initpts_RZPhi.shape[0]
    initps_RZPhi_flattened = np.reshape(initpts_RZPhi, (-1), order='F')
    dPhidPhi = np.ones((pts_num,))

    if phi_increasing:
        def dXRXZdPhi(t, y):
            R_ = y[:pts_num]
            Z_ = y[pts_num:2 * pts_num]
            Phi_ = y[2 * pts_num:3 * pts_num] % (2 * np.pi)
            pts = np.stack((R_, Z_, Phi_), axis=1)
            return np.concatenate((RBRdBPhi_interp(pts), RBZdBPhi_interp(pts), dPhidPhi))
    else:
        def dXRXZdPhi(t, y):
            R_ = y[:pts_num]
            Z_ = y[pts_num:2 * pts_num]
            Phi_ = y[2 * pts_num:3 * pts_num] % (2 * np.pi)
            pts = np.stack((R_, Z_, Phi_), axis=1)
            return np.concatenate((-RBRdBPhi_interp(pts), -RBZdBPhi_interp(pts), -dPhidPhi))

    def out_of_grid(t, y):
        R_ = y[:pts_num]
        Z_ = y[pts_num:2 * pts_num]
        return min(
            min(R_) - R[1], R[-2] - max(R_),
            min(Z_) - Z[1], Z[-2] - max(Z_),
        )

    out_of_grid.terminal = True

    fltres = solve_ivp(
        dXRXZdPhi,
        [0.0, total_deltaPhi],
        initps_RZPhi_flattened,
        events=out_of_grid,
        dense_output=True,
        *arg,
        **kwarg,
    )
    fltres.sol.pts_num = pts_num
    fltres.phi_increasing = phi_increasing
    return fltres


def save_Poincare_orbits(filename: str, list_of_arrRZPhi):
    np.savez(filename, *list_of_arrRZPhi)


def load_Poincare_orbits(filename: str):
    orbits = []
    data = np.load(filename)
    for var in data.files:
        orbits.append(data[var])
    return orbits


# ---------------------------------------------------------------------------
# New API — RK4 integrator
# ---------------------------------------------------------------------------

def _rk4_step(f: Callable, y: np.ndarray, dt: float) -> np.ndarray:
    """Single fixed-step 4th-order Runge-Kutta step.

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
            y = _rk4_step(self.field_func, y, dt)
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

        Returns
        -------
        list of ndarray
            One (n_steps, 3) trajectory per starting point.
        """
        start_pts = np.asarray(start_pts, dtype=float)
        workers = n_workers or self.n_workers

        def _trace_one(pt):
            return self.trace(pt, t_max)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_trace_one, start_pts))
        return results

    # ------------------------------------------------------------------
    # Legacy shim
    # ------------------------------------------------------------------

    def bundle_tracing_with_t_as_DeltaPhi(self, start_pts, t_max, **kwargs):
        """Deprecated shim — use :meth:`trace_many` instead."""
        import warnings
        warnings.warn(
            "bundle_tracing_with_t_as_DeltaPhi is deprecated; use trace_many",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.trace_many(start_pts, t_max)


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
        ``'cuda'``   — :class:`~pyna.flt_cuda.CUDAFieldLineTracer`
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
        from pyna.flt_cuda import CUDAFieldLineTracer  # noqa: PLC0415
        return CUDAFieldLineTracer(**kwargs)
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
