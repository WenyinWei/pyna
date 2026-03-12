"""Field-line tracer with parallel support.

Provides the existing :func:`bundle_tracing_with_t_as_DeltaPhi` interface
(backward compatible) plus a new :class:`FieldLineTracer` class with
RK4 integration and optional parallelism via
:class:`concurrent.futures.ProcessPoolExecutor` or
:class:`concurrent.futures.ThreadPoolExecutor` (Python 3.13 free-threading).

Functions
---------
get_backend(mode)
    Return a backend object suitable for field-line tracing.
"""
from __future__ import annotations

import sysconfig
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Optional

from pyna.field import RegualrCylindricalGridField

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import OdeSolution
def _mat_interp(self, t):
    return self.__call__(t).reshape( (self.pts_num, 3), order='F')
OdeSolution.mat_interp = _mat_interp
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Legacy API (backward compatible)
# ---------------------------------------------------------------------------

def bundle_tracing_with_t_as_DeltaPhi(afield:RegualrCylindricalGridField, total_deltaPhi, initpts_RZPhi, phi_increasing:bool, *arg, **kwarg):
    R, Z, Phi, BR, BZ, BPhi = afield.R, afield.Z, afield.Phi, afield.BR, afield.BZ, afield.BPhi
    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_interp = RegularGridInterpolator( 
        (R, Z, Phi), RBRdBPhi[...,:,:,:],
        method="linear", bounds_error=True )
    RBZdBPhi_interp = RegularGridInterpolator( 
        (R, Z, Phi), RBZdBPhi[...,:,:,:],
        method="linear", bounds_error=True )
    
    pts_num = initpts_RZPhi.shape[0] 
    initps_RZPhi_flattened = np.reshape(initpts_RZPhi, (-1), order='F')

    dPhidPhi = np.ones((pts_num))
    if phi_increasing:
        def dXRXZdPhi(t, y):
            R_ = y[:pts_num]
            Z_ = y[pts_num:2*pts_num]
            Phi_ = y[2*pts_num:3*pts_num] % (2*np.pi)
            pts_RZPhi = np.stack( (R_, Z_, Phi_) , axis=1)
            dXRdPhi = RBRdBPhi_interp(pts_RZPhi) 
            dXZdPhi = RBZdBPhi_interp(pts_RZPhi) 
            return np.concatenate((dXRdPhi, dXZdPhi, dPhidPhi))
    else:
        def dXRXZdPhi(t, y):
            R_ = y[:pts_num]
            Z_ = y[pts_num:2*pts_num]
            Phi_ = y[2*pts_num:3*pts_num] % (2*np.pi)
            pts_RZPhi = np.stack( (R_, Z_, Phi_) , axis=1)
            dXRdPhi =-RBRdBPhi_interp(pts_RZPhi) 
            dXZdPhi =-RBZdBPhi_interp(pts_RZPhi) 
            return np.concatenate((dXRdPhi, dXZdPhi,-dPhidPhi))
        
    def out_of_grid(t, y):
        R_, Z_ = y[:pts_num], y[pts_num:2*pts_num]
        R_max, R_min = max(R_), min(R_)
        Z_max, Z_min = max(Z_), min(Z_)
        return min( 
            R_min - R[1], R[-2] - R_max, 
            Z_min - Z[1], Z[-2] - Z_max, )
    out_of_grid.terminal = True
    
    fltres = solve_ivp(
        dXRXZdPhi, 
        [0.0, total_deltaPhi], 
        initps_RZPhi_flattened, events=out_of_grid, dense_output=True, *arg, **kwarg)
    
    fltres.sol.pts_num = pts_num
    fltres.phi_increasing = phi_increasing
    return fltres


def save_Poincare_orbits(filename:str, list_of_arrRZPhi):
    np.savez(filename, *list_of_arrRZPhi)


def load_Poincare_orbits(filename:str):
    Poincare_orbits_list = [ ]
    Poincare_orbits_npz = np.load(filename)
    for var in Poincare_orbits_npz.files:
        Poincare_orbits_list.append( Poincare_orbits_npz[var] )
    return Poincare_orbits_list


# ---------------------------------------------------------------------------
# New API
# ---------------------------------------------------------------------------

def _rk4_step(f, y, dt):
    """Single RK4 step."""
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _trace_one(args):
    """Top-level function for subprocess-based parallelism."""
    field_func, start_pt, t_max, dt, RZlimit = args
    tracer = FieldLineTracer(field_func, dt=dt, RZlimit=RZlimit)
    return tracer.trace(start_pt, t_max)


class FieldLineTracer:
    """RK4 field-line integrator.

    Parameters
    ----------
    field_func : callable
        ``field_func(rzphi) → (dR, dZ, dphi)`` — unit tangent vector.
    dt : float
        Integration step size (arc length).
    RZlimit : tuple or None
        Optional ``(R_min, R_max, Z_min, Z_max)`` bounding box.
        Integration stops if the trajectory leaves this domain.
    """

    def __init__(self, field_func, dt: float = 0.04, RZlimit=None) -> None:
        self.field_func = field_func
        self.dt = dt
        self.RZlimit = RZlimit

    def trace(self, start_pt, t_max: float) -> np.ndarray:
        """Trace a single field line.

        Parameters
        ----------
        start_pt : array-like of length 3
            Starting point (R, Z, φ).
        t_max : float
            Maximum arc-length parameter.

        Returns
        -------
        ndarray of shape (N, 3)
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
        """Trace multiple field lines in parallel.

        Uses :class:`ThreadPoolExecutor` when the Python 3.13+ free-threading
        build is detected (GIL disabled), otherwise falls back to
        :class:`ProcessPoolExecutor`.

        Parameters
        ----------
        start_pts : array-like of shape (N, 3)
            Starting points.
        t_max : float
            Maximum arc-length for each field line.
        n_workers : int or None
            Number of parallel workers.  ``None`` → CPU count.

        Returns
        -------
        list of ndarray
        """
        start_pts = np.asarray(start_pts, dtype=float)
        gil_disabled = sysconfig.get_config_var('Py_GIL_DISABLED') == 1

        if gil_disabled:
            Executor = ThreadPoolExecutor
        else:
            # Use threads anyway — ProcessPoolExecutor has pickling issues
            # with closures.  For heavier workloads users can pass
            # pre-serialisable field functions.
            Executor = ThreadPoolExecutor

        with Executor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(self.trace, pt, t_max)
                for pt in start_pts
            ]
            return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def get_backend(mode: str = 'cpu', **kwargs):
    """Return a field-line tracing backend.

    Parameters
    ----------
    mode : str
        ``'cpu'``    — :class:`FieldLineTracer` (RK4, CPU).
        ``'cuda'``   — raises :exc:`NotImplementedError`.
        ``'opencl'`` — raises :exc:`NotImplementedError`.
    **kwargs
        Passed to :class:`FieldLineTracer` constructor.

    Returns
    -------
    FieldLineTracer (for mode='cpu').
    """
    if mode == 'cpu':
        return _CpuBackend(**kwargs)
    elif mode == 'cuda':
        raise NotImplementedError(
            "CUDA backend not yet implemented; install cupy and pyna[cuda]"
        )
    elif mode == 'opencl':
        raise NotImplementedError(
            "OpenCL backend reserved; not yet implemented"
        )
    else:
        raise ValueError(f"Unknown backend mode: {mode!r}")


class _CpuBackend:
    """Thin wrapper returned by ``get_backend('cpu')``."""

    def __init__(self, dt: float = 0.04, RZlimit=None) -> None:
        self._dt = dt
        self._RZlimit = RZlimit

    def trace_many(self, field_func, start_pts, t_max: float, dt=None, RZlimit=None):
        dt = dt or self._dt
        RZlimit = RZlimit or self._RZlimit
        tracer = FieldLineTracer(field_func, dt=dt, RZlimit=RZlimit)
        return tracer.trace_many(start_pts, t_max)
