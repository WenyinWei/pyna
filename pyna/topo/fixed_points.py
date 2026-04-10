"""
fixed_points.py 鈥?Newton-method fixed-point locator for Poincar茅 maps.

For a field-line Poincar茅 map P: (R,Z) 鈫?(R,Z) and its m-th iterate P^m,
Newton's method finds fixed points x* satisfying P^m(x*) = x* by iterating:

    x_{k+1} = x_k 鈭?(DP^m(x_k) 鈭?I)^{-1} 路 (P^m(x_k) 鈭?x_k)

The monodromy matrix DP^m is computed via variational equations integrated
alongside the field-line ODE.

Public API
----------
find_magnetic_axis(tracer, R_guess, Z_guess, ...)
    Newton method for the magnetic axis (m=1, O-point).

find_fixed_point_newton(tracer, R_guess, Z_guess, phi_sec, period, ...)
    Newton method for an m-periodic fixed point (X or O point).

refine_fixed_points_from_pkl(pkl_path, tracer, phi_sections, ...)
    Batch-refine all fixed points stored in a pickle file.

find_island_chain_fixed_points(tracer, R_ax, Z_ax, period, phi_sec, ...)
    Grid-scan + Newton to find all X/O points of an m-period island chain.

propagate_island_chain(tracer, R0, Z0, phi0, period, section_phis, ...)
    P^1 propagation from one known fixed point to map out the full chain.
"""

from __future__ import annotations

import math
import pickle
import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np

from pyna.topo.variational import PoincareMapVariationalEquations

from pyna.topo.fixed_point import FixedPoint
try:
    from pyna._cyna import find_fixed_points_batch as _cyna_find_fixed_points_batch
    _CYNA_AVAILABLE = True
except ImportError:
    _CYNA_AVAILABLE = False

__all__ = [
    "find_magnetic_axis",
    "find_fixed_point_newton",
    "refine_fixed_points_from_pkl",
    "find_island_chain_fixed_points",
    "propagate_island_chain",
    # Legacy API (field_func-based, used by test_fixed_points.py and __init__.py)
    "find_periodic_orbit",
    "classify_fixed_point",
    "classify_fixed_point_higher_order",
    "poincare_map",
    "scan_fixed_point_seeds",
    "refine_fixed_point",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi

# Physical bounds for sanity check
_R_MIN, _R_MAX = 0.5, 1.5
_Z_ABS_MAX = 0.5


def _make_field_func(tracer):
    """Extract a phi-parameterised field-line RHS from a FieldlineTracer.

    Returns a callable  f(r, z, phi) 鈫?(dR/dphi, dZ/dphi).
    """
    def _f(r, z, phi):
        RZ = np.array([[r, z]])
        out = tracer._rhs(phi, RZ, direction=1)  # shape (1, 2)
        return out[0]  # (dR/dphi, dZ/dphi)
    return _f


def _trace_one(tracer, R0: float, Z0: float, phi_sec: float, m: int):
    """Trace m toroidal turns and return the final (R, Z).

    Uses tracer.trace_poincare for a single seed without wall termination.

    Returns (R_end, Z_end) or (None, None) if the field line was lost.
    """
    R_seeds = np.array([R0])
    Z_seeds = np.array([Z0])
    try:
        result = tracer.trace_poincare(
            R_seeds, Z_seeds, phi_sec, N_turns=m,
            use_wall=False, verbose=False,
        )
    except Exception as exc:
        warnings.warn(f"trace_poincare failed: {exc}")
        return None, None

    # result[0] is (R_arr, Z_arr) for seed 0, where R_arr and Z_arr are 1D
    R_arr, Z_arr = result[0]
    if len(R_arr) < m:
        # Field line terminated early (hit wall or left grid)
        return None, None

    # last point is after m turns
    R_end, Z_end = float(R_arr[-1]), float(Z_arr[-1])
    return R_end, Z_end


def _compute_DPm(tracer, R0: float, Z0: float, phi_sec: float, m: int):
    """Compute the m-turn Poincar茅-map Jacobian DP^m at (R0, Z0).

    Strategy:
    1. Try PoincareMapVariationalEquations (always available).
       Integrates from phi_sec to phi_sec + m * 2蟺.

    Returns DP^m as ndarray shape (2, 2), or None on failure.
    """
    field_func = _make_field_func(tracer)
    vq = PoincareMapVariationalEquations(field_func)
    phi_start = phi_sec
    phi_end = phi_sec + m * _TWO_PI
    try:
        DPm = vq.jacobian_matrix([R0, Z0], (phi_start, phi_end))
        return DPm
    except Exception as exc:
        warnings.warn(f"_compute_DPm variational failed: {exc}")
        return None


def _classify(DPm: np.ndarray) -> str:
    """Return 'X' for hyperbolic (|Tr| > 2) or 'O' for elliptic (|Tr| <= 2)."""
    tr = np.trace(DPm)
    return 'X' if abs(tr) > 2.0 else 'O'


def _in_bounds(R: float, Z: float) -> bool:
    return (_R_MIN <= R <= _R_MAX) and (abs(Z) <= _Z_ABS_MAX)


def _extract_field_cache(tracer) -> Optional[dict]:
    """Extract BR/BPhi/BZ grid arrays from a FieldlineTracer.

    Returns a dict with keys BR, BPhi, BZ, R_grid, Z_grid, Phi_grid,
    or None if extraction fails.
    """
    try:
        import numpy as np
        BR   = tracer._grid_values(tracer.itp_BR)
        BPhi = tracer._grid_values(tracer.itp_BPhi)
        BZ   = tracer._grid_values(tracer.itp_BZ)
        R_grid   = np.ascontiguousarray(tracer.R_grid,   dtype=np.float64)
        Z_grid   = np.ascontiguousarray(tracer.Z_grid,   dtype=np.float64)
        Phi_grid = np.ascontiguousarray(tracer.Phi_grid, dtype=np.float64)
        # Reshape flat arrays to (nPhi, nR, nZ) matching cyna convention
        # RegularGridInterpolator values shape: (len(R_grid), len(Z_grid), len(Phi_grid)) or similar
        # cyna expects BR[nPhi, nR, nZ] 鈥?check actual shape
        nR, nZ, nPhi = len(R_grid), len(Z_grid), len(Phi_grid)
        BR   = BR.reshape(nR, nZ, nPhi)
        BPhi = BPhi.reshape(nR, nZ, nPhi)
        BZ   = BZ.reshape(nR, nZ, nPhi)
        return dict(BR=BR, BPhi=BPhi, BZ=BZ,
                    R_grid=R_grid, Z_grid=Z_grid, Phi_grid=Phi_grid)
    except Exception as exc:
        warnings.warn(f"_extract_field_cache failed: {exc}")
        return None


def _cyna_result_to_fp(R_out, Z_out, res, conv, DPm_flat, eig_r, eig_i, ptype, idx: int):
    """Unpack cyna batch output for a single index idx."""
    R = float(R_out[idx])
    Z = float(Z_out[idx])
    converged = bool(conv[idx])
    DPm = DPm_flat[idx].reshape(2, 2)
    pt = int(ptype[idx])
    if not converged or pt == -1:
        return None, None, None, None
    kind = 'X' if pt == 1 else 'O'
    return R, Z, DPm, kind


# ---------------------------------------------------------------------------
# Core Newton iterator
# ---------------------------------------------------------------------------

def _newton_iterate(
    tracer,
    R0: float,
    Z0: float,
    phi_sec: float,
    period: int,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    """Core Newton loop for P^m(x) - x = 0.

    Returns (R*, Z*, DP^m) on convergence, or (None, None, None) on failure.
    """
    R, Z = float(R0), float(Z0)
    prev_res = None

    for k in range(max_iter):
        # --- compute P^m(x) and DP^m ----
        DPm = _compute_DPm(tracer, R, Z, phi_sec, period)
        if DPm is None:
            if verbose:
                print(f"  iter {k}: DPm computation failed, aborting")
            return None, None, None

        R_end, Z_end = _trace_one(tracer, R, Z, phi_sec, period)
        if R_end is None:
            if verbose:
                print(f"  iter {k}: trace lost field line, aborting")
            return None, None, None

        # residual vector F = P^m(x) - x
        F = np.array([R_end - R, Z_end - Z])
        res = np.linalg.norm(F)

        if verbose:
            print(f"  iter {k}: R={R:.8f} Z={Z:.8f}  |F|={res:.3e}  Tr(DP^m)={np.trace(DPm):.6f}")

        if res < tol:
            if verbose:
                print(f"  Converged in {k} iterations, |F|={res:.3e}")
            return R, Z, DPm

        # --- Newton step: 未x = -(DP^m - I)^{-1} F ---
        J = DPm - np.eye(2)
        try:
            delta = -np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  iter {k}: singular Jacobian, aborting")
            return None, None, None

        # --- line search / damping ---
        step = 1.0
        for _ in range(5):
            R_new = R + step * delta[0]
            Z_new = Z + step * delta[1]
            if not _in_bounds(R_new, Z_new):
                step *= 0.5
                continue
            # evaluate new residual
            R_end_new, Z_end_new = _trace_one(tracer, R_new, Z_new, phi_sec, period)
            if R_end_new is None:
                step *= 0.5
                continue
            F_new = np.array([R_end_new - R_new, Z_end_new - Z_new])
            res_new = np.linalg.norm(F_new)
            if prev_res is None or res_new < res:
                break
            step *= 0.5
        else:
            # All damping attempts failed; take a tiny step anyway
            R_new = R + step * delta[0]
            Z_new = Z + step * delta[1]
            if not _in_bounds(R_new, Z_new):
                if verbose:
                    print(f"  iter {k}: out of bounds after damping, aborting")
                return None, None, None

        prev_res = res
        R, Z = R_new, Z_new

    # max_iter reached without convergence
    # Return best estimate with the final DPm
    DPm_final = _compute_DPm(tracer, R, Z, phi_sec, period)
    if verbose:
        R_end, Z_end = _trace_one(tracer, R, Z, phi_sec, period)
        res_final = np.linalg.norm([R_end - R, Z_end - Z]) if R_end is not None else float('inf')
        print(f"  max_iter={max_iter} reached without convergence, |F|={res_final:.3e}")
    return None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_magnetic_axis(
    tracer,
    R_guess: float,
    Z_guess: float,
    phi_sec: float = 0.0,
    max_iter: int = 40,
    tol: float = 1e-9,
    verbose: bool = False,
    field_cache: Optional[dict] = None,
) -> tuple[float, float, np.ndarray]:
    """Locate the magnetic axis (m=1 O-point).

    Uses cyna C++ ``find_fixed_points_batch`` when available (fast, parallel),
    falls back to Python Newton iteration otherwise.

    Parameters
    ----------
    tracer : FieldlineTracer
        Field-line tracer with a ``trace_poincare`` method and ``_rhs``.
    R_guess, Z_guess : float
        Initial guess close to the magnetic axis.
    phi_sec : float
        Toroidal angle of the Poincar茅 section (radians). Default 0.
    max_iter : int
        Maximum Newton iterations. Default 40.
    tol : float
        Convergence tolerance on |P(x) - x|. Default 1e-9.
    verbose : bool
        Print iteration progress. Default False.
    field_cache : dict, optional
        Pre-extracted field arrays (BR, BPhi, BZ, R_grid, Z_grid, Phi_grid).
        If None, extracted automatically from tracer (or Python fallback used).

    Returns
    -------
    R_ax, Z_ax : float
        Magnetic axis coordinates.
    DPm : ndarray, shape (2, 2)
        Monodromy matrix (DP^1) at the converged point.

    Raises
    ------
    RuntimeError
        If Newton iteration fails to converge.
    """
    if _CYNA_AVAILABLE:
        fc = field_cache or _extract_field_cache(tracer)
        if fc is not None:
            if verbose:
                print(f"find_magnetic_axis [cyna]: seed (R={R_guess:.4f}, Z={Z_guess:.4f}), phi={phi_sec:.4f}")
            R_out, Z_out, res, conv, DPm_flat, eig_r, eig_i, ptype = _cyna_find_fixed_points_batch(
                np.array([R_guess], dtype=np.float64),
                np.array([Z_guess], dtype=np.float64),
                float(phi_sec), 1,
                max_iter=max_iter, tol=tol,
                BR=fc['BR'], BPhi=fc['BPhi'], BZ=fc['BZ'],
                R_grid=fc['R_grid'], Z_grid=fc['Z_grid'], Phi_grid=fc['Phi_grid'],
            )
            R, Z, DPm, kind = _cyna_result_to_fp(R_out, Z_out, res, conv, DPm_flat, eig_r, eig_i, ptype, 0)
            if R is not None:
                if verbose:
                    print(f"  Converged: R={R:.8f} Z={Z:.8f} res={res[0]:.2e} kind={kind}")
                return R, Z, DPm
            # fall through to Python fallback
            warnings.warn("cyna find_fixed_points_batch did not converge for magnetic axis; falling back to Python Newton")

    # Python fallback
    if verbose:
        print(f"find_magnetic_axis [Python]: seed (R={R_guess:.4f}, Z={Z_guess:.4f}), phi={phi_sec:.4f}")
    R, Z, DPm = _newton_iterate(
        tracer, R_guess, Z_guess, phi_sec,
        period=1, max_iter=max_iter, tol=tol, verbose=verbose,
    )
    if R is None:
        raise RuntimeError(
            f"Newton iteration failed to converge to the magnetic axis "
            f"starting from (R={R_guess}, Z={Z_guess})"
        )
    return R, Z, DPm


def find_fixed_point_newton(
    tracer,
    R_guess: float,
    Z_guess: float,
    phi_sec: float,
    period: int,
    max_iter: int = 40,
    tol: float = 1e-9,
    verbose: bool = False,
    field_cache: Optional[dict] = None,
) -> tuple[float, float, np.ndarray, str]:
    """Locate an m-periodic fixed point (X or O point).

    Uses cyna C++ ``find_fixed_points_batch`` when available (fast, parallel),
    falls back to Python Newton iteration otherwise.

    Parameters
    ----------
    tracer : FieldlineTracer
        Field-line tracer with a ``trace_poincare`` method and ``_rhs``.
    R_guess, Z_guess : float
        Initial guess near the target fixed point.
    phi_sec : float
        Toroidal angle of the Poincar茅 section (radians).
    period : int
        Period m of the fixed point (number of Poincar茅 turns per cycle).
    max_iter : int
        Maximum Newton iterations. Default 40.
    tol : float
        Convergence tolerance on |P^m(x) - x|. Default 1e-9.
    verbose : bool
        Print iteration progress. Default False.
    field_cache : dict, optional
        Pre-extracted field arrays (BR, BPhi, BZ, R_grid, Z_grid, Phi_grid).
        If None, extracted automatically from tracer (or Python fallback used).

    Returns
    -------
    R, Z : float
        Converged fixed point coordinates.
    DPm : ndarray, shape (2, 2)
        Monodromy matrix DP^m at the converged point.
    kind : str
        ``'X'`` for hyperbolic (|Tr(DP^m)| > 2) or ``'O'`` for elliptic.

    Raises
    ------
    RuntimeError
        If Newton iteration fails to converge.
    """
    if _CYNA_AVAILABLE:
        fc = field_cache or _extract_field_cache(tracer)
        if fc is not None:
            if verbose:
                print(f"find_fixed_point_newton [cyna]: m={period}, seed (R={R_guess:.4f}, Z={Z_guess:.4f}), phi={phi_sec:.4f}")
            R_out, Z_out, res, conv, DPm_flat, eig_r, eig_i, ptype = _cyna_find_fixed_points_batch(
                np.array([R_guess], dtype=np.float64),
                np.array([Z_guess], dtype=np.float64),
                float(phi_sec), int(period),
                max_iter=max_iter, tol=tol,
                BR=fc['BR'], BPhi=fc['BPhi'], BZ=fc['BZ'],
                R_grid=fc['R_grid'], Z_grid=fc['Z_grid'], Phi_grid=fc['Phi_grid'],
            )
            R, Z, DPm, kind = _cyna_result_to_fp(R_out, Z_out, res, conv, DPm_flat, eig_r, eig_i, ptype, 0)
            if R is not None:
                if verbose:
                    print(f"  Converged: R={R:.8f} Z={Z:.8f} res={res[0]:.2e} kind={kind}")
                return R, Z, DPm, kind
            warnings.warn(f"cyna find_fixed_points_batch did not converge (period={period}); falling back to Python Newton")

    # Python fallback
    if verbose:
        print(f"find_fixed_point_newton [Python]: m={period}, seed "
              f"(R={R_guess:.4f}, Z={Z_guess:.4f}), phi={phi_sec:.4f}")
    R, Z, DPm = _newton_iterate(
        tracer, R_guess, Z_guess, phi_sec,
        period=period, max_iter=max_iter, tol=tol, verbose=verbose,
    )
    if R is None:
        raise RuntimeError(
            f"Newton iteration failed to converge (period={period}) "
            f"starting from (R={R_guess}, Z={Z_guess})"
        )
    kind = _classify(DPm)
    return R, Z, DPm, kind


def refine_fixed_points_from_pkl(
    pkl_path: str,
    tracer,
    phi_sections: list,
    tol: float = 1e-8,
    period_map: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """Refine all fixed points in a pickle file with Newton's method.

    Reads a pickle file containing fixed-point data in the format produced
    by ``topoquest.hao_starting_cfg_v4.load_fp_pkl()``, uses each stored
    point as an initial guess for Newton iteration, and returns a refined
    dictionary in the same format.

    The pkl format is assumed to be::

        {phi_sec: {'xpts': [(R, Z, DPm_or_None), ...],
                   'opts': [(R, Z, DPm_or_None), ...]}, ...}

    OR the post-processed format with _FP objects::

        {phi_sec: {'xpts': [_FP, ...], 'opts': [_FP, ...]}, ...}

    Parameters
    ----------
    pkl_path : str
        Path to the pickle file.
    tracer : FieldlineTracer
        Field-line tracer.
    phi_sections : list of float
        Toroidal section angles to process.
    tol : float
        Newton convergence tolerance. Default 1e-8.
    period_map : dict, optional
        Maps phi_sec 鈫?island period m for island-chain fixed points.
        If None, period=1 is assumed for all points.
    verbose : bool
        Print progress. Default False.

    Returns
    -------
    refined : dict
        Same structure as input:
        ``{phi_sec: {'xpts': [(R, Z, DPm), ...], 'opts': [(R, Z, DPm), ...]}}``
    """
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    if period_map is None:
        period_map = {}

    refined = {}

    for phi_sec in phi_sections:
        k = float(phi_sec)
        # Find matching key (allow small float mismatch)
        if k not in raw:
            k = min(raw.keys(), key=lambda x: abs(x - k), default=None)
        if k is None or k not in raw:
            refined[float(phi_sec)] = {'xpts': [], 'opts': []}
            continue

        sec = raw[k]
        m = period_map.get(float(phi_sec), 1)

        refined_sec = {'xpts': [], 'opts': []}

        for kind_key in ('xpts', 'opts'):
            for entry in sec.get(kind_key, []):
                # Support both raw tuples and _FP objects
                if hasattr(entry, 'R'):
                    R0, Z0 = entry.R, entry.Z
                else:
                    R0, Z0 = float(entry[0]), float(entry[1])

                if verbose:
                    print(f"  Refining {kind_key[:-1].upper()} at "
                          f"phi={phi_sec:.4f}  R={R0:.5f} Z={Z0:.5f}  m={m}")

                try:
                    R_r, Z_r, DPm = _newton_iterate(
                        tracer, R0, Z0, float(phi_sec),
                        period=m, max_iter=20, tol=tol, verbose=verbose,
                    )
                    if R_r is None:
                        # Keep original if Newton failed
                        warnings.warn(
                            f"Newton failed for {kind_key[:-1].upper()} at "
                            f"phi={phi_sec:.4f} R={R0:.5f} Z={Z0:.5f}; "
                            f"keeping original"
                        )
                        R_r, Z_r = R0, Z0
                        DPm = (np.array(entry.DPm) if hasattr(entry, 'DPm')
                               else np.eye(2))
                    refined_sec[kind_key].append((R_r, Z_r, DPm))
                except Exception as exc:
                    warnings.warn(f"Unexpected error refining fixed point: {exc}")
                    refined_sec[kind_key].append((R0, Z0, np.eye(2)))

        refined[float(phi_sec)] = refined_sec

    return refined


# ===========================================================================
# Legacy API 鈥?field_func-based fixed-point tools (used by old tests and
# topology analysis code that passes field_func directly rather than a tracer)
# ===========================================================================

from scipy.optimize import root          # noqa: E402
from scipy.signal import argrelmin       # noqa: E402
from pyna.topo._rk4 import rk4_integrate # noqa: E402

FieldFunc2D = Callable[[float, float, float], np.ndarray]


def poincare_map(
    x0: np.ndarray,
    field_func: FieldFunc2D,
    n_turns: int = 1,
    phi_start: float = 0.0,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    method: str = "DOP853",
) -> np.ndarray:
    """Integrate n_turns toroidal turns and return endpoint (R, Z)."""
    x0 = np.asarray(x0, dtype=float)

    def rhs(phi, y):
        return field_func(y[0], y[1], phi)

    max_step = 0.05
    sol = rk4_integrate(
        rhs,
        [phi_start, phi_start + 2.0 * np.pi * n_turns],
        x0,
        max_step=max_step,
        dense_output=False,
    )
    return sol.y[:, -1]


def scan_fixed_point_seeds(
    field_func: FieldFunc2D,
    R_center: float,
    Z_center: float,
    r_scan: float,
    n_turns: int,
    n_scan: int = 200,
    *,
    order: int = 5,
) -> List[np.ndarray]:
    """Scan a ring and return candidate seeds near period-n fixed points."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_scan, endpoint=False)
    R_ring = R_center + r_scan * np.cos(thetas)
    Z_ring = Z_center + r_scan * np.sin(thetas)

    residuals = np.array([
        np.linalg.norm(poincare_map([R, Z], field_func, n_turns) - [R, Z])
        for R, Z in zip(R_ring, Z_ring)
    ])

    idx_mins = argrelmin(residuals, order=order)[0]
    seeds = sorted(
        [(residuals[i], np.array([R_ring[i], Z_ring[i]])) for i in idx_mins]
    )
    return [x for _, x in seeds]


def refine_fixed_point(
    seed: np.ndarray,
    field_func: FieldFunc2D,
    n_turns: int,
    *,
    tol: float = 1e-12,
    maxfev: int = 400,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> Optional[np.ndarray]:
    """Refine a seed to a true period-n fixed point via scipy.optimize.root."""
    def residual(x):
        return poincare_map(x, field_func, n_turns, rtol=rtol, atol=atol) - x

    sol = root(residual, seed, method="hybr", tol=tol,
               options={"maxfev": maxfev})
    if sol.success and np.linalg.norm(sol.fun) < 1e-8:
        return sol.x.copy()
    return None


def find_periodic_orbit(
    field_func: FieldFunc2D,
    seed: np.ndarray,
    n_turns: int,
    *,
    r_scan: Optional[float] = None,
    n_scan: int = 200,
    tol: float = 1e-12,
    maxfev: int = 400,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    dedup_tol: float = 1e-4,
    verbose: bool = False,
) -> List[np.ndarray]:
    """Find all period-n fixed points near a given seed (scan + Newton)."""
    seed = np.asarray(seed, dtype=float)
    R_center, Z_center = float(seed[0]), float(seed[1])

    if r_scan is None:
        r_scan = max(0.03, 0.15 * np.sqrt(R_center**2 + Z_center**2) / 10)
        if verbose:
            print(f"r_scan not given; using {r_scan:.4f} m")

    if verbose:
        print(f"Scanning ring: centre=({R_center:.4f}, {Z_center:.4f}), "
              f"r={r_scan:.4f} m, n_turns={n_turns}, n_scan={n_scan}")

    seeds = scan_fixed_point_seeds(
        field_func, R_center, Z_center, r_scan, n_turns,
        n_scan=n_scan, order=5,
    )

    if verbose:
        print(f"  {len(seeds)} candidate seeds found")

    fixed_points: List[np.ndarray] = []
    for s in seeds:
        fp = refine_fixed_point(s, field_func, n_turns,
                                tol=tol, maxfev=maxfev, rtol=rtol, atol=atol)
        if fp is not None:
            if all(np.linalg.norm(fp - q) > dedup_tol for q in fixed_points):
                fixed_points.append(fp)
                if verbose:
                    res = np.linalg.norm(
                        poincare_map(fp, field_func, n_turns) - fp)
                    print(f"  Fixed point: R={fp[0]:.6f}  Z={fp[1]:.6f}  "
                          f"|residual|={res:.2e}")

    if verbose:
        print(f"Total: {len(fixed_points)} distinct period-{n_turns} fixed points")

    return fixed_points


def classify_fixed_point(
    fp: np.ndarray,
    field_func: FieldFunc2D,
    n_turns: int,
    *,
    fd_eps: float = 1e-6,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Tuple[str, np.ndarray, float]:
    """Classify a fixed point as 'X' or 'O' via the monodromy matrix."""
    from pyna.topo.variational import PoincareMapVariationalEquations

    fp = np.asarray(fp, dtype=float)

    def _wrap(R, Z, phi):
        return field_func(R, Z, phi)

    vq = PoincareMapVariationalEquations(_wrap, fd_eps=fd_eps)
    phi_span = (0.0, 2.0 * np.pi * n_turns)
    J = vq.jacobian_matrix(
        fp, phi_span,
        solve_ivp_kwargs=dict(method="DOP853", rtol=rtol, atol=atol),
    )

    det_J = float(np.linalg.det(J))
    eigvals = np.linalg.eigvals(J)
    lam_abs = sorted(np.abs(eigvals))

    if np.all(np.isreal(eigvals)):
        lam_real = np.sort(np.real(eigvals))
        if lam_real[0] < 0.99 and lam_real[1] > 1.01:
            fp_type = "X"
        else:
            fp_type = "unknown"
    else:
        if abs(lam_abs[0] - 1.0) < 0.05 and abs(lam_abs[1] - 1.0) < 0.05:
            fp_type = "O"
        else:
            fp_type = "unknown"

    return fp_type, J, det_J


def classify_fixed_point_higher_order(
    fp: np.ndarray,
    field_func: FieldFunc2D,
    n_turns: int,
    *,
    parabolic_tol: float = 1e-2,
    fd_eps: float = 1e-6,
    fd_eps2: float = 1e-5,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict:
    """Extended fixed-point classifier (handles degenerate/non-conservative cases)."""
    from pyna.topo.variational import PoincareMapVariationalEquations

    fp = np.asarray(fp, dtype=float)
    phi_span = (0.0, 2.0 * np.pi * n_turns)

    vq = PoincareMapVariationalEquations(field_func, fd_eps=fd_eps, fd_eps2=fd_eps2)
    ivp_kw = dict(method="DOP853", rtol=rtol, atol=atol)

    J = vq.jacobian_matrix(fp, phi_span, solve_ivp_kwargs=ivp_kw)
    det_J = float(np.linalg.det(J))
    eigvals = np.linalg.eigvals(J)
    lam_abs = np.abs(eigvals)
    stability_index = float(np.trace(J) / 2.0)

    result = dict(J=J, T=None, det_J=det_J,
                  eigenvalues=eigvals, stability_index=stability_index)

    _det_tol = 0.05

    if abs(det_J - 1.0) > _det_tol and abs(det_J + 1.0) > _det_tol:
        all_contracting = np.all(lam_abs < 1.0 - _det_tol)
        all_expanding   = np.all(lam_abs > 1.0 + _det_tol)
        if all_contracting:
            result["type"] = "sink"
        elif all_expanding:
            result["type"] = "source"
        else:
            result["type"] = "saddle"
        return result

    J_minus_I = np.linalg.norm(J - np.eye(2), "fro")
    J_plus_I  = np.linalg.norm(J + np.eye(2), "fro")
    is_near_I  = J_minus_I < parabolic_tol
    is_near_mI = J_plus_I  < parabolic_tol

    if is_near_I or is_near_mI:
        _, _, T = vq.tangent_map(fp, phi_span, order=2, solve_ivp_kwargs=ivp_kw)
        result["T"] = T
        H_eff = T[0]
        disc = H_eff[0, 0] * H_eff[1, 1] - H_eff[0, 1] ** 2
        T_norm = float(np.linalg.norm(T))
        if T_norm < 1e-6:
            result["type"] = "parabolic"
        elif disc > 0:
            result["type"] = "degenerate_O"
        elif disc < 0:
            result["type"] = "degenerate_X"
        else:
            result["type"] = "parabolic"
        return result

    if np.all(np.isreal(eigvals)):
        lam_real = np.sort(np.real(eigvals))
        if lam_real[0] < 1.0 - _det_tol and lam_real[1] > 1.0 + _det_tol:
            result["type"] = "X"
        else:
            result["type"] = "unknown"
    else:
        if (abs(lam_abs[0] - 1.0) < _det_tol and abs(lam_abs[1] - 1.0) < _det_tol):
            result["type"] = "O"
        else:
            result["type"] = "unknown"

    return result


# ===========================================================================
# Island-chain fixed-point search 鈥?integrated from island_xo_v3 & island_fast
# ===========================================================================

def find_island_chain_fixed_points(
    tracer,
    R_ax: float,
    Z_ax: float,
    period: int,
    phi_sec: float = 0.0,
    *,
    r_min: float = 0.02,
    r_max: float = 0.25,
    n_r: int = 8,
    n_ang: int = 48,
    coarse_tol: float = 0.008,
    period1_tol: float = 0.001,
    dedup_tol: float = 1e-3,
    top_n_candidates: int = 20,
    max_iter: int = 30,
    tol: float = 1e-9,
    verbose: bool = False,
) -> List[dict]:
    """Search for all period-m X/O fixed points of an island chain.

    Combines the grid-scan + top-N filtering strategy from ``island_xo_v3``
    with deduplication.  Does NOT trace manifolds (use the manifold module
    for that).

    Algorithm
    ---------
    1. Build an (R, 胃) polar grid centred on (R_ax, Z_ax) in the range
       r/a 鈭?[r_min, r_max] 脳 [0, 2蟺).
    2. For each grid point x0, evaluate ``|P^m(x0) - x0|``.  Keep only
       those with residual < ``coarse_tol`` *and* ``|P^1(x0) - x0| > period1_tol``
       (to exclude the magnetic axis / period-1 points).
    3. Sort by coarse residual, take the best ``top_n_candidates``.
    4. Run Newton iteration (``_newton_iterate``) for each candidate.
    5. Deduplicate converged points within ``dedup_tol``.
    6. Classify each converged point as 'X' or 'O'.
    7. Return list sorted by R.

    Parameters
    ----------
    tracer : FieldlineTracer
        Field-line tracer (needs ``trace_poincare`` / ``_rhs``).
    R_ax, Z_ax : float
        Approximate magnetic axis position used to centre the search grid.
    period : int
        Island chain period m (number of Poincar茅 turns per cycle).
    phi_sec : float
        Poincar茅 section angle [rad]. Default 0.
    r_min, r_max : float
        Radial range [m] relative to the magnetic axis. Default [0.02, 0.25].
    n_r : int
        Number of radial grid points. Default 8.
    n_ang : int
        Number of angular grid points. Default 48.
    coarse_tol : float
        Maximum |P^m(x) - x| [m] to accept as a candidate. Default 8 mm.
    period1_tol : float
        Minimum |P^1(x) - x| [m] to *exclude* period-1 fixed points.
        Default 1 mm.
    dedup_tol : float
        Distance threshold [m] for deduplication. Default 1 mm.
    top_n_candidates : int
        Maximum number of candidates passed to Newton. Default 20.
    max_iter : int
        Maximum Newton iterations per candidate. Default 30.
    tol : float
        Newton convergence tolerance. Default 1e-9.
    verbose : bool
        Print progress. Default False.

    Returns
    -------
    list of dict
        Each dict has keys:
        ``{'R': float, 'Z': float, 'DPm': ndarray(2,2), 'kind': str,
           'trace': float, 'eigenvalues': ndarray}``.
        Sorted by R (ascending).
    """
    # --- build search grid ---
    r_vals = np.linspace(r_min, r_max, n_r)
    ang_vals = np.linspace(0.0, _TWO_PI, n_ang, endpoint=False)
    total = n_r * n_ang
    if verbose:
        print(f"[find_island_chain_fixed_points] m={period}, phi={phi_sec:.4f}")
        print(f"  Grid: {n_r}脳{n_ang}={total} pts  r=[{r_min:.3f},{r_max:.3f}]")
        print(f"  Coarse criterion: |P^m-x|<{coarse_tol*1000:.1f}mm AND "
              f"|P^1-x|>{period1_tol*1000:.1f}mm")

    candidates: List[Tuple[float, float, np.ndarray]] = []  # (res_m, res1, x0)

    # --- Fast coarse scan via cyna batch (replaces slow Python _trace_one loop) ---
    fc = _extract_field_cache(tracer) if _CYNA_AVAILABLE else None
    if _CYNA_AVAILABLE and fc is not None:
        # Build all seed points
        R_seeds_all, Z_seeds_all = [], []
        for r in r_vals:
            for ang in ang_vals:
                R0 = R_ax + r * np.cos(ang)
                Z0 = Z_ax + r * np.sin(ang)
                if _in_bounds(R0, Z0):
                    R_seeds_all.append(R0)
                    Z_seeds_all.append(Z0)
        R_seeds_all = np.array(R_seeds_all, dtype=np.float64)
        Z_seeds_all = np.array(Z_seeds_all, dtype=np.float64)
        if verbose:
            print(f"  [cyna coarse] Batch P^m on {len(R_seeds_all)} seeds (period={period})...")
        # Run Newton with loose tolerance to get residuals quickly
        R_c, Z_c, res_c, conv_c, DPm_c, _, _, ptype_c = _cyna_find_fixed_points_batch(
            R_seeds_all, Z_seeds_all,
            float(phi_sec), int(period),
            max_iter=8, tol=coarse_tol * 0.1,
            BR=fc['BR'], BPhi=fc['BPhi'], BZ=fc['BZ'],
            R_grid=fc['R_grid'], Z_grid=fc['Z_grid'], Phi_grid=fc['Phi_grid'],
        )
        # period-1 exclusion via cyna (single turn)
        R_1c, Z_1c, res_1c, _, _, _, _, _ = _cyna_find_fixed_points_batch(
            R_seeds_all, Z_seeds_all,
            float(phi_sec), 1,
            max_iter=8, tol=period1_tol * 0.1,
            BR=fc['BR'], BPhi=fc['BPhi'], BZ=fc['BZ'],
            R_grid=fc['R_grid'], Z_grid=fc['Z_grid'], Phi_grid=fc['Phi_grid'],
        )
        for i in range(len(R_seeds_all)):
            if res_c[i] >= coarse_tol:
                continue
            if res_1c[i] < period1_tol:
                continue  # period-1 point
            x0 = np.array([R_seeds_all[i], Z_seeds_all[i]])
            candidates.append((res_c[i], res_1c[i], x0))
    else:
        # Python fallback (slow)
        for r in r_vals:
            for ang in ang_vals:
                x0 = np.array([R_ax + r * np.cos(ang), Z_ax + r * np.sin(ang)])
                if not _in_bounds(x0[0], x0[1]):
                    continue
                R_m, Z_m = _trace_one(tracer, x0[0], x0[1], phi_sec, period)
                if R_m is None:
                    continue
                res_m = np.linalg.norm([R_m - x0[0], Z_m - x0[1]])
                if res_m >= coarse_tol:
                    continue
                R_1, Z_1 = _trace_one(tracer, x0[0], x0[1], phi_sec, 1)
                if R_1 is None:
                    continue
                res_1 = np.linalg.norm([R_1 - x0[0], Z_1 - x0[1]])
                if res_1 < period1_tol:
                    continue
                candidates.append((res_m, res_1, x0.copy()))

    candidates.sort(key=lambda t: t[0])
    if verbose:
        print(f"  Candidates after coarse filter: {len(candidates)}")
        for res_m, res_1, x0 in candidates[:5]:
            print(f"    R={x0[0]:.4f} Z={x0[1]:.4f} "
                  f"|P^m-x|={res_m*1000:.1f}mm |P^1-x|={res_1*1000:.1f}mm")

    # --- Newton refinement ---
    if verbose:
        print(f"  Newton refinement: top-{min(top_n_candidates, len(candidates))} candidates")

    fixed_points: List[dict] = []
    seen: List[np.ndarray] = []

    top_cands = candidates[:top_n_candidates]

    if _CYNA_AVAILABLE and fc is not None and top_cands:
        # Batch Newton with cyna 鈥?one call for all candidates
        R_seeds = np.array([c[2][0] for c in top_cands], dtype=np.float64)
        Z_seeds = np.array([c[2][1] for c in top_cands], dtype=np.float64)
        if verbose:
            print(f"  [cyna] Batch Newton: {len(R_seeds)} seeds")
        R_out, Z_out, res_out, conv_out, DPm_flat, eig_r, eig_i, ptype = _cyna_find_fixed_points_batch(
            R_seeds, Z_seeds,
            float(phi_sec), int(period),
            max_iter=max_iter, tol=tol,
            BR=fc['BR'], BPhi=fc['BPhi'], BZ=fc['BZ'],
            R_grid=fc['R_grid'], Z_grid=fc['Z_grid'], Phi_grid=fc['Phi_grid'],
        )
        for idx in range(len(R_seeds)):
            R_fp, Z_fp, DPm, kind = _cyna_result_to_fp(
                R_out, Z_out, res_out, conv_out, DPm_flat, eig_r, eig_i, ptype, idx)
            if R_fp is None:
                continue

            xfp = np.array([R_fp, Z_fp])
            if any(np.linalg.norm(xfp - xs) < dedup_tol for xs in seen):
                continue

            # final period-1 check
            R_1fp, Z_1fp = _trace_one(tracer, R_fp, Z_fp, phi_sec, 1)
            if R_1fp is not None:
                res_1fp = np.linalg.norm([R_1fp - R_fp, Z_1fp - Z_fp])
                if res_1fp < period1_tol * 0.5:
                    if verbose:
                        print(f"  [skip] period-1 point at R={R_fp:.5f} Z={Z_fp:.5f}")
                    continue

            tr = float(np.trace(DPm))
            evals = np.linalg.eigvals(DPm)
            seen.append(xfp)
            fp_dict = {
                'R': R_fp, 'Z': Z_fp,
                'DPm': DPm.copy(),
                'kind': kind,
                'trace': tr,
                'eigenvalues': evals,
            }
            fixed_points.append(fp_dict)
            if verbose:
                print(f"  [{kind}] R={R_fp:.6f} Z={Z_fp:.6f} "
                      f"tr={tr:.4f} 位={evals[0]:.4f},{evals[1]:.4f}")
    else:
        # Python fallback: sequential Newton
        for res_m, res_1, x0 in top_cands:
            R_fp, Z_fp, DPm = _newton_iterate(
                tracer, x0[0], x0[1], phi_sec,
                period=period, max_iter=max_iter, tol=tol, verbose=False,
            )
            if R_fp is None:
                continue
            if DPm is None:
                continue

            # deduplication
            xfp = np.array([R_fp, Z_fp])
            if any(np.linalg.norm(xfp - xs) < dedup_tol for xs in seen):
                continue

            # final period-1 check
            R_1fp, Z_1fp = _trace_one(tracer, R_fp, Z_fp, phi_sec, 1)
            if R_1fp is not None:
                res_1fp = np.linalg.norm([R_1fp - R_fp, Z_1fp - Z_fp])
                if res_1fp < period1_tol * 0.5:
                    if verbose:
                        print(f"  [skip] period-1 point at R={R_fp:.5f} Z={Z_fp:.5f}")
                    continue

            kind = _classify(DPm)
            tr = float(np.trace(DPm))
            evals = np.linalg.eigvals(DPm)
            seen.append(xfp)
            fp_dict = {
                'R': R_fp, 'Z': Z_fp,
                'DPm': DPm.copy(),
                'kind': kind,
                'trace': tr,
                'eigenvalues': evals,
            }
            fixed_points.append(fp_dict)
            if verbose:
                print(f"  [{kind}] R={R_fp:.6f} Z={Z_fp:.6f} "
                      f"tr={tr:.4f} 位={evals[0]:.4f},{evals[1]:.4f}")

    # sort by R
    fixed_points.sort(key=lambda d: d['R'])
    if verbose:
        n_X = sum(1 for d in fixed_points if d['kind'] == 'X')
        n_O = sum(1 for d in fixed_points if d['kind'] == 'O')
        print(f"  Result: {len(fixed_points)} fixed points (X={n_X}, O={n_O})")
    return fixed_points


# ===========================================================================
# P^1 propagation strategy 鈥?from island_p1_propagate
# ===========================================================================

def propagate_island_chain(
    tracer,
    R0: float,
    Z0: float,
    phi0: float,
    period: int,
    section_phis: Optional[List[float]] = None,
    *,
    refine: bool = True,
    max_iter: int = 30,
    tol: float = 1e-9,
    dedup_tol: float = 1e-5,
    period1_tol: float = 0.5e-3,
    verbose: bool = False,
) -> dict:
    """Map out an island chain by iterating P^1 from one known fixed point.

    For an m-period island chain: starting from one fixed point at phi=phi0,
    one full toroidal turn (P^1) maps to the NEXT point on the chain.
    After m turns we recover all m distinct period-m fixed points at phi=phi0.

    This is the "sequential P^1" strategy from ``island_fast.py``:
    it is much cheaper than doing m-turn integration for each starting point
    because each single-turn trace is already available.

    If ``section_phis`` is given, crossings at those intermediate sections
    are also collected (by recording P^1 intermediate positions).  When
    ``refine=True``, each collected crossing is Newton-refined with P^m.

    Parameters
    ----------
    tracer : FieldlineTracer
        Field-line tracer.
    R0, Z0 : float
        One known fixed point of P^m at toroidal angle phi0.
    phi0 : float
        Toroidal section angle of the seed [rad].
    period : int
        Island chain period m.
    section_phis : list of float, optional
        Extra toroidal sections at which to collect chain points.
        If None (default), only phi0 is processed.
    refine : bool
        Newton-refine each collected crossing. Default True.
    max_iter : int
        Max Newton iterations. Default 30.
    tol : float
        Newton convergence tolerance. Default 1e-9.
    dedup_tol : float
        Deduplication distance [m]. Default 1e-5.
    period1_tol : float
        Threshold |P^1 - x| below which a point is flagged as period-1.
    verbose : bool
        Print progress. Default False.

    Returns
    -------
    dict
        ``{phi: [{'R': float, 'Z': float, 'DPm': ndarray, 'kind': str,
                  'trace': float, 'eigenvalues': ndarray}, ...]}``.
        Each list contains (up to) m distinct fixed points at that section,
        sorted by R.

    Notes
    -----
    For sections other than phi0 this function uses the tracer's single-turn
    ``_trace_one`` helper.  Intermediate crossings are estimated from the
    P^1 orbit and then Newton-refined.  If ``refine=False``, raw crossing
    estimates are returned without refinement.
    """
    if section_phis is None:
        section_phis = [phi0]

    all_phis = sorted(set([float(phi0)] + [float(p) for p in section_phis]))

    # --- Step 1: iterate P^1 from seed to get all period-m points at phi0 ---
    if verbose:
        print(f"[propagate_island_chain] m={period}, seed=(R={R0:.5f}, Z={Z0:.5f}), phi0={phi0:.4f}")
        print(f"  Iterating P^1 脳 {period} to collect chain points at phi={phi0:.4f}")

    chain_at_phi0: List[np.ndarray] = [np.array([R0, Z0])]
    xc = np.array([R0, Z0], dtype=float)

    for step_i in range(period - 1):
        R_next, Z_next = _trace_one(tracer, xc[0], xc[1], phi0, 1)
        if R_next is None:
            warnings.warn(f"P^1 mapping lost field line at step {step_i + 1}; "
                          f"stopping propagation at {len(chain_at_phi0)} points.")
            break
        xc = np.array([R_next, Z_next])
        chain_at_phi0.append(xc.copy())
        if verbose:
            print(f"  chain[{step_i+1}]: R={R_next:.6f} Z={Z_next:.6f}")

    # closure check
    if len(chain_at_phi0) == period and verbose:
        R_close, Z_close = _trace_one(tracer, chain_at_phi0[-1][0], chain_at_phi0[-1][1], phi0, 1)
        if R_close is not None:
            err = np.linalg.norm([R_close - R0, Z_close - Z0])
            print(f"  Closure check: |P^1(last) - seed| = {err*1000:.3f} mm")

    # --- Step 2: optionally Newton-refine each collected point at phi0 ---
    def _refine_and_classify(R_init, Z_init, phi_s):
        """Newton-refine a point at phi_s and classify it."""
        if refine:
            R_fp, Z_fp, DPm = _newton_iterate(
                tracer, R_init, Z_init, phi_s,
                period=period, max_iter=max_iter, tol=tol, verbose=False,
            )
            if R_fp is None:
                warnings.warn(f"Newton did not converge for crossing near "
                              f"(R={R_init:.5f}, Z={Z_init:.5f}) at phi={phi_s:.4f}")
                R_fp, Z_fp = R_init, Z_init
                DPm = _compute_DPm(tracer, R_fp, Z_fp, phi_s, period)
        else:
            R_fp, Z_fp = R_init, Z_init
            DPm = _compute_DPm(tracer, R_fp, Z_fp, phi_s, period)

        if DPm is None:
            DPm = np.full((2, 2), np.nan)
            kind, tr, evals = '?', float('nan'), np.array([float('nan'), float('nan')])
        else:
            kind = _classify(DPm)
            tr = float(np.trace(DPm))
            evals = np.linalg.eigvals(DPm)

        return {
            'R': R_fp, 'Z': Z_fp,
            'DPm': DPm.copy() if DPm is not None else np.full((2,2), np.nan),
            'kind': kind, 'trace': tr, 'eigenvalues': evals,
        }

    def _dedup_fp_list(fp_list):
        kept = []
        for fp in fp_list:
            xfp = np.array([fp['R'], fp['Z']])
            if not any(np.linalg.norm(xfp - np.array([k['R'], k['Z']])) < dedup_tol
                       for k in kept):
                kept.append(fp)
        return sorted(kept, key=lambda d: d['R'])

    result = {}

    # phi0: refine raw chain points
    phi0_f = float(phi0)
    pts_phi0 = []
    for xpt in chain_at_phi0:
        fp = _refine_and_classify(xpt[0], xpt[1], phi0_f)
        # flag obvious period-1 points
        R_1, Z_1 = _trace_one(tracer, fp['R'], fp['Z'], phi0_f, 1)
        if R_1 is not None:
            res_1 = np.linalg.norm([R_1 - fp['R'], Z_1 - fp['Z']])
            if res_1 < period1_tol:
                if verbose:
                    print(f"  [skip phi0] period-1 at R={fp['R']:.5f}")
                continue
        pts_phi0.append(fp)
    result[phi0_f] = _dedup_fp_list(pts_phi0)

    # additional sections: re-run P^1 orbit and record crossings
    other_phis = [p for p in all_phis if abs(p - phi0_f) > 1e-8]
    if other_phis:
        # For each additional phi section, propagate from each phi0 chain
        # point by single turn and collect the crossing nearest to that section.
        # Since we only have a RK4 fixed-step tracer, we approximate: run one
        # turn from each chain point at phi0 and collect the recorded endpoint.
        # Then Newton-refine that endpoint at the target phi.
        #
        # NOTE: a proper multi-section crossing would require the tracer to
        # support intermediate section recording.  If the tracer has a
        # trace_poincare_multi method, use it; otherwise fall back to the
        # single-section approach.
        for phi_s in other_phis:
            phi_s_f = float(phi_s)
            pts_s = []
            for xpt in chain_at_phi0:
                # Propagate from phi0 to phi_s (one turn) to get a seed
                R_seed, Z_seed = _trace_one(tracer, xpt[0], xpt[1], phi_s_f, 1)
                if R_seed is None:
                    continue
                fp = _refine_and_classify(R_seed, Z_seed, phi_s_f)
                R_1, Z_1 = _trace_one(tracer, fp['R'], fp['Z'], phi_s_f, 1)
                if R_1 is not None:
                    res_1 = np.linalg.norm([R_1 - fp['R'], Z_1 - fp['Z']])
                    if res_1 < period1_tol:
                        continue
                pts_s.append(fp)
            result[phi_s_f] = _dedup_fp_list(pts_s)
            if verbose:
                n_conv = sum(1 for p in result[phi_s_f] if p['kind'] != '?')
                print(f"  phi={phi_s_f:.4f}: {n_conv} fixed points collected")

    if verbose:
        for phi_s, pts in sorted(result.items()):
            n_X = sum(1 for p in pts if p['kind'] == 'X')
            n_O = sum(1 for p in pts if p['kind'] == 'O')
            print(f"  phi={phi_s:.4f}: {len(pts)} pts (X={n_X}, O={n_O})")

    return result


# ---------------------------------------------------------------------------
# group_fixed_points_by_orbit
# ---------------------------------------------------------------------------

def group_fixed_points_by_orbit(
    fixed_points,
    tracer,
    m: int,
    phi0: float = 0.0,
    tol: float = 1e-4,
) -> "list[list[int]]":
    """Group fixed points by which periodic orbit they belong to.

    Two fixed points belong to the same orbit when one can be reached from
    the other by applying the single-turn Poincare map P^1 exactly k times
    (1 <= k < m).  This is the physically correct criterion for orbit
    identity -- not a spatial distance threshold between points.

    The algorithm propagates P^1 from each point and checks whether any
    known point is reached within ``tol`` metres.  Union-Find collects
    connected components.

    Parameters
    ----------
    fixed_points : sequence of FixedPoint or dict with keys 'R','Z'
        Already-converged Newton fixed points at section ``phi0``.
    tracer : callable
        Poincare tracer with signature ``R1, Z1 = tracer(R0, Z0, phi0, 1)``.
        The cyna C++ tracer satisfies this interface.
    m : int
        Orbit period (number of toroidal turns to close the orbit).
    phi0 : float
        Toroidal angle of the Poincare section [rad].  Default 0.
    tol : float
        Distance threshold [m] for point identity under P^1.  Default 1e-4.

    Returns
    -------
    list of list of int
        Each inner list contains the indices (into ``fixed_points``) of the
        points on the same periodic orbit, ordered by visitation sequence
        starting from the lowest index.  Singleton lists for isolated points.

    Notes
    -----
    Complexity: O(m * N) tracer calls where N = len(fixed_points).
    For cyna-accelerated tracers this is negligible (< 1 ms for N~20, m~10).

    Example
    -------
    For a m=10, n=3 island chain with 10 X-points at phi=0::

        groups = group_fixed_points_by_orbit(x_fps, tracer, m=10)
        # Returns [[0, 3, 6, 9, 2, 5, 8, 1, 4, 7]]  (one orbit, all 10 pts)
    """
    import numpy as np

    n = len(fixed_points)
    if n == 0:
        return []

    def _RZ(fp):
        if hasattr(fp, 'R'):
            return float(fp.R), float(fp.Z)
        return float(fp['R']), float(fp['Z'])

    # Union-Find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    RZs = [_RZ(fp) for fp in fixed_points]

    for i in range(n):
        R, Z = RZs[i]
        cur_R, cur_Z = R, Z
        for _ in range(m - 1):
            try:
                next_R, next_Z = tracer(cur_R, cur_Z, phi0, 1)
            except Exception:
                break
            if next_R is None or (hasattr(next_R, '__len__') is False and
                                   (next_R != next_R)):  # NaN check
                break
            cur_R, cur_Z = float(next_R), float(next_Z)
            for j in range(n):
                if find(i) == find(j):
                    continue
                Rj, Zj = RZs[j]
                if (cur_R - Rj) ** 2 + (cur_Z - Zj) ** 2 < tol * tol:
                    union(i, j)

    # Collect groups (preserve visitation order within each group)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(i)

    # Sort groups by smallest index; sort members by visitation order
    # (follow P^1 chain from the smallest index)
    result = []
    for root in sorted(buckets):
        members = sorted(buckets[root])
        # Reconstruct visitation sequence starting from members[0]
        start = members[0]
        seq = [start]
        visited_set = {start}
        R, Z = RZs[start]
        cur_R, cur_Z = R, Z
        for _ in range(len(members) - 1):
            try:
                next_R, next_Z = tracer(cur_R, cur_Z, phi0, 1)
            except Exception:
                break
            if next_R is None:
                break
            cur_R, cur_Z = float(next_R), float(next_Z)
            for j in members:
                if j in visited_set:
                    continue
                Rj, Zj = RZs[j]
                if (cur_R - Rj) ** 2 + (cur_Z - Zj) ** 2 < tol * tol:
                    seq.append(j)
                    visited_set.add(j)
                    break
        # Append any members not reached by the walk (shouldn't happen for
        # well-converged points, but guards against tracer noise)
        for j in members:
            if j not in visited_set:
                seq.append(j)
        result.append(seq)

    return result
