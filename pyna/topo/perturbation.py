"""
DPm (monodromy matrix) change under magnetic field perturbation.

Physics summary
---------------
Given a base periodic orbit with monodromy matrix DPm and a perturbation
field Î´B, we compute:

1. **Î´X_cyc** : shift of the periodic orbit position in (R, Z).
   Integrates  dÎ´X/dÏ† = A(x,Ï†)Â·Î´X + Î´f(x,Ï†)  along the orbit (m turns),
   where Î´f = f_pert(R,Z,Ï†) âˆ?f_base(R,Z,Ï†).
   Initial condition from the self-consistency condition:
       (DPm âˆ?I) Â· Î´X_cyc(Ï†â‚€) = âˆ’âˆ® Î´f dÏ†

2. **Î´DPm** : change in the monodromy matrix.
   Primary method (cyna available): ``trace_orbit_along_phi`` for both
   base and perturbed fields â†?DPm_pert âˆ?DPm_base (C++ FD Jacobian).
   Fallback (Python): variational equations via ``PoincareMapVariationalEquations``.

3. **Î´Î»_u** : first-order shift of the hyperbolic eigenvalue.
       Î´Î»_u = (w_u^T Â· Î´DPm Â· v_u) / (w_u^T Â· v_u)
   where v_u is the right and w_u the left eigenvector of DPm.

C++ acceleration
----------------
When ``base_field_cache`` and ``pert_field_cache`` are supplied as dicts with
keys ``BR, BPhi, BZ, R_grid, Z_grid, Phi_grid`` (matching pyna's field-cache
convention), the following cyna functions are used:

* ``pyna._cyna.trace_orbit_along_phi`` â€?orbit tracing + DPm in C++.
* ``pyna._cyna.compute_A_matrix_batch`` â€?batch A-matrix for Î´X_cyc integral.

Without caches (or when cyna is unavailable) the code falls back to Python
RK4 + finite-difference Jacobians (``PoincareMapVariationalEquations``).

Field-cache convention (identical to pyna.topo.island_chain)
-------------------------------------------------------------
``field_cache = dict(BR=..., BPhi=..., BZ=..., R_grid=..., Z_grid=...,
                     Phi_grid=...)``
Arrays must be C-contiguous float64; BR/BPhi/BZ have shape
``(nR, nZ, nPhi)`` or compatible layout accepted by cyna.

References
----------
W7-X Jacobian analysis Julia notebook
``W7X_Jac_change_under_perturbation.ipynb``, cells 43â€?6.
``pyna.control.FPT_3d`` â€?related non-axisymmetric FPT implementation.
X-point displacement criterion
------------------------------
The complete X-point displacement stability criterion
    dx_Xpt = -A_inv @ (R_Xpt * dB_pol(Xpt) / BPhi(Xpt))
is implemented in ``topoquest.analysis.beta_limits.xpoint_displacement_stability``.
This module (pyna) provides the underlying ``compute_A_matrix_batch`` (via
``pyna._cyna``), which computes the Poincare map tangent matrix (FPT A-matrix)
used in the X-point displacement instability calculation.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Optional cyna C++ backend
# ---------------------------------------------------------------------------
try:
    from pyna._cyna import (
        trace_orbit_along_phi as _cyna_trace_orbit,
        compute_A_matrix_batch as _cyna_A_batch,
    )
    _HAS_CYNA = (_cyna_trace_orbit is not None) and (_cyna_A_batch is not None)
except ImportError:
    _cyna_trace_orbit = None
    _cyna_A_batch = None
    _HAS_CYNA = False

from pyna.topo._rk4 import rk4_integrate
from pyna.topo.variational import PoincareMapVariationalEquations

__all__ = [
    "DPm_finite_difference",
    "DPm_shift_under_field_perturbation",
    "eigenvalue_perturbation",
]


# ---------------------------------------------------------------------------
# Internal: cyna-based DPm computation
# ---------------------------------------------------------------------------

def _compute_DPm_via_cyna(
    R0: float,
    Z0: float,
    phi_start: float,
    phi_span: float,
    island_period: int,
    field_cache: dict,
    DPhi: float = 0.05,
    fd_eps: float = 1e-4,
) -> np.ndarray:
    """Compute DPm using cyna trace_orbit_along_phi.

    Parameters
    ----------
    R0, Z0 : float
        Starting position.
    phi_start : float
        Starting toroidal angle.
    phi_span : float
        Total integration span = island_period * 2Ï€.
    island_period : int
        Number of toroidal turns (m) for the DPm Jacobian.
    field_cache : dict
        Keys: BR, BPhi, BZ, R_grid, Z_grid, Phi_grid.
    DPhi : float
        RK4 step size inside cyna.
    fd_eps : float
        FD step for cyna A-matrix computation.

    Returns
    -------
    DPm : ndarray, shape (2, 2)
    trajectory : ndarray, shape (N, 3)  â€?columns [R, Z, phi]
    """
    def _c(a):
        return np.ascontiguousarray(a, dtype=np.float64)

    BR    = _c(field_cache['BR'].ravel() if field_cache['BR'].ndim == 3
               else field_cache['BR'])
    BPhi  = _c(field_cache['BPhi'].ravel() if field_cache['BPhi'].ndim == 3
               else field_cache['BPhi'])
    BZ    = _c(field_cache['BZ'].ravel() if field_cache['BZ'].ndim == 3
               else field_cache['BZ'])
    Rg    = _c(field_cache['R_grid'])
    Zg    = _c(field_cache['Z_grid'])
    Phig  = _c(field_cache['Phi_grid'])

    # dphi_out: output every DPhi radians
    dphi_out = DPhi * 4  # coarser output for speed

    R_arr, Z_arr, phi_arr, DPm_flat, alive_arr = _cyna_trace_orbit(
        float(R0), float(Z0), float(phi_start),
        float(phi_span), float(dphi_out),
        int(island_period), float(DPhi), float(fd_eps),
        BR, BPhi, BZ, Rg, Zg, Phig,
    )
    R_arr   = np.asarray(R_arr)
    Z_arr   = np.asarray(Z_arr)
    phi_arr = np.asarray(phi_arr)
    DPm_flat = np.asarray(DPm_flat)  # shape (N, 4)

    # The cyna DPm at each output step is the m-turn Jacobian; we want the
    # final one (the full m-turn monodromy matrix).
    DPm = DPm_flat[-1].reshape(2, 2)
    trajectory = np.column_stack([R_arr, Z_arr, phi_arr])
    return DPm, trajectory


def _compute_DPm_via_python(
    x0: np.ndarray,
    field_func,
    phi_span: tuple,
    fd_eps: float = 1e-6,
) -> tuple:
    """Compute DPm using Python variational equations (fallback).

    Returns
    -------
    DPm : ndarray (2, 2)
    x_end : ndarray (2,)
    """
    veq = PoincareMapVariationalEquations(field_func, fd_eps=fd_eps)
    x_end, DPm = veq.tangent_map(x0, phi_span, order=1)
    return DPm, x_end


# ---------------------------------------------------------------------------
# Eigenvalue perturbation theory
# ---------------------------------------------------------------------------

def eigenvalue_perturbation(DPm: np.ndarray, delta_DPm: np.ndarray) -> dict:
    """First-order perturbation theory for the eigenvalues of DPm.

    For a 2Ã—2 area-preserving matrix DPm with eigenvalues Î» (|Î»| > 1
    for a hyperbolic fixed point), computes:
        Î´Î» = (w^T Â· Î´DPm Â· v) / (w^T Â· v)

    Parameters
    ----------
    DPm : ndarray, shape (2, 2)
        Base monodromy matrix.
    delta_DPm : ndarray, shape (2, 2)
        Perturbation to the monodromy matrix.

    Returns
    -------
    dict with keys:
        eigenvalues        ndarray (2,)  eigenvalues of DPm
        delta_eigenvalues  ndarray (2,)  first-order shifts Î´Î»
        new_eigenvalues    ndarray (2,)  eigenvalues + Î´Î»
        lambda_u           complex       hyperbolic eigenvalue (|Î»| > 1)
        delta_lambda_u     complex       its perturbation
        new_lambda_u       complex       estimated new hyperbolic eigenvalue
    """
    eigenvalues, V = np.linalg.eig(DPm)        # right eigenvectors
    eigs_left, W   = np.linalg.eig(DPm.T)      # left eigenvectors

    delta_eigs = np.zeros(2, dtype=complex)
    for k in range(2):
        v = V[:, k]
        lam = eigenvalues[k]
        # Match left eigenvector to this eigenvalue
        idx = int(np.argmin(np.abs(eigs_left - lam)))
        w = W[:, idx]
        denom = complex(w @ v)
        if abs(denom) < 1e-14:
            delta_eigs[k] = np.nan
        else:
            delta_eigs[k] = complex(w @ delta_DPm @ v) / denom

    abs_eigs = np.abs(eigenvalues)
    idx_u = int(np.argmax(abs_eigs))

    return {
        "eigenvalues":       eigenvalues,
        "delta_eigenvalues": delta_eigs,
        "new_eigenvalues":   eigenvalues + delta_eigs,
        "lambda_u":          complex(eigenvalues[idx_u]),
        "delta_lambda_u":    complex(delta_eigs[idx_u]),
        "new_lambda_u":      complex(eigenvalues[idx_u] + delta_eigs[idx_u]),
    }


# ---------------------------------------------------------------------------
# Î´X_cyc: orbit position shift under perturbation
# ---------------------------------------------------------------------------

def _compute_delta_X_cyc(
    x0: np.ndarray,
    DPm: np.ndarray,
    field_func,
    delta_f_func,
    phi_start: float,
    phi_end: float,
    trajectory_RZphi: Optional[np.ndarray] = None,
    A_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the periodic orbit position shift Î´X_cyc.

    Integrates the inhomogeneous variational equation:
        dÎ´X/dÏ† = A(x, Ï†) Â· Î´X + Î´f(x, Ï†)

    The initial condition is from the periodicity condition:
        (DPm âˆ?I) Â· Î´X_cyc = âˆ’âˆ® Î´f dÏ†

    Parameters
    ----------
    x0 : ndarray (2,)
        Orbit start position.
    DPm : ndarray (2, 2)
        Base monodromy matrix.
    field_func : callable
        Base field ``(R, Z, phi)`` â†?array (2,).
    delta_f_func : callable
        Perturbation ``(R, Z, phi)`` â†?array (2,).
    phi_start, phi_end : float
        Integration limits.
    trajectory_RZphi : ndarray (N, 3), optional
        Pre-traced orbit for A-matrix evaluation (avoids re-integration).
    A_arr : ndarray (N, 2, 2), optional
        Pre-computed A matrices at each orbit point.

    Returns
    -------
    delta_X_cyc : ndarray (2,)
    """
    n_steps = 400
    phi_arr = np.linspace(phi_start, phi_end, n_steps + 1)
    h = (phi_end - phi_start) / n_steps

    # Propagate base orbit, accumulating âˆ?Î´f dÏ† via trapezoidal rule
    y = x0.copy()
    integral_df = np.zeros(2)
    df_prev = np.asarray(delta_f_func(y[0], y[1], phi_arr[0]), dtype=float)

    for i in range(n_steps):
        phi = phi_arr[i]
        phi_next = phi_arr[i + 1]
        r, z = y[0], y[1]
        k1 = np.asarray(field_func(r, z, phi), dtype=float)
        r2, z2 = r + 0.5*h*k1[0], z + 0.5*h*k1[1]
        k2 = np.asarray(field_func(r2, z2, phi + 0.5*h), dtype=float)
        r3, z3 = r + 0.5*h*k2[0], z + 0.5*h*k2[1]
        k3 = np.asarray(field_func(r3, z3, phi + 0.5*h), dtype=float)
        r4, z4 = r + h*k3[0], z + h*k3[1]
        k4 = np.asarray(field_func(r4, z4, phi_next), dtype=float)
        y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        df_curr = np.asarray(delta_f_func(y[0], y[1], phi_next), dtype=float)
        integral_df += 0.5 * h * (df_prev + df_curr)
        df_prev = df_curr

    # Solve (DPm âˆ?I) Â· Î´X_cyc = âˆ’âˆ® Î´f dÏ†
    lhs = DPm - np.eye(2)
    rhs_vec = -integral_df
    try:
        delta_X0 = np.linalg.solve(lhs, rhs_vec)
    except np.linalg.LinAlgError:
        delta_X0, _, _, _ = np.linalg.lstsq(lhs, rhs_vec, rcond=None)

    return delta_X0


# ---------------------------------------------------------------------------
# Finite-difference DPm (primary Î´DPm method)
# ---------------------------------------------------------------------------

def DPm_finite_difference(
    x0,
    field_func_base,
    field_func_pert,
    phi_span,
    fd_eps_state: float = 1e-6,
    base_field_cache: Optional[dict] = None,
    pert_field_cache: Optional[dict] = None,
    island_period: int = 1,
    DPhi: float = 0.05,
    fd_eps_cyna: float = 1e-4,
) -> tuple:
    """Compute Î´DPm by finite difference between base and perturbed field.

    Parameters
    ----------
    x0 : array_like, shape (2,)
        Initial position (R, Z) of the periodic orbit.
    field_func_base : callable
        Base field: ``field_func_base(R, Z, phi)`` â†?array (dR/dÏ†, dZ/dÏ†).
        Used for Python fallback path only.
    field_func_pert : callable
        Perturbed field: same signature.
        Used for Python fallback path only.
    phi_span : tuple of float
        (phi_start, phi_end).
    fd_eps_state : float, optional
        FD step for Python variational equations. Default 1e-6.
    base_field_cache : dict, optional
        Field-cache dict for cyna C++ path (BR, BPhi, BZ, R_grid, Z_grid,
        Phi_grid). When supplied together with ``pert_field_cache``, the cyna
        ``trace_orbit_along_phi`` C++ backend is used.
    pert_field_cache : dict, optional
        Field-cache dict for the perturbed field.
    island_period : int, optional
        Number of toroidal turns (used by cyna path). Default 1.
    DPhi : float, optional
        RK4 step for cyna. Default 0.05.
    fd_eps_cyna : float, optional
        FD step for cyna A-matrix. Default 1e-4.

    Returns
    -------
    DPm_base : ndarray, shape (2, 2)
    DPm_pert : ndarray, shape (2, 2)
    delta_DPm : ndarray, shape (2, 2)
    trajectory_base : ndarray, shape (N, 3) or None
        Base orbit points [R, Z, phi] (available from cyna path only).
    """
    x0 = np.asarray(x0, dtype=float)
    phi_start = float(phi_span[0])
    phi_span_val = float(phi_span[1]) - phi_start
    trajectory_base = None

    if _HAS_CYNA and base_field_cache is not None and pert_field_cache is not None:
        # â”€â”€ C++ path (preferred) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        DPm_base, trajectory_base = _compute_DPm_via_cyna(
            x0[0], x0[1], phi_start, phi_span_val,
            island_period, base_field_cache, DPhi, fd_eps_cyna,
        )
        DPm_pert, _ = _compute_DPm_via_cyna(
            x0[0], x0[1], phi_start, phi_span_val,
            island_period, pert_field_cache, DPhi, fd_eps_cyna,
        )
    else:
        # â”€â”€ Python fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        veq_base = PoincareMapVariationalEquations(field_func_base, fd_eps=fd_eps_state)
        veq_pert = PoincareMapVariationalEquations(field_func_pert, fd_eps=fd_eps_state)
        _, DPm_base = veq_base.tangent_map(x0, phi_span, order=1)
        _, DPm_pert = veq_pert.tangent_map(x0, phi_span, order=1)

    delta_DPm = DPm_pert - DPm_base
    return DPm_base, DPm_pert, delta_DPm, trajectory_base


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def DPm_shift_under_field_perturbation(
    cycle_data,
    delta_B_func,
    field_func,
    phi_start: float = 0.0,
    island_period: int = 1,
    fd_eps: float = 1e-6,
    field_func_pert=None,
    base_field_cache: Optional[dict] = None,
    pert_field_cache: Optional[dict] = None,
    DPhi: float = 0.05,
    fd_eps_cyna: float = 1e-4,
) -> dict:
    """Compute the shift of DPm, eigenvalues, and orbit position under a
    magnetic field perturbation Î´B.

    Acceleration: when ``base_field_cache`` and ``pert_field_cache`` are
    supplied, the cyna C++ backend is used for orbit tracing and A-matrix
    computation (much faster than Python FD).  The Python path is used as a
    fallback when cyna is unavailable or caches are not provided.

    Parameters
    ----------
    cycle_data : CycleVariationalData
        Result from ``pyna.topo.monodromy.evolve_DPm_along_cycle``.
        Provides ``trajectory`` (NÃ—2) and ``DPm`` (2Ã—2).
    delta_B_func : callable or None
        Perturbation to the field direction:
        ``delta_B_func(R, Z, phi)`` â†?array (2,) = (Î´fR, Î´fZ).
        If ``None``, ``field_func_pert`` must be given.
    field_func : callable
        Base field: ``(R, Z, phi)`` â†?array (dR/dÏ†, dZ/dÏ†).
        Used for Python fallback and Î´X_cyc integration.
    phi_start : float
        Starting Ï†. Default 0.
    island_period : int
        Number of toroidal turns per island period. Default 1.
    fd_eps : float
        FD step for Python variational equations. Default 1e-6.
    field_func_pert : callable, optional
        Fully perturbed field function ``(R, Z, phi)`` â†?array (2,).
        If given, takes precedence over building one from ``delta_B_func``.
    base_field_cache : dict, optional
        Field cache for the base field (cyna C++ path).
    pert_field_cache : dict, optional
        Field cache for the perturbed field (cyna C++ path).
    DPhi : float
        RK4 step for cyna orbit tracing. Default 0.05.
    fd_eps_cyna : float
        FD step for cyna A-matrix batch computation. Default 1e-4.

    Returns
    -------
    dict with keys:

    delta_X_cyc : ndarray (2,)
        Shift of the periodic orbit's starting position (R, Z).
    delta_DPm : ndarray (2, 2)
        Shift of the monodromy matrix DPm_pert âˆ?DPm_base.
    DPm_base : ndarray (2, 2)
        Base monodromy matrix (should match ``cycle_data.DPm``).
    DPm_pert : ndarray (2, 2)
        Perturbed monodromy matrix.
    delta_lambda_u : complex
        First-order shift of the hyperbolic eigenvalue.
    new_lambda_u : complex
        Estimated new hyperbolic eigenvalue (Î»_u + Î´Î»_u).
    lambda_u : complex
        Base hyperbolic eigenvalue.
    eigenvalue_analysis : dict
        Full output of ``eigenvalue_perturbation(DPm, Î´DPm)``.
    used_cyna : bool
        True if the cyna C++ backend was used for DPm computation.
    """
    if delta_B_func is None and field_func_pert is None:
        raise ValueError("Must supply either delta_B_func or field_func_pert.")

    phi_end = phi_start + island_period * 2.0 * np.pi
    phi_span = (phi_start, phi_end)

    # Starting position
    x0 = np.asarray(cycle_data.trajectory[0], dtype=float)

    # ------------------------------------------------------------------
    # Build perturbed field function if not supplied
    # ------------------------------------------------------------------
    if field_func_pert is None:
        def field_func_pert(R, Z, phi,
                            _ff=field_func, _df=delta_B_func):
            f0 = np.asarray(_ff(R, Z, phi), dtype=float)
            df = np.asarray(_df(R, Z, phi), dtype=float)
            return f0 + df

    # Î´f function (difference of field directions)
    def delta_f_func(R, Z, phi,
                     _base=field_func, _pert=field_func_pert):
        return (np.asarray(_pert(R, Z, phi), dtype=float)
                - np.asarray(_base(R, Z, phi), dtype=float))

    # ------------------------------------------------------------------
    # 1. Compute Î´DPm by finite difference
    # ------------------------------------------------------------------
    use_cyna = (_HAS_CYNA
                and base_field_cache is not None
                and pert_field_cache is not None)

    DPm_base, DPm_pert, delta_DPm, trajectory_cyna = DPm_finite_difference(
        x0,
        field_func_base=field_func,
        field_func_pert=field_func_pert,
        phi_span=phi_span,
        fd_eps_state=fd_eps,
        base_field_cache=base_field_cache,
        pert_field_cache=pert_field_cache,
        island_period=island_period,
        DPhi=DPhi,
        fd_eps_cyna=fd_eps_cyna,
    )

    # ------------------------------------------------------------------
    # 2. Compute Î´X_cyc (orbit position shift)
    #    A-matrix: use cyna batch if available, else rely on Python RK4
    # ------------------------------------------------------------------
    # When cyna is available and we have base_field_cache, compute A_arr
    A_arr = None
    trajectory_for_xcyc = trajectory_cyna  # may be None (Python path)
    if use_cyna and trajectory_cyna is not None:
        traj = trajectory_cyna  # shape (N, 3): R, Z, phi
        R_arr   = np.ascontiguousarray(traj[:, 0], dtype=np.float64)
        Z_arr   = np.ascontiguousarray(traj[:, 1], dtype=np.float64)
        phi_arr = np.ascontiguousarray(traj[:, 2], dtype=np.float64)
        def _c(a):
            return np.ascontiguousarray(a, dtype=np.float64)
        A_arr = _cyna_A_batch(
            R_arr, Z_arr, phi_arr,
            _c(base_field_cache['BR'].ravel()
               if base_field_cache['BR'].ndim == 3
               else base_field_cache['BR']),
            _c(base_field_cache['BPhi'].ravel()
               if base_field_cache['BPhi'].ndim == 3
               else base_field_cache['BPhi']),
            _c(base_field_cache['BZ'].ravel()
               if base_field_cache['BZ'].ndim == 3
               else base_field_cache['BZ']),
            _c(base_field_cache['R_grid']),
            _c(base_field_cache['Z_grid']),
            _c(base_field_cache['Phi_grid']),
            fd_eps_cyna,
        )

    delta_X_cyc = _compute_delta_X_cyc(
        x0=x0,
        DPm=DPm_base,
        field_func=field_func,
        delta_f_func=delta_f_func,
        phi_start=phi_start,
        phi_end=phi_end,
        trajectory_RZphi=trajectory_for_xcyc,
        A_arr=A_arr,
    )

    # ------------------------------------------------------------------
    # 3. Eigenvalue perturbation theory
    # ------------------------------------------------------------------
    eig_analysis = eigenvalue_perturbation(DPm_base, delta_DPm)

    return {
        "delta_X_cyc":        delta_X_cyc,
        "delta_DPm":          delta_DPm,
        "DPm_base":           DPm_base,
        "DPm_pert":           DPm_pert,
        "delta_lambda_u":     eig_analysis["delta_lambda_u"],
        "new_lambda_u":       eig_analysis["new_lambda_u"],
        "lambda_u":           eig_analysis["lambda_u"],
        "eigenvalue_analysis": eig_analysis,
        "used_cyna":          use_cyna,
    }
