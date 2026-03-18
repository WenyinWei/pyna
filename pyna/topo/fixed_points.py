"""Tools for locating and refining periodic orbits (X- and O-points) of
the Poincaré map of a magnetic field in cylindrical coordinates.

The Poincaré map P^n maps an initial condition (R, Z) at toroidal
angle φ=0 to its position after n full toroidal turns. Fixed points
of P^n satisfy P^n(x) = x and correspond to magnetic island X-points
(hyperbolic) and O-points (elliptic).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.signal import argrelmin


# Type alias for 2D field function: (R, Z, phi) -> [dR/dphi, dZ/dphi]
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
    """Integrate n_turns toroidal turns and return endpoint (R, Z).

    Parameters
    ----------
    x0 : array-like, shape (2,)
        Initial condition [R, Z].
    field_func : callable
        2D field function (R, Z, phi) -> [dR/dphi, dZ/dphi].
    n_turns : int
        Number of toroidal turns.
    phi_start : float
        Starting toroidal angle (rad).

    Returns
    -------
    ndarray, shape (2,)
        Endpoint [R, Z] after n_turns.
    """
    x0 = np.asarray(x0, dtype=float)

    def rhs(phi, y):
        return field_func(y[0], y[1], phi)

    sol = solve_ivp(
        rhs,
        [phi_start, phi_start + 2.0 * np.pi * n_turns],
        x0,
        method=method,
        rtol=rtol,
        atol=atol,
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
    """Scan a ring of points and return seeds near period-n fixed points.

    Evaluates |P^n(x) - x| along a ring of radius r_scan centred at
    (R_center, Z_center) and returns the positions of local minima,
    sorted by residual magnitude (best candidates first).

    Parameters
    ----------
    R_center, Z_center : float
        Centre of the search ring (m). Typically the rational surface
        location (R0 + r_res, 0) for an island chain.
    r_scan : float
        Ring radius (m). Typically the minor radius of the rational surface.
    n_turns : int
        Orbit period (n in P^n).
    n_scan : int
        Number of points on the ring.
    order : int
        Half-window size for local minimum detection.

    Returns
    -------
    list of ndarray
        Candidate seed points, sorted by |P^n(x)-x|. Each element is
        shape (2,) array [R, Z].
    """
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
    """Refine a seed to a true period-n fixed point via Newton's method.

    Parameters
    ----------
    seed : array-like, shape (2,)
        Initial guess for the fixed point.
    field_func : callable
        2D field function.
    n_turns : int
        Orbit period.
    tol : float
        Convergence tolerance for the root finder.
    maxfev : int
        Maximum function evaluations.

    Returns
    -------
    ndarray shape (2,) if converged, None otherwise.
    """
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
    """Find all period-n fixed points near a given seed.

    This is the main user-facing function. It combines ring scanning and
    Newton refinement to locate all period-n fixed points (both X and O
    type) near the seed location.

    Parameters
    ----------
    field_func : callable
        2D field (R, Z, phi) -> [dR/dphi, dZ/dphi].
    seed : array-like, shape (2,)
        Approximate location of an island chain (e.g. a point on the
        rational surface).
    n_turns : int
        Island chain period (toroidal mode number n for q = m/n islands).
    r_scan : float or None
        Ring scan radius. If None, estimated as 10% of distance from
        seed to the magnetic axis (or 0.1 m if that fails).
    n_scan : int
        Points on the scan ring.
    tol, maxfev : float, int
        Newton solver parameters.
    rtol, atol : float
        ODE integration tolerances.
    dedup_tol : float
        Distance threshold for de-duplicating fixed points (m).
    verbose : bool
        Print progress messages.

    Returns
    -------
    list of ndarray
        De-duplicated list of converged fixed points [R, Z].
        Typically contains 2n points (n X-points + n O-points for an m/n island).

    Examples
    --------
    >>> from pyna.topo.fixed_points import find_periodic_orbit
    >>> fps = find_periodic_orbit(field_func_2d, seed=[3.07, 0.07], n_turns=3, r_scan=0.07)
    >>> print(f"Found {len(fps)} fixed points")
    """
    seed = np.asarray(seed, dtype=float)
    R_center, Z_center = float(seed[0]), float(seed[1])

    if r_scan is None:
        # heuristic: use 15% of seed distance from cylinder axis
        r_scan = max(0.03, 0.15 * np.sqrt(R_center**2 + Z_center**2) / 10)
        if verbose:
            print(f"r_scan not given; using {r_scan:.4f} m")

    if verbose:
        print(f"Scanning ring: centre=({R_center:.4f}, {Z_center:.4f}), "
              f"r={r_scan:.4f} m, n_turns={n_turns}, n_scan={n_scan}")

    seeds = scan_fixed_point_seeds(
        field_func, R_center, Z_center, r_scan, n_turns,
        n_scan=n_scan, order=5
    )

    if verbose:
        print(f"  {len(seeds)} candidate seeds found")

    fixed_points: List[np.ndarray] = []
    for s in seeds:
        fp = refine_fixed_point(s, field_func, n_turns,
                                tol=tol, maxfev=maxfev, rtol=rtol, atol=atol)
        if fp is not None:
            # De-duplicate
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
    """Classify a fixed point as X-point or O-point from its monodromy matrix.

    Parameters
    ----------
    fp : array-like, shape (2,)
        Fixed point to classify (must be a true fixed point to machine precision).
    field_func : callable
        2D field function.
    n_turns : int
        Orbit period.
    fd_eps : float
        Finite-difference step for monodromy matrix computation.
    rtol, atol : float
        ODE tolerances.

    Returns
    -------
    (fp_type, monodromy, det_J) : tuple
        fp_type : str
            'X' for hyperbolic (X-point), 'O' for elliptic (O-point),
            'unknown' if eigenvalues are complex or det deviates from 1.
        monodromy : ndarray, shape (2, 2)
            The monodromy matrix DP^n.
        det_J : float
            Determinant of the monodromy matrix (should be 1.0).
    """
    from pyna.topo.variational import PoincareMapVariationalEquations

    fp = np.asarray(fp, dtype=float)

    def _field_func_2d_wrap(R, Z, phi):
        return field_func(R, Z, phi)

    vq = PoincareMapVariationalEquations(_field_func_2d_wrap, fd_eps=fd_eps)
    phi_span = (0.0, 2.0 * np.pi * n_turns)
    J = vq.jacobian_matrix(
        fp, phi_span,
        solve_ivp_kwargs=dict(method="DOP853", rtol=rtol, atol=atol)
    )

    det_J = float(np.linalg.det(J))
    eigvals = np.linalg.eigvals(J)
    lam_abs = sorted(np.abs(eigvals))

    # X-point: real eigenvalues with |λ_s| < 1 < |λ_u|
    # O-point: complex conjugate eigenvalues on unit circle
    if np.all(np.isreal(eigvals)):
        lam_real = np.sort(np.real(eigvals))
        if lam_real[0] < 0.99 and lam_real[1] > 1.01:
            fp_type = "X"
        else:
            fp_type = "unknown"
    else:
        # Check if on unit circle (elliptic)
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
    """Extended fixed-point classifier handling degenerate and non-conservative cases.

    This function goes beyond the standard X/O dichotomy by handling:

    1. **Non-area-preserving maps** (det(DP^n) ≠ 1): classifies fixed points as
       ``'sink'``, ``'source'``, or ``'saddle'``, covering e.g. dissipative or
       non-divergence-free vector fields.
    2. **Higher-order (parabolic / degenerate) fixed points** where DP^n ≈ I
       and the first-order monodromy does not discriminate the neighbourhood.
       In this case the order-2 tensor T = D²P^n is computed and used.

    The type hierarchy (from most specific to most general):

    ============  =======  ===========================================================
    Type string   det(J)   Eigenvalue pattern
    ============  =======  ===========================================================
    ``'O'``       ≈ 1      complex conjugate pair on unit circle (elliptic)
    ``'X'``       ≈ 1      real, one < 1 and one > 1 (hyperbolic)
    ``'degenerate_O'``  ≈ 1  both eigenvalues ≈ +1; T indicates elliptic neighbourhood
    ``'degenerate_X'``  ≈ 1  both eigenvalues ≈ +1; T indicates hyperbolic neighbourhood
    ``'parabolic'``  ≈ 1   J ≈ I, T is zero or inconclusive
    ``'sink'``    < 1      all |λ| < 1 (contracting; non-conservative)
    ``'source'``  > 1      all |λ| > 1 (expanding; non-conservative)
    ``'saddle'``  ≈ ±1     one |λ| < 1, one |λ| > 1 (non-area-preserving saddle)
    ``'unknown'`` —        none of the above categories
    ============  =======  ===========================================================

    Parameters
    ----------
    fp : array-like, shape (2,)
        Fixed point [R, Z] (should satisfy P^n(fp) = fp to good precision).
    field_func : callable
        2D field function ``(R, Z, phi) → [dR/dφ, dZ/dφ]``.
    n_turns : int
        Orbit period (number of full turns in the Poincaré map).
    parabolic_tol : float
        Tolerance on ``||J - I||_F`` for declaring the point parabolic
        (J ≈ I).  Default 0.01.
    fd_eps : float
        Finite-difference step for the Jacobian A = ∂f/∂x (order-1).
    fd_eps2 : float
        Finite-difference step for the Hessian H = ∂²f/∂x² (order-2).
    rtol, atol : float
        ODE integration tolerances.

    Returns
    -------
    result : dict with keys
        ``type`` : str
            Fixed-point type string (see table above).
        ``J`` : ndarray, shape (2, 2)
            Monodromy matrix DP^n.
        ``T`` : ndarray, shape (2, 2, 2) or None
            Second-derivative tensor D²P^n.  Computed only when J ≈ I;
            ``None`` otherwise.
        ``det_J`` : float
            Determinant of J.
        ``eigenvalues`` : ndarray, shape (2,)
            Eigenvalues of J.
        ``stability_index`` : float
            Tr(J) / 2 (for 2×2 symplectic maps: |k| < 1 ↔ elliptic).

    Notes
    -----
    The degenerate-O / degenerate-X discrimination from T uses the
    criterion that the Hessian of the effective Hamiltonian at the fixed
    point is positive-definite (O-type) or indefinite (X-type).  For the
    2×2 Poincaré map, this is assessed via the sign of
    ``T[0, 0, 0] * T[0, 1, 1] - T[0, 0, 1]²`` (a leading-order
    discriminant of the second-order normal form).

    Examples
    --------
    >>> info = classify_fixed_point_higher_order(fp, field_func, n_turns=3)
    >>> info['type']
    'O'
    >>> info['stability_index']   # doctest: +SKIP
    0.42
    """
    from pyna.topo.variational import PoincareMapVariationalEquations

    fp = np.asarray(fp, dtype=float)
    phi_span = (0.0, 2.0 * np.pi * n_turns)

    vq = PoincareMapVariationalEquations(
        field_func, fd_eps=fd_eps, fd_eps2=fd_eps2
    )
    ivp_kw = dict(method="DOP853", rtol=rtol, atol=atol)

    # --- order-1 monodromy ---
    J = vq.jacobian_matrix(fp, phi_span, solve_ivp_kwargs=ivp_kw)
    det_J = float(np.linalg.det(J))
    eigvals = np.linalg.eigvals(J)
    lam_abs = np.abs(eigvals)
    stability_index = float(np.trace(J) / 2.0)

    result = dict(J=J, T=None, det_J=det_J,
                  eigenvalues=eigvals, stability_index=stability_index)

    # Shared tolerance for det and eigenvalue magnitude deviations from 1.
    _det_tol = 0.05

    # ── 1. Non-conservative branch (det deviates significantly from ±1) ──
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

    # ── 2. Area-preserving branch ──
    # Check for parabolic / higher-order (J ≈ I or J ≈ -I)
    J_minus_I  = np.linalg.norm(J - np.eye(2), "fro")
    J_plus_I   = np.linalg.norm(J + np.eye(2), "fro")
    is_near_I  = J_minus_I  < parabolic_tol
    is_near_mI = J_plus_I   < parabolic_tol

    if is_near_I or is_near_mI:
        # Compute order-2 tensor to discriminate degenerate type
        _, _, T = vq.tangent_map(fp, phi_span, order=2, solve_ivp_kwargs=ivp_kw)
        result["T"] = T

        # Discriminant: for component i=0, check if T is sign-definite in jk
        # Effective Hessian H_jk = T[0, j, k]  (leading component)
        H_eff = T[0]   # shape (2, 2)
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

    # ── 3. Standard area-preserving classification ──
    if np.all(np.isreal(eigvals)):
        lam_real = np.sort(np.real(eigvals))
        if lam_real[0] < 1.0 - _det_tol and lam_real[1] > 1.0 + _det_tol:
            result["type"] = "X"
        else:
            result["type"] = "unknown"
    else:
        # Elliptic: complex conjugate pair on the unit circle.
        # Use the same _det_tol for eigenvalue magnitude deviation from 1.
        if (abs(lam_abs[0] - 1.0) < _det_tol and abs(lam_abs[1] - 1.0) < _det_tol):
            result["type"] = "O"
        else:
            result["type"] = "unknown"

    return result
