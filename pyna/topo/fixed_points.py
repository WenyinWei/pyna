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
