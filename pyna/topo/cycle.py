"""Periodic field-line orbits (fixed points of the Poincaré map).

A period-n cycle is a field line trajectory that returns to its starting
point after exactly n toroidal turns: X(φ + 2πn) = X(φ).

Finding cycles: Newton-Raphson on G(x0) = P^n(x0) - x0 = 0
where P is the Poincaré map (one-turn map).

Robustness: if Newton diverges (leaves domain), automatically try
fallback seed points from a grid around the initial guess.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class PeriodicOrbit:
    """A periodic field-line orbit.

    Attributes
    ----------
    rzphi0 : ndarray, shape (3,)
        Starting point (R, Z, phi0).
    period_n : int
        Number of toroidal turns (period of Poincaré map).
    trajectory : ndarray, shape (N, 3)
        Full orbit trajectory (R, Z, phi).
    monodromy : ndarray, shape (2, 2)
        Monodromy matrix M = J(2π·n). Eigenvalues characterize stability.
    """
    rzphi0: np.ndarray
    period_n: int
    trajectory: np.ndarray
    monodromy: np.ndarray

    @property
    def is_stable(self) -> bool:
        """True if |eigenvalues| ≤ 1 (elliptic, O-point type)."""
        eigvals = np.linalg.eigvals(self.monodromy)
        return bool(np.all(np.abs(eigvals) <= 1.0 + 1e-6))

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.monodromy)

    @property
    def stability_index(self) -> float:
        """Tr(M)/2 for a 2x2 symplectic map. |k|<1 → elliptic, |k|>1 → hyperbolic."""
        return float(np.trace(self.monodromy) / 2.0)


# ---------------------------------------------------------------------------
# Core integration helpers
# ---------------------------------------------------------------------------

def _field_func_phi_parameterized(field_func: Callable, rzphi: np.ndarray) -> np.ndarray:
    """Convert field_func(rzphi)→(dR/dl, dZ/dl, dphi/dl) to phi-parameterized.

    Returns (dR/dphi, dZ/dphi) = (dR/dl)/(dphi/dl).
    """
    f = np.asarray(field_func(rzphi), dtype=float)
    # f[2] = dphi/dl; f[0]/f[2] = dR/dphi; f[1]/f[2] = dZ/dphi
    dphi_dl = f[2]
    if abs(dphi_dl) < 1e-30:
        return np.array([0.0, 0.0])
    return np.array([f[0] / dphi_dl, f[1] / dphi_dl])


def poincare_map_n(
    field_func: Callable,
    rzphi0,
    n_turns: int,
    dt: float = 0.05,
    RZlimit: Optional[Tuple] = None,
) -> Tuple[float, float]:
    """Integrate field line for n toroidal turns, return final (R, Z).

    The integration uses the toroidal angle φ as the independent variable.

    Parameters
    ----------
    field_func : callable
        ``field_func(rzphi) → (dR/dl, dZ/dl, dphi/dl)``.
    rzphi0 : array-like (3,)
        Starting point (R, Z, phi).
    n_turns : int
        Number of toroidal turns to integrate (φ increases by 2π·n_turns).
    dt : float
        Step size in φ (radians) for the integrator.
    RZlimit : tuple or None
        Optional ``(R_min, R_max, Z_min, Z_max)`` domain boundary.

    Returns
    -------
    (R_final, Z_final) or (nan, nan) if field left domain.
    """
    rzphi0 = np.asarray(rzphi0, dtype=float)
    R0, Z0, phi0 = rzphi0[0], rzphi0[1], rzphi0[2]
    phi_end = phi0 + n_turns * 2.0 * np.pi

    def rhs(phi, rz):
        rzphi = np.array([rz[0], rz[1], phi])
        return _field_func_phi_parameterized(field_func, rzphi)

    events = []
    if RZlimit is not None:
        R_min, R_max, Z_min, Z_max = RZlimit

        def hit_boundary(phi, rz):
            return min(rz[0] - R_min, R_max - rz[0], rz[1] - Z_min, Z_max - rz[1])
        hit_boundary.terminal = True
        hit_boundary.direction = -1
        events.append(hit_boundary)

    try:
        sol = solve_ivp(
            rhs,
            (phi0, phi_end),
            [R0, Z0],
            method="DOP853",
            max_step=dt,
            rtol=1e-8,
            atol=1e-9,
            events=events if events else None,
        )
    except Exception:
        return float("nan"), float("nan")

    if not sol.success:
        return float("nan"), float("nan")

    # Check if boundary event triggered
    if events and sol.t_events[0].size > 0:
        return float("nan"), float("nan")

    return float(sol.y[0, -1]), float(sol.y[1, -1])


def poincare_map_n_trajectory(
    field_func: Callable,
    rzphi0,
    n_turns: int,
    dt: float = 0.05,
    RZlimit: Optional[Tuple] = None,
) -> np.ndarray:
    """Integrate field line for n_turns, return full (R, Z, phi) trajectory."""
    rzphi0 = np.asarray(rzphi0, dtype=float)
    R0, Z0, phi0 = rzphi0[0], rzphi0[1], rzphi0[2]
    phi_end = phi0 + n_turns * 2.0 * np.pi

    def rhs(phi, rz):
        rzphi = np.array([rz[0], rz[1], phi])
        return _field_func_phi_parameterized(field_func, rzphi)

    n_points = max(int((phi_end - phi0) / dt), 10)
    t_eval = np.linspace(phi0, phi_end, n_points)

    events = []
    if RZlimit is not None:
        R_min, R_max, Z_min, Z_max = RZlimit

        def hit_boundary(phi, rz):
            return min(rz[0] - R_min, R_max - rz[0], rz[1] - Z_min, Z_max - rz[1])
        hit_boundary.terminal = True
        hit_boundary.direction = -1
        events.append(hit_boundary)

    sol = solve_ivp(
        rhs,
        (phi0, phi_end),
        [R0, Z0],
        method="DOP853",
        max_step=dt,
        rtol=1e-8,
        atol=1e-9,
        t_eval=t_eval,
        events=events if events else None,
    )

    # Build (N, 3) trajectory
    traj = np.column_stack([sol.y[0], sol.y[1], sol.t])
    return traj


def jacobian_of_poincare_map(
    field_func: Callable,
    rzphi0,
    n_turns: int,
    dt: float = 0.05,
    eps: float = 1e-5,
) -> np.ndarray:
    """Finite-difference Jacobian ∂(R_f, Z_f)/∂(R_0, Z_0) of n-turn Poincaré map.

    Parameters
    ----------
    field_func : callable
    rzphi0 : array-like (3,)
    n_turns : int
    dt : float
        Integration step size.
    eps : float
        Finite-difference perturbation.

    Returns
    -------
    J : ndarray, shape (2, 2)
        Jacobian of the n-turn Poincaré map. det(J) ≈ 1 for area-preserving.
    """
    R0, Z0, phi0 = float(rzphi0[0]), float(rzphi0[1]), float(rzphi0[2])
    R_f, Z_f = poincare_map_n(field_func, rzphi0, n_turns, dt)

    # Perturb R
    R_fR, Z_fR = poincare_map_n(field_func, [R0 + eps, Z0, phi0], n_turns, dt)
    # Perturb Z
    R_fZ, Z_fZ = poincare_map_n(field_func, [R0, Z0 + eps, phi0], n_turns, dt)

    J = np.array([
        [(R_fR - R_f) / eps, (R_fZ - R_f) / eps],
        [(Z_fR - Z_f) / eps, (Z_fZ - Z_f) / eps],
    ])
    return J


# ---------------------------------------------------------------------------
# Cycle finder
# ---------------------------------------------------------------------------

def _try_find_cycle_from_seed(
    field_func: Callable,
    seed_rzphi: np.ndarray,
    n_turns: int,
    dt: float,
    RZlimit: Optional[Tuple],
    max_iter: int,
    tol: float,
    damping: float = 0.5,
) -> Optional[PeriodicOrbit]:
    """Internal Newton-Raphson cycle finder from a single seed."""
    x0 = np.array([seed_rzphi[0], seed_rzphi[1]], dtype=float)
    phi0 = float(seed_rzphi[2])

    for _ in range(max_iter):
        rzphi = np.array([x0[0], x0[1], phi0])
        R_f, Z_f = poincare_map_n(field_func, rzphi, n_turns, dt, RZlimit)

        if np.isnan(R_f) or np.isnan(Z_f):
            return None

        G = np.array([R_f - x0[0], Z_f - x0[1]])
        if np.linalg.norm(G) < tol:
            # Converged — compute trajectory and monodromy
            traj = poincare_map_n_trajectory(field_func, rzphi, n_turns, dt, RZlimit)
            J = jacobian_of_poincare_map(field_func, rzphi, n_turns, dt)
            return PeriodicOrbit(
                rzphi0=rzphi.copy(),
                period_n=n_turns,
                trajectory=traj,
                monodromy=J,
            )

        # Jacobian of G = P^n - I: dG/dx0 = J_poincare - I
        J_p = jacobian_of_poincare_map(field_func, rzphi, n_turns, dt)
        dGdx = J_p - np.eye(2)

        try:
            delta = np.linalg.solve(dGdx, -G)
        except np.linalg.LinAlgError:
            return None

        x0 = x0 + damping * delta

        # Domain check
        if RZlimit is not None:
            R_min, R_max, Z_min, Z_max = RZlimit
            if not (R_min < x0[0] < R_max and Z_min < x0[1] < Z_max):
                return None
        # Also cap step size to avoid wild Newton excursions
        step_norm = np.linalg.norm(delta)
        if step_norm > 0.1:
            return None

    return None


def find_cycle(
    field_func: Callable,
    init_rzphi: np.ndarray,
    n_turns: int = 1,
    dt: float = 0.05,
    RZlimit: Optional[Tuple] = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    n_fallback_seeds: int = 12,
    fallback_radius: float = 0.05,
) -> Optional[PeriodicOrbit]:
    """Find a periodic orbit starting from init_rzphi using Newton-Raphson.

    G(x0) = P^n(x0) - x0 = 0

    If Newton diverges or leaves domain, automatically tries n_fallback_seeds
    alternative starting points distributed on a circle of radius fallback_radius
    around init_rzphi.

    Parameters
    ----------
    field_func : callable
        Field function ``f(rzphi) → (dR/dl, dZ/dl, dphi/dl)``.
    init_rzphi : array (3,)
        Initial guess (R0, Z0, phi0).
    n_turns : int
        Period (number of toroidal turns).
    dt : float
        Integration step size in φ.
    RZlimit : tuple or None
        Domain limits (R_min, R_max, Z_min, Z_max).
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on |G(x0)|.
    n_fallback_seeds : int
        Number of fallback seeds to try if primary Newton fails.
    fallback_radius : float
        Radius around init_rzphi for fallback seeds.

    Returns
    -------
    PeriodicOrbit or None if not found.
    """
    init_rzphi = np.asarray(init_rzphi, dtype=float)

    # Primary attempt
    orbit = _try_find_cycle_from_seed(
        field_func, init_rzphi, n_turns, dt, RZlimit, max_iter, tol
    )
    if orbit is not None:
        return orbit

    # Fallback: try seeds on a circle around the initial guess
    R0, Z0, phi0 = init_rzphi[0], init_rzphi[1], init_rzphi[2]
    angles = np.linspace(0, 2 * np.pi, n_fallback_seeds, endpoint=False)
    for ang in angles:
        seed = np.array([
            R0 + fallback_radius * np.cos(ang),
            Z0 + fallback_radius * np.sin(ang),
            phi0,
        ])
        orbit = _try_find_cycle_from_seed(
            field_func, seed, n_turns, dt, RZlimit, max_iter, tol
        )
        if orbit is not None:
            return orbit

    return None


# ---------------------------------------------------------------------------
# Find all cycles near a resonant surface
# ---------------------------------------------------------------------------

def find_all_cycles_near_resonance(
    field_func: Callable,
    equilibrium,
    m: int,
    n: int,
    n_seeds: int = 8,
    dt: float = 0.05,
    RZlimit: Optional[Tuple] = None,
) -> list:
    """Find all O- and X-point cycles near the q=m/n resonant surface.

    For a q = m/n resonance, there are m O-points and m X-points at
    equally spaced angular positions around the resonant surface. This
    function seeds the Newton-Raphson solver at 2·m·n_seeds angular
    positions around the resonant flux surface and deduplicates the
    resulting orbits.

    Parameters
    ----------
    field_func : callable
    equilibrium : SimpleStellarartor or similar
        Must have ``resonant_psi(m, n)`` and ``R0``, ``r0`` attributes.
    m, n : int
        Mode numbers defining the resonance q = m/n.
    n_seeds : int
        Number of angular seeds per expected fixed point.
    dt : float
        Integration step.
    RZlimit : tuple or None

    Returns
    -------
    list of PeriodicOrbit
    """
    psi_list = equilibrium.resonant_psi(m, n)
    if not psi_list:
        warnings.warn(f"q={m}/{n} resonance not found in equilibrium")
        return []

    psi_res = psi_list[0]
    r_res = float(np.sqrt(psi_res)) * equilibrium.r0
    R0 = equilibrium.R0

    # Seed angles: 2*m angular positions around the resonant circle
    total_seeds = 2 * m * n_seeds
    seed_angles = np.linspace(0, 2 * np.pi, total_seeds, endpoint=False)

    found_orbits: list[PeriodicOrbit] = []

    # For this stellarator model, q = m/n (tokamak convention: q = toroidal/poloidal).
    # The orbit period in toroidal turns equals m (the numerator), not n.
    # dθ/dφ = 1/q = n/m → after m toroidal turns, θ advances by 2πn → closed.
    orbit_period = m

    for theta in seed_angles:
        seed = np.array([
            R0 + r_res * np.cos(theta),
            r_res * np.sin(theta),
            0.0,
        ])
        orbit = find_cycle(
            field_func, seed, n_turns=orbit_period, dt=dt, RZlimit=RZlimit,
            max_iter=50, tol=1e-8,
            n_fallback_seeds=6, fallback_radius=0.3 * r_res,
        )
        if orbit is None:
            continue

        # Deduplicate: check if this orbit is already in the list
        duplicate = False
        for existing in found_orbits:
            dist = np.linalg.norm(orbit.rzphi0[:2] - existing.rzphi0[:2])
            if dist < 1e-4:
                duplicate = True
                break
        if not duplicate:
            found_orbits.append(orbit)

    return found_orbits
