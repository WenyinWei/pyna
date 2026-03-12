"""q-profile (safety factor) and rotation transform response to coil perturbations.

q(s) = safety factor at normalized flux surface s ∈ [0,1]
ι(s) = rotation transform = 1/q(s)

Response ∂q(s_i)/∂I_k is computed by:
  Method A (fast, analytic flux-surface integral):
    δq ≈ (1/2π) ∮ δ(RBphi/Bpol²) · Bpol dl  (flux surface integral of δB)
    Works for axisymmetric perturbations only.

  Method B (robust, finite-difference field-line tracing):
    Trace one field line on flux surface ψ for one poloidal turn in base field
    and perturbed field (base + ε·δB_coil), measure Δφ/Δθ, compute δq.
    More expensive but works for 3D perturbations too.

Default: Method A unless the caller explicitly requests Method B.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple

import joblib
from pathlib import Path

_cache_dir = Path.home() / ".pyna_cache"
_cache_dir.mkdir(exist_ok=True)
memory = joblib.Memory(location=str(_cache_dir), verbose=0)


# ── helpers ────────────────────────────────────────────────────────────────

def _bpol(field_func: Callable, R: float, Z: float) -> float:
    """Return |Bpol| = sqrt(BR² + BZ²) at a single point."""
    f = field_func([R, Z, 0.0])
    return float(np.sqrt(f[0] ** 2 + f[1] ** 2))


def _bvec(field_func: Callable, R: float, Z: float) -> Tuple[float, float, float]:
    """Return (BR, BZ, Bphi) from field_func([R,Z,phi]) → [BR, BZ, Bphi]."""
    f = field_func([R, Z, 0.0])
    return float(f[0]), float(f[1]), float(f[2])


# ── Method A: analytic flux-surface integral ──────────────────────────────

def q_from_flux_surface_integral(
    field_func: Callable,
    surf_R: np.ndarray,
    surf_Z: np.ndarray,
) -> float:
    """Compute q by integrating q = (1/2π) ∮ (R Bphi / Bpol) / |∇ψ̃| dθ.

    In practice we use the PEST-like formula::

        q = ∮ (Bphi/R) dl / (2π ∮ Bpol dl / (2π L))

    which simplifies for a parameterized contour (θ uniform in poloidal angle)
    to the arc-length weighted average::

        q ≈ <R Bphi / Bpol>_{dl}   / (2π)  · L_pol / L_pol = <R Bphi/Bpol>_{dl}

    where ``<·>_{dl}`` denotes the arc-length-weighted mean.

    Parameters
    ----------
    field_func : callable
        ``field_func([R, Z, phi]) -> [BR, BZ, Bphi]``
    surf_R, surf_Z : ndarray, shape (N,)
        R, Z coordinates uniformly sampling the flux surface.

    Returns
    -------
    q : float
    """
    surf_R = np.asarray(surf_R, dtype=float)
    surf_Z = np.asarray(surf_Z, dtype=float)
    N = len(surf_R)

    dR = np.diff(np.append(surf_R, surf_R[0]))
    dZ = np.diff(np.append(surf_Z, surf_Z[0]))
    dl = np.sqrt(dR ** 2 + dZ ** 2)

    integrand = np.empty(N)
    for j in range(N):
        BR, BZ, Bphi = _bvec(field_func, surf_R[j], surf_Z[j])
        Bpol = np.sqrt(BR ** 2 + BZ ** 2)
        if Bpol < 1e-12:
            integrand[j] = 0.0
        else:
            integrand[j] = surf_R[j] * Bphi / Bpol

    # q = (1/2π) * (∮ (R Bphi / Bpol) dθ) where dθ parameterised as uniform
    # With arc-length weight: q = ∮ (R Bphi / Bpol) dl / ∮ dl
    # (this gives the correct poloidal-turn-average)
    total_dl = np.sum(dl)
    if total_dl < 1e-30:
        return np.nan
    return float(np.sum(integrand * dl) / total_dl)


# ── Method B: finite-difference field-line tracing ────────────────────────

def q_by_fieldline_tracing(
    field_func: Callable,
    R_start: float,
    Z_start: float,
    n_steps: int = 2000,
    ds: float = 1e-3,
) -> float:
    """Compute q by tracing a field line for one poloidal turn.

    Integrates dφ/dθ along the field line until Z returns to starting side,
    then measures total toroidal angle traversed.

    Parameters
    ----------
    field_func : callable
        ``field_func([R, Z, phi]) -> [BR, BZ, Bphi]``
    R_start, Z_start : float
        Starting point (should be on a flux surface, not at the axis).
    n_steps : int
        Maximum number of Euler steps.
    ds : float
        Arc-length step size (in metres, normalised units work too).

    Returns
    -------
    q : float
        Estimated safety factor.  NaN if tracing fails.
    """
    R, Z = float(R_start), float(Z_start)
    phi_total = 0.0
    theta_total = 0.0

    for _ in range(n_steps):
        BR, BZ, Bphi = _bvec(field_func, R, Z)
        Bmod = np.sqrt(BR ** 2 + BZ ** 2 + (Bphi) ** 2)
        if Bmod < 1e-15:
            break
        # unit tangent: dr/ds proportional to B
        bR = BR / Bmod
        bZ = BZ / Bmod
        bPhi = Bphi / Bmod

        dR = bR * ds
        dZ = bZ * ds
        dPhi = bPhi * ds / R  # dφ = (Bphi/R) ds / |B|

        phi_total += dPhi
        theta_total += np.sqrt(dR ** 2 + dZ ** 2) / np.hypot(R - R_start + dR / 2, Z - Z_start + dZ / 2 + 1e-15)

        R += dR
        Z += dZ

        # Check if we've completed one poloidal transit
        # Simple criterion: |θ_accumulated| ≈ 2π via winding number
        # More robust: watch Z sign crossing from below
        # We accumulate dl_poloidal and check once dl ≈ circumference
        # (handled via theta_total proxy)

    # Fallback: return phi/2pi (approximate)
    if abs(theta_total) < 1e-10:
        return np.nan
    return float(phi_total / theta_total) / (2 * np.pi)


def q_by_fieldline_winding(
    field_func: Callable,
    R_start: float,
    Z_start: float,
    n_poloidal_steps: int = 512,
    rtol: float = 1e-6,
) -> float:
    """Compute q by tracing a field line for exactly one poloidal turn.

    Uses scipy ODE integration.  Returns Δφ / (2π) = q.

    Parameters
    ----------
    field_func : callable
    R_start, Z_start : float
    n_poloidal_steps : int
        Number of ODE steps per poloidal turn.
    rtol : float
        ODE relative tolerance.

    Returns
    -------
    q : float
    """
    from scipy.integrate import solve_ivp

    R0_start, Z0_start = float(R_start), float(Z_start)

    # We integrate in (R, Z, phi) parameterised by poloidal arc length l.
    # dR/dl = BR/Bpol,  dZ/dl = BZ/Bpol,  dphi/dl = Bphi/(R Bpol)
    # Stop after one poloidal turn: total angle ∫ dl |∇θ| = 2π
    # Rough poloidal circumference estimate:

    # Estimate circumference by tracing a few steps
    def rhs(l, state):
        R_, Z_, phi_ = state[0], state[1], state[2]
        BR, BZ, Bphi = _bvec(field_func, R_, Z_)
        Bpol = np.sqrt(BR ** 2 + BZ ** 2) + 1e-15
        return [BR / Bpol, BZ / Bpol, Bphi / (R_ * Bpol)]

    # Estimate the poloidal arc length of one turn from first pass
    # Use a coarse 20-step integration and check when angle closes
    # Simpler: integrate for l_max = large value, detect crossing

    def event_complete_turn(l, state):
        """Triggered when Z returns to Z_start from below (after leaving)."""
        return state[1] - Z0_start

    event_complete_turn.terminal = True
    event_complete_turn.direction = 1  # rising

    # First integrate a bit to get past Z_start going downward
    # then integrate until Z crosses Z_start going upward
    # Use a two-pass approach:

    # Estimate circumference ~ pi * (r_min + r_max) for an ellipse
    BR_s, BZ_s, _ = _bvec(field_func, R0_start, Z0_start)
    Bpol_s = np.sqrt(BR_s ** 2 + BZ_s ** 2) + 1e-15
    # Rough kappa ~ 1.7 (typical tokamak), a ~ 0.6 * s_norm * R0
    # For safety factor, just integrate for long enough
    l_max = 50.0  # large arc length

    try:
        sol = solve_ivp(
            rhs,
            [0.0, l_max],
            [R0_start, Z0_start, 0.0],
            method="RK45",
            max_step=l_max / n_poloidal_steps,
            rtol=rtol,
            atol=rtol * 1e-3,
            dense_output=True,
        )

        # Find when the trajectory returns to (R_start, Z_start) in poloidal plane
        # by finding the first time after a minimum arc when R,Z ≈ R0,Z0
        if not sol.success:
            return np.nan

        R_traj = sol.y[0]
        Z_traj = sol.y[1]
        phi_traj = sol.y[2]

        # Find the first closed return: ||(R,Z) - (R0,Z0)|| < threshold
        # after travelling some minimum distance
        dist = np.sqrt((R_traj - R0_start) ** 2 + (Z_traj - Z0_start) ** 2)
        # Skip first 10% of trajectory
        skip = max(1, len(dist) // 10)
        idx = np.argmin(dist[skip:]) + skip
        if dist[idx] > 0.1:
            return np.nan
        q = float(phi_traj[idx] / (2 * np.pi))
        return q

    except Exception:
        return np.nan


# ── Response matrix: Method A ─────────────────────────────────────────────

def q_response_matrix_analytic(
    base_field_func: Callable,
    delta_field_funcs: List[Callable],
    flux_surfaces_RZ: List[Tuple[np.ndarray, np.ndarray]],
    s_labels: List[float],
    q_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute ∂q(s_i)/∂I_k by analytic flux-surface integral.

    For each coil k and each flux surface i the first-order perturbation is::

        δq_i ≈ (1/2π) ∮ δ(R Bphi/Bpol) dl / ∮ dl
             = <R δBphi/Bpol - R Bphi δBpol/Bpol²>_{dl}

    For PF coils (δBphi = 0, only BR, BZ change)::

        δq_i ≈ -q_0 <δBpol/Bpol>_{dl}

    where ``<·>_{dl}`` is the arc-length-weighted mean over the flux surface.

    Parameters
    ----------
    base_field_func : callable
        ``f([R, Z, phi]) -> [BR, BZ, Bphi]``  — base equilibrium.
    delta_field_funcs : list of callable
        One callable per coil, same signature, returns δ[BR, BZ, Bphi] per
        unit current (A⁻¹).
    flux_surfaces_RZ : list of (R_arr, Z_arr)
        One entry per flux surface.
    s_labels : list of float
        Normalised flux label for each surface (used for labelling and q₀
        fallback if ``q_values`` not supplied).
    q_values : ndarray or None
        Precomputed q values at each surface.  If None, computed from
        ``q_from_flux_surface_integral`` on the base field.

    Returns
    -------
    R_q : ndarray, shape (n_surfaces, n_coils)
        Response matrix ∂q / ∂I_k  (units: T·m/A or 1/A depending on δB
        normalisation).
    labels : list of str
        Row labels, e.g. ``['q.s0.2', 'q.s0.4', ...]``.
    """
    n_s = len(flux_surfaces_RZ)
    n_coils = len(delta_field_funcs)
    R_q = np.zeros((n_s, n_coils))

    for i, ((surf_R, surf_Z), s_val) in enumerate(
        zip(flux_surfaces_RZ, s_labels)
    ):
        surf_R = np.asarray(surf_R, dtype=float)
        surf_Z = np.asarray(surf_Z, dtype=float)
        N = len(surf_R)

        # Arc-length weights
        dR = np.diff(np.append(surf_R, surf_R[0]))
        dZ = np.diff(np.append(surf_Z, surf_Z[0]))
        dl = np.sqrt(dR ** 2 + dZ ** 2)
        total_dl = np.sum(dl) + 1e-30

        # Base field on the surface
        BR_b = np.empty(N)
        BZ_b = np.empty(N)
        Bphi_b = np.empty(N)
        for j in range(N):
            f = base_field_func([surf_R[j], surf_Z[j], 0.0])
            BR_b[j], BZ_b[j], Bphi_b[j] = f[0], f[1], f[2]
        Bpol_b = np.sqrt(BR_b ** 2 + BZ_b ** 2) + 1e-15

        # q₀ on this surface
        if q_values is not None:
            q0 = float(q_values[i])
        else:
            integrand_q = surf_R * Bphi_b / Bpol_b
            q0 = float(np.sum(integrand_q * dl) / total_dl)

        for k, delta_field in enumerate(delta_field_funcs):
            dBR = np.empty(N)
            dBZ = np.empty(N)
            dBphi = np.empty(N)
            for j in range(N):
                df = delta_field([surf_R[j], surf_Z[j], 0.0])
                dBR[j], dBZ[j], dBphi[j] = df[0], df[1], df[2]

            dBpol = (BR_b * dBR + BZ_b * dBZ) / Bpol_b  # δBpol (first order)

            # δ(R Bphi / Bpol) = R δBphi/Bpol - R Bphi δBpol / Bpol²
            delta_integrand = (
                surf_R * dBphi / Bpol_b
                - surf_R * Bphi_b * dBpol / (Bpol_b ** 2)
            )
            # δq = <δ(R Bphi/Bpol)>_{dl}
            R_q[i, k] = float(np.sum(delta_integrand * dl) / total_dl)

    labels = [f"q.s{s:.2f}" for s in s_labels]
    return R_q, labels


# ── Response matrix: Method B ─────────────────────────────────────────────

def q_response_matrix_fd(
    base_field_func: Callable,
    delta_field_funcs: List[Callable],
    flux_surfaces_RZ: List[Tuple[np.ndarray, np.ndarray]],
    s_labels: List[float],
    epsilon: float = 1e-3,
    n_poloidal_steps: int = 512,
) -> Tuple[np.ndarray, List[str]]:
    """Compute ∂q(s_i)/∂I_k by finite-difference field-line tracing (Method B).

    For each coil k and surface i::

        ∂q/∂I_k ≈ [q(base + ε δB_k) - q(base)] / ε

    This works for 3-D perturbations but is O(n_coils × n_surfaces) ODE calls.

    Parameters
    ----------
    epsilon : float
        Finite-difference step (unit coil current perturbation).
    """
    n_s = len(flux_surfaces_RZ)
    n_coils = len(delta_field_funcs)
    R_q = np.zeros((n_s, n_coils))

    for i, ((surf_R, surf_Z), s_val) in enumerate(
        zip(flux_surfaces_RZ, s_labels)
    ):
        surf_R = np.asarray(surf_R, dtype=float)
        surf_Z = np.asarray(surf_Z, dtype=float)
        # Choose starting point at outer midplane
        idx_max_R = int(np.argmax(surf_R))
        R0, Z0 = float(surf_R[idx_max_R]), float(surf_Z[idx_max_R])

        q_base = q_by_fieldline_winding(
            base_field_func, R0, Z0, n_poloidal_steps=n_poloidal_steps
        )
        if not np.isfinite(q_base):
            continue

        for k, delta_field in enumerate(delta_field_funcs):

            def perturbed_field(x, _eps=epsilon, _df=delta_field):
                fb = base_field_func(x)
                df = _df(x)
                return [fb[0] + _eps * df[0], fb[1] + _eps * df[1], fb[2] + _eps * df[2]]

            q_pert = q_by_fieldline_winding(
                perturbed_field, R0, Z0, n_poloidal_steps=n_poloidal_steps
            )
            if np.isfinite(q_pert):
                R_q[i, k] = (q_pert - q_base) / epsilon

    labels = [f"q.s{s:.2f}" for s in s_labels]
    return R_q, labels


# ── Rotation-transform response ───────────────────────────────────────────

def iota_response_matrix(
    base_field_func: Callable,
    delta_field_funcs: List[Callable],
    flux_surfaces_RZ: List[Tuple[np.ndarray, np.ndarray]],
    s_labels: List[float],
    q_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """∂ι(s_i)/∂I_k where ι = 1/q.

    Uses the chain rule: ∂ι/∂I = -1/q² · ∂q/∂I.

    Parameters and returns are analogous to :func:`q_response_matrix_analytic`.
    """
    R_q, labels = q_response_matrix_analytic(
        base_field_func,
        delta_field_funcs,
        flux_surfaces_RZ,
        s_labels,
        q_values=q_values,
    )

    if q_values is None:
        # Compute q₀ from surface integrals
        q_arr = np.zeros(len(s_labels))
        for i, ((surf_R, surf_Z), _) in enumerate(zip(flux_surfaces_RZ, s_labels)):
            q_arr[i] = q_from_flux_surface_integral(base_field_func, surf_R, surf_Z)
    else:
        q_arr = np.asarray(q_values, dtype=float)

    # Guard against q = 0
    q_safe = np.where(np.abs(q_arr) < 1e-6, 1e-6, q_arr)
    R_iota = -R_q / q_safe[:, np.newaxis] ** 2
    iota_labels = [lbl.replace("q.", "iota.") for lbl in labels]
    return R_iota, iota_labels


# ── Top-level convenience ──────────────────────────────────────────────────

def build_qprofile_response(
    base_field_func: Callable,
    delta_field_funcs: List[Callable],
    equilibrium,
    s_values: tuple = (0.2, 0.4, 0.6, 0.8, 1.0),
    method: str = "A",
    epsilon: float = 1e-3,
) -> Tuple[np.ndarray, List[str]]:
    """Compute the q-profile response matrix ∂q(s_i)/∂I_k.

    Parameters
    ----------
    base_field_func : callable
    delta_field_funcs : list of callable
    equilibrium : object
        Must provide ``.flux_surface(s) -> (R_arr, Z_arr)`` for Method A/B
        initialisation.  Falls back to an elliptical approximation if the
        method is absent or raises.
    s_values : tuple of float
        Normalised flux labels (0 < s < 1).
    method : ``'A'`` or ``'B'``
        Method A = analytic flux-surface integral (fast).
        Method B = finite-difference field-line tracing (robust, slow).
    epsilon : float
        FD step for Method B.

    Returns
    -------
    R_q : ndarray, shape (n_s, n_coils)
    labels : list of str
    """
    flux_surfaces: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in s_values:
        try:
            R_fs, Z_fs = equilibrium.flux_surface(s)
            flux_surfaces.append((np.asarray(R_fs), np.asarray(Z_fs)))
        except Exception:
            # Fallback: circular/elliptical approximation
            theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            R0 = getattr(equilibrium, "R0", 1.86)
            a = getattr(equilibrium, "a", 0.6)
            kappa = getattr(equilibrium, "kappa", 1.7)
            R_fs = R0 + a * s * np.cos(theta)
            Z_fs = a * s * kappa * np.sin(theta)
            flux_surfaces.append((R_fs, Z_fs))

    if method == "B":
        return q_response_matrix_fd(
            base_field_func,
            delta_field_funcs,
            flux_surfaces,
            list(s_values),
            epsilon=epsilon,
        )
    return q_response_matrix_analytic(
        base_field_func,
        delta_field_funcs,
        flux_surfaces,
        list(s_values),
    )
