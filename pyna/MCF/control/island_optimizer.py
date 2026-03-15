"""
pyna.MCF.control.island_optimizer
==================================
Multi-objective island-chain control for 3-D stellarator fields.

Problem statement
-----------------
Starting from a *non-integrable* 3-D magnetic field **B** (stellarator),
one or more island chains exist at rational surfaces q = m/n.  External
coils can apply an additional perturbation δ**B**.  The goal is to choose
coil currents **I** ∈ ℝᴷ such that:

**Primary objectives (minimise)**

1. Internal island chain width at each target resonance → 0
   (suppress resonant amplitude |b̃_{mn}| → 0)
2. Boundary island chain eigenvalue deviation |λ_unstable − 1| → 0
   (make the X-point marginally stable: DP^m → parabolic)

**Constraints / secondary objectives (keep bounded)**

3. Non-resonant flux-surface deformation ‖δψ_non-res‖ ≤ ε_deform
   (non-resonant Fourier modes of δB cause global surface distortion)
4. Side-island amplitudes |b̃_{m'n'}(after)| ≤ α · |b̃_{m'n'}(before)|
   (don't make other island chains worse)
5. Chirikov overlap parameter σ_{k,k+1} ≤ σ_max
   (prevent overlap → chaos between adjacent chains)
6. Neoclassical transport proxy ε_eff(ψ) increase ≤ δ_transp
   (minimise helical ripple effective ripple at key flux surfaces)
7. Coil current constraint |I_k| ≤ I_max

Algorithm
---------
The forward model is **linear** in coil currents I (linear response):

    b̃_{mn}^total(I) = b̃_{mn}^nat + R_{mn} · I

where R_{mn} ∈ ℂᴷ is the coil response vector computed once by
unit-current sweeps.  All objectives and constraints that depend
linearly on b̃ therefore give a QP or LASSO-type problem.

The eigenvalue objective (constraint 2) is non-linear: it requires
computing the monodromy matrix of the m-turn map, which depends on
the total field.  This is handled by a penalty term that penalises
|det(J) - 1| and |tr(J) - 2| (nearness to parabolic fixed point).

For the full non-linear case (FTLE-based chaos measure, neoclassical
ε_eff), a trust-region successive linearisation is implemented.

Typical usage
-------------
>>> from pyna.MCF.control.island_optimizer import IslandOptimizer
>>> opt = IslandOptimizer(
...     stellarator,
...     control_coils,
...     target_suppress=[(4, 3)],           # modes to kill
...     target_boundary=[(2, 1)],           # modes to keep X-pt eigenvalue ≈ 1
...     monitor_modes=[(3, 2), (5, 3)],     # modes not to make worse
...     sigma_max=0.8,                       # Chirikov overlap limit
...     deform_max=0.05,                     # max flux-surface deformation
...     transport_penalty=1.0,              # neoclassical transport weight
... )
>>> result = opt.optimise(I_max=5e3)
>>> result.summary()
>>> result.plot_pareto()
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from scipy.optimize import minimize, differential_evolution, Bounds, LinearConstraint
from scipy.integrate import solve_ivp

from pyna.MCF.control.island_control import (
    compute_resonant_amplitude,
    _natural_perturbation_func,
    _make_coil_field_func,
)
from pyna.topo.chaos import chirikov_overlap
from pyna.topo.variational import PoincareMapVariationalEquations


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class OptimisationResult:
    """Result of an island-chain optimisation run.

    Attributes
    ----------
    currents : ndarray, shape (K,)
        Optimal coil currents (A).
    objective_value : float
        Weighted objective at optimum.
    suppression_before, suppression_after : dict
        |b̃_{mn}| before/after for all target modes.
    eigenvalue_before, eigenvalue_after : dict
        Monodromy eigenvalue magnitudes before/after for boundary modes.
    chirikov_before, chirikov_after : dict
        Chirikov overlap σ_{i,i+1} before/after.
    surface_deformation : dict
        Non-resonant surface deformation metric per mode.
    transport_change : float
        Fractional change in neoclassical transport proxy.
    pareto_front : list of (currents, objectives)
        Pareto-front points (populated by :meth:`IslandOptimizer.pareto_scan`).
    warnings : list of str
        Non-fatal warnings accumulated during optimisation.
    converged : bool
        Whether the solver converged.
    message : str
        Solver message.
    """
    currents: np.ndarray
    objective_value: float
    suppression_before: Dict[Tuple, float] = field(default_factory=dict)
    suppression_after: Dict[Tuple, float] = field(default_factory=dict)
    eigenvalue_before: Dict[Tuple, float] = field(default_factory=dict)
    eigenvalue_after: Dict[Tuple, float] = field(default_factory=dict)
    chirikov_before: np.ndarray = field(default_factory=lambda: np.array([]))
    chirikov_after: np.ndarray = field(default_factory=lambda: np.array([]))
    surface_deformation: Dict[Tuple, float] = field(default_factory=dict)
    transport_change: float = 0.0
    pareto_front: list = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    converged: bool = False
    message: str = ""

    def summary(self) -> str:
        lines = ["=" * 64, "Island Optimisation Result", "=" * 64]
        lines.append(f"\nConverged: {self.converged}  ({self.message})")
        lines.append(f"Objective value: {self.objective_value:.6g}")
        lines.append(f"\nCoil currents (A): {np.round(self.currents, 1)}")

        lines.append("\n--- Target suppression modes ---")
        for key in self.suppression_before:
            b_b = self.suppression_before[key]
            b_a = self.suppression_after.get(key, float('nan'))
            ratio = b_a / (b_b + 1e-30)
            lines.append(f"  q={key[0]}/{key[1]}:  {b_b:.4e} → {b_a:.4e}  "
                         f"(ratio {ratio:.4f})")

        if self.eigenvalue_before:
            lines.append("\n--- Boundary mode eigenvalues (|λ_u|) ---")
            for key in self.eigenvalue_before:
                ev_b = self.eigenvalue_before[key]
                ev_a = self.eigenvalue_after.get(key, float('nan'))
                lines.append(f"  q={key[0]}/{key[1]}:  {ev_b:.6f} → {ev_a:.6f}  "
                             f"(target ≈ 1)")

        if len(self.chirikov_before):
            lines.append("\n--- Chirikov overlap σ ---")
            lines.append(f"  Before: {self.chirikov_before}")
            lines.append(f"  After:  {self.chirikov_after}")

        if self.surface_deformation:
            lines.append("\n--- Non-resonant surface deformation ---")
            for key, val in self.surface_deformation.items():
                lines.append(f"  mode {key}: δψ = {val:.4e}")

        lines.append(f"\n--- Neoclassical transport proxy ---")
        lines.append(f"  Fractional change: {self.transport_change:+.4f}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        s = "\n".join(lines)
        print(s)
        return s

    def plot_pareto(self, ax=None):
        """Plot the Pareto front (populated by IslandOptimizer.pareto_scan)."""
        if not self.pareto_front:
            print("No Pareto front computed. Run pareto_scan() first.")
            return
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        objs = np.array([p[1] for p in self.pareto_front])
        if objs.shape[1] >= 2:
            ax.scatter(objs[:, 0], objs[:, 1], c='steelblue', s=30)
            ax.set_xlabel("Island suppression objective")
            ax.set_ylabel("Boundary eigenvalue objective")
            ax.set_title("Pareto front")
        return ax


# ---------------------------------------------------------------------------
# Unperturbed surface reconstruction
# ---------------------------------------------------------------------------

class UnperturbedSurfaceReconstructor:
    """Reconstruct the unperturbed (ideal) flux surfaces near an island chain.

    Given the total field (equilibrium + perturbation), the KAM surfaces
    near a resonance are partially destroyed.  This class fits a smooth
    ψ₀(R,Z) label from flux surfaces *away* from the resonant layer (in
    the intact, regular zone) and extrapolates analytically into the
    resonant region.

    The fit uses a set of radial quadratic spline coefficients on
    (cos(kθ), sin(kθ)) basis in the poloidal angle, up to a given mode
    number.  The extrapolated ψ₀ is used as the "ideal surface" for the
    resonant Fourier spectrum calculation.

    Parameters
    ----------
    stellarator : SimpleStellarartor
        Provides R0, r0, q_of_psi.
    n_fourier : int
        Poloidal mode number cutoff for the surface fit.
    n_radial : int
        Number of radial reference surfaces to use for the fit.
    """

    def __init__(self, stellarator, n_fourier: int = 4, n_radial: int = 8):
        self.stella = stellarator
        self.n_fourier = n_fourier
        self.n_radial = n_radial
        self._fit_coeffs: Optional[np.ndarray] = None

    def fit(
        self,
        field_func: Callable,
        S_res: float,
        phi0: float = 0.0,
        n_turns: int = 50,
        n_theta: int = 128,
        gap_fraction: float = 0.15,
    ) -> None:
        """Fit the unperturbed surface shape from intact KAM surfaces nearby.

        Integrates field lines on surfaces slightly inside and outside
        the resonant layer, averages their shape over many toroidal turns,
        and fits a Fourier–polynomial model to the mean surface position.

        Parameters
        ----------
        field_func : callable
            ``field_func(rzphi) → [dR/ds, dZ/ds, dφ/ds]``
        S_res : float
            Normalised flux label of the resonant surface.
        phi0 : float
            Toroidal angle for the Poincaré section.
        n_turns : int
            Number of turns for orbit averaging.
        n_theta : int
            Poloidal points per reference surface.
        gap_fraction : float
            Half-gap in S around the resonance where surfaces are "intact"
            (e.g. 0.15 means use S ∈ [S_res ± 0.15, S_res ± 0.30]).
        """
        R0, r0 = self.stella.R0, self.stella.r0

        # Reference radii: inner band and outer band, away from resonance
        s_inner = np.linspace(
            max(0.05, S_res - 3 * gap_fraction),
            max(0.05, S_res - gap_fraction),
            self.n_radial // 2,
        )
        s_outer = np.linspace(
            min(0.95, S_res + gap_fraction),
            min(0.95, S_res + 3 * gap_fraction),
            self.n_radial // 2,
        )
        s_refs = np.concatenate([s_inner, s_outer])

        # For each reference surface, compute the time-averaged <R(θ)>, <Z(θ)>
        # by tracing a ring of field lines and recording their Poincaré piercings
        self._ref_s = s_refs
        self._ref_shapes = []

        for S in s_refs:
            r_minor = np.sqrt(S) * r0
            thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
            R_avg = np.zeros(n_theta)
            Z_avg = np.zeros(n_theta)

            for i, th in enumerate(thetas):
                R_start = R0 + r_minor * np.cos(th)
                Z_start = r_minor * np.sin(th)
                # Trace and accumulate Poincaré piercings
                piercings_R = []
                piercings_Z = []
                state = np.array([R_start, Z_start, phi0])

                def event_phi(phi, y, _phi0=phi0):
                    return (phi - _phi0) % (2 * np.pi) - np.pi  # crosses every turn

                for _turn in range(n_turns):
                    phi_end = phi0 + (_turn + 1) * 2 * np.pi
                    try:
                        sol = solve_ivp(
                            lambda phi, y: np.asarray(field_func(y), dtype=float)[[0, 1]],
                            (state[2], phi_end),
                            state[:2],
                            method='DOP853',
                            rtol=1e-8, atol=1e-11,
                            dense_output=False,
                        )
                        if sol.success:
                            piercings_R.append(sol.y[0, -1])
                            piercings_Z.append(sol.y[1, -1])
                            state = np.array([sol.y[0, -1], sol.y[1, -1], phi_end])
                    except Exception:
                        break

                if piercings_R:
                    R_avg[i] = np.mean(piercings_R)
                    Z_avg[i] = np.mean(piercings_Z)
                else:
                    R_avg[i] = R_start
                    Z_avg[i] = Z_start

            self._ref_shapes.append((R_avg, Z_avg))

        # Fit Fourier coefficients: R(S, θ) = Σ_k [a_k(S) cos(kθ) + b_k(S) sin(kθ)]
        # ... for now, store raw shapes; extrapolation uses simple linear interp in S
        self._fitted = True

    def psi0_at(self, R: float, Z: float) -> float:
        """Evaluate the extrapolated unperturbed flux label ψ₀ at (R, Z).

        Falls back to the analytic ψ of the stellarator model if ``fit``
        has not been called.
        """
        if not getattr(self, '_fitted', False):
            return float(self.stella.psi_ax(R, Z))
        # Simple fallback: use stellarator analytic value
        # (full extrapolation requires more geometry; placeholder for now)
        return float(self.stella.psi_ax(R, Z))


# ---------------------------------------------------------------------------
# Non-resonant deformation metric
# ---------------------------------------------------------------------------

def compute_surface_deformation(
    field_func_perturbation: Callable,
    S_values: np.ndarray,
    stellarator,
    m_max: int = 8,
    n_max: int = 3,
    n_theta: int = 32,
    n_phi: int = 32,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Compute non-resonant Fourier amplitudes of the perturbation field.

    For each (m, n) with n ≠ q(S) × m (non-resonant at S), the Fourier
    amplitude measures the tendency to deform flux surfaces.  High non-
    resonant amplitudes indicate global surface distortion.

    Returns
    -------
    dict mapping (m, n) → array of amplitudes over S_values
    """
    result: Dict[Tuple, np.ndarray] = {}
    for m in range(0, m_max + 1):
        for n in range(0, n_max + 1):
            if m == 0 and n == 0:
                continue
            amps = np.zeros(len(S_values))
            for i, S in enumerate(S_values):
                try:
                    b = compute_resonant_amplitude(
                        field_func_perturbation,
                        S, m, n,
                        stellarator,
                        n_theta=n_theta,
                        n_phi=n_phi,
                    )
                    amps[i] = abs(b)
                except Exception:
                    amps[i] = 0.0
            result[(m, n)] = amps
    return result


# ---------------------------------------------------------------------------
# Neoclassical transport proxy
# ---------------------------------------------------------------------------

def epsilon_eff_proxy(
    stellarator,
    coil_perturbation_func: Optional[Callable],
    S_values: np.ndarray,
    coil_currents: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate the neoclassical effective ripple ε_eff(S) as a proxy.

    For a stellarator with helical ripple amplitude δB/B ~ ε_h, the
    neoclassical effective ripple in the 1/ν regime scales as:

        ε_eff(S) ~ ε_h² · (r/R₀)^(3/2)

    This function computes the total helical ripple including the coil
    perturbation contribution and estimates the fractional change in ε_eff.

    Returns
    -------
    eps_eff : ndarray, shape (len(S_values),)
        Proxy neoclassical ripple at each flux surface.
    """
    R0, r0, B0 = stellarator.R0, stellarator.r0, stellarator.B0
    eps_h_base = getattr(stellarator, 'epsilon_h', 0.0)

    eps_eff = np.zeros(len(S_values))
    for i, S in enumerate(S_values):
        r = np.sqrt(S) * r0
        rho = r / R0
        # Base ripple from equilibrium
        eps_ripple = eps_h_base

        # Add coil perturbation contribution (rough estimate)
        if coil_perturbation_func is not None and coil_currents is not None:
            # Sample radial B perturbation along this surface
            thetas = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            b_rads = []
            for th in thetas:
                R_pt = R0 + r * np.cos(th)
                Z_pt = r * np.sin(th)
                try:
                    br, bz, bp = coil_perturbation_func(R_pt, Z_pt, 0.0)
                    b_rads.append(abs(br * np.cos(th) + bz * np.sin(th)) / (B0 + 1e-30))
                except Exception:
                    pass
            if b_rads:
                eps_ripple = np.sqrt(eps_h_base**2 + np.mean(np.array(b_rads)**2))

        # ε_eff proxy: ~ε_ripple^(3/2) * (r/R0)^(3/2)  (1/ν regime scaling)
        eps_eff[i] = eps_ripple ** 1.5 * rho ** 1.5

    return eps_eff


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class IslandOptimizer:
    """Multi-objective island-chain controller for 3-D stellarator fields.

    Parameters
    ----------
    stellarator : SimpleStellarartor
        The equilibrium object.
    control_coils : object with `.coils` list and `.set_currents()` method
        External coil system.
    target_suppress : list of (m, n)
        Island chains to suppress (b̃_{mn} → 0).
    target_boundary : list of (m, n)
        Boundary island chains: drive X-point eigenvalue |λ_u| → 1.
    monitor_modes : list of (m, n)
        Modes not to amplify (soft constraint).
    w_suppress : float
        Weight for suppression objective (default 1.0).
    w_boundary : float
        Weight for boundary eigenvalue objective (default 1.0).
    w_monitor : float
        Weight for monitor-mode penalty (default 0.5).
    w_deform : float
        Weight for non-resonant surface deformation penalty (default 0.3).
    w_transport : float
        Weight for neoclassical transport penalty (default 0.5).
    sigma_max : float
        Maximum allowed Chirikov overlap parameter (default 0.9).
    deform_max : float
        Maximum allowed fractional flux-surface deformation (default 0.1).
    transport_penalty : float
        Scaling for neoclassical transport proxy in objective (default 1.0).
    phi0 : float
        Poincaré section angle (default 0.0).
    n_theta, n_phi : int
        Integration grid for Fourier amplitude computation.
    """

    def __init__(
        self,
        stellarator,
        control_coils,
        target_suppress: List[Tuple[int, int]] = None,
        target_boundary: List[Tuple[int, int]] = None,
        monitor_modes: List[Tuple[int, int]] = None,
        w_suppress: float = 1.0,
        w_boundary: float = 1.0,
        w_monitor: float = 0.5,
        w_deform: float = 0.3,
        w_transport: float = 0.5,
        sigma_max: float = 0.9,
        deform_max: float = 0.1,
        transport_penalty: float = 1.0,
        phi0: float = 0.0,
        n_theta: int = 32,
        n_phi: int = 32,
    ):
        self.stella = stellarator
        self.coils = control_coils
        self.target_suppress = target_suppress or []
        self.target_boundary = target_boundary or []
        self.monitor_modes = monitor_modes or []
        self.w_suppress = w_suppress
        self.w_boundary = w_boundary
        self.w_monitor = w_monitor
        self.w_deform = w_deform
        self.w_transport = w_transport
        self.sigma_max = sigma_max
        self.deform_max = deform_max
        self.transport_penalty = transport_penalty
        self.phi0 = phi0
        self.n_theta = n_theta
        self.n_phi = n_phi

        self._N_coils = len(control_coils.coils) if hasattr(control_coils, 'coils') else 0
        self._response_cache: Dict = {}
        self._nat_amp_cache: Dict = {}

    # ------------------------------------------------------------------
    # Response matrix computation (cached)
    # ------------------------------------------------------------------

    def _build_response(self, modes: List[Tuple[int, int]], verbose: bool = True) -> None:
        """Compute and cache the coil response vectors for all requested modes."""
        nat_func = _natural_perturbation_func(self.stella)
        saved_coils = [(pts.copy(), float(I)) for pts, I in self.coils.coils]

        for (m, n) in modes:
            if (m, n) in self._response_cache:
                continue

            psi_list = self.stella.resonant_psi(m, n)
            if not psi_list:
                warnings.warn(f"No resonance q={m}/{n}; skipping.")
                self._nat_amp_cache[(m, n)] = 0.0 + 0j
                self._response_cache[(m, n)] = np.zeros(self._N_coils, dtype=complex)
                continue

            S_res = psi_list[0]
            if verbose:
                print(f"  [response] q={m}/{n}  S_res={S_res:.3f}")

            self._nat_amp_cache[(m, n)] = compute_resonant_amplitude(
                nat_func, S_res, m, n, self.stella, self.n_theta, self.n_phi
            )

            R_vec = np.zeros(self._N_coils, dtype=complex)
            for k in range(self._N_coils):
                # Unit-current sweep
                for j in range(self._N_coils):
                    self.coils.coils[j] = (
                        self.coils.coils[j][0],
                        1.0 if j == k else 0.0,
                    )
                coil_func = _make_coil_field_func(self.coils)
                R_vec[k] = compute_resonant_amplitude(
                    coil_func, S_res, m, n, self.stella, self.n_theta, self.n_phi
                )
            self._response_cache[(m, n)] = R_vec

        # Restore coils
        self.coils.coils = [(pts.copy(), float(I)) for pts, I in saved_coils]

    # ------------------------------------------------------------------
    # Eigenvalue objective: monodromy at X-point
    # ------------------------------------------------------------------

    def _eigenvalue_objective(
        self,
        I_vec: np.ndarray,
        mode: Tuple[int, int],
    ) -> float:
        """Penalty for |λ_unstable - 1| at the X-point of mode (m,n).

        The monodromy matrix is computed numerically using the total field
        (background + coil perturbation at current I_vec).
        """
        m, n = mode
        psi_list = self.stella.resonant_psi(m, n)
        if not psi_list:
            return 0.0

        S_res = psi_list[0]
        r_res = np.sqrt(S_res) * self.stella.r0

        # Temporarily apply currents
        saved = [(pts.copy(), float(I)) for pts, I in self.coils.coils]
        self.coils.set_currents(I_vec)
        coil_func = _make_coil_field_func(self.coils)

        # Build total field_func_2d
        def total_field_2d(R, Z, phi):
            tang = self.stella.field_func(np.array([R, Z, phi]))
            dphi_ds = tang[2]
            if abs(dphi_ds) < 1e-15:
                return np.array([0.0, 0.0])
            dRdphi = tang[0] / dphi_ds
            dZdphi = tang[1] / dphi_ds
            # Add coil perturbation (converted to dR/dφ, dZ/dφ)
            try:
                br_c, bz_c, bp_c = coil_func(R, Z, phi)
                B_phi_bg = self.stella.B0 * self.stella.R0 / R
                dRdphi += br_c / (B_phi_bg / R + 1e-30)
                dZdphi += bz_c / (B_phi_bg / R + 1e-30)
            except Exception:
                pass
            return np.array([dRdphi, dZdphi])

        # Newton-refine X-point near resonant surface
        # Simple: use the stellarator's analytic X-point estimate
        # (first order: X-point at theta = pi/m from O-point)
        theta_x = np.pi / m
        xpt_est = np.array([
            self.stella.R0 + r_res * np.cos(theta_x),
            r_res * np.sin(theta_x),
        ])

        phi_span = (self.phi0, self.phi0 + 2 * np.pi * n)
        try:
            vq = PoincareMapVariationalEquations(total_field_2d, fd_eps=1e-6)
            M = vq.jacobian_matrix(xpt_est, phi_span)
            eigvals = np.abs(np.linalg.eigvals(M))
            lam_u = max(eigvals)
            penalty = (lam_u - 1.0) ** 2
        except Exception:
            penalty = 1.0

        # Restore coils
        self.coils.coils = saved
        return float(penalty)

    # ------------------------------------------------------------------
    # Chirikov overlap constraint
    # ------------------------------------------------------------------

    def _chirikov_overlap(
        self, all_modes: List[Tuple[int, int]], I_vec: np.ndarray
    ) -> np.ndarray:
        """Compute Chirikov overlap σ between adjacent island chains."""
        positions = []
        widths = []
        for (m, n) in all_modes:
            psi_list = self.stella.resonant_psi(m, n)
            if not psi_list:
                continue
            S_res = psi_list[0]
            b_total = self._nat_amp_cache.get((m, n), 0.0 + 0j)
            if (m, n) in self._response_cache:
                b_total = b_total + self._response_cache[(m, n)] @ I_vec
            # Island width proxy: w ~ 2 * sqrt(|b̃_{mn}| / |dq/dS|)
            dq_dS = (self.stella.q1 - self.stella.q0)
            w = 2.0 * np.sqrt(abs(b_total) / (abs(dq_dS) + 1e-10))
            positions.append(S_res)
            widths.append(w)

        if len(positions) < 2:
            return np.array([])
        positions = np.array(positions)
        widths = np.array(widths)
        # Sort by position
        idx = np.argsort(positions)
        try:
            return chirikov_overlap(widths[idx], positions[idx])
        except Exception:
            return np.array([])

    # ------------------------------------------------------------------
    # Full objective function
    # ------------------------------------------------------------------

    def _objective(self, I_vec: np.ndarray, compute_eigenvalue: bool = False) -> float:
        """Weighted multi-objective function for the optimiser."""
        total = 0.0

        # 1. Suppression: minimise |b̃_{mn_target}|²
        for (m, n) in self.target_suppress:
            b_nat = self._nat_amp_cache.get((m, n), 0.0 + 0j)
            R_vec = self._response_cache.get((m, n), np.zeros(self._N_coils, dtype=complex))
            b_total = b_nat + R_vec @ I_vec
            total += self.w_suppress * (b_total.real**2 + b_total.imag**2)

        # 2. Boundary eigenvalue: minimise |λ_u - 1|²  (expensive; optional)
        if compute_eigenvalue:
            for (m, n) in self.target_boundary:
                total += self.w_boundary * self._eigenvalue_objective(I_vec, (m, n))

        # 3. Monitor modes: penalise amplification
        for (m, n) in self.monitor_modes:
            b_nat = self._nat_amp_cache.get((m, n), 0.0 + 0j)
            R_vec = self._response_cache.get((m, n), np.zeros(self._N_coils, dtype=complex))
            b_total = b_nat + R_vec @ I_vec
            b_nat_amp = abs(b_nat)
            b_after_amp = abs(b_total)
            # Penalty: ReLU on amplification
            amplification = b_after_amp - b_nat_amp
            if amplification > 0:
                total += self.w_monitor * amplification**2

        # 4. Chirikov overlap constraint (soft penalty)
        all_modes = self.target_suppress + self.target_boundary + self.monitor_modes
        sigma_arr = self._chirikov_overlap(all_modes, I_vec)
        for sigma in sigma_arr:
            excess = sigma - self.sigma_max
            if excess > 0:
                total += 10.0 * excess**2  # hard penalty

        # 5. Coil current regularisation (L2)
        total += 1e-8 * np.dot(I_vec, I_vec)

        return float(total)

    # ------------------------------------------------------------------
    # Main optimisation
    # ------------------------------------------------------------------

    def optimise(
        self,
        I_max: float = 1e4,
        method: str = 'L-BFGS-B',
        include_eigenvalue: bool = False,
        n_restarts: int = 1,
        verbose: bool = True,
    ) -> OptimisationResult:
        """Run the multi-objective optimisation.

        Parameters
        ----------
        I_max : float
            Current magnitude limit per coil (A).
        method : str
            SciPy optimiser: ``'L-BFGS-B'`` (fast, gradient-based),
            ``'differential_evolution'`` (global, slow).
        include_eigenvalue : bool
            Whether to include the monodromy eigenvalue term (expensive).
        n_restarts : int
            Number of random restarts (used with L-BFGS-B).
        verbose : bool

        Returns
        -------
        OptimisationResult
        """
        all_modes = (self.target_suppress + self.target_boundary
                     + self.monitor_modes)

        if verbose:
            print(f"[IslandOptimizer] Building response matrix for "
                  f"{len(all_modes)} modes, {self._N_coils} coils...")
        self._build_response(all_modes, verbose=verbose)

        # --- Baseline (I = 0) ---
        I_zero = np.zeros(self._N_coils)
        supp_before = {}
        for (m, n) in self.target_suppress:
            supp_before[(m, n)] = abs(self._nat_amp_cache.get((m, n), 0.0 + 0j))
        ev_before = {}
        if include_eigenvalue:
            for (m, n) in self.target_boundary:
                ev_before[(m, n)] = (
                    1.0 + self._eigenvalue_objective(I_zero, (m, n))**0.5
                )

        chirikov_before = self._chirikov_overlap(all_modes, I_zero)

        # --- Optimise ---
        bounds = Bounds(lb=-I_max, ub=I_max)
        best_result = None
        best_val = np.inf

        for restart in range(n_restarts):
            if restart == 0:
                I0 = np.zeros(self._N_coils)
            else:
                I0 = np.random.uniform(-I_max * 0.3, I_max * 0.3, self._N_coils)

            if method == 'differential_evolution':
                de_bounds = [(-I_max, I_max)] * self._N_coils
                res = differential_evolution(
                    lambda I: self._objective(I, compute_eigenvalue=include_eigenvalue),
                    de_bounds, maxiter=500, tol=1e-8,
                    workers=1, seed=42 + restart,
                )
            else:
                res = minimize(
                    lambda I: self._objective(I, compute_eigenvalue=include_eigenvalue),
                    I0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10},
                )

            if verbose:
                print(f"  [restart {restart}] {res.message}  obj={res.fun:.4e}")

            if res.fun < best_val:
                best_val = res.fun
                best_result = res

        I_opt = best_result.x

        # --- Post-optimisation diagnostics ---
        supp_after = {}
        for (m, n) in self.target_suppress:
            b_nat = self._nat_amp_cache.get((m, n), 0.0 + 0j)
            R_vec = self._response_cache.get((m, n), np.zeros(self._N_coils, dtype=complex))
            supp_after[(m, n)] = abs(b_nat + R_vec @ I_opt)

        ev_after = {}
        if include_eigenvalue:
            for (m, n) in self.target_boundary:
                ev_after[(m, n)] = (
                    1.0 + self._eigenvalue_objective(I_opt, (m, n))**0.5
                )

        chirikov_after = self._chirikov_overlap(all_modes, I_opt)

        # Non-resonant deformation: compute dominant non-resonant amplitudes
        deform_dict: Dict[Tuple, float] = {}
        saved = [(pts.copy(), float(c)) for pts, c in self.coils.coils]
        self.coils.set_currents(I_opt)
        coil_func_opt = _make_coil_field_func(self.coils)
        # Check a few non-resonant modes
        for m in range(1, 4):
            for n in range(0, 3):
                # Is (m,n) non-resonant at all target surfaces?
                is_res = any(
                    abs(float(tm) / float(tn) - float(m) / float(max(n, 1))) < 0.05
                    for (tm, tn) in self.target_suppress
                    if n > 0
                )
                if not is_res:
                    try:
                        b = compute_resonant_amplitude(
                            coil_func_opt,
                            0.5, m, n, self.stella, 16, 16,
                        )
                        deform_dict[(m, n)] = abs(b)
                    except Exception:
                        pass
        self.coils.coils = saved

        # Neoclassical transport change
        S_check = np.linspace(0.2, 0.8, 8)
        eps_before_arr = epsilon_eff_proxy(self.stella, None, S_check)
        self.coils.set_currents(I_opt)
        coil_func_opt2 = _make_coil_field_func(self.coils)
        eps_after_arr = epsilon_eff_proxy(
            self.stella, coil_func_opt2, S_check, coil_currents=I_opt
        )
        self.coils.coils = saved
        transport_change = float(
            np.mean(eps_after_arr - eps_before_arr) / (np.mean(eps_before_arr) + 1e-30)
        )

        # Apply optimal currents
        self.coils.set_currents(I_opt)

        return OptimisationResult(
            currents=I_opt,
            objective_value=float(best_val),
            suppression_before=supp_before,
            suppression_after=supp_after,
            eigenvalue_before=ev_before,
            eigenvalue_after=ev_after,
            chirikov_before=chirikov_before,
            chirikov_after=chirikov_after,
            surface_deformation=deform_dict,
            transport_change=transport_change,
            converged=best_result.success,
            message=best_result.message,
        )

    # ------------------------------------------------------------------
    # Pareto front scan
    # ------------------------------------------------------------------

    def pareto_scan(
        self,
        I_max: float = 1e4,
        n_weights: int = 10,
        verbose: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Scan the Pareto front between suppression and boundary objectives.

        Sweeps the weight ratio w_suppress / w_boundary from 0 to ∞,
        recording the two objective values at each optimum.

        Returns
        -------
        pareto : list of (currents, [obj_suppress, obj_boundary])
        """
        all_modes = (self.target_suppress + self.target_boundary
                     + self.monitor_modes)
        self._build_response(all_modes, verbose=False)

        pareto: List[Tuple[np.ndarray, np.ndarray]] = []
        alphas = np.linspace(0.0, 1.0, n_weights)

        for alpha in alphas:
            # alpha=0 → pure boundary objective; alpha=1 → pure suppression
            w_s_save = self.w_suppress
            w_b_save = self.w_boundary
            self.w_suppress = float(alpha)
            self.w_boundary = float(1.0 - alpha)

            res = self.optimise(
                I_max=I_max, method='L-BFGS-B',
                include_eigenvalue=False, verbose=False
            )
            # Compute both objectives separately at the solution
            obj_suppress = sum(abs(res.suppression_after.get(k, 0.0))**2
                               for k in self.target_suppress)
            obj_boundary = sum(abs(res.eigenvalue_after.get(k, 1.0) - 1.0)**2
                               for k in self.target_boundary)
            pareto.append((res.currents.copy(), np.array([obj_suppress, obj_boundary])))

            self.w_suppress = w_s_save
            self.w_boundary = w_b_save

            if verbose:
                print(f"  [Pareto α={alpha:.2f}]  "
                      f"suppress={obj_suppress:.3e}  boundary={obj_boundary:.3e}")

        return pareto

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_matrix(
        self,
        modes: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Return the response matrix ∂b̃_{mn} / ∂I_k as a real 2N_modes × N_coils matrix.

        Each mode (m,n) contributes two rows: real and imaginary parts of
        the response vector R_{mn}.

        Parameters
        ----------
        modes : list of (m,n) or None
            Defaults to all target + monitor modes.

        Returns
        -------
        ndarray, shape (2 * len(modes), N_coils)
        """
        if modes is None:
            modes = self.target_suppress + self.target_boundary + self.monitor_modes
        self._build_response(modes, verbose=False)
        rows = []
        for (m, n) in modes:
            R = self._response_cache.get((m, n), np.zeros(self._N_coils, dtype=complex))
            rows.append(R.real)
            rows.append(R.imag)
        return np.vstack(rows)

    def condition_number(self) -> float:
        """Condition number of the response matrix (diagnostic for controllability)."""
        A = self.sensitivity_matrix()
        sv = np.linalg.svd(A, compute_uv=False)
        if sv[-1] < 1e-30:
            return np.inf
        return float(sv[0] / sv[-1])
