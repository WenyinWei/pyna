"""

pyna.MCF.control.island_optimizer

==================================

Multi-objective island-chain control for 3-D stellarator fields.



Problem statement

-----------------

Starting from a *non-integrable* 3-D magnetic field **B** (stellarator),

one or more island chains exist at rational surfaces q = m/n.  External

coils can apply an additional perturbation Î´**B**.  The goal is to choose

coil currents **I** â?âá´· such that:



**Primary objectives (minimise)**



1. Internal island chain width at each target resonance â?0

   (suppress resonant amplitude |bÌ_{mn}| â?0)

2. Boundary island chain eigenvalue deviation |Î»_unstable â?1| â?0

   (make the X-point marginally stable: DP^m â?parabolic)



**Constraints / secondary objectives (keep bounded)**



3. Non-resonant flux-surface deformation âÎ´Ï_non-resâ?â?Îµ_deform

   (non-resonant Fourier modes of Î´B cause global surface distortion)

4. Side-island amplitudes |bÌ_{m'n'}(after)| â?Î± Â· |bÌ_{m'n'}(before)|

   (don't make other island chains worse)

5. Chirikov overlap parameter Ï_{k,k+1} â?Ï_max

   (prevent overlap â?chaos between adjacent chains)

6. Neoclassical transport proxy Îµ_eff(Ï) increase â?Î´_transp

   (minimise helical ripple effective ripple at key flux surfaces)

7. Coil current constraint |I_k| â?I_max



Algorithm

---------

The forward model is **linear** in coil currents I (linear response):



    bÌ_{mn}^total(I) = bÌ_{mn}^nat + R_{mn} Â· I



where R_{mn} â?âá´· is the coil response vector computed once by

unit-current sweeps.  All objectives and constraints that depend

linearly on bÌ therefore give a QP or LASSO-type problem.



The eigenvalue objective (constraint 2) is non-linear: it requires

computing the monodromy matrix of the m-turn map, which depends on

the total field.  This is handled by a penalty term that penalises

|det(J) - 1| and |tr(J) - 2| (nearness to parabolic fixed point).



For the full non-linear case (FTLE-based chaos measure, neoclassical

Îµ_eff), a trust-region successive linearisation is implemented.



Typical usage

-------------

>>> from pyna.MCF.control.island_optimizer import IslandOptimizer

>>> opt = IslandOptimizer(

...     stellarator,

...     control_coils,

...     target_suppress=[(4, 3)],           # modes to kill

...     target_boundary=[(2, 1)],           # modes to keep X-pt eigenvalue â?1

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

from pyna.flt import FieldLineTracer as _FieldLineTracer



from pyna.MCF.control.island_control import (

    compute_resonant_amplitude,

    _natural_perturbation_func,

    _make_coil_field_func,

)

from pyna.topo.chaos import chirikov_overlap

from pyna.topo.variational import PoincareMapVariationalEquations, _fd_jacobian





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

        |bÌ_{mn}| before/after for all target modes.

    eigenvalue_before, eigenvalue_after : dict

        Monodromy eigenvalue magnitudes before/after for boundary modes.

    chirikov_before, chirikov_after : dict

        Chirikov overlap Ï_{i,i+1} before/after.

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

            lines.append(f"  q={key[0]}/{key[1]}:  {b_b:.4e} â?{b_a:.4e}  "

                         f"(ratio {ratio:.4f})")



        if self.eigenvalue_before:

            lines.append("\n--- Boundary mode eigenvalues (|Î»_u|) ---")

            for key in self.eigenvalue_before:

                ev_b = self.eigenvalue_before[key]

                ev_a = self.eigenvalue_after.get(key, float('nan'))

                lines.append(f"  q={key[0]}/{key[1]}:  {ev_b:.6f} â?{ev_a:.6f}  "

                             f"(target â?1)")



        if len(self.chirikov_before):

            lines.append("\n--- Chirikov overlap Ï ---")

            lines.append(f"  Before: {self.chirikov_before}")

            lines.append(f"  After:  {self.chirikov_after}")



        if self.surface_deformation:

            lines.append("\n--- Non-resonant surface deformation ---")

            for key, val in self.surface_deformation.items():

                lines.append(f"  mode {key}: Î´Ï = {val:.4e}")



        lines.append(f"\n--- Neoclassical transport proxy ---")

        lines.append(f"  Fractional change: {self.transport_change:+.4f}")



        if self.warnings:

            lines.append("\nWarnings:")

            for w in self.warnings:

                lines.append(f"  â?{w}")



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

    Ïâ(R,Z) label from flux surfaces *away* from the resonant layer (in

    the intact, regular zone) and extrapolates analytically into the

    resonant region.



    The fit uses a set of radial quadratic spline coefficients on

    (cos(kÎ¸), sin(kÎ¸)) basis in the poloidal angle, up to a given mode

    number.  The extrapolated Ïâ is used as the "ideal surface" for the

    resonant Fourier spectrum calculation.



    Parameters

    ----------

    stellarator : StellaratorSimple

        Provides R0, r0, q_of_psi.

    n_Fourier : int

        Poloidal mode number cutoff for the surface fit.

    n_radial : int

        Number of radial reference surfaces to use for the fit.

    """



    def __init__(self, stellarator, n_Fourier: int = 4, n_radial: int = 8):

        self.stella = stellarator

        self.n_Fourier = n_Fourier

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

        and fits a Fourierâpolynomial model to the mean surface position.



        Parameters

        ----------

        field_func : callable

            ``field_func(rzphi) â?[dR/ds, dZ/ds, dÏ/ds]``

        S_res : float

            Normalised flux label of the resonant surface.

        phi0 : float

            Toroidal angle for the PoincarÃ© section.

        n_turns : int

            Number of turns for orbit averaging.

        n_theta : int

            Poloidal points per reference surface.

        gap_fraction : float

            Half-gap in S around the resonance where surfaces are "intact"

            (e.g. 0.15 means use S â?[S_res Â± 0.15, S_res Â± 0.30]).

        """

        from scipy.interpolate import CubicSpline



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



        # ODE right-hand side: dy/dphi = [dR/dphi, dZ/dphi]

        # field_func(rzphi) â?[dR/ds, dZ/ds, dphi/ds]; convert to per-phi

        def ode_rzphi(phi, y):

            rzphi = np.array([y[0], y[1], phi])

            tang = np.asarray(field_func(rzphi), dtype=float)

            dphi_ds = tang[2]

            if abs(dphi_ds) < 1e-15:

                return np.array([0.0, 0.0])

            return np.array([tang[0] / dphi_ds, tang[1] / dphi_ds])



        self._ref_s = s_refs

        # Store Fourier coefficients: list of (R_coeffs, Z_coeffs) per surface

        # R_coeffs, Z_coeffs: complex arrays of length n_Fourier+1

        n_f = self.n_Fourier + 1

        R_coeff_arr = np.zeros((len(s_refs), n_f), dtype=complex)

        Z_coeff_arr = np.zeros((len(s_refs), n_f), dtype=complex)



        for si, S in enumerate(s_refs):

            r_minor = np.sqrt(S) * r0

            thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)



            # Collect all PoincarÃ© piercings for this surface

            all_R = []

            all_Z = []



            for th in thetas:

                R_start = R0 + r_minor * np.cos(th)

                Z_start = r_minor * np.sin(th)

                # Trace fieldline with FieldLineTracer and sample Poincaré crossings
                # Estimate arc-length for n_turns toroidal revolutions
                _start_tang = np.asarray(field_func(np.array([R_start, Z_start, phi0])), dtype=float)
                _dphi_ds = abs(_start_tang[2]) if abs(_start_tang[2]) > 1e-15 else 0.3
                _t_max = n_turns * 2 * np.pi / _dphi_ds * 1.5  # generous factor
                _tracer = _FieldLineTracer(field_func, dt=0.05)
                _traj = _tracer.trace(np.array([R_start, Z_start, phi0]), _t_max)
                # Extract Poincaré crossings at phi = phi0 + k*2*pi for k=1..n_turns
                for _k in range(1, n_turns + 1):
                    _phi_cross = phi0 + _k * 2 * np.pi
                    _mask = (_traj[:-1, 2] < _phi_cross) & (_traj[1:, 2] >= _phi_cross)
                    _idx = np.where(_mask)[0]
                    if len(_idx) == 0:
                        break
                    _i = _idx[0]
                    _alpha = (_phi_cross - _traj[_i, 2]) / (_traj[_i + 1, 2] - _traj[_i, 2] + 1e-30)
                    _R_cross = _traj[_i, 0] + _alpha * (_traj[_i + 1, 0] - _traj[_i, 0])
                    _Z_cross = _traj[_i, 1] + _alpha * (_traj[_i + 1, 1] - _traj[_i, 1])
                    all_R.append(_R_cross)
                    all_Z.append(_Z_cross)



            if len(all_R) >= 4:

                all_R = np.array(all_R)

                all_Z = np.array(all_Z)

                # Compute poloidal angle relative to magnetic axis

                angles = np.arctan2(all_Z, all_R - R0)

                # Sort by angle

                sort_idx = np.argsort(angles)

                angles_s = angles[sort_idx]

                R_s = all_R[sort_idx]

                Z_s = all_Z[sort_idx]

                # Interpolate onto uniform theta grid

                # Wrap to handle periodicity

                angles_ext = np.concatenate([angles_s - 2 * np.pi, angles_s, angles_s + 2 * np.pi])

                R_ext = np.concatenate([R_s, R_s, R_s])

                Z_ext = np.concatenate([Z_s, Z_s, Z_s])

                theta_uniform = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)

                R_uniform = np.interp(theta_uniform, angles_ext, R_ext)

                Z_uniform = np.interp(theta_uniform, angles_ext, Z_ext)

                # Extract Fourier coefficients via rfft

                R_fft = np.fft.rfft(R_uniform) / n_theta

                Z_fft = np.fft.rfft(Z_uniform) / n_theta

                R_coeff_arr[si, :] = R_fft[:n_f]

                Z_coeff_arr[si, :] = Z_fft[:n_f]

            else:

                # Fallback: use circular approximation

                R_coeff_arr[si, 0] = R0

                if n_f > 1:

                    R_coeff_arr[si, 1] = r_minor / 2.0  # cos mode

                    Z_coeff_arr[si, 1] = -1j * r_minor / 2.0  # sin mode via imag



        # Fit splines of Fourier coefficients vs S and store for extrapolation

        self._R_splines_re = []

        self._R_splines_im = []

        self._Z_splines_re = []

        self._Z_splines_im = []

        for k in range(n_f):

            try:

                self._R_splines_re.append(CubicSpline(s_refs, R_coeff_arr[:, k].real, extrapolate=True))

                self._R_splines_im.append(CubicSpline(s_refs, R_coeff_arr[:, k].imag, extrapolate=True))

                self._Z_splines_re.append(CubicSpline(s_refs, Z_coeff_arr[:, k].real, extrapolate=True))

                self._Z_splines_im.append(CubicSpline(s_refs, Z_coeff_arr[:, k].imag, extrapolate=True))

            except Exception:

                # Not enough points for cubic spline; use linear

                from scipy.interpolate import interp1d

                self._R_splines_re.append(interp1d(s_refs, R_coeff_arr[:, k].real, fill_value='extrapolate'))

                self._R_splines_im.append(interp1d(s_refs, R_coeff_arr[:, k].imag, fill_value='extrapolate'))

                self._Z_splines_re.append(interp1d(s_refs, Z_coeff_arr[:, k].real, fill_value='extrapolate'))

                self._Z_splines_im.append(interp1d(s_refs, Z_coeff_arr[:, k].imag, fill_value='extrapolate'))



        self._S_res = S_res

        self._phi0 = phi0

        self._n_theta_fit = n_theta

        self._s_refs = s_refs

        self._fitted = True



    def _surface_contour(self, S: float, n_theta: int = 64) -> Tuple[np.ndarray, np.ndarray]:

        """Return R(Î¸), Z(Î¸) for the extrapolated surface at flux label S."""

        n_f = self.n_Fourier + 1

        theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)

        R_fft = np.zeros(n_theta // 2 + 1, dtype=complex)

        Z_fft = np.zeros(n_theta // 2 + 1, dtype=complex)

        for k in range(min(n_f, n_theta // 2 + 1)):

            R_fft[k] = self._R_splines_re[k](S) + 1j * self._R_splines_im[k](S)

            Z_fft[k] = self._Z_splines_re[k](S) + 1j * self._Z_splines_im[k](S)

        R_arr = np.fft.irfft(R_fft * n_theta, n=n_theta)

        Z_arr = np.fft.irfft(Z_fft * n_theta, n=n_theta)

        return R_arr, Z_arr



    def psi0_at(self, R: float, Z: float) -> float:

        """Evaluate the extrapolated unperturbed flux label Ïâ at (R, Z).



        Falls back to the analytic Ï of the stellarator model if ``fit``

        has not been called.  Near S_res, extrapolates the fitted Fourier

        surface model; elsewhere uses the analytic value.

        """

        if not getattr(self, '_fitted', False):

            return float(self.stella.psi_ax(R, Z))



        # For points near S_res, use the fitted/extrapolated surfaces.

        # Strategy: find the S value such that (R,Z) lies on the extrapolated

        # contour at S, using a distance-bisection approach.

        #

        # For robustness, we bracket using the reference surfaces:

        # compute the mean radius of each reference surface and interpolate.

        R0 = self.stella.R0

        r_query = np.sqrt((R - R0) ** 2 + Z ** 2)  # minor radius of query point



        # Mean minor radius of each reference surface

        r_refs = []

        for k, S in enumerate(self._s_refs):

            try:

                R_c, Z_c = self._surface_contour(S, n_theta=32)

                r_mean = np.mean(np.sqrt((R_c - R0) ** 2 + Z_c ** 2))

            except Exception:

                r_mean = np.sqrt(S) * self.stella.r0

            r_refs.append(r_mean)

        r_refs = np.array(r_refs)



        # Also include S_res using extrapolation

        try:

            R_res, Z_res = self._surface_contour(self._S_res, n_theta=32)

            r_res_mean = np.mean(np.sqrt((R_res - R0) ** 2 + Z_res ** 2))

            s_all = np.append(self._s_refs, self._S_res)

            r_all = np.append(r_refs, r_res_mean)

        except Exception:

            s_all = self._s_refs

            r_all = r_refs



        # Sort by r_mean and interpolate to get S(r_query)

        sort_idx = np.argsort(r_all)

        r_sorted = r_all[sort_idx]

        s_sorted = s_all[sort_idx]



        if r_query <= r_sorted[0]:

            psi0 = float(s_sorted[0])

        elif r_query >= r_sorted[-1]:

            psi0 = float(s_sorted[-1])

        else:

            psi0 = float(np.interp(r_query, r_sorted, s_sorted))



        # Clamp to valid range

        psi0 = float(np.clip(psi0, 1e-4, 1.0 - 1e-4))

        return psi0





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



    For each (m, n) with n â?q(S) Ã m (non-resonant at S), the Fourier

    amplitude measures the tendency to deform flux surfaces.  High non-

    resonant amplitudes indicate global surface distortion.



    Returns

    -------

    dict mapping (m, n) â?array of amplitudes over S_values

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

    """Compute the neoclassical effective ripple Îµ_eff(S) via flux-surface average.



    In the 1/Î½ regime, the effective ripple is defined as:



        Îµ_eff(Ï) = sqrt( <(|B| - <|B|>)Â²>_FSA ) / <|B|>_FSA



    where <Â·>_FSA is the flux surface average (averaged over poloidal angle

    at phi=0 cross-section).



    For each flux surface labelled by S, n_theta=16 points are sampled along

    the poloidal circle at phi=0. The background |B| is computed from

    field_func tangent vector using B_phi = B0*R0/R, and any coil perturbation

    is added vectorially.



    Returns

    -------

    eps_eff : ndarray, shape (len(S_values),)

        Neoclassical effective ripple at each flux surface.

    """

    R0, r0, B0 = stellarator.R0, stellarator.r0, stellarator.B0

    n_theta = 16

    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    phi = 0.0  # sample at phi=0 cross-section



    eps_eff = np.zeros(len(S_values))

    for i, S in enumerate(S_values):

        r = np.sqrt(S) * r0

        B_mags = np.zeros(n_theta)



        for k, th in enumerate(thetas):

            R_pt = R0 + r * np.cos(th)

            Z_pt = r * np.sin(th)



            # Get background B magnitude from field_func tangent vector

            # field_func returns unit tangent [dR/ds, dZ/ds, dphi/ds]

            # B_phi = B0 * R0 / R  (tokamak/stellarator toroidal field)

            # dphi/ds = B_phi / (R * B_mag) => B_mag = B_phi / (R * dphi_ds)

            try:

                tang = stellarator.field_func(np.array([R_pt, Z_pt, phi]))

                dphi_ds = tang[2]

                if abs(dphi_ds) > 1e-30:

                    B_phi = B0 * R0 / R_pt

                    B_mag = B_phi / (R_pt * dphi_ds)

                else:

                    B_mag = B0

            except Exception:

                B_mag = B0



            # Add coil perturbation field if provided

            if coil_perturbation_func is not None:

                try:

                    br_coil, bz_coil, bp_coil = coil_perturbation_func(R_pt, Z_pt, phi)

                    # Reconstruct full B vector: background + perturbation

                    if abs(tang[2]) > 1e-30:

                        B_R_bg = tang[0] * B_mag

                        B_Z_bg = tang[1] * B_mag

                        B_phi_bg = B0 * R0 / R_pt

                    else:

                        B_R_bg, B_Z_bg, B_phi_bg = 0.0, 0.0, B0

                    B_R_tot = B_R_bg + br_coil

                    B_Z_tot = B_Z_bg + bz_coil

                    B_phi_tot = B_phi_bg + bp_coil

                    B_mag = np.sqrt(B_R_tot**2 + B_Z_tot**2 + B_phi_tot**2)

                except Exception:

                    pass



            B_mags[k] = B_mag



        # Flux surface average: Îµ_eff = sqrt(Var(|B|)) / <|B|>

        B_mean = np.mean(B_mags)

        B_var = np.mean((B_mags - B_mean) ** 2)

        eps_eff[i] = np.sqrt(B_var) / (B_mean + 1e-30)



    return eps_eff





# ---------------------------------------------------------------------------

# Main optimizer class

# ---------------------------------------------------------------------------



class IslandOptimizer:

    """Multi-objective island-chain controller for 3-D stellarator fields.



    Parameters

    ----------

    stellarator : StellaratorSimple

        The equilibrium object.

    control_coils : object with `.coils` list and `.set_currents()` method

        External coil system.

    target_suppress : list of (m, n)

        Island chains to suppress (bÌ_{mn} â?0).

    target_boundary : list of (m, n)

        Boundary island chains: drive X-point eigenvalue |Î»_u| â?1.

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

        PoincarÃ© section angle (default 0.0).

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

        from joblib import Parallel, delayed



        nat_func = _natural_perturbation_func(self.stella)

        saved_coils = [(pts.copy(), float(I)) for pts, I in self.coils.coils]



        def _unit_response(k, coil_pts_list, S_res, m, n, stella, n_theta, n_phi):

            """Compute response for coil k at unit current (fully self-contained)."""

            from pyna.MCF.control.island_control import (

                compute_resonant_amplitude,

                _make_coil_field_func,

            )

            # Build a lightweight coil object with only coil k active

            class _TmpCoils:

                pass

            tmp = _TmpCoils()

            tmp.coils = [

                (pts.copy(), 1.0 if j == k else 0.0)

                for j, pts in enumerate(coil_pts_list)

            ]

            coil_func = _make_coil_field_func(tmp)

            return compute_resonant_amplitude(coil_func, S_res, m, n, stella, n_theta, n_phi)



        coil_pts_list = [pts.copy() for pts, _ in saved_coils]



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



            # Parallel unit-current sweep over coils

            r_vals = Parallel(n_jobs=-1, backend='loky')(

                delayed(_unit_response)(

                    k, coil_pts_list, S_res, m, n,

                    self.stella, self.n_theta, self.n_phi

                )

                for k in range(self._N_coils)

            )

            self._response_cache[(m, n)] = np.array(r_vals, dtype=complex)



        # Restore coils

        self.coils.coils = [(pts.copy(), float(I)) for pts, I in saved_coils]



    # ------------------------------------------------------------------

    # Eigenvalue objective: monodromy at X-point

    # ------------------------------------------------------------------



    def _refine_xpoint(

        self,

        xpt_est: np.ndarray,

        total_field_2d: Callable,

        phi_span: Tuple[float, float],

        n_iter: int = 10,

        tol: float = 1e-9,

    ) -> np.ndarray:

        """Newton iteration to refine an X-point (unstable fixed point) of the

        m-turn PoincarÃ© map.



        Starting from an analytic estimate ``xpt_est``, iterates:



            x_{k+1} = x_k + (J - I)^{-1} (x_end - x_k)



        where J is the monodromy (Jacobian) of the integrated map and

        x_end is the endpoint of the integrated orbit.  Converges when

        |x_end - x_k| < tol.



        Parameters

        ----------

        xpt_est : ndarray, shape (2,)

            Initial estimate [R, Z] of the X-point.

        total_field_2d : callable

            ``total_field_2d(R, Z, phi) â?[dR/dphi, dZ/dphi]``

        phi_span : (phi0, phi1)

            Integration range (one full period = phi0 + 2Ï*n).

        n_iter : int

            Maximum Newton iterations.

        tol : float

            Convergence tolerance on |x_end - x_start|.



        Returns

        -------

        xpt : ndarray, shape (2,)

            Refined X-point.  If Newton fails, returns the initial estimate.

        """

        x = np.array(xpt_est, dtype=float)

        vq = PoincareMapVariationalEquations(total_field_2d, fd_eps=1e-6)



        for _k in range(n_iter):

            try:

                M = vq.jacobian_matrix(x, phi_span)

                # Integrate the orbit to get x_end using FieldLineTracer
                # total_field_2d(R, Z, phi) -> [dR/dphi, dZ/dphi]; convert to arc-length form
                _phi0_xpt, _phi1_xpt = float(phi_span[0]), float(phi_span[1])
                def _flt_func_xpt(rzphi):
                    _rr, _zz, _pp = rzphi[0], rzphi[1], rzphi[2]
                    _drdphi, _dzdphi = np.asarray(total_field_2d(_rr, _zz, _pp), dtype=float)
                    # |dr/dl|² = (dR/dphi)² + (dZ/dphi)² + R²
                    _scale = np.sqrt(_drdphi**2 + _dzdphi**2 + _rr**2) + 1e-15
                    return [_drdphi / _scale, _dzdphi / _scale, 1.0 / _scale]
                _phi_span_len = abs(_phi1_xpt - _phi0_xpt)
                _t_max_xpt = _phi_span_len * 5.0  # generous
                _xpt_tracer = _FieldLineTracer(_flt_func_xpt, dt=0.02)
                _xpt_traj = _xpt_tracer.trace(np.array([x[0], x[1], _phi0_xpt]), _t_max_xpt)
                # Find crossing at phi = phi1
                _phi_col = _xpt_traj[:, 2]
                _cross_mask = (_phi_col[:-1] < _phi1_xpt) & (_phi_col[1:] >= _phi1_xpt)
                _cross_idx = np.where(_cross_mask)[0]
                if len(_cross_idx) == 0:
                    break
                _ci = _cross_idx[-1]
                _alpha_xpt = (_phi1_xpt - _phi_col[_ci]) / (_phi_col[_ci + 1] - _phi_col[_ci] + 1e-30)
                x_end = (_xpt_traj[_ci, :2] + _alpha_xpt * (_xpt_traj[_ci + 1, :2] - _xpt_traj[_ci, :2]))

                residual = x_end - x

                if np.linalg.norm(residual) < tol:

                    return x_end  # converged to fixed point

                # Newton step: (J - I) dx = -(x_end - x_start) re-arranged as

                # x_new = x + (J - I)^{-1} (x_end - x)

                A = M - np.eye(2)

                try:

                    dx = np.linalg.solve(A, residual)

                except np.linalg.LinAlgError:

                    break

                x = x + dx

            except Exception:

                break



        return x



    def _eigenvalue_objective(

        self,

        I_vec: np.ndarray,

        mode: Tuple[int, int],

    ) -> float:

        """Penalty for |Î»_unstable - 1|Â² at the Newton-refined X-point of mode (m,n).



        Uses Newton iteration (see :meth:`_refine_xpoint`) to locate the

        X-point to high precision before computing the monodromy matrix.

        The total field includes the background equilibrium plus the coil

        perturbation at current ``I_vec``.

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



        # Build total field_func_2d: (R, Z, phi) â?[dR/dphi, dZ/dphi]

        def total_field_2d(R, Z, phi):

            tang = self.stella.field_func(np.array([R, Z, phi]))

            dphi_ds = tang[2]

            if abs(dphi_ds) < 1e-15:

                return np.array([0.0, 0.0])

            dRdphi = tang[0] / dphi_ds

            dZdphi = tang[1] / dphi_ds

            # Add coil perturbation (converted to dR/dÏ, dZ/dÏ)

            try:

                br_c, bz_c, bp_c = coil_func(R, Z, phi)

                B_phi_bg = self.stella.B0 * self.stella.R0 / R

                dRdphi += br_c / (B_phi_bg / R + 1e-30)

                dZdphi += bz_c / (B_phi_bg / R + 1e-30)

            except Exception:

                pass

            return np.array([dRdphi, dZdphi])



        # Analytic first-order estimate: X-point at theta = pi/m

        theta_x = np.pi / m

        xpt_est = np.array([

            self.stella.R0 + r_res * np.cos(theta_x),

            r_res * np.sin(theta_x),

        ])



        phi_span = (self.phi0, self.phi0 + 2 * np.pi * n)

        try:

            # Newton-refine X-point to |residual| < 1e-9

            xpt = self._refine_xpoint(xpt_est, total_field_2d, phi_span,

                                      n_iter=10, tol=1e-9)

            # Compute monodromy matrix at refined X-point

            vq = PoincareMapVariationalEquations(total_field_2d, fd_eps=1e-6)

            M = vq.jacobian_matrix(xpt, phi_span)

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

        """Compute Chirikov overlap Ï between adjacent island chains."""

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

            # Island width proxy: w ~ 2 * sqrt(|bÌ_{mn}| / |dq/dS|)

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



        # 1. Suppression: minimise |bÌ_{mn_target}|Â²

        for (m, n) in self.target_suppress:

            b_nat = self._nat_amp_cache.get((m, n), 0.0 + 0j)

            R_vec = self._response_cache.get((m, n), np.zeros(self._N_coils, dtype=complex))

            b_total = b_nat + R_vec @ I_vec

            total += self.w_suppress * (b_total.real**2 + b_total.imag**2)



        # 2. Boundary eigenvalue: minimise |Î»_u - 1|Â²  (expensive; optional)

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



        Sweeps the weight ratio w_suppress / w_boundary from 0 to â?

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

            # alpha=0 â?pure boundary objective; alpha=1 â?pure suppression

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

                print(f"  [Pareto Î±={alpha:.2f}]  "

                      f"suppress={obj_suppress:.3e}  boundary={obj_boundary:.3e}")



        return pareto



    # ------------------------------------------------------------------

    # Sensitivity analysis

    # ------------------------------------------------------------------



    def sensitivity_matrix(

        self,

        modes: Optional[List[Tuple[int, int]]] = None,

    ) -> np.ndarray:

        """Return the response matrix âbÌ_{mn} / âI_k as a real 2N_modes Ã N_coils matrix.



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

        mat = np.vstack(rows)

        if not np.isfinite(mat).all():

            import warnings

            warnings.warn("sensitivity_matrix contains non-finite values; "

                          "check Biot-Savart near-wire regions.")

        return mat



    def condition_number(self) -> float:

        """Condition number of the response matrix (diagnostic for controllability)."""

        A = self.sensitivity_matrix()

        # Guard against NaN/inf from Biot-Savart near-wire or degenerate coils

        if not np.isfinite(A).all():

            bad_cols = np.where(~np.isfinite(A).all(axis=0))[0]

            import warnings

            warnings.warn(f"Response matrix has {len(bad_cols)} non-finite column(s) "

                          f"at coil indices {bad_cols}; dropping them for SVD.")

            A = A[:, np.isfinite(A).all(axis=0)]

        if A.size == 0 or A.shape[1] == 0:

            return np.inf

        try:

            sv = np.linalg.svd(A, compute_uv=False)

        except np.linalg.LinAlgError:

            # Fall back to robust SVD via scipy

            from scipy.linalg import svd as scipy_svd

            sv = scipy_svd(A, compute_uv=False, check_finite=False, lapack_driver='gesdd')

        if len(sv) == 0 or sv[-1] < 1e-30:

            return np.inf

        return float(sv[0] / sv[-1])

