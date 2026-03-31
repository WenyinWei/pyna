"""Island-chain connectivity and orbit-based fixed-point propagation.

Continuous-time perspective on discrete-map fixed points
---------------------------------------------------------
Consider a field with toroidal period Np (i.e. the field is invariant under
φ → φ + 2π/Np). A Poincaré section placed at φ = φ₀ induces a map P. An
m/n island chain means a family of periodic orbits that close after exactly
m toroidal turns: P^m(x) = x. The chain contains exactly n_island = m/gcd(m,n)
distinct Poincaré points per section (for a full m/n chain with n the
poloidal winding number).

Key insight: connectivity via continuous-time flow
--------------------------------------------------
All fixed points of P^m on the same island chain are connected by the
continuous-time field-line flow. Concretely:

    Starting from a fixed point x* at φ₀, follow the field line for
    Δφ = 2π/Np (one field period). Because the field is Np-periodic and
    x* is a period-m fixed point of the *full* map, the image x_1 =
    Φ_{Δφ}(x*) is **another fixed point of P^m at the same section φ₀
    after the symmetry reduction** — but evaluated at the section
    φ₁ = φ₀ + Δφ. In other words, x_1 is the section-φ₁ representative
    of the same orbit.

    After Np steps of size Δφ = 2π/Np one full toroidal turn is
    completed, so after m*Np steps we return to x*. The entire chain of
    fixed points at sections φ₀, φ₀+Δφ, φ₀+2Δφ, … is swept out by a
    single orbit integration.

DPm evolution along the chain (conjugation / similarity)
---------------------------------------------------------
Let Φ_k denote the linearised flow from φ₀ to φ₀ + k·Δφ (i.e. DX_pol
accumulated over k field periods from the starting fixed point). Then
the monodromy matrix at the k-th section point x_k is similar to DPm(x*):

    DPm(x_k) = Φ_k · DPm(x*) · Φ_k⁻¹

This follows directly from the chain rule applied to the composition
P^m = Φ_{Np·m} = Φ_k ∘ P^m ∘ Φ_k⁻¹ (when restricted to the chain).

Consequence: eigenvalues are **invariant** across all points of a chain
(as expected for a conjugate family), and one monodromy computation at a
single X/O-point fully determines the stability of the entire chain.

In practice (MCF with cached fields)
-------------------------------------
The continuous-time integration uses a parallel RK4 (via the cyna C++
backend when available, or scipy.integrate.solve_ivp as fallback). The
DX_pol variational equation is integrated simultaneously:

    d(DX_pol)/dφ = A(R, Z, φ) · DX_pol,    DX_pol(φ₀) = I

where A_ij = ∂(R·B_pol_i / Bφ) / ∂x_j.

API summary
-----------
Main entry point for MCF use::

    chain = IslandChainOrbit.from_single_fixedpoint(
        R0, Z0, phi0,         # one known X- or O-point
        field_func,           # callable rzphi -> (dR/dl, dZ/dl, dphi/dl)
        Np,                   # field period
        m,                    # orbit period (toroidal turns)
        n_sections=4,         # how many phi sections to return
        section_phis=None,    # explicit section angles (overrides n_sections)
    )

    for fp in chain.fixed_points:
        print(fp.phi, fp.R, fp.Z, fp.DPm)

Note on p/q vs m/n (for general dynamical systems):
----------------------------------------------------
In the general dynamical-systems literature a periodic orbit of period q of a
map is sometimes called a q-cycle, with the rational rotation number written
p/q (p poloidal windings per q toroidal turns). In MCF, the same orbit is
labelled m/n with m = number of toroidal turns (= q) and n = poloidal
winding number (= p). The correspondence is therefore: p = n, q = m.
This module uses the MCF convention (m, n) throughout.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChainFixedPoint:
    """A single fixed point of the m-turn Poincaré map, with monodromy.

    Attributes
    ----------
    phi : float
        Toroidal angle of the Poincaré section [rad].
    R, Z : float
        Poloidal coordinates [m].
    DPm : ndarray, shape (2, 2)
        Monodromy matrix at this section.  Eigenvalues are identical for
        all points on the same chain (similarity invariant).
    DX_pol_accum : ndarray, shape (2, 2)
        Accumulated linearised flow Φ_k from the seed point to this
        section: DPm = Φ_k · DPm_seed · Φ_k⁻¹.
    kind : str
        ``'X'`` (hyperbolic, |Tr| > 2) or ``'O'`` (elliptic, |Tr| ≤ 2).
    """
    phi: float
    R: float
    Z: float
    DPm: np.ndarray
    DX_pol_accum: np.ndarray
    kind: str = field(init=False)

    def __post_init__(self) -> None:
        tr = float(np.trace(self.DPm))
        self.kind = 'X' if abs(tr) > 2.0 else 'O'

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.DPm)

    @property
    def stability_index(self) -> float:
        """Tr(DPm)/2.  |k| > 1 → hyperbolic (X), |k| ≤ 1 → elliptic (O)."""
        return float(np.trace(self.DPm)) / 2.0

    @property
    def greene_residue(self) -> float:
        """Greene's residue R = (2 - Tr)/4.  R < 0 → hyperbolic."""
        return (2.0 - float(np.trace(self.DPm))) / 4.0

    @property
    def unstable_eigenvec(self) -> Optional[np.ndarray]:
        """Unit eigenvector along the unstable manifold (X-points only)."""
        if self.kind != 'X':
            return None
        ev, evec = np.linalg.eig(self.DPm)
        ev = ev.real; evec = evec.real
        iu = int(np.argmax(np.abs(ev)))
        v = evec[:, iu].copy()
        if v[1] < 0:
            v = -v
        return v / (np.linalg.norm(v) + 1e-30)

    @property
    def stable_eigenvec(self) -> Optional[np.ndarray]:
        """Unit eigenvector along the stable manifold (X-points only)."""
        if self.kind != 'X':
            return None
        ev, evec = np.linalg.eig(self.DPm)
        ev = ev.real; evec = evec.real
        is_ = int(np.argmin(np.abs(ev)))
        v = evec[:, is_].copy()
        if v[1] < 0:
            v = -v
        return v / (np.linalg.norm(v) + 1e-30)


@dataclass
class IslandChainOrbit:
    """All Poincaré-section representatives of one island-chain orbit.

    Attributes
    ----------
    m : int
        Orbit period (toroidal turns to close the orbit).
    n : int
        Poloidal winding number (informational; does not affect computation).
    Np : int
        Field toroidal period (stellarator symmetry).
    fixed_points : list of ChainFixedPoint
        Fixed points at each requested Poincaré section, in order of
        increasing φ.  Length = number of requested sections × number of
        chain points per section.
    seed_phi : float
        Toroidal angle of the seed fixed point.
    seed_RZ : tuple
        (R, Z) of the seed fixed point.
    """
    m: int
    n: int
    Np: int
    fixed_points: List[ChainFixedPoint]
    seed_phi: float
    seed_RZ: tuple

    # ------------------------------------------------------------------ #
    # Constructor                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_single_fixedpoint(
        cls,
        R0: float,
        Z0: float,
        phi0: float,
        field_func: Callable,
        Np: int,
        m: int,
        n: int = 0,
        *,
        section_phis: Optional[Sequence[float]] = None,
        n_sections: int = 4,
        dt: float = 0.05,
        rtol: float = 1e-9,
        atol: float = 1e-10,
        newton_tol: float = 1e-9,
        newton_maxiter: int = 40,
        newton_eps: float = 1e-4,
        refine: bool = True,
    ) -> "IslandChainOrbit":
        """Build the full island-chain orbit from one known fixed point.

        Starting from a single converged fixed point (R0, Z0) at
        Poincaré section φ₀, this method:

        1. Integrates the field line and the variational equation
           dDX_pol/dφ = A·DX_pol simultaneously from φ₀ to
           φ₀ + m·2π, recording section crossings at the requested φ
           values (``section_phis``).

        2. At each section crossing φ_k, applies the conjugation formula::

               DPm(x_k) = Φ_k · DPm(x*) · Φ_k⁻¹

           where Φ_k = DX_pol(φ₀ → φ_k) is accumulated automatically by
           the variational equation.

        3. Optionally Newton-refines each section point so that
           G(x_k) = P^m(x_k) - x_k = 0 is satisfied to ``newton_tol``.

        Parameters
        ----------
        R0, Z0 : float
            Seed fixed-point coordinates [m].
        phi0 : float
            Toroidal angle of the seed Poincaré section [rad].
        field_func : callable
            ``field_func(rzphi) → (dR/dl, dZ/dl, dphi/dl)`` (arc-length
            parameterised; field direction, not magnitude).
        Np : int
            Field toroidal period of the device.
        m : int
            Orbit period: number of toroidal turns required to close.
        n : int, optional
            Poloidal winding number (informational).
        section_phis : sequence of float, optional
            Explicit toroidal angles at which to record section crossings.
            Defaults to ``n_sections`` equally spaced angles in
            [φ₀, φ₀ + 2π/Np).
        n_sections : int
            Number of sections if ``section_phis`` is not given.
        dt : float
            Maximum step size for the ODE integrator [rad].
        rtol, atol : float
            ODE solver tolerances.
        newton_tol : float
            Convergence tolerance for Newton refinement at each section.
        newton_maxiter : int
            Maximum Newton iterations per section point.
        newton_eps : float
            Finite-difference step for Newton Jacobian.
        refine : bool
            If True (default), Newton-refine each section crossing.

        Returns
        -------
        IslandChainOrbit

        Notes
        -----
        The monodromy DPm(x*) at the seed is computed by a preliminary
        full-orbit integration (φ₀ → φ₀ + 2π·m) with DX_pol(φ₀) = I.
        DX_pol at φ_end equals DPm. This preliminary DPm is then used as
        the monodromy reference for conjugating all downstream section
        points.
        """
        # ── Step 0: compute DPm at the seed ──────────────────────────
        DPm_seed = _compute_DPm_at_seed(
            R0, Z0, phi0, field_func, m, dt=dt, rtol=rtol, atol=atol
        )

        # ── Step 1: determine section angles ─────────────────────────
        if section_phis is None:
            dphi_period = 2.0 * np.pi / Np
            section_phis = [phi0 + k * dphi_period
                            for k in range(n_sections)]
        section_phis = list(section_phis)

        # ── Step 2: integrate orbit + DX_pol over m full turns ───────
        #    Record crossings whenever φ passes through any section_phi
        fps = _propagate_chain(
            R0, Z0, phi0, field_func, m, DPm_seed,
            section_phis=section_phis,
            dt=dt, rtol=rtol, atol=atol,
        )

        # ── Step 3: Newton refinement at each section point ──────────
        if refine:
            fps = _refine_chain_points(
                fps, field_func, m, phi0,
                tol=newton_tol, maxiter=newton_maxiter, eps=newton_eps,
                dt=dt,
            )

        return cls(
            m=m, n=n, Np=Np,
            fixed_points=fps,
            seed_phi=phi0,
            seed_RZ=(R0, Z0),
        )

    # ------------------------------------------------------------------ #
    # Convenience accessors                                               #
    # ------------------------------------------------------------------ #

    def at_section(self, phi: float, tol: float = 1e-6) -> List[ChainFixedPoint]:
        """Return fixed points at the section closest to ``phi``."""
        if not self.fixed_points:
            return []
        phis = np.array([fp.phi for fp in self.fixed_points])
        best = float(np.min(np.abs(phis - phi)))
        return [fp for fp in self.fixed_points
                if abs(fp.phi - phi) <= best + tol]

    def xpoints(self, phi: Optional[float] = None) -> List[ChainFixedPoint]:
        """All X-points, optionally filtered by section phi."""
        fps = self.at_section(phi) if phi is not None else self.fixed_points
        return [fp for fp in fps if fp.kind == 'X']

    def opoints(self, phi: Optional[float] = None) -> List[ChainFixedPoint]:
        """All O-points, optionally filtered by section phi."""
        fps = self.at_section(phi) if phi is not None else self.fixed_points
        return [fp for fp in fps if fp.kind == 'O']

    def summary(self) -> str:
        """One-line summary string."""
        sections = sorted({fp.phi for fp in self.fixed_points})
        lines = [f"IslandChainOrbit  m={self.m}  n={self.n}  Np={self.Np}",
                 f"  Seed: (R={self.seed_RZ[0]:.5f}, Z={self.seed_RZ[1]:.5f})"
                 f"  phi0={self.seed_phi:.4f}"]
        for phi in sections:
            fps = self.at_section(phi)
            x = [f for f in fps if f.kind == 'X']
            o = [f for f in fps if f.kind == 'O']
            ev = fps[0].eigenvalues if fps else []
            lines.append(f"  phi={phi:.4f}:  {len(x)} X  {len(o)} O"
                         f"  (eigenvalues ≈ {[f'{v.real:.4f}' for v in ev]})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # MCF cyna fast path                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_cyna_cache(
        cls,
        R0: float,
        Z0: float,
        phi0: float,
        field_cache: dict,
        Np: int,
        m: int,
        n: int = 0,
        *,
        section_phis: Optional[Sequence[float]] = None,
        n_sections: int = 4,
        DPhi: float = 0.05,
        fd_eps: float = 1e-4,
        newton_maxiter: int = 80,
        newton_tol: float = 1e-9,
        refine: bool = True,
        n_threads: int = 0,
    ) -> "IslandChainOrbit":
        """Build island-chain orbit using the cyna C++ parallel backend.

        This is the MCF-optimised constructor. Instead of scipy.integrate,
        it uses ``pyna._cyna.trace_orbit_along_phi`` to propagate (R, Z)
        and accumulate DX_pol along the orbit in a single C++ call, then
        ``pyna._cyna.find_fixed_points_batch`` for Newton refinement at
        each section.

        The conjugation formula is applied to the DPm matrices returned
        by cyna (which stores DPm[k] = DX_pol(phi0 → phi_k))::

            DPm(x_k) = DX_pol_k · DPm_seed · DX_pol_k⁻¹

        Parameters
        ----------
        R0, Z0 : float
            Seed fixed-point coordinates (must already be Newton-converged).
        phi0 : float
            Toroidal angle of the seed section [rad].
        field_cache : dict
            Dict with keys ``'BR', 'BPhi', 'BZ', 'R_grid', 'Z_grid',
            'Phi_grid'`` (as returned by topoquest field-cache builders).
            Phi_grid may be the original (will be extended by 2π internally).
        Np : int
            Field toroidal period.
        m : int
            Orbit period (map turns).
        n : int, optional
            Poloidal winding number.
        section_phis : sequence of float, optional
            Target Poincaré sections. Defaults to Np equally-spaced angles
            in [phi0, phi0 + 2π/Np).
        n_sections : int
            Number of sections if ``section_phis`` is None.
        DPhi : float
            RK4 step size for cyna integrator [rad].
        fd_eps : float
            Finite-difference eps for Newton Jacobian [m].
        newton_maxiter : int
            Maximum Newton iterations per point.
        newton_tol : float
            Newton convergence tolerance.
        refine : bool
            Whether to Newton-refine each section crossing.
        n_threads : int
            Number of parallel threads (0 = auto).

        Returns
        -------
        IslandChainOrbit
        """
        try:
            from pyna.MCF.flt import trace_orbit_along_phi, find_fixed_points_batch
        except ImportError as exc:
            raise ImportError(
                "pyna.MCF.flt (cyna C++ backend) not available; use from_single_fixedpoint instead"
            ) from exc

        # Extend Phi_grid for periodicity
        Phi_grid = field_cache['Phi_grid']
        if abs(Phi_grid[-1] - 2 * np.pi) > 1e-6:
            Phi_ext = np.append(Phi_grid, 2 * np.pi)
        else:
            Phi_ext = np.asarray(Phi_grid, dtype=np.float64)

        def _ext(a):
            return np.concatenate([a, a[:, :, :1]], axis=2)

        BR_c   = np.ascontiguousarray(_ext(field_cache['BR']),   dtype=np.float64)
        BPhi_c = np.ascontiguousarray(_ext(field_cache['BPhi']), dtype=np.float64)
        BZ_c   = np.ascontiguousarray(_ext(field_cache['BZ']),   dtype=np.float64)
        Rg = np.ascontiguousarray(field_cache['R_grid'], dtype=np.float64)
        Zg = np.ascontiguousarray(field_cache['Z_grid'], dtype=np.float64)
        Pg = np.ascontiguousarray(Phi_ext, dtype=np.float64)

        field_kw = dict(BR=BR_c, BPhi=BPhi_c, BZ=BZ_c,
                        R_grid=Rg, Z_grid=Zg, Phi_grid=Pg)

        # ── Target sections ───────────────────────────────────────────
        if section_phis is None:
            dphi = 2.0 * np.pi / Np
            section_phis = [phi0 + k * dphi for k in range(n_sections)]
        section_phis = list(section_phis)

        # Span needed: from phi0 to max section + small margin
        phi_max_target = max(
            phi0 + ((p - phi0) % (2 * np.pi)) for p in section_phis
        )
        phi_span = phi_max_target - phi0 + 0.05

        # dphi_out = smallest gap between consecutive targets (use pi/4 if uniform)
        s_sorted = sorted(
            phi0 + ((p - phi0) % (2 * np.pi)) for p in section_phis
        )
        dphi_out = float(min(np.diff(s_sorted))) if len(s_sorted) > 1 else float(2 * np.pi / Np)

        # ── cyna orbit propagation ────────────────────────────────────
        # Returns (R_arr, Z_arr, phi_arr, DPm_arr[N,4], alive_arr)
        R_arr, Z_arr, phi_arr, DPm_arr, alive_arr = trace_orbit_along_phi(
            float(R0), float(Z0), float(phi0),
            float(phi_span), float(dphi_out),
            int(m), float(DPhi), float(fd_eps),
            BR_c, BPhi_c, BZ_c, Rg, Zg, Pg,
        )
        R_arr   = np.asarray(R_arr)
        Z_arr   = np.asarray(Z_arr)
        phi_arr = np.asarray(phi_arr)
        DPm_arr = np.asarray(DPm_arr)   # shape (N, 4)
        alive_arr = np.asarray(alive_arr, dtype=bool)

        # ── DPm seed (at phi0): compute via find_fixed_points_batch FD ─
        #    (5-point FD in batch, very fast)
        _, _, _, conv_s, DPm_seed_fp, _, _, _ = find_fixed_points_batch(
            np.array([R0]), np.array([Z0]),
            phi_section=float(phi0), n_turns=int(m),
            DPhi=float(DPhi), fd_eps=float(fd_eps),
            max_iter=40, tol=newton_tol, n_threads=n_threads,
            **field_kw,
        )
        if conv_s[0]:
            DPm_seed = np.asarray(DPm_seed_fp[0]).reshape(2, 2)
        else:
            # fallback: use DPm from cyna orbit at phi0 step
            DPm_seed = DPm_arr[0].reshape(2, 2) if len(DPm_arr) > 0 else np.eye(2)

        # ── Match orbit output to requested sections ──────────────────
        fps: List[ChainFixedPoint] = []

        for phi_tgt in section_phis:
            phi_abs = phi0 + ((phi_tgt - phi0) % (2 * np.pi))
            # Find nearest alive output point
            if alive_arr.sum() == 0:
                continue
            dists = np.where(alive_arr, np.abs(phi_arr - phi_abs), np.inf)
            idx = int(np.argmin(dists))
            if not alive_arr[idx] or np.isnan(R_arr[idx]):
                warnings.warn(f"Orbit dead at phi={phi_tgt:.4f}; skipping.")
                continue

            R_k = float(R_arr[idx])
            Z_k = float(Z_arr[idx])
            # DPm_arr[idx] = Φ_k flattened (DX_pol from phi0 to phi_k)
            Phi_k = DPm_arr[idx].reshape(2, 2)

            # Conjugation: DPm(x_k) = Φ_k · DPm_seed · Φ_k⁻¹
            try:
                Phi_k_inv = np.linalg.inv(Phi_k)
                DPm_k = Phi_k @ DPm_seed @ Phi_k_inv
            except np.linalg.LinAlgError:
                DPm_k = DPm_seed.copy()
                Phi_k_inv = np.eye(2)

            phi_section = float(phi_abs) % (2 * np.pi)

            if refine:
                # Newton refinement at this section
                R_fp, Z_fp, _, conv, DPm_fp, _, _, pt_fp = find_fixed_points_batch(
                    np.array([R_k]), np.array([Z_k]),
                    phi_section=phi_section, n_turns=int(m),
                    DPhi=float(DPhi), fd_eps=float(fd_eps),
                    max_iter=newton_maxiter, tol=newton_tol,
                    n_threads=n_threads, **field_kw,
                )
                if conv[0] and not np.isnan(float(R_fp[0])):
                    R_k = float(R_fp[0])
                    Z_k = float(Z_fp[0])
                    DPm_k = np.asarray(DPm_fp[0]).reshape(2, 2)
                else:
                    warnings.warn(
                        f"Newton refinement failed at phi={phi_section:.4f} "
                        f"(R={R_k:.5f}, Z={Z_k:.5f}); using propagated position."
                    )

            fps.append(ChainFixedPoint(
                phi=phi_section,
                R=R_k, Z=Z_k,
                DPm=DPm_k,
                DX_pol_accum=Phi_k,
            ))

        return cls(
            m=m, n=n, Np=Np,
            fixed_points=fps,
            seed_phi=phi0,
            seed_RZ=(R0, Z0),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _field_direction_phi(field_func: Callable, R: float, Z: float, phi: float):
    """φ-parameterised field direction: (dR/dφ, dZ/dφ)."""
    f = np.asarray(field_func(np.array([R, Z, phi])), dtype=float)
    dphi_dl = f[2]
    if abs(dphi_dl) < 1e-30:
        return np.zeros(2)
    return np.array([f[0] / dphi_dl, f[1] / dphi_dl])


def _build_A_func(field_func: Callable, eps: float = 1e-4) -> Callable:
    """Return A(R, Z, φ) = Jacobian of φ-direction w.r.t. (R, Z)."""
    def A_func(R: float, Z: float, phi: float) -> np.ndarray:
        f0 = _field_direction_phi(field_func, R, Z, phi)
        fR = _field_direction_phi(field_func, R + eps, Z, phi)
        fZ = _field_direction_phi(field_func, R, Z + eps, phi)
        return np.array([
            [(fR[0] - f0[0]) / eps, (fZ[0] - f0[0]) / eps],
            [(fR[1] - f0[1]) / eps, (fZ[1] - f0[1]) / eps],
        ])
    return A_func


def _compute_DPm_at_seed(
    R0: float, Z0: float, phi0: float,
    field_func: Callable, m: int,
    dt: float = 0.05, rtol: float = 1e-9, atol: float = 1e-10,
) -> np.ndarray:
    """Integrate DX_pol over m turns from (R0, Z0, phi0). Return DPm = DX_pol(phi_end)."""
    A_func = _build_A_func(field_func)
    phi_end = phi0 + m * 2.0 * np.pi

    def rhs(phi, y):
        R, Z = y[0], y[1]
        DX = y[2:6].reshape(2, 2)
        dRZ = _field_direction_phi(field_func, R, Z, phi)
        A = A_func(R, Z, phi)
        dDX = A @ DX
        return np.concatenate([dRZ, dDX.ravel()])

    y0 = np.array([R0, Z0, 1., 0., 0., 1.])   # DX_pol = I
    sol = solve_ivp(rhs, (phi0, phi_end), y0,
                    method='DOP853', max_step=dt, rtol=rtol, atol=atol)
    if not sol.success:
        warnings.warn(f"DPm seed integration failed: {sol.message}")
    return sol.y[2:6, -1].reshape(2, 2)


def _propagate_chain(
    R0: float, Z0: float, phi0: float,
    field_func: Callable, m: int,
    DPm_seed: np.ndarray,
    section_phis: List[float],
    dt: float = 0.05, rtol: float = 1e-9, atol: float = 1e-10,
) -> List[ChainFixedPoint]:
    """
    Integrate orbit + DX_pol from phi0 to phi0 + 2π·m, recording
    fixed-point data at each section_phi crossing.

    At each recorded section phi_k with accumulated linearised flow Φ_k:
        DPm(x_k) = Φ_k · DPm_seed · Φ_k⁻¹

    This is the similarity/conjugation formula expressing that all points
    on the same chain are related by continuous-time flow, and the
    monodromy matrix transforms as a covariant 2-tensor under the flow map.
    """
    A_func = _build_A_func(field_func)
    phi_end = phi0 + m * 2.0 * np.pi

    # Sort and deduplicate target sections within [phi0, phi_end)
    targets = sorted(set(float(p) % (2 * np.pi) for p in section_phis))
    # Convert to absolute phis within the integration window
    abs_targets = []
    for t in targets:
        phi_abs = phi0 + ((t - phi0) % (2 * np.pi))
        if phi_abs < phi0:
            phi_abs += 2 * np.pi
        abs_targets.append(phi_abs)
    abs_targets = sorted(abs_targets)

    def rhs(phi, y):
        R, Z = y[0], y[1]
        DX = y[2:6].reshape(2, 2)
        dRZ = _field_direction_phi(field_func, R, Z, phi)
        A = A_func(R, Z, phi)
        dDX = A @ DX
        return np.concatenate([dRZ, dDX.ravel()])

    # Integrate with t_eval at the target sections only (much cheaper than dense_output)
    y0 = np.array([R0, Z0, 1., 0., 0., 1.])
    t_eval_arr = np.array(sorted(set([phi0] + abs_targets + [phi_end])))
    sol = solve_ivp(rhs, (phi0, phi_end), y0,
                    method='DOP853', max_step=dt, rtol=rtol, atol=atol,
                    t_eval=t_eval_arr)
    if not sol.success:
        warnings.warn(f"Chain propagation integration failed: {sol.message}")

    # Build interpolation from sparse t_eval output
    from scipy.interpolate import interp1d as _interp1d
    _interp = _interp1d(sol.t, sol.y, kind='linear', axis=1,
                        bounds_error=False, fill_value='extrapolate')

    fps: List[ChainFixedPoint] = []

    for phi_k in abs_targets:
        phi_k_clipped = float(np.clip(phi_k, phi0, phi_end))
        y_k = _interp(phi_k_clipped)
        R_k, Z_k = float(y_k[0]), float(y_k[1])
        Phi_k = y_k[2:6].reshape(2, 2)  # DX_pol(phi0 → phi_k) = Φ_k

        # Conjugation: DPm(x_k) = Φ_k · DPm_seed · Φ_k⁻¹
        try:
            Phi_k_inv = np.linalg.inv(Phi_k)
            DPm_k = Phi_k @ DPm_seed @ Phi_k_inv
        except np.linalg.LinAlgError:
            DPm_k = DPm_seed.copy()
            Phi_k_inv = np.eye(2)

        # Normalise phi_k to [0, 2π) for the stored section label
        phi_section = float(phi_k_clipped) % (2 * np.pi)

        fps.append(ChainFixedPoint(
            phi=phi_section,
            R=R_k, Z=Z_k,
            DPm=DPm_k,
            DX_pol_accum=Phi_k,
        ))

    return fps


def _poincare_map_m(
    R0: float, Z0: float, phi0: float,
    field_func: Callable, m: int,
    dt: float = 0.05,
) -> tuple:
    """Integrate m turns and return (R_final, Z_final)."""
    phi_end = phi0 + m * 2.0 * np.pi

    def rhs(phi, rz):
        return _field_direction_phi(field_func, rz[0], rz[1], phi)

    sol = solve_ivp(rhs, (phi0, phi_end), [R0, Z0],
                    method='DOP853', max_step=dt, rtol=1e-9, atol=1e-10)
    if not sol.success or len(sol.y[0]) == 0:
        return float('nan'), float('nan')
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def _refine_chain_points(
    fps: List[ChainFixedPoint],
    field_func: Callable,
    m: int,
    phi0: float,
    tol: float = 1e-9,
    maxiter: int = 40,
    eps: float = 1e-4,
    dt: float = 0.05,
) -> List[ChainFixedPoint]:
    """Newton-refine each ChainFixedPoint so that P^m(x_k) = x_k.

    Each section point x_k lies at Poincaré section φ_k. We run Newton on
    G(x) = P^m_{φ_k}(x) - x = 0, where P^m_{φ_k} is the m-turn map
    starting at section φ_k.

    The Jacobian dG/dx = DPm(x) - I is estimated by finite differences.
    The DPm stored in each ChainFixedPoint (from conjugation) is used as
    the initial Jacobian guess (analytical) and updated by FD after the
    first Newton step.
    """
    refined: List[ChainFixedPoint] = []

    for fp in fps:
        R, Z = fp.R, fp.Z
        phi_k = float(fp.phi)
        # Absolute phi for integration: nearest absolute phi >= phi0
        phi_abs = phi0 + ((phi_k - phi0) % (2 * np.pi))

        converged = False
        for it in range(maxiter):
            Rf, Zf = _poincare_map_m(R, Z, phi_abs, field_func, m, dt=dt)
            if np.isnan(Rf):
                break
            G = np.array([Rf - R, Zf - Z])
            if np.linalg.norm(G) < tol:
                converged = True
                break

            # Finite-difference Jacobian of G = P^m - I
            RfR, ZfR = _poincare_map_m(R + eps, Z, phi_abs, field_func, m, dt=dt)
            RfZ, ZfZ = _poincare_map_m(R, Z + eps, phi_abs, field_func, m, dt=dt)
            if np.any(np.isnan([RfR, ZfR, RfZ, ZfZ])):
                break
            DPm_fd = np.array([
                [(RfR - Rf) / eps, (RfZ - Rf) / eps],
                [(ZfR - Zf) / eps, (ZfZ - Zf) / eps],
            ])
            dGdx = DPm_fd - np.eye(2)
            try:
                delta = np.linalg.solve(dGdx, -G)
            except np.linalg.LinAlgError:
                break
            step = np.linalg.norm(delta)
            if step > 0.15:          # cap Newton step
                delta *= 0.15 / step
            R += delta[0]
            Z += delta[1]

        if not converged:
            warnings.warn(
                f"Newton refinement did not converge at phi={phi_k:.4f} "
                f"(R={fp.R:.5f}, Z={fp.Z:.5f}); using propagated position."
            )
            refined.append(fp)
            continue

        # Recompute DPm by FD at the refined position
        Rf, Zf = _poincare_map_m(R, Z, phi_abs, field_func, m, dt=dt)
        RfR, ZfR = _poincare_map_m(R + eps, Z, phi_abs, field_func, m, dt=dt)
        RfZ, ZfZ = _poincare_map_m(R, Z + eps, phi_abs, field_func, m, dt=dt)
        DPm_refined = np.array([
            [(RfR - Rf) / eps, (RfZ - Rf) / eps],
            [(ZfR - Zf) / eps, (ZfZ - Zf) / eps],
        ])

        refined.append(ChainFixedPoint(
            phi=phi_k, R=R, Z=Z,
            DPm=DPm_refined,
            DX_pol_accum=fp.DX_pol_accum,
        ))

    return refined
