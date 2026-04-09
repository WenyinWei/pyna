"""Island-chain connectivity and orbit-based fixed-point propagation.

Continuous-time perspective on discrete-map fixed points
---------------------------------------------------------
Consider a field with toroidal period Np (i.e. the field is invariant under
φ �?φ + 2π/Np). A Poincaré section placed at φ = φ₀ induces a map P. An
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
    after the symmetry reduction** �?but evaluated at the section
    φ�?= φ₀ + Δφ. In other words, x_1 is the section-φ�?representative
    of the same orbit.

    After Np steps of size Δφ = 2π/Np one full toroidal turn is
    completed, so after m*Np steps we return to x*. The entire chain of
    fixed points at sections φ₀, φ₀+Δφ, φ₀+2Δφ, �?is swept out by a
    single orbit integration.

DPm evolution along the chain (conjugation / similarity)
---------------------------------------------------------
Let Φ_k denote the linearised flow from φ₀ to φ₀ + k·Δφ (i.e. DX_pol
accumulated over k field periods from the starting fixed point). Then
the monodromy matrix at the k-th section point x_k is similar to DPm(x*):

    DPm(x_k) = Φ_k · DPm(x*) · Φ_k⁻�?

This follows directly from the chain rule applied to the composition
P^m = Φ_{Np·m} = Φ_k �?P^m �?Φ_k⁻�?(when restricted to the chain).

Consequence: eigenvalues are **invariant** across all points of a chain
(as expected for a conjugate family), and one monodromy computation at a
single X/O-point fully determines the stability of the entire chain.

In practice (MCF with cached fields)
-------------------------------------
The continuous-time integration uses a parallel RK4 (via the cyna C++
backend when available, or numpy RK4 as fallback). The
DX_pol variational equation is integrated simultaneously:

    d(DX_pol)/dφ = A(R, Z, φ) · DX_pol,    DX_pol(φ₀) = I

where A_ij = �?R·B_pol_i / Bφ) / ∂x_j.

API summary
-----------
Main entry point for MCF use::

    chain = PeriodicOrbit.from_single_fixedpoint(
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
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from pyna.topo._rk4 import rk4_integrate
from pyna.topo._base import InvariantObject


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FixedPoint:
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
        section: DPm = Φ_k · DPm_seed · Φ_k⁻�?
    kind : str
        ``'X'`` (hyperbolic, |Tr| > 2) or ``'O'`` (elliptic, |Tr| �?2).
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
        """Tr(DPm)/2.  |k| > 1 �?hyperbolic (X), |k| �?1 �?elliptic (O)."""
        return float(np.trace(self.DPm)) / 2.0

    @property
    def greene_residue(self) -> float:
        """Greene's residue R = (2 - Tr)/4.  R < 0 �?hyperbolic."""
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
class PeriodicOrbit(InvariantObject):
    """All Poincaré-section representatives of one island-chain orbit.

    This IS-A InvariantObject: a periodic orbit of the Poincaré map P^m(x*) = x*.

    Attributes
    ----------
    m : int
        Orbit period (toroidal turns to close the orbit).
    n : int
        Poloidal winding number (informational; does not affect computation).
    Np : int
        Field toroidal period (stellarator symmetry).
    fixed_points : list of FixedPoint
        Fixed points at each requested Poincaré section, in order of
        increasing φ.  Length = number of requested sections × number of
        chain points per section.
    seed_phi : float
        Toroidal angle of the seed fixed point.
    seed_RZ : tuple
        (R, Z) of the seed fixed point.
    role : str or None
        Optional semantic role label for this orbit (e.g. ``'X'``, ``'O'``,
        ``'separatrix'``, or any user-defined tag).  Not used internally;
        provided as a bridge point for semantic/higher-level layers.
    island_chain : object or None
        Optional back-reference to a higher-level semantic ``IslandChain``
        object.  Stored as ``object`` to avoid circular imports; callers
        may cast to ``pyna.topo.island.IslandChain`` as needed.
    """
    m: int
    n: int
    Np: int
    fixed_points: List[FixedPoint]
    seed_phi: float
    seed_RZ: tuple
    role: Optional[str] = None
    island_chain: Optional[object] = None
    section_phis: Optional[List[float]] = None
    orbit_R: Optional[np.ndarray] = None
    orbit_Z: Optional[np.ndarray] = None
    orbit_phi: Optional[np.ndarray] = None
    orbit_alive: Optional[np.ndarray] = None

    def attach_island_chain(self, chain: object, role: Optional[str] = None) -> None:
        """Attach a semantic *IslandChain* back-reference to this orbit.

        Parameters
        ----------
        chain : IslandChain (or any object)
            The higher-level semantic chain object.  Stored without type
            checking to prevent circular imports.
        role : str, optional
            Semantic role label (e.g. ``'X'``, ``'O'``).  Updates
            ``self.role`` if provided.
        """
        self.island_chain = chain
        if role is not None:
            self.role = role

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
    ) -> "PeriodicOrbit":
        """Build the full island-chain orbit from one known fixed point.

        Starting from a single converged fixed point (R0, Z0) at
        Poincaré section φ₀, this method:

        1. Integrates the field line and the variational equation
           dDX_pol/dφ = A·DX_pol simultaneously from φ₀ to
           φ₀ + m·2π, recording section crossings at the requested φ
           values (``section_phis``).

        2. At each section crossing φ_k, applies the conjugation formula::

               DPm(x_k) = Φ_k · DPm(x*) · Φ_k⁻�?

           where Φ_k = DX_pol(φ₀ �?φ_k) is accumulated automatically by
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
            ``field_func(rzphi) �?(dR/dl, dZ/dl, dphi/dl)`` (arc-length
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
        PeriodicOrbit

        Notes
        -----
        The monodromy DPm(x*) at the seed is computed by a preliminary
        full-orbit integration (φ₀ �?φ₀ + 2π·m) with DX_pol(φ₀) = I.
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
            section_phis=list(section_phis),
        )

    # ------------------------------------------------------------------ #
    # InvariantObject interface                                            #
    # ------------------------------------------------------------------ #

    @property
    def label(self) -> Optional[str]:
        """Human-readable label: role if set, else 'm/n orbit'."""
        if self.role is not None:
            return self.role
        if self.n:
            return f"{self.m}/{self.n} orbit"
        return f"{self.m}-period orbit"

    def section_cut(self, section) -> list:
        """Return the section crossing of this orbit.

        Returns a list of Island objects (when pyna.topo.island is available)
        or FixedPoints. Each Island carries a back-reference island.periodic_orbit = self.
        """
        if isinstance(section, (int, float)):
            phi = float(section)
        elif hasattr(section, 'phi'):
            phi = float(section.phi)
        else:
            return []
        fps = self.fixed_points_at_section(phi)
        try:
            from pyna.topo.island import Island as _Island
            islands = []
            for fp in fps:
                isl = _Island(
                    period_n=self.m,
                    O_point=np.array([float(fp.R), float(fp.Z)], dtype=float),
                    X_points=[],
                    halfwidth=float('nan'),
                    label=self.label,
                )
                isl.periodic_orbit = self
                islands.append(isl)
            return islands
        except Exception:
            return fps

    # ── Backward compatibility: wrapper-style API ─────────────────────────────

    @classmethod
    def from_island_chain_orbit(
        cls,
        orbit: "PeriodicOrbit",
        *,
        label: Optional[str] = None,
        poincare_map=None,
    ) -> "PeriodicOrbit":
        """Return orbit as-is (backward compat: previously wrapped IslandChainOrbit).

        Since PeriodicOrbit IS IslandChainOrbit now, this is a no-op that
        optionally updates the label/role for semantic purposes.
        """
        if label is not None:
            orbit.role = label
        return orbit

    @property
    def orbit(self) -> "PeriodicOrbit":
        """Self-reference for backward compatibility (was: the wrapped IslandChainOrbit)."""
        return self

    @property
    def resonance(self):
        """ResonanceNumber(m, n) for this orbit."""
        from pyna.topo.resonance import ResonanceNumber
        return ResonanceNumber(self.m, self.n)

    @property
    def stability(self) -> str:
        """'X' (hyperbolic), 'O' (elliptic), 'mixed', or 'unknown'."""
        kind_totals = {'X': 0, 'O': 0}
        for fp in self.fixed_points:
            if fp.kind in kind_totals:
                kind_totals[fp.kind] += 1
        mixed = kind_totals['X'] > 0 and kind_totals['O'] > 0
        if mixed:
            return 'mixed'
        if kind_totals['X'] > kind_totals['O']:
            return 'X'
        if kind_totals['O'] > kind_totals['X']:
            return 'O'
        return 'unknown'

    @property
    def greene_residue(self) -> float:
        """Greene's residue at the first fixed point."""
        if not self.fixed_points:
            return float('nan')
        return float(self.fixed_points[0].greene_residue)

    @property
    def eigenvalues(self) -> "np.ndarray":
        """Eigenvalues of DPm at the first fixed point."""
        if not self.fixed_points:
            return np.array([float('nan'), float('nan')])
        return self.fixed_points[0].eigenvalues

    def _raw_diagnostics(self) -> Dict[str, Any]:
        """Alias for diagnostics() (backward compat)."""
        return self.diagnostics()

    # ------------------------------------------------------------------ #
    # Convenience accessors                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _phi_distance(phi_a: float, phi_b: float) -> float:
        """Wrapped angular distance in [0, ?]."""
        return abs(((float(phi_a) - float(phi_b) + np.pi) % (2 * np.pi)) - np.pi)

    def fixed_points_at_section(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        """Return fixed points whose stored section matches ``phi`` within ``tol``."""
        return [fp for fp in self.fixed_points
                if self._phi_distance(fp.phi, phi) <= tol]

    def at_section(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        """Return fixed points at the requested section.

        First tries an exact wrapped-angle match within ``tol``. If no exact
        match exists, falls back to the nearest stored section for backward
        compatibility with earlier callers.
        """
        if not self.fixed_points:
            return []
        exact = self.fixed_points_at_section(phi, tol=tol)
        if exact:
            return exact
        phis = np.array([fp.phi for fp in self.fixed_points])
        best = float(np.min([self._phi_distance(p, phi) for p in phis]))
        return [fp for fp in self.fixed_points
                if self._phi_distance(fp.phi, phi) <= best + tol]

    def xpoints(self, phi: Optional[float] = None) -> List[FixedPoint]:
        """All X-points, optionally filtered by section phi."""
        fps = self.at_section(phi) if phi is not None else self.fixed_points
        return [fp for fp in fps if fp.kind == 'X']

    def opoints(self, phi: Optional[float] = None) -> List[FixedPoint]:
        """All O-points, optionally filtered by section phi."""
        fps = self.at_section(phi) if phi is not None else self.fixed_points
        return [fp for fp in fps if fp.kind == 'O']

    # ------------------------------------------------------------------ #
    # Connectivity properties                                             #
    # ------------------------------------------------------------------ #

    @property
    def n_independent_orbits(self) -> int:
        """Number of independent field-line trajectories = gcd(m, n).

        For HAO m=10, n=3: gcd=1 �?all 10 islands are on one orbit.
        For W7X m=5, n=5: gcd=5 �?5 disconnected flux tubes.
        """
        from math import gcd
        return gcd(self.m, self.n) if self.n > 0 else 1

    @property
    def is_connected(self) -> bool:
        """True when all islands form a single connected orbit (gcd(m,n)==1)."""
        return self.n_independent_orbits == 1

    @property
    def n_points_per_orbit(self) -> int:
        """Number of distinct Poincaré fixed points per independent orbit = m // gcd(m,n)."""
        return self.m // self.n_independent_orbits

    def visit_sequence(self) -> list:
        """Return the order in which fixed points are visited by each orbit.

        Returns a list of lists. Each inner list gives the island indices
        (0-based, into the fixed_points at a single phi section) visited
        by one independent orbit.

        For a connected chain (gcd=1), returns [[0, n_step, 2*n_step, ...] mod m].
        For disconnected chains, returns [[0], [1], ..., [gcd-1]].

        Examples
        --------
        HAO m=10, n=3: [[0, 3, 6, 9, 2, 5, 8, 1, 4, 7]]
        W7X m=5, n=5: [[0], [1], [2], [3], [4]]
        6/4 chain:    [[0, 2, 4], [1, 3, 5]]
        """
        n_step = self.n // self.n_independent_orbits
        m_per_orbit = self.n_points_per_orbit
        orbits = []
        for start in range(self.n_independent_orbits):
            orbit = [(start + k * n_step) % self.m for k in range(m_per_orbit)]
            orbits.append(orbit)
        return orbits

    def summary(self) -> str:
        """Human-readable summary of the chain contents."""
        sections = sorted({fp.phi for fp in self.fixed_points})
        lines = [f"PeriodicOrbit  m={self.m}  n={self.n}  Np={self.Np}",
                 f"  Seed: (R={self.seed_RZ[0]:.5f}, Z={self.seed_RZ[1]:.5f})"
                 f"  phi0={self.seed_phi:.4f}"]
        for phi in sections:
            fps = self.fixed_points_at_section(phi)
            x = [f for f in fps if f.kind == 'X']
            o = [f for f in fps if f.kind == 'O']
            ev = fps[0].eigenvalues if fps else []
            lines.append(f"  phi={phi:.4f}:  {len(x)} X  {len(o)} O"
                         f"  (eigenvalues ? {[f'{v.real:.4f}' for v in ev]})")
        return "\n".join(lines)

    @property
    def orbit_debug_available(self) -> bool:
        """Whether raw orbit samples are available for debugging."""
        return (self.orbit_R is not None and self.orbit_Z is not None and
                self.orbit_phi is not None)

    def orbit_xyz(self) -> Optional[np.ndarray]:
        """Return sampled 3-D orbit points as ``(N, 3) = (X, Y, Z)``."""
        if not self.orbit_debug_available:
            return None
        R = np.asarray(self.orbit_R, dtype=float)
        Z = np.asarray(self.orbit_Z, dtype=float)
        phi = np.asarray(self.orbit_phi, dtype=float)
        if self.orbit_alive is not None:
            alive = np.asarray(self.orbit_alive, dtype=bool)
            R = R[alive]
            Z = Z[alive]
            phi = phi[alive]
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        return np.column_stack([X, Y, Z])

    def section_count_map(
        self,
        requested_phis: Optional[Sequence[float]] = None,
        tol: float = 1e-6,
    ) -> Dict[float, int]:
        """Return the number of fixed points found at each requested section."""
        if requested_phis is None:
            requested_phis = self.section_phis or sorted({fp.phi for fp in self.fixed_points})
        return {
            float(phi): len(self.fixed_points_at_section(float(phi), tol=tol))
            for phi in requested_phis
        }

    def diagnostics(
        self,
        requested_phis: Optional[Sequence[float]] = None,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """Return a structured completeness / debug report for the chain."""
        if requested_phis is None:
            requested_phis = self.section_phis or sorted({fp.phi for fp in self.fixed_points})
        requested_phis = [float(phi) for phi in requested_phis]
        counts = self.section_count_map(requested_phis, tol=tol)
        missing = [phi for phi, cnt in counts.items() if cnt == 0]
        multiple = {phi: cnt for phi, cnt in counts.items() if cnt > 1}

        kind_totals = {'X': 0, 'O': 0}
        section_kind_counts: Dict[float, Dict[str, int]] = {}
        section_points: Dict[float, List[tuple]] = {}
        for phi in requested_phis:
            fps = self.fixed_points_at_section(phi, tol=tol)
            section_kind_counts[float(phi)] = {
                'X': sum(fp.kind == 'X' for fp in fps),
                'O': sum(fp.kind == 'O' for fp in fps),
            }
            kind_totals['X'] += section_kind_counts[float(phi)]['X']
            kind_totals['O'] += section_kind_counts[float(phi)]['O']
            section_points[float(phi)] = [
                (float(fp.R), float(fp.Z), fp.kind, float(fp.greene_residue))
                for fp in fps
            ]

        mixed_kind = kind_totals['X'] > 0 and kind_totals['O'] > 0
        dominant_kind = None
        if kind_totals['X'] > kind_totals['O']:
            dominant_kind = 'X'
        elif kind_totals['O'] > kind_totals['X']:
            dominant_kind = 'O'

        orbit_info: Dict[str, Any] = {'available': False}
        xyz = self.orbit_xyz()
        if xyz is not None and len(xyz):
            phi_raw = np.asarray(self.orbit_phi, dtype=float)
            R_raw = np.asarray(self.orbit_R, dtype=float)
            Z_raw = np.asarray(self.orbit_Z, dtype=float)
            mask = np.isfinite(phi_raw) & np.isfinite(R_raw) & np.isfinite(Z_raw)
            if self.orbit_alive is not None:
                mask &= np.asarray(self.orbit_alive, dtype=bool)
            phi_use = phi_raw[mask]
            R_use = R_raw[mask]
            Z_use = Z_raw[mask]
            orbit_info = {
                'available': True,
                'n_samples': int(len(xyz)),
                'phi_span': (
                    float(np.min(phi_use)),
                    float(np.max(phi_use)),
                ),
                'R_range': (
                    float(np.min(R_use)),
                    float(np.max(R_use)),
                ),
                'Z_range': (
                    float(np.min(Z_use)),
                    float(np.max(Z_use)),
                ),
                'xyz_bbox': (
                    tuple(np.min(xyz, axis=0).tolist()),
                    tuple(np.max(xyz, axis=0).tolist()),
                ),
            }

        return {
            'm': int(self.m),
            'n': int(self.n),
            'Np': int(self.Np),
            'seed_phi': float(self.seed_phi),
            'seed_RZ': tuple(float(v) for v in self.seed_RZ),
            'n_fixed_points': int(len(self.fixed_points)),
            'requested_sections': requested_phis,
            'section_counts': counts,
            'missing_sections': missing,
            'multiple_points_sections': multiple,
            'kind_totals': kind_totals,
            'dominant_kind': dominant_kind,
            'mixed_kind': mixed_kind,
            'section_kind_counts': section_kind_counts,
            'section_points': section_points,
            'orbit_info': orbit_info,
            # InvariantObject enrichment
            'invariant_type': 'PeriodicOrbit',
            'stability': 'mixed' if mixed_kind else (dominant_kind or 'unknown'),
            'greene_residue': float(self.fixed_points[0].greene_residue) if self.fixed_points else float('nan'),
            'resonance': f"{self.m}/{self.n}",
        }

    def is_complete(
        self,
        requested_phis: Optional[Sequence[float]] = None,
        *,
        expected_kind: Optional[str] = None,
        expected_count_per_section: int = 1,
        tol: float = 1e-6,
    ) -> bool:
        """Return True if the chain is complete and kind-consistent."""
        diag = self.diagnostics(requested_phis=requested_phis, tol=tol)
        if diag['missing_sections']:
            return False
        if any(cnt != expected_count_per_section for cnt in diag['section_counts'].values()):
            return False
        if expected_kind is not None:
            if diag['mixed_kind']:
                return False
            if diag['dominant_kind'] != expected_kind:
                return False
        return True

    def debug_summary(
        self,
        requested_phis: Optional[Sequence[float]] = None,
        *,
        expected_kind: Optional[str] = None,
        tol: float = 1e-6,
    ) -> str:
        """Return a multi-line diagnostics string for warnings / logs."""
        diag = self.diagnostics(requested_phis=requested_phis, tol=tol)
        lines = [
            f"PeriodicOrbit diagnostics  m={self.m} n={self.n} Np={self.Np}",
            f"  seed=(R={self.seed_RZ[0]:.5f}, Z={self.seed_RZ[1]:.5f}) phi0={self.seed_phi:.4f}",
            f"  n_fixed_points={diag['n_fixed_points']}  kind_totals={diag['kind_totals']}  mixed_kind={diag['mixed_kind']}",
            f"  requested_sections={[round(p, 6) for p in diag['requested_sections']]}",
            f"  section_counts={{ {', '.join(f'{round(k, 6)}: {v}' for k, v in diag['section_counts'].items())} }}",
        ]
        if expected_kind is not None:
            lines.append(f"  expected_kind={expected_kind}  dominant_kind={diag['dominant_kind']}")
        if diag['missing_sections']:
            lines.append(f"  missing_sections={[round(p, 6) for p in diag['missing_sections']]}")
        if diag['multiple_points_sections']:
            lines.append(
                f"  multiple_points_sections={{ {', '.join(f'{round(k, 6)}: {v}' for k, v in diag['multiple_points_sections'].items())} }}"
            )
        for phi in diag['requested_sections']:
            pts = diag['section_points'][float(phi)]
            if not pts:
                continue
            lines.append(
                f"  phi={phi:.4f}: " + "; ".join(
                    f"{kind}(R={R:.5f}, Z={Z:.5f}, G={G:.5f})"
                    for R, Z, kind, G in pts
                )
            )
        orbit_info = diag['orbit_info']
        if orbit_info.get('available', False):
            lines.append(
                f"  orbit_samples={orbit_info['n_samples']}  phi_span={orbit_info['phi_span']}  "
                f"R_range={orbit_info['R_range']}  Z_range={orbit_info['Z_range']}"
            )
            lines.append(f"  xyz_bbox={orbit_info['xyz_bbox']}")
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
    ) -> "PeriodicOrbit":
        """Build island-chain orbit using the cyna C++ parallel backend.

        This is the MCF-optimised constructor. Instead of scipy.integrate,
        it uses ``pyna._cyna.trace_orbit_along_phi`` to propagate (R, Z)
        and accumulate DX_pol along the orbit in a single C++ call, then
        ``pyna._cyna.find_fixed_points_batch`` for Newton refinement at
        each section.

        The conjugation formula is applied to the DPm matrices returned
        by cyna (which stores DPm[k] = DX_pol(phi0 �?phi_k))::

            DPm(x_k) = DX_pol_k · DPm_seed · DX_pol_k⁻�?

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
        PeriodicOrbit
        """
        try:
            import pyna._cyna as _cyna_mod
            trace_orbit_along_phi = _cyna_mod.trace_orbit_along_phi
            find_fixed_points_batch = _cyna_mod.find_fixed_points_batch
            if trace_orbit_along_phi is None or find_fixed_points_batch is None:
                raise ImportError("pyna._cyna functions not available")
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "pyna._cyna (cyna C++ backend) not available; use from_single_fixedpoint instead"
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
        # Direct C++ binding (pyna._cyna.trace_orbit_along_phi), 14-arg signature:
        # (R0, Z0, phi0, phi_span, dphi_out, n_turns_DPm, DPhi, fd_eps,
        #  BR, BPhi, BZ, R_grid, Z_grid, Phi_grid) -> (R, Z, phi, DPm_flat, alive)
        # MCF convention: BR, BZ, BPhi (polar first, toroidal last).
        # Note: cyna ABI ordering is BR, BPhi, BZ (not BR, BZ, BPhi).
        # Python wrapper in pyna.MCF.flt has a different (older) 11-arg signature.
        try:
            import pyna._cyna as _cyna_direct
            _raw = _cyna_direct.trace_orbit_along_phi(
                float(R0), float(Z0), float(phi0),
                float(phi_span), float(dphi_out),
                int(m), float(DPhi), float(fd_eps),
                BR_c, BPhi_c, BZ_c, Rg, Zg, Pg,
            )
            R_arr     = np.asarray(_raw[0])
            Z_arr     = np.asarray(_raw[1])
            phi_arr   = np.asarray(_raw[2])
            DPm_arr   = np.asarray(_raw[3])   # shape (N, 4)
            alive_arr = np.asarray(_raw[4], dtype=bool)
        except Exception as exc:
            raise RuntimeError(
                f"pyna._cyna.trace_orbit_along_phi failed: {exc}. "
                "Ensure cyna is built and field_cache has correct shape."
            ) from exc

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
        fps: List[FixedPoint] = []

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

            # Conjugation: DPm(x_k) = Φ_k · DPm_seed · Φ_k⁻�?
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

            fps.append(FixedPoint(
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
            section_phis=list(section_phis),
            orbit_R=np.asarray(R_arr).copy(),
            orbit_Z=np.asarray(Z_arr).copy(),
            orbit_phi=np.asarray(phi_arr).copy(),
            orbit_alive=np.asarray(alive_arr, dtype=bool).copy(),
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
    sol = rk4_integrate(rhs, (phi0, phi_end), y0, max_step=dt)
    if not sol.success:
        warnings.warn(f"DPm seed integration failed: {sol.message}")
    return sol.y[2:6, -1].reshape(2, 2)


def _propagate_chain(
    R0: float, Z0: float, phi0: float,
    field_func: Callable, m: int,
    DPm_seed: np.ndarray,
    section_phis: List[float],
    dt: float = 0.05, rtol: float = 1e-9, atol: float = 1e-10,
) -> List[FixedPoint]:
    """
    Integrate orbit + DX_pol from phi0 to phi0 + 2π·m, recording
    fixed-point data at each section_phi crossing.

    At each recorded section phi_k with accumulated linearised flow Φ_k:
        DPm(x_k) = Φ_k · DPm_seed · Φ_k⁻�?

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
    sol = rk4_integrate(rhs, (phi0, phi_end), y0, max_step=dt, t_eval=t_eval_arr)
    if not sol.success:
        warnings.warn(f"Chain propagation integration failed: {sol.message}")

    # Build interpolation from sparse t_eval output
    from scipy.interpolate import interp1d as _interp1d
    _interp = _interp1d(sol.t, sol.y, kind='linear', axis=1,
                        bounds_error=False, fill_value='extrapolate')

    fps: List[FixedPoint] = []

    for phi_k in abs_targets:
        phi_k_clipped = float(np.clip(phi_k, phi0, phi_end))
        y_k = _interp(phi_k_clipped)
        R_k, Z_k = float(y_k[0]), float(y_k[1])
        Phi_k = y_k[2:6].reshape(2, 2)  # DX_pol(phi0 �?phi_k) = Φ_k

        # Conjugation: DPm(x_k) = Φ_k · DPm_seed · Φ_k⁻�?
        try:
            Phi_k_inv = np.linalg.inv(Phi_k)
            DPm_k = Phi_k @ DPm_seed @ Phi_k_inv
        except np.linalg.LinAlgError:
            DPm_k = DPm_seed.copy()
            Phi_k_inv = np.eye(2)

        # Normalise phi_k to [0, 2π) for the stored section label
        phi_section = float(phi_k_clipped) % (2 * np.pi)

        fps.append(FixedPoint(
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

    sol = rk4_integrate(rhs, (phi0, phi_end), [R0, Z0], max_step=dt)
    if not sol.success or len(sol.y[0]) == 0:
        return float('nan'), float('nan')
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def _refine_chain_points(
    fps: List[FixedPoint],
    field_func: Callable,
    m: int,
    phi0: float,
    tol: float = 1e-9,
    maxiter: int = 40,
    eps: float = 1e-4,
    dt: float = 0.05,
) -> List[FixedPoint]:
    """Newton-refine each FixedPoint so that P^m(x_k) = x_k.

    Each section point x_k lies at Poincaré section φ_k. We run Newton on
    G(x) = P^m_{φ_k}(x) - x = 0, where P^m_{φ_k} is the m-turn map
    starting at section φ_k.

    The Jacobian dG/dx = DPm(x) - I is estimated by finite differences.
    The DPm stored in each FixedPoint (from conjugation) is used as
    the initial Jacobian guess (analytical) and updated by FD after the
    first Newton step.
    """
    refined: List[FixedPoint] = []

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

        refined.append(FixedPoint(
            phi=phi_k, R=R, Z=Z,
            DPm=DPm_refined,
            DX_pol_accum=fp.DX_pol_accum,
        ))

    return refined


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------
IslandChainOrbit = PeriodicOrbit  # backward compat
ChainFixedPoint = FixedPoint  # backward compat
