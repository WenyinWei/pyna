"""pyna.topo.cycle - Continuous-time periodic orbit (Cycle) abstraction.

Design rationale
================

Discrete / Poincaré-map layer:
  FixedPoint          -- one point on a section (R, Z, phi), with DPm
  IslandChain (topo)  -- all FixedPoints of one resonance on one section

Continuous-time / 3D layer:
  Cycle               -- the 3D periodic orbit itself (THIS MODULE)
  Tube                -- one connected family of Cycles (O or X type)
  TubeChain           -- all Tubes of one resonance (m/n)

Relationship:
  A Cycle is the continuous-time counterpart of a FixedPoint.
  Where FixedPoint lives on one Poincaré section, a Cycle is the
  full 3D closed curve that FixedPoint belongs to.

  FixedPoint  ↔  Cycle.at_section(phi)   (forward projection)
  Cycle       ↔  IslandChainOrbit        (same object, different emphasis)

The key insight Wenyin articulated:
  Every "point" we draw on a Poincaré plot is the INTERSECTION of a 3D
  object with a 2D plane. The 3D object is primary; the section cut is
  derived. This means:

    - All section cuts of the same Cycle share IDENTICAL eigenvalues
      (similarity invariance of DPm under conjugation)
    - Manifolds W^u, W^s are 3D invariant manifolds; each section plot
      shows their 2D cross-section
    - Computing manifolds per-section separately is redundant — one
      msp.run_fwd_rev call covers all sections simultaneously

Architecture vision:
  The Poincaré plotting pipeline should be:
    1. Build 3D objects: Cycle, Tube, TubeChain, ResonanceStructure
    2. "Slice" them at each phi_section → section views (fast)
    3. Plot the section views

  Step 2 is the key: given a 3D Cycle, getting its section cut at
  any phi is just reading off ChainFixedPoint at that phi (already
  stored in IslandChainOrbit.fixed_points).

Public API
----------
Cycle(orbit, kind)
    Wraps an IslandChainOrbit with semantic kind ('X' or 'O').
    Provides clean access to stability, manifold seeds, section cuts.

    Properties:
        .m, .n, .Np         -- resonance numbers
        .kind               -- 'X' (hyperbolic) or 'O' (elliptic)
        .eigenvalues        -- (λ_s, λ_u) shared across all sections
        .stability_index    -- Tr(DPm)/2
        .greene_residue     -- (2 - Tr)/4
        .at_section(phi)    -- ChainFixedPoint at this section
        .section_phis       -- list of available sections

    Methods for manifold seeding:
        .unstable_seeds(phi, n_seeds, init_length) -> (R_arr, Z_arr)
        .stable_seeds(phi, n_seeds, init_length)   -> (R_arr, Z_arr)

    Methods for section cut:
        .to_fixed_point(phi) -> topoquest.plot.topology.FixedPoint
        .section_cut(phi)    -> (R, Z, DPm, kind)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pyna.topo.island_chain import IslandChainOrbit, ChainFixedPoint


@dataclass
class Cycle:
    """Continuous-time periodic orbit — the 3D counterpart of FixedPoint.

    Parameters
    ----------
    orbit : IslandChainOrbit
        The underlying cyna-computed periodic orbit, with fixed points
        at all requested Poincaré sections.
    kind : str
        'X' (hyperbolic) or 'O' (elliptic).  If None, inferred from
        the first available ChainFixedPoint.
    label : str, optional
        Human-readable label.
    """
    orbit: IslandChainOrbit
    kind: Optional[str] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind is None:
            fps = self.orbit.fixed_points
            if fps:
                self.kind = fps[0].kind
        if self.kind not in ('X', 'O', None):
            raise ValueError(f"kind must be 'X', 'O' or None, got {self.kind!r}")

    # ── Resonance numbers ─────────────────────────────────────────────────────

    @property
    def m(self) -> int:
        return self.orbit.m

    @property
    def n(self) -> int:
        return self.orbit.n

    @property
    def Np(self) -> int:
        return self.orbit.Np

    @property
    def section_phis(self) -> List[float]:
        return list(self.orbit.section_phis or [])

    # ── Stability (section-independent — similarity invariant) ───────────────

    @property
    def DPm(self) -> np.ndarray:
        """Monodromy matrix (2×2). Same eigenvalues at all sections."""
        fps = self.orbit.fixed_points
        if not fps:
            return np.eye(2)
        return np.asarray(fps[0].DPm, dtype=float)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of DPm: (λ_s, λ_u) for X-cycle, both on unit circle for O."""
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
    def is_hyperbolic(self) -> bool:
        return self.kind == 'X'

    def _eigenvectors(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (stable_evec, unstable_evec) unit vectors, or (None, None)."""
        if not self.is_hyperbolic:
            return None, None
        evals, evecs = np.linalg.eig(self.DPm)
        mods = np.abs(evals)
        iu = int(np.argmax(mods))
        is_ = 1 - iu
        evec_u = np.real(evecs[:, iu]); evec_u /= np.linalg.norm(evec_u)
        evec_s = np.real(evecs[:, is_]); evec_s /= np.linalg.norm(evec_s)
        # Canonical sign: Z-component >= 0
        if evec_u[1] < 0: evec_u = -evec_u
        if evec_s[1] < 0: evec_s = -evec_s
        return evec_s, evec_u

    @property
    def stable_eigenvec(self) -> Optional[np.ndarray]:
        return self._eigenvectors()[0]

    @property
    def unstable_eigenvec(self) -> Optional[np.ndarray]:
        return self._eigenvectors()[1]

    # ── Section cut ──────────────────────────────────────────────────────────

    def at_section(self, phi: float, tol: float = 0.08) -> Optional[ChainFixedPoint]:
        """Return the ChainFixedPoint at the nearest stored section."""
        fps = self.orbit.fixed_points
        if not fps:
            return None
        best = min(fps, key=lambda fp: abs(fp.phi - phi))
        if abs(best.phi - phi) > tol:
            return None
        return best

    def section_RZ(self, phi: float) -> Optional[Tuple[float, float]]:
        """(R, Z) coordinates at this section, or None."""
        fp = self.at_section(phi)
        return (float(fp.R), float(fp.Z)) if fp else None

    def to_fixed_point(self, phi: float):
        """Convert to topoquest.plot.topology.FixedPoint for plotting."""
        from topoquest.plot.topology import FixedPoint as PlotFP
        fp = self.at_section(phi)
        if fp is None:
            return None
        return PlotFP(R=fp.R, Z=fp.Z, phi=phi, DPm=np.asarray(fp.DPm),
                      kind=self.kind or fp.kind)

    # ── Manifold seed generation ──────────────────────────────────────────────

    def unstable_seeds(
        self,
        phi: float,
        n_seeds: int = 40,
        init_length: float = 1e-6,
        max_length: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Seed points along the unstable eigenvector at section phi.

        Returns (R_seeds, Z_seeds) — log-spaced along ±evec_u.
        Used to seed msp.run for W^u tracing.
        """
        fp = self.at_section(phi)
        evec = self.unstable_eigenvec
        if fp is None or evec is None:
            return np.array([]), np.array([])
        eps = np.logspace(np.log10(init_length), np.log10(max_length), n_seeds)
        R_all = np.concatenate([fp.R + eps * evec[0], fp.R - eps * evec[0]])
        Z_all = np.concatenate([fp.Z + eps * evec[1], fp.Z - eps * evec[1]])
        return R_all, Z_all

    def stable_seeds(
        self,
        phi: float,
        n_seeds: int = 40,
        init_length: float = 1e-6,
        max_length: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Seed points along the stable eigenvector at section phi.

        Returns (R_seeds, Z_seeds) — log-spaced along ±evec_s.
        Used to seed msp.run_fwd_rev (reverse trace) for W^s tracing.
        """
        fp = self.at_section(phi)
        evec = self.stable_eigenvec
        if fp is None or evec is None:
            return np.array([]), np.array([])
        eps = np.logspace(np.log10(init_length), np.log10(max_length), n_seeds)
        R_all = np.concatenate([fp.R + eps * evec[0], fp.R - eps * evec[0]])
        Z_all = np.concatenate([fp.Z + eps * evec[1], fp.Z - eps * evec[1]])
        return R_all, Z_all

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        kind = self.kind or '?'
        TR = float(np.trace(self.DPm))
        lam = sorted(np.abs(np.linalg.eigvals(self.DPm)))
        return (
            f"Cycle(m={self.m}, n={self.n}, kind={kind}, "
            f"label={self.label!r})\n"
            f"  Tr(DPm)={TR:.4f}  eigenvalues={lam[0]:.4f},{lam[1]:.4f}  "
            f"Greene_R={self.greene_residue:.4f}\n"
            f"  sections: {self.section_phis}"
        )


# ---------------------------------------------------------------------------
# CycleChain: all Cycles of one resonance family
# ---------------------------------------------------------------------------

class CycleChain:
    """All periodic orbits (Cycles) of one resonance (m/n).

    This is the continuous-time counterpart of IslandChain (discrete).
    It wraps a list of Cycles (each from one IslandChainOrbit seed) and
    provides section-view queries and manifold seed generation.

    The key property: all Cycles in a CycleChain share the same DPm
    eigenvalues (conjugation invariance), so stability is a property
    of the resonance, not individual section points.
    """

    def __init__(
        self,
        cycles: List[Cycle],
        m: int,
        n: int,
        Np: int,
        kind: Optional[str] = None,
        label: Optional[str] = None,
    ):
        self.cycles = list(cycles)
        self.m = m
        self.n = n
        self.Np = Np
        self.kind = kind or (cycles[0].kind if cycles else None)
        self.label = label

    @property
    def expected_n_cycles(self) -> int:
        """Expected number of distinct cycles per section: m // gcd(m, n)."""
        return self.m // gcd(self.m, self.n)

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    @property
    def is_complete(self) -> bool:
        return self.n_cycles == self.expected_n_cycles

    def section_fixed_points(
        self,
        phi: float,
        tol: float = 0.08,
    ) -> List[ChainFixedPoint]:
        """All ChainFixedPoints at section phi from all Cycles."""
        fps = []
        for c in self.cycles:
            fp = c.at_section(phi, tol=tol)
            if fp is not None:
                fps.append(fp)
        return fps

    def section_plot_fixed_points(self, phi: float):
        """All topoquest.plot.topology.FixedPoint objects at section phi."""
        return [fp for c in self.cycles
                for fp in ([c.to_fixed_point(phi)] if c.to_fixed_point(phi) else [])]

    def all_manifold_seeds(
        self,
        phi: float,
        n_seeds: int = 40,
        r_min: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate unstable and stable seeds for all outer X-cycles at phi.

        Returns (R_u, Z_u, R_s, Z_s) — combined seed arrays for one
        msp.run / msp.run_fwd_rev call covering all cycles.

        r_min : float
            Minimum R to include (filter inner X-points, keep boundary ones).
        """
        R_u_all, Z_u_all, R_s_all, Z_s_all = [], [], [], []
        for c in self.cycles:
            if c.kind != 'X':
                continue
            fp = c.at_section(phi)
            if fp is None or float(fp.R) < r_min:
                continue
            Ru, Zu = c.unstable_seeds(phi, n_seeds=n_seeds)
            Rs, Zs = c.stable_seeds(phi, n_seeds=n_seeds)
            R_u_all.append(Ru); Z_u_all.append(Zu)
            R_s_all.append(Rs); Z_s_all.append(Zs)
        if not R_u_all:
            return np.array([]), np.array([]), np.array([]), np.array([])
        return (np.concatenate(R_u_all), np.concatenate(Z_u_all),
                np.concatenate(R_s_all), np.concatenate(Z_s_all))

    @classmethod
    def from_orbits(
        cls,
        orbits: List[IslandChainOrbit],
        kind: str,
        label: Optional[str] = None,
    ) -> "CycleChain":
        """Build a CycleChain from a list of IslandChainOrbit objects."""
        if not orbits:
            raise ValueError("CycleChain.from_orbits: empty orbit list")
        m, n, Np = orbits[0].m, orbits[0].n, orbits[0].Np
        cycles = [Cycle(orbit=o, kind=kind) for o in orbits]
        return cls(cycles=cycles, m=m, n=n, Np=Np, kind=kind, label=label)

    def summary(self) -> str:
        return (
            f"CycleChain(m={self.m}, n={self.n}, kind={self.kind}, "
            f"cycles={self.n_cycles}/{self.expected_n_cycles}, "
            f"complete={self.is_complete})"
        )


# ---------------------------------------------------------------------------
# ResonanceCycles: X + O cycle chains for one resonance
# ---------------------------------------------------------------------------

class ResonanceCycles:
    """Full continuous-time resonance structure: X-CycleChain + O-CycleChain.

    This is the top-level 3D object representing one resonance (m/n).
    Section plots are derived by slicing this 3D structure.

    Replaces the combination of:
      ResonanceStructure (TubeChain-based)  ← still valid, this is a
                                               cleaner higher-level wrapper
    """

    def __init__(
        self,
        x_chain: Optional[CycleChain] = None,
        o_chain: Optional[CycleChain] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
        Np: Optional[int] = None,
        label: Optional[str] = None,
    ):
        self.x_chain = x_chain
        self.o_chain = o_chain
        first = x_chain or o_chain
        self.m  = m  or (first.m  if first else None)
        self.n  = n  or (first.n  if first else None)
        self.Np = Np or (first.Np if first else None)
        self.label = label

    def section_xpoints(self, phi: float) -> list:
        if self.x_chain is None: return []
        return self.x_chain.section_plot_fixed_points(phi)

    def section_opoints(self, phi: float) -> list:
        if self.o_chain is None: return []
        return self.o_chain.section_plot_fixed_points(phi)

    def fp_by_sec(self, phi_sections: List[float]) -> dict:
        """Return {phi: {'xpts': [...], 'opts': [...]}} for plot_poincare_2x2."""
        return {
            float(phi): {
                'xpts': self.section_xpoints(float(phi)),
                'opts': self.section_opoints(float(phi)),
            }
            for phi in phi_sections
        }

    def manifold_seeds_all_sections(
        self,
        phi_sections: List[float],
        n_seeds: int = 40,
        r_min: float = 1.0,
    ) -> dict:
        """Return {phi: (R_u, Z_u, R_s, Z_s)} for manifold tracing.

        All seeds are derived from the 3D Cycle objects (continuous time).
        The caller passes these to msp.run / run_fwd_rev once per section.
        """
        if self.x_chain is None:
            return {float(p): (np.array([]), np.array([]),
                               np.array([]), np.array([])) for p in phi_sections}
        return {
            float(phi): self.x_chain.all_manifold_seeds(float(phi), n_seeds, r_min)
            for phi in phi_sections
        }

    @classmethod
    def from_orbits(
        cls,
        x_orbits: Optional[List[IslandChainOrbit]] = None,
        o_orbits: Optional[List[IslandChainOrbit]] = None,
        label: Optional[str] = None,
    ) -> "ResonanceCycles":
        x_chain = CycleChain.from_orbits(x_orbits, kind='X') if x_orbits else None
        o_chain = CycleChain.from_orbits(o_orbits, kind='O') if o_orbits else None
        return cls(x_chain=x_chain, o_chain=o_chain, label=label)

    def summary(self) -> str:
        xstr = self.x_chain.summary() if self.x_chain else "X: none"
        ostr = self.o_chain.summary() if self.o_chain else "O: none"
        return f"ResonanceCycles(m={self.m}/n={self.n})\n  {xstr}\n  {ostr}"
