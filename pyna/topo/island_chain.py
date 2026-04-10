"""pyna.topo.island_chain — compatibility shim (to be removed).

``ChainFixedPoint`` and ``IslandChainOrbit`` were intermediate coupling
classes that tied ``IslandChain`` and ``PeriodicOrbit`` together.  They
violate high-cohesion / low-coupling principles and are being phased out:

* ``ChainFixedPoint`` → ``pyna.topo.invariants.FixedPoint``
* ``IslandChainOrbit`` → ``pyna.topo.tube.Tube`` (direct field storage)

This module keeps them alive only to let old call sites continue importing
without changes while migration proceeds.  It re-exports ``FixedPoint`` as
``ChainFixedPoint`` and provides a thin ``IslandChainOrbit`` wrapper class
whose constructor signature exactly matches the old one.

Do NOT add new features here.  New code should import ``FixedPoint`` and
``Tube`` directly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pyna.topo.invariants import FixedPoint

# ---------------------------------------------------------------------------
# ChainFixedPoint — straight alias for FixedPoint
# ---------------------------------------------------------------------------

#: Alias: ``ChainFixedPoint`` is ``FixedPoint``.
#: Import ``FixedPoint`` from ``pyna.topo.invariants`` in new code.
ChainFixedPoint = FixedPoint


# ---------------------------------------------------------------------------
# IslandChainOrbit
# ---------------------------------------------------------------------------

class IslandChainOrbit:
    """Compatibility wrapper for the old IslandChainOrbit interface.

    .. deprecated::
        Use ``pyna.topo.tube.Tube`` directly.  This class exists only to
        keep old call-sites from breaking during migration.

    Parameters
    ----------
    m, n : int
        Resonance numbers (q = m/n).
    Np : int
        Discrete rotational symmetry of the device.
    fixed_points : list of FixedPoint
        Poincaré fixed points at the recorded sections.
    seed_phi : float
        Toroidal angle of the seed section [rad].
    seed_RZ : (float, float)
        (R, Z) of the seed point.
    section_phis : list of float, optional
        Toroidal angles of all recorded sections.
    orbit_R, orbit_Z, orbit_phi : array-like, optional
        Raw orbit trajectory arrays (for debug / general-section cut).
    orbit_alive : array-like of bool, optional
        Mask for alive trajectory points.
    """

    def __init__(
        self,
        m: int,
        n: int,
        Np: int,
        fixed_points: List[FixedPoint],
        seed_phi: float,
        seed_RZ: Tuple[float, float],
        section_phis: Optional[List[float]] = None,
        orbit_R: Optional[np.ndarray] = None,
        orbit_Z: Optional[np.ndarray] = None,
        orbit_phi: Optional[np.ndarray] = None,
        orbit_alive: Optional[np.ndarray] = None,
    ):
        self.m = int(m)
        self.n = int(n)
        self.Np = int(Np)
        self.fixed_points: List[FixedPoint] = list(fixed_points)
        self.seed_phi = float(seed_phi)
        self.seed_RZ = tuple(seed_RZ)

        # section_phis: deduce from fixed_points if not given
        if section_phis is not None:
            self.section_phis: List[float] = [float(p) for p in section_phis]
        else:
            seen: List[float] = []
            for fp in self.fixed_points:
                if not any(abs(fp.phi - s) < 1e-9 for s in seen):
                    seen.append(float(fp.phi))
            self.section_phis = seen

        # Raw orbit trajectory (optional; used by Tube._general_section_fps)
        self.orbit_R = np.asarray(orbit_R, dtype=float) if orbit_R is not None else None
        self.orbit_Z = np.asarray(orbit_Z, dtype=float) if orbit_Z is not None else None
        self.orbit_phi = np.asarray(orbit_phi, dtype=float) if orbit_phi is not None else None
        self.orbit_alive = (
            np.asarray(orbit_alive, dtype=bool)
            if orbit_alive is not None
            else (np.ones(len(self.orbit_R), dtype=bool) if self.orbit_R is not None else None)
        )

    # ── Tube-compatible properties (used by Tube proxy attributes) ────────────

    @property
    def orbit_debug_available(self) -> bool:
        return self.orbit_R is not None and len(self.orbit_R) > 0

    def fixed_points_at_section(
        self, phi: float, tol: float = 1e-6
    ) -> List[FixedPoint]:
        """Return all FixedPoints whose section matches ``phi``."""
        return [fp for fp in self.fixed_points if abs(fp.phi - phi) < tol]

    def diagnostics(
        self, requested_phis: Optional[Sequence[float]] = None
    ) -> Dict:
        """Return a completeness/diagnostic dictionary."""
        phis = list(requested_phis) if requested_phis is not None else self.section_phis
        fps_by_phi: Dict[float, List[FixedPoint]] = {}
        for phi in phis:
            fps_by_phi[phi] = self.fixed_points_at_section(phi)

        missing = [phi for phi, fps in fps_by_phi.items() if not fps]
        all_kinds = [fp.kind for fp in self.fixed_points]
        kind_counts: Dict[str, int] = {}
        for k in all_kinds:
            kind_counts[k] = kind_counts.get(k, 0) + 1
        mixed_kind = len(set(all_kinds) - {''}) > 1

        dominant_kind: Optional[str] = None
        if not mixed_kind and all_kinds:
            dominant_kind = all_kinds[0]

        return {
            'm': self.m,
            'n': self.n,
            'Np': self.Np,
            'seed_phi': self.seed_phi,
            'seed_RZ': self.seed_RZ,
            'n_fixed_points': len(self.fixed_points),
            'section_phis': self.section_phis,
            'missing_sections': missing,
            'kind_totals': kind_counts,
            'mixed_kind': mixed_kind,
            'dominant_kind': dominant_kind,
            'orbit_debug_available': self.orbit_debug_available,
        }

    def is_complete(
        self,
        expected_phis: Sequence[float],
        expected_kind: Optional[str] = None,
        expected_count_per_section: int = 1,
    ) -> bool:
        """True when every expected section has the right number of fixed points."""
        for phi in expected_phis:
            fps = self.fixed_points_at_section(float(phi))
            if len(fps) < expected_count_per_section:
                return False
            if expected_kind and any(fp.kind != expected_kind for fp in fps):
                return False
        return True

    def debug_summary(
        self,
        requested_phis: Optional[Sequence[float]] = None,
        expected_kind: Optional[str] = None,
    ) -> str:
        d = self.diagnostics(requested_phis)
        lines = [
            f"IslandChainOrbit m={self.m} n={self.n} Np={self.Np}",
            f"  seed: phi={self.seed_phi:.4f} RZ={self.seed_RZ}",
            f"  fixed_points={d['n_fixed_points']}  mixed_kind={d['mixed_kind']}",
            f"  kind_totals={d['kind_totals']}",
            f"  missing_sections={d['missing_sections']}",
        ]
        if expected_kind:
            lines.append(f"  expected_kind={expected_kind}")
        return "\n".join(lines)

    def orbit_xyz(self) -> Optional[np.ndarray]:
        """Return (3, N) array of (R, Z, phi) orbit samples, or None."""
        if not self.orbit_debug_available:
            return None
        return np.vstack([self.orbit_R, self.orbit_Z, self.orbit_phi])

    # ── Connectivity (same logic as IslandChain, for backward compat) ─────────

    @property
    def n_independent_orbits(self) -> int:
        """Number of topologically independent field-line trajectories = gcd(m, n)."""
        return gcd(self.m, self.n) if self.n > 0 else 1

    @property
    def is_connected(self) -> bool:
        """True when all m Poincaré points lie on a single orbit (gcd==1)."""
        return self.n_independent_orbits == 1

    @property
    def n_points_per_orbit(self) -> int:
        """Number of section-cut points per independent orbit = m // gcd(m,n)."""
        return self.m // self.n_independent_orbits

    def visit_sequence(self) -> List[List[int]]:
        """Return the Poincaré visitation order for each independent orbit.

        Under the single-turn Poincaré map P^1, starting from section-cut
        index 0 and stepping by n (mod m) gives the sequence of Poincaré
        points visited on one field-line orbit.

        Returns
        -------
        list of list of int
            One inner list per independent orbit.  Each inner list is the
            ordered sequence of section-cut indices visited on that orbit.

        Examples
        --------
        m=10, n=3  →  [[0, 3, 6, 9, 2, 5, 8, 1, 4, 7]]   (one orbit)
        m=5,  n=5  →  [[0], [1], [2], [3], [4]]            (5 independent orbits)
        m=6,  n=4  →  [[0, 2, 4], [1, 3, 5]]              (2 independent orbits)
        """
        n_orbs = self.n_independent_orbits
        step = self.n % self.m if self.m > 0 else 1

        result: List[List[int]] = []
        visited = set()

        for start in range(n_orbs):
            if start in visited:
                continue
            seq: List[int] = []
            idx = start
            while idx not in visited:
                seq.append(idx)
                visited.add(idx)
                idx = (idx + step) % self.m
            result.append(seq)

        return result

    def summary(self) -> str:
        """One-line human-readable summary."""
        return self.debug_summary()

    def __repr__(self) -> str:
        return (
            f"IslandChainOrbit(m={self.m}, n={self.n}, Np={self.Np}, "
            f"n_fps={len(self.fixed_points)}, seed_phi={self.seed_phi:.4f})"
        )
