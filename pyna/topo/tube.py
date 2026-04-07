"""Continuous-time tube and tube-chain abstractions.

These classes provide the continuous-time counterparts of the discrete
Poincaré-map objects ``Island`` / ``IslandChain``.

Conceptual correspondence
-------------------------
- ``Tube`` wraps one continuous-time periodic orbit (typically represented in
  practice by an :class:`~pyna.topo.island_chain.IslandChainOrbit`).
  Cutting the tube by a Poincaré section produces one discrete fixed point;
  for an O-type tube this becomes an ``Island`` centre on that section.
- ``TubeChain`` is a collection of tubes belonging to the same m/n resonance.
  Cutting the full chain by a Poincaré section produces a discrete
  ``IslandChain``.

This layer is useful when the continuous-time connectivity is more robust than
per-section Newton searches: if one section misses an island, the raw 3-D tube
still exists and can be used to recover / debug the missing discrete point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Any, Dict, List, Optional, Sequence
import warnings

import numpy as np

from pyna.topo.island import Island, IslandChain
from pyna.topo.island_chain import ChainFixedPoint, IslandChainOrbit


@dataclass
class Tube:
    """Continuous-time counterpart of one discrete island / fixed-point family.

    Parameters
    ----------
    orbit : IslandChainOrbit
        Underlying continuous-time orbit object.  One tube corresponds to one
        connected periodic orbit in the 3-D field.
    label : str, optional
        Human-readable label.
    debug_info : dict
        Additional metadata for diagnostics.
    """

    orbit: IslandChainOrbit
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_orbit(cls, orbit: IslandChainOrbit, label: Optional[str] = None) -> "Tube":
        return cls(orbit=orbit, label=label)

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
    def seed_phi(self) -> float:
        return self.orbit.seed_phi

    @property
    def seed_RZ(self) -> tuple:
        return self.orbit.seed_RZ

    @property
    def section_phis(self) -> Optional[List[float]]:
        return self.orbit.section_phis

    @property
    def kind(self) -> Optional[str]:
        diag = self.orbit.diagnostics(self.section_phis)
        if diag['mixed_kind']:
            return None
        return diag['dominant_kind']

    def at_section(self, phi: float, tol: float = 1e-6) -> List[ChainFixedPoint]:
        return self.orbit.fixed_points_at_section(phi, tol=tol)

    def diagnostics(self, requested_phis: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        diag = self.orbit.diagnostics(requested_phis=requested_phis)
        diag['label'] = self.label
        diag['tube_kind'] = self.kind
        return diag

    def summary(self) -> str:
        kind = self.kind or 'mixed'
        base = self.orbit.summary()
        return f"Tube(kind={kind}, label={self.label})\n{base}"

    def orbit_xyz(self) -> Optional[np.ndarray]:
        return self.orbit.orbit_xyz()

    def to_island(
        self,
        phi: float,
        *,
        x_points: Optional[List[np.ndarray]] = None,
        level: int = 1,
        label: Optional[str] = None,
        tol: float = 1e-6,
    ) -> Island:
        """Cut the tube by a section and convert it to a discrete ``Island``.

        This is meaningful primarily for O-type tubes.  X-type tubes can still
        be cut, but the resulting object should usually be used as neighbour
        information for an O-type island instead.
        """
        fps = self.at_section(phi, tol=tol)
        if len(fps) != 1:
            raise ValueError(
                f"Tube.to_island expected exactly 1 section point at phi={phi:.6f}, got {len(fps)}"
            )
        fp = fps[0]
        isl = Island(
            period_n=self.m,
            O_point=np.array([fp.R, fp.Z], dtype=float),
            X_points=[] if x_points is None else [np.asarray(x, dtype=float) for x in x_points],
            halfwidth=float('nan'),
            level=level,
            label=label or self.label,
            debug_info={
                'tube_kind': self.kind,
                'phi_section': float(phi),
                'greene_residue': float(fp.greene_residue),
                'seed_RZ': tuple(float(v) for v in self.seed_RZ),
            },
        )
        return isl


@dataclass
class TubeChain:
    """Continuous-time counterpart of a discrete ``IslandChain``.

    A ``TubeChain`` contains one tube per connected periodic orbit on the same
    m/n resonance.  Cutting the chain by a Poincaré section produces the full
    set of discrete fixed points on that section.
    """

    m: int
    n: int
    Np: int
    tubes: List[Tube] = field(default_factory=list)
    kind: Optional[str] = None
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_orbits(
        cls,
        orbits: Sequence[IslandChainOrbit],
        *,
        expected_kind: Optional[str] = None,
        label: Optional[str] = None,
    ) -> "TubeChain":
        if not orbits:
            raise ValueError("TubeChain.from_orbits requires at least one orbit")
        m = orbits[0].m
        n = orbits[0].n
        Np = orbits[0].Np
        tubes = [Tube.from_orbit(orbit, label=f"tube[{i}]") for i, orbit in enumerate(orbits)]
        chain_kind = expected_kind
        if chain_kind is None:
            kinds = {tube.kind for tube in tubes}
            kinds.discard(None)
            chain_kind = next(iter(kinds)) if len(kinds) == 1 else None
        return cls(m=m, n=n, Np=Np, tubes=tubes, kind=chain_kind, label=label)

    @property
    def expected_n_tubes(self) -> int:
        return self.m // gcd(self.m, self.n)

    @property
    def n_tubes(self) -> int:
        return len(self.tubes)

    def section_fixed_points(self, phi: float, tol: float = 1e-6) -> List[ChainFixedPoint]:
        fps: List[ChainFixedPoint] = []
        for tube in self.tubes:
            fps.extend(tube.at_section(phi, tol=tol))
        return fps

    def diagnostics(
        self,
        requested_phis: Optional[Sequence[float]] = None,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        tube_diags = [tube.diagnostics(requested_phis=requested_phis) for tube in self.tubes]
        per_section_counts: Dict[float, int] = {}
        if requested_phis is None:
            requested_phis = self.tubes[0].section_phis if self.tubes else []
        if requested_phis is None:
            requested_phis = []
        for phi in requested_phis:
            per_section_counts[float(phi)] = len(self.section_fixed_points(float(phi), tol=tol))
        return {
            'm': int(self.m),
            'n': int(self.n),
            'Np': int(self.Np),
            'kind': self.kind,
            'label': self.label,
            'expected_n_tubes': int(self.expected_n_tubes),
            'n_tubes': int(self.n_tubes),
            'complete': bool(self.n_tubes == self.expected_n_tubes),
            'section_counts': per_section_counts,
            'tube_kinds': [tube.kind for tube in self.tubes],
            'tube_diagnostics': tube_diags,
        }

    def summary(self) -> str:
        diag = self.diagnostics(self.tubes[0].section_phis if self.tubes else None)
        return (
            f"TubeChain(kind={self.kind}, label={self.label}, m={self.m}, n={self.n}, Np={self.Np}) "
            f"tubes={diag['n_tubes']}/{diag['expected_n_tubes']} section_counts={diag['section_counts']}"
        )

    def warn_if_incomplete(self, requested_phis: Optional[Sequence[float]] = None) -> None:
        diag = self.diagnostics(requested_phis=requested_phis)
        if not diag['complete']:
            warnings.warn(
                f"TubeChain incomplete for m/n={self.m}/{self.n}: "
                f"found {diag['n_tubes']} tubes, expected {diag['expected_n_tubes']}"
            )

    def to_island_chain(
        self,
        phi: float,
        *,
        x_tubechain: Optional["TubeChain"] = None,
        proximity_tol: float = 1.0,
        tol: float = 1e-6,
    ) -> IslandChain:
        """Cut the continuous-time chain by one section and form ``IslandChain``.

        For an O-type tube chain this returns the usual discrete island chain.
        If an X-type ``x_tubechain`` is supplied, its section cuts are used as
        candidate X-points when attaching neighbours to each island.
        """
        O_points = [np.array([fp.R, fp.Z], dtype=float)
                    for fp in self.section_fixed_points(phi, tol=tol)]
        X_points: List[np.ndarray] = []
        if x_tubechain is not None:
            X_points = [np.array([fp.R, fp.Z], dtype=float)
                        for fp in x_tubechain.section_fixed_points(phi, tol=tol)]
        chain = IslandChain.from_fixed_points(
            O_points=O_points,
            X_points=X_points,
            m=self.m,
            n=self.n,
            proximity_tol=proximity_tol,
        )
        chain.warn_if_incomplete(prefix="TubeChain.to_island_chain: ")
        return chain


__all__ = ["Tube", "TubeChain"]
