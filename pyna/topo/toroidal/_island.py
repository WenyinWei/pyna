"""toroidal._island — Island, IslandChain, ChainRole, _ToriMixin."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from math import gcd
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np

from pyna.topo.core import Island as _Island, IslandChain as _IslandChain
from ._fixed_point import FixedPoint
from ._cycle import PeriodicOrbit


class _ToriMixin:
    """Mixin for objects that manage a radial stack of tori."""

    def __init__(self):
        self._tori: List[Any] = []
        self._r_vals: List[float] = []

    def add_torus(self, torus: Any, r: float):
        import bisect
        idx = bisect.bisect_left(self._r_vals, r)
        self._r_vals.insert(idx, r)
        self._tori.insert(idx, torus)
        if "rotation_profile" in self.__dict__:
            del self.__dict__["rotation_profile"]

    def _central_rotation_vector(self) -> Tuple[float, ...]:
        raise NotImplementedError

    @cached_property
    def rotation_profile(self):
        r_vals = [0.0] + list(self._r_vals)
        rv_list = [self._central_rotation_vector()] + [t.rotation_vector for t in self._tori]
        r_arr = np.array(r_vals)
        dim = len(rv_list[0])
        rv_arr = np.array([[rv[i] for rv in rv_list] for i in range(dim)])

        if len(r_arr) < 2:
            const_val = tuple(float(rv_list[0][i]) for i in range(dim))
            def profile_constant(r: float) -> Tuple[float, ...]:
                return const_val
            return profile_constant

        try:
            from scipy.interpolate import PchipInterpolator
            interps = [PchipInterpolator(r_arr, rv_arr[i]) for i in range(dim)]
        except ImportError:
            interps = [lambda x, i=i: np.interp(x, r_arr, rv_arr[i]) for i in range(dim)]

        def profile(r: float) -> Tuple[float, ...]:
            return tuple(float(interps[i](r)) for i in range(dim))

        return profile


class ChainRole(Enum):
    """Semantic role of an IslandChain in the island hierarchy."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    NESTED = "nested"


@dataclass(eq=False)
class Island(_ToriMixin, _Island):
    """Toroidal magnetic island on a Poincaré section.

    Inherits from core.Island and adds toroidal-specific fields.
    """

    tube: Optional[Any] = field(default=None, repr=False)
    tube_chain: Optional[Any] = field(default=None, repr=False)
    resonance_index: Optional[int] = field(default=None, repr=False)
    section: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        _ToriMixin.__init__(self)

    @property
    def O_point(self) -> FixedPoint:
        return self.O_orbit[0]

    @property
    def X_points(self) -> List[FixedPoint]:
        return [orb[0] for orb in self.X_orbits if len(orb) > 0]

    @property
    def R(self) -> float:
        return self.O_point.R

    @property
    def Z(self) -> float:
        return self.O_point.Z

    @property
    def phi(self) -> float:
        return self.O_point.phi

    @property
    def m(self) -> int:
        if self.tube_chain is not None:
            return self.tube_chain.m
        return 0

    @property
    def n(self) -> int:
        if self.tube_chain is not None:
            return self.tube_chain.n
        return 0

    def _set_next(self, island: "Island") -> None:
        self._next = island

    def _set_prev(self, island: "Island") -> None:
        self._prev = island

    def _set_last(self, island: "Island") -> None:
        self._prev = island

    def _central_rotation_vector(self) -> Tuple[float, ...]:
        mono = self.O_orbit.monodromy
        if mono is None:
            return (0.0,)
        import cmath
        eigs = mono.eigenvalues
        for eig in eigs:
            if abs(eig.imag) > 1e-10:
                angle = abs(cmath.phase(eig)) / (2 * np.pi)
                return (angle,)
        return (0.0,)

    def root_island(self) -> "Island":
        isl = self
        while (isl.parent_chain is not None
               and hasattr(isl.parent_chain, 'parent_island')
               and isl.parent_chain.parent_island is not None):
            isl = isl.parent_chain.parent_island
        return isl

    def axis_fixed_point(self) -> FixedPoint:
        return self.root_island().O_point

    def section_cut(self, section=None) -> list:
        if section is not None:
            raise ValueError("Island is already a discrete section object; cut the parent Tube instead.")
        return [self]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'Island',
            'label': self.label,
            'R': self.O_point.R,
            'Z': self.O_point.Z,
            'phi': self.O_point.phi,
            'kind': self.O_point.kind,
            'n_X_orbits': len(self.X_orbits),
            'n_child_chains': len(self.child_chains),
        }


@dataclass(eq=False)
class IslandChain(_IslandChain):
    """Toroidal island chain: all islands of a resonance on one section."""

    m: int = 0
    n: int = 0
    parent_tube: Optional[Any] = None
    role: Optional[ChainRole] = None

    @property
    def winding(self) -> Tuple[int, ...]:
        return (self.m, self.n) if self.n else (self.m,)

    @property
    def expected_n_islands(self) -> int:
        islands_per_tube = self.m // gcd(self.m, self.n) if self.n > 0 else self.m
        n_tubes_included = int(self.metadata.get('n_tubes_included', 1))
        return islands_per_tube * n_tubes_included

    @property
    def n_independent_orbits(self) -> int:
        return gcd(self.m, self.n) if self.n > 0 else 1

    @property
    def is_connected(self) -> bool:
        return self.n_independent_orbits == 1

    @property
    def orbit_groups(self) -> List[List[Island]]:
        n_orbs = self.n_independent_orbits
        groups: List[List[Island]] = [[] for _ in range(n_orbs)]
        for idx, isl in enumerate(self.islands):
            groups[idx % n_orbs].append(isl)
        return groups

    @property
    def q_rational(self) -> float:
        return self.m / self.n if self.n > 0 else float('inf')

    @property
    def depth(self) -> int:
        d = 0
        p = self.parent_island
        while p is not None:
            d += 1
            p = p.parent_chain.parent_island if (p.parent_chain is not None
                 and hasattr(p.parent_chain, 'parent_island')) else None
        return d

    @property
    def O_points(self) -> List[FixedPoint]:
        return [isl.O_point for isl in self.islands]

    @property
    def X_points(self) -> List[FixedPoint]:
        return [fp for isl in self.islands for fp in isl.X_points]

    @property
    def section_phi(self) -> Optional[float]:
        return float(self.islands[0].O_point.phi) if self.islands else None

    @classmethod
    def from_O_X_points(
        cls,
        O_points: Optional[Sequence[Any]] = None,
        X_points: Optional[Sequence[Any]] = None,
        *,
        m: int = 1,
        n: int = 1,
        proximity_tol: float = 1.0,
    ) -> "IslandChain":
        def _to_fp(pt, phi=0.0):
            if isinstance(pt, FixedPoint):
                return pt
            arr = np.asarray(pt, dtype=float).ravel()
            return FixedPoint(phi=phi, R=float(arr[0]), Z=float(arr[1]), DPm=np.eye(2), kind='O')

        def _to_xfp(pt, phi=0.0):
            if isinstance(pt, FixedPoint):
                return pt
            arr = np.asarray(pt, dtype=float).ravel()
            return FixedPoint(phi=phi, R=float(arr[0]), Z=float(arr[1]),
                              DPm=np.array([[2.0, 0.0], [0.0, 0.5]]), kind='X')

        o_fps = [_to_fp(p) for p in (O_points or [])]
        x_fps = [_to_xfp(p) for p in (X_points or [])]

        islands: List[Island] = []
        for o_fp in o_fps:
            nearby_x = [x_fp for x_fp in x_fps
                        if np.hypot(x_fp.R - o_fp.R, x_fp.Z - o_fp.Z) < proximity_tol]
            islands.append(Island(
                O_orbit=PeriodicOrbit(points=[o_fp]),
                X_orbits=[PeriodicOrbit(points=[xfp]) for xfp in nearby_x],
            ))

        return cls(m=m, n=n, islands=islands,
                   metadata={'n_tubes_included': len(islands) // max(1, (m // gcd(m, n))) if islands else 0})

    @classmethod
    def from_seed_points(cls, O_points=None, X_points=None, *, m=1, n=1, proximity_tol=1.0) -> "IslandChain":
        return cls.from_O_X_points(O_points=O_points, X_points=X_points, m=m, n=n, proximity_tol=proximity_tol)

    def summary(self) -> str:
        return f"IslandChain(m={self.m}, n={self.n}, islands={self.n_islands}/{self.expected_n_islands}, connected={self.is_connected})"

    def split_into_subchains(self, connectivity_groups: List[List[int]]) -> None:
        pass

    def warn_if_incomplete(self, prefix: str = "") -> None:
        if self.n_islands < self.expected_n_islands:
            warnings.warn(
                f"{prefix}IslandChain incomplete: expected {self.expected_n_islands} "
                f"islands for m/n={self.m}/{self.n}, found {self.n_islands}."
            )

    def section_cut(self, section=None) -> list:
        if section is not None:
            raise ValueError("IslandChain is already a discrete section object; cut the parent TubeChain instead.")
        return list(self.islands)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'IslandChain',
            'label': self.label,
            'm': self.m, 'n': self.n,
            'n_islands': self.n_islands,
            'expected_n_islands': self.expected_n_islands,
            'is_connected': self.is_connected,
            'n_independent_orbits': self.n_independent_orbits,
            'depth': self.depth,
            'role': self.role.value if self.role else None,
        }
