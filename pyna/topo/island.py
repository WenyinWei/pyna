from __future__ import annotations

"""Generic island / island-chain objects for discrete maps.

These classes are coordinate-system agnostic.  They represent the reduced
objects obtained after cutting continuous-time geometry with a section and
viewing the induced Poincaré map.
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pyna.topo._base import InvariantSet
from pyna.topo.core import PeriodicOrbit, SectionPoint


class ChainRole(enum.Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    NESTED = "nested"


@dataclass(eq=False)
class Island(InvariantSet):
    """One island of a discrete Poincaré-map resonance structure."""

    O_orbit: PeriodicOrbit = field(default_factory=PeriodicOrbit)
    X_orbits: List[PeriodicOrbit] = field(default_factory=list)
    child_chains: List["IslandChain"] = field(default_factory=list)
    parent_chain: Optional["IslandChain"] = field(default=None, repr=False)
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _next: Optional["Island"] = field(default=None, init=False, repr=False)
    _prev: Optional["Island"] = field(default=None, init=False, repr=False)

    @property
    def O_point(self) -> Optional[SectionPoint]:
        return self.O_orbit.points[0] if self.O_orbit.points else None

    @property
    def X_points(self) -> List[SectionPoint]:
        return [orb.points[0] for orb in self.X_orbits if orb.points]

    def step(self) -> "Island":
        if self._next is None:
            raise RuntimeError("Island not linked inside an IslandChain")
        return self._next

    def step_back(self) -> "Island":
        if self._prev is None:
            raise RuntimeError("Island not linked inside an IslandChain")
        return self._prev

    def add_child_chain(self, chain: "IslandChain") -> None:
        chain.parent_island = self
        self.child_chains.append(chain)

    def section_cut(self, section=None) -> list:
        return [self]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "Island",
            "label": self.label,
            "has_O_orbit": bool(self.O_orbit.points),
            "n_X_orbits": len(self.X_orbits),
            "n_child_chains": len(self.child_chains),
        }


@dataclass(eq=False)
class IslandChain(InvariantSet):
    """Discrete island chain obtained on a section of a flow or map."""

    islands: List[Island] = field(default_factory=list)
    period: Optional[int] = None
    label: Optional[str] = None
    role: Optional[ChainRole] = None
    parent_island: Optional[Island] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for i, isl in enumerate(self.islands):
            isl.parent_chain = self
            if self.islands:
                isl._next = self.islands[(i + 1) % len(self.islands)]
                isl._prev = self.islands[(i - 1) % len(self.islands)]
        if self.period is None and self.islands:
            self.period = len(self.islands)

    @property
    def n_islands(self) -> int:
        return len(self.islands)

    @property
    def O_points(self) -> List[SectionPoint]:
        return [isl.O_point for isl in self.islands if isl.O_point is not None]

    @property
    def X_points(self) -> List[SectionPoint]:
        pts: List[SectionPoint] = []
        for isl in self.islands:
            pts.extend(isl.X_points)
        return pts

    def add_island(self, island: Island) -> None:
        island.parent_chain = self
        if self.islands:
            prev = self.islands[-1]
            prev._next = island
            island._prev = prev
            island._next = self.islands[0]
            self.islands[0]._prev = island
        else:
            island._next = island
            island._prev = island
        self.islands.append(island)
        if self.period is None:
            self.period = len(self.islands)

    def section_cut(self, section=None) -> list:
        return list(self.islands)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "IslandChain",
            "label": self.label,
            "period": self.period,
            "n_islands": self.n_islands,
            "n_O_points": len(self.O_points),
            "n_X_points": len(self.X_points),
            "role": None if self.role is None else self.role.value,
        }


__all__ = ["ChainRole", "Island", "IslandChain"]
