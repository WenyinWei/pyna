"""Builder objects for finite-dimensional topology geometry.

Builders encode explicit promotion rules.  A sampled trajectory is just a
trajectory until a caller asks to build a ``Cycle`` and passes the required
closure policy.  The same idea applies to discrete ``Orbit`` versus
``PeriodicOrbit``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np

from pyna.topo.adapters import (
    as_cycle,
    as_orbit,
    as_periodic_orbit,
    as_section_point,
    as_trajectory,
)
from pyna.topo.core import (
    Cycle,
    Island,
    IslandChain,
    Orbit,
    PeriodicOrbit,
    SectionPoint,
    Trajectory,
    Tube,
    TubeChain,
)


def _representative_state(orbit: PeriodicOrbit) -> Optional[np.ndarray]:
    if orbit.representative_state is not None:
        return np.asarray(orbit.representative_state, dtype=float)
    for pt in orbit.points:
        if pt.state is not None:
            return np.asarray(pt.state, dtype=float)
    if orbit.orbit_trace is not None and orbit.orbit_trace.n_samples:
        return orbit.orbit_trace.states[0]
    return None


def _assign_x_orbits(
    o_orbits: Sequence[PeriodicOrbit],
    x_orbits: Sequence[PeriodicOrbit],
    *,
    proximity_tol: Optional[float],
) -> List[List[PeriodicOrbit]]:
    assignments: List[List[PeriodicOrbit]] = [[] for _ in o_orbits]
    if not o_orbits or not x_orbits:
        return assignments

    if proximity_tol is None and len(o_orbits) == len(x_orbits):
        for idx, x_orbit in enumerate(x_orbits):
            assignments[idx].append(x_orbit)
        return assignments

    o_states = [_representative_state(orbit) for orbit in o_orbits]
    for x_orbit in x_orbits:
        x_state = _representative_state(x_orbit)
        if x_state is None:
            continue

        best_idx = None
        best_dist = np.inf
        for idx, o_state in enumerate(o_states):
            if o_state is None or o_state.shape != x_state.shape:
                continue
            dist = float(np.linalg.norm(x_state - o_state))
            if dist < best_dist:
                best_idx = idx
                best_dist = dist

        if best_idx is not None and (proximity_tol is None or best_dist <= float(proximity_tol)):
            assignments[best_idx].append(x_orbit)

    return assignments


@dataclass
class GeometryBuilder:
    """Convenience builder for core trajectory/orbit/cycle objects."""

    closure_tol: float = 1e-8

    def trajectory(self, obj: Any, **kwargs: Any) -> Trajectory:
        return as_trajectory(obj, **kwargs)

    def orbit(self, obj: Any, **kwargs: Any) -> Orbit:
        return as_orbit(obj, **kwargs)

    def cycle(
        self,
        obj: Any,
        *,
        require_closed: bool = True,
        closure_tol: Optional[float] = None,
        **kwargs: Any,
    ) -> Cycle:
        return as_cycle(
            obj,
            require_closed=require_closed,
            closure_tol=self.closure_tol if closure_tol is None else float(closure_tol),
            **kwargs,
        )

    def periodic_orbit(
        self,
        obj: Any,
        *,
        verify: bool = True,
        closure_tol: Optional[float] = None,
        **kwargs: Any,
    ) -> PeriodicOrbit:
        return as_periodic_orbit(
            obj,
            verify=verify,
            closure_tol=self.closure_tol if closure_tol is None else float(closure_tol),
            **kwargs,
        )

    def section_point(self, obj: Any, **kwargs: Any) -> SectionPoint:
        return as_section_point(obj, **kwargs)


@dataclass
class IslandChainBuilder:
    """Builder for generic discrete island chains."""

    label: Optional[str] = None
    period: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    proximity_tol: Optional[float] = None
    _islands: List[Island] = field(default_factory=list, init=False, repr=False)

    def add_island(
        self,
        O_orbit: Any,
        X_orbits: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> "IslandChainBuilder":
        o_orbit = as_periodic_orbit(O_orbit, verify=False)
        x_orbits = [as_periodic_orbit(orb, verify=False) for orb in (X_orbits or [])]
        island = Island(
            O_orbit=o_orbit,
            X_orbits=x_orbits,
            label=label or self.label,
            metadata=dict(metadata or {}),
        )
        self._islands.append(island)
        return self

    def add_from_orbit_clouds(
        self,
        O_orbits: Sequence[Any],
        X_orbits: Optional[Sequence[Any]] = None,
    ) -> "IslandChainBuilder":
        o_orbits = [as_periodic_orbit(orb, verify=False) for orb in O_orbits]
        x_orbits = [as_periodic_orbit(orb, verify=False) for orb in (X_orbits or [])]
        assignments = _assign_x_orbits(o_orbits, x_orbits, proximity_tol=self.proximity_tol)
        for o_orbit, assigned in zip(o_orbits, assignments):
            self.add_island(o_orbit, assigned)
        return self

    def build(self) -> IslandChain:
        chain = IslandChain(
            islands=list(self._islands),
            period=self.period if self.period is not None else len(self._islands),
            label=self.label,
            metadata=dict(self.metadata),
        )
        return chain

    @classmethod
    def from_periodic_orbits(
        cls,
        O_orbits: Sequence[Any],
        X_orbits: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
        period: Optional[int] = None,
        proximity_tol: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> IslandChain:
        builder = cls(
            label=label,
            period=period,
            metadata=dict(metadata or {}),
            proximity_tol=proximity_tol,
        )
        builder.add_from_orbit_clouds(O_orbits, X_orbits)
        return builder.build()


@dataclass
class TubeChainBuilder:
    """Builder for generic continuous-time tube chains."""

    label: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    _tubes: List[Tube] = field(default_factory=list, init=False, repr=False)

    def add_tube(
        self,
        O_cycle: Any,
        X_cycles: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
        require_closed: bool = False,
        closure_tol: float = 1e-8,
        debug_info: Optional[dict] = None,
    ) -> "TubeChainBuilder":
        o_cycle = as_cycle(O_cycle, require_closed=require_closed, closure_tol=closure_tol)
        x_cycles = [
            as_cycle(cyc, require_closed=require_closed, closure_tol=closure_tol)
            for cyc in (X_cycles or [])
        ]
        self._tubes.append(
            Tube(
                O_cycle=o_cycle,
                X_cycles=x_cycles,
                label=label or self.label,
                debug_info=dict(debug_info or {}),
            )
        )
        return self

    def build(self) -> TubeChain:
        return TubeChain(tubes=list(self._tubes), label=self.label, debug_info=dict(self.metadata))

    @classmethod
    def from_cycles(
        cls,
        O_cycles: Sequence[Any],
        X_cycles: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
        require_closed: bool = False,
        closure_tol: float = 1e-8,
        metadata: Optional[dict] = None,
    ) -> TubeChain:
        builder = cls(label=label, metadata=dict(metadata or {}))
        x_cycles = list(X_cycles or [])
        for idx, o_cycle in enumerate(O_cycles):
            assigned = [x_cycles[idx]] if idx < len(x_cycles) else []
            builder.add_tube(
                o_cycle,
                assigned,
                require_closed=require_closed,
                closure_tol=closure_tol,
            )
        return builder.build()


__all__ = [
    "GeometryBuilder",
    "IslandChainBuilder",
    "TubeChainBuilder",
]
