"""DynamicsContext — a factory for building invariant objects with correct dimensionality."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pyna.topo.invariants import (
    Cycle,
    FixedPoint,
    Island,
    IslandChain,
    MonodromyData,
    PeriodicOrbit,
    StableManifold,
    Tube,
    TubeChain,
    UnstableManifold,
)


class DynamicsContext:
    """Factory that creates invariant objects threaded with the correct ambient_dim."""

    def __init__(self, flow: Any, section: Any = None):
        self.flow = flow
        self.section = section
        self._phase_dim = getattr(flow, "phase_dim", None)

    @property
    def ambient_dim(self) -> Optional[int]:
        return self._phase_dim

    @property
    def section_dim(self) -> Optional[int]:
        return self._phase_dim - 1 if self._phase_dim else None

    def cycle(
        self,
        winding: tuple,
        sections: Optional[Dict[float, FixedPoint]] = None,
        monodromy: Optional[MonodromyData] = None,
    ) -> Cycle:
        return Cycle(
            winding=winding,
            sections=sections or {},
            monodromy=monodromy,
            ambient_dim=self.ambient_dim,
        )

    def fixed_point(self, phi: float, R: float, Z: float, DPm) -> FixedPoint:
        return FixedPoint(
            phi=phi, R=R, Z=Z, DPm=DPm, ambient_dim=self.section_dim
        )

    def island(self, O_point: FixedPoint, X_points: Optional[List[FixedPoint]] = None) -> Island:
        x_fps = X_points or []
        return Island(
            O_orbit=PeriodicOrbit(points=[O_point]),
            X_orbits=[PeriodicOrbit(points=[xfp]) for xfp in x_fps],
            ambient_dim=self.section_dim,
        )

    def island_chain(
        self,
        O_points: Optional[List[FixedPoint]] = None,
        X_points: Optional[List[FixedPoint]] = None,
    ) -> IslandChain:
        return IslandChain(
            islands=[],
            ambient_dim=self.section_dim,
        )

    def tube(self, O_cycle: Cycle, X_cycles: Optional[List[Cycle]] = None) -> Tube:
        return Tube(
            O_cycle=O_cycle,
            X_cycles=X_cycles or [],
            ambient_dim=self.ambient_dim,
        )

    def tube_chain(
        self,
        O_cycles: Optional[List[Cycle]] = None,
        X_cycles: Optional[List[Cycle]] = None,
    ) -> TubeChain:
        return TubeChain(
            O_cycles=O_cycles or [],
            X_cycles=X_cycles or [],
            ambient_dim=self.ambient_dim,
        )

    def stable_manifold(self, cycle: Cycle, branches: Optional[list] = None) -> StableManifold:
        return StableManifold(
            cycle=cycle,
            branches=branches or [],
            ambient_dim=self.ambient_dim,
        )

    def unstable_manifold(self, cycle: Cycle, branches: Optional[list] = None) -> UnstableManifold:
        return UnstableManifold(
            cycle=cycle,
            branches=branches or [],
            ambient_dim=self.ambient_dim,
        )
