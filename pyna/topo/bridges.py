"""Bridge objects between continuous and discrete topology layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from pyna.topo.core import Island, IslandChain, PeriodicOrbit, Tube, TubeChain


@runtime_checkable
class SectionCutBridge(Protocol):
    """Bridge from continuous geometry to section-level discrete geometry."""

    def cut_tube(self, tube: Tube, section: Any, **kwargs: Any) -> IslandChain: ...

    def cut_tube_chain(self, tube_chain: TubeChain, section: Any, **kwargs: Any) -> IslandChain: ...


@dataclass
class CoreSectionCutBridge:
    """Default bridge for ``pyna.topo.core`` geometry objects."""

    def cut_cycle(self, cycle: Any, section: Any) -> PeriodicOrbit:
        return cycle.section_cut(section)

    def cut_tube(self, tube: Tube, section: Any, **kwargs: Any) -> IslandChain:
        """Cut a generic ``Tube`` into a generic ``IslandChain``."""

        o_po = self.cut_cycle(tube.O_cycle, section)
        x_pos = [self.cut_cycle(x_cycle, section) for x_cycle in tube.X_cycles]
        chain = IslandChain(label=tube.label, metadata={"bridge": self.__class__.__name__})

        if o_po.points:
            for i, o_pt in enumerate(o_po.points):
                x_pts = []
                for x_po in x_pos:
                    if i < len(x_po.points):
                        x_pts.append(x_po.points[i])
                chain.add_island(
                    Island(
                        O_orbit=PeriodicOrbit(points=[o_pt], period=1),
                        X_orbits=[PeriodicOrbit(points=[x_pt], period=1) for x_pt in x_pts],
                        label=tube.label,
                    )
                )
        else:
            for x_po in x_pos:
                for x_pt in x_po.points:
                    chain.add_island(
                        Island(
                            O_orbit=PeriodicOrbit(points=[], period=0),
                            X_orbits=[PeriodicOrbit(points=[x_pt], period=1)],
                            label=tube.label,
                        )
                    )
        return chain

    def cut_tube_chain(self, tube_chain: TubeChain, section: Any, **kwargs: Any) -> IslandChain:
        """Cut all tubes and merge their islands into one generic chain."""

        chain = IslandChain(
            label=tube_chain.label,
            metadata={
                "bridge": self.__class__.__name__,
                "n_tubes_included": len(tube_chain.tubes),
            },
        )
        for tube in tube_chain.tubes:
            sub = self.cut_tube(tube, section, **kwargs)
            for island in sub.islands:
                chain.add_island(island)
        return chain


__all__ = [
    "SectionCutBridge",
    "CoreSectionCutBridge",
]
