"""pyna.topo.toroidal — Toroidal / MCF specializations.

All toroidal classes inherit from their generic counterparts in
:mod:`pyna.topo.core` following the "parent-name + suffix" convention.

Class hierarchy
---------------
core.SectionPoint
  → FixedPoint          (+R/Z/phi/DPm/kind/monodromy)

core.PeriodicOrbit
  → PeriodicOrbit       (+monodromy via FixedPoint.DPm)

core.Cycle
  → Cycle               (+winding, .sections dict, .section_points)

core.Island
  → Island              (+tube/tube_chain/R/Z/phi, _ToriMixin, root_island)

core.IslandChain
  → IslandChain         (+winding/m/n, gcd analysis, q_rational, constructors)

core.Tube
  → Tube                (+m/n, seed_phi, cyna-accelerated section_cut, TubeCutPoint)

core.TubeChain
  → TubeChain           (+expected_n_tubes, from_XO_fixed_points, SectionView)
"""
from pyna.topo._base import InvariantSet, InvariantManifold, SectionCuttable
from pyna.topo.core import Stability

from ._monodromy import MonodromyData
from ._fixed_point import FixedPoint
from ._cycle import PeriodicOrbit, Cycle
from ._island import ChainRole, Island, IslandChain
from ._tube import TubeCutPoint, Tube, TubeChain
from ._manifold import StableManifold, UnstableManifold

# Backward-compat aliases
ToroidalSectionPoint = FixedPoint
ToroidalPeriodicOrbit = PeriodicOrbit
ToroidalCycle = Cycle

__all__ = [
    "Stability",
    "MonodromyData",
    "FixedPoint",
    "PeriodicOrbit",
    "Cycle",
    "ChainRole",
    "TubeCutPoint",
    "Island",
    "IslandChain",
    "Tube",
    "TubeChain",
    "StableManifold",
    "UnstableManifold",
    "ToroidalSectionPoint",
    "ToroidalPeriodicOrbit",
    "ToroidalCycle",
]
