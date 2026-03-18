"""pyna.topo — topology analysis subpackage."""

from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.topo.chaos import (
    chirikov_overlap,
    ftle_field,
    chaotic_boundary_estimate,
)
from pyna.topo.topology_analysis import analyse_topology, TopologyReport
from pyna.topo.fixed_points import (
    find_periodic_orbit,
    classify_fixed_point,
    classify_fixed_point_higher_order,
)
from pyna.topo.island import Island, IslandChain

__all__ = [
    "PoincareMapVariationalEquations",
    "StableManifold",
    "UnstableManifold",
    # chaos diagnostics
    "chirikov_overlap",
    "ftle_field",
    "chaotic_boundary_estimate",
    # high-level facade
    "analyse_topology",
    "TopologyReport",
    # fixed points
    "find_periodic_orbit",
    "classify_fixed_point",
    "classify_fixed_point_higher_order",
    # island topology
    "Island",
    "IslandChain",
]
