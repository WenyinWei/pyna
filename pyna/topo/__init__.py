"""pyna.topo — topology analysis subpackage."""

from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.topo.chaos import (
    chirikov_overlap,
    ftle_field,
    chaotic_boundary_estimate,
)
from pyna.topo.topology_analysis import analyse_topology, TopologyReport

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
]
