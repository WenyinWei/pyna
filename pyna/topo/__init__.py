"""pyna.topo — topology analysis subpackage."""

from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold, ScipyStableManifold, ScipyUnstableManifold, CynaStableManifold, CynaUnstableManifold
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
from pyna.topo.island_extract import detect_residual_islands
from pyna.topo.poincare import rotational_transform_from_trajectory
from pyna.topo.monodromy import (
    evolve_DPm_along_cycle,
    CycleVariationalData,
    orbit_shift_under_perturbation,
    monodromy_change_under_perturbation,
    second_order_orbit_variation,
    monodromy_matrix,
)

__all__ = [
    "PoincareMapVariationalEquations",
    "StableManifold",
    "UnstableManifold",
    "ScipyStableManifold",
    "ScipyUnstableManifold",
    "CynaStableManifold",
    "CynaUnstableManifold",
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
    "detect_residual_islands",
    # rotational transform
    "rotational_transform_from_trajectory",
    # monodromy / variational
    "evolve_DPm_along_cycle",
    "CycleVariationalData",
    "orbit_shift_under_perturbation",
    "monodromy_change_under_perturbation",
    "second_order_orbit_variation",
    "monodromy_matrix",
]
