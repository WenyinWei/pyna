"""pyna.topo �?topology analysis subpackage."""

from pyna.topo.trajectory3d import ToroidalTrajectory3D, trace_toroidal_trajectory

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
    find_magnetic_axis,
    find_fixed_point_newton,
    refine_fixed_points_from_pkl,
    group_fixed_points_by_orbit,
)
from pyna.topo.island import Island, IslandChain, ChainRole
from pyna.topo.flux_surface import FluxSurface, FluxSurfaceMap, XPointOrbit
from pyna.topo.island_extract import detect_residual_islands
from pyna.topo.poincare import rotational_transform_from_trajectory, PoincareAccumulator
from pyna.topo.monodromy import (
    evolve_DPm_along_cycle,
    CycleVariationalData,
    orbit_shift_under_perturbation,
    monodromy_change_under_perturbation,
    second_order_orbit_variation,
    monodromy_matrix,
)
from pyna.topo.identity import ResonanceID, TubeID, IslandID
from pyna.topo.section_view import SectionViewPoint, SectionCorrespondence, SectionView, SectionViewBuilder
from pyna.topo.tube import TubeCutPoint, Tube, TubeChain
from pyna.plot.island import plot_island, island_section_points
from pyna.plot.island_chain import plot_island_chain, island_chain_section_points
from pyna.topo.fast_metrics import (
    lcfs_effective_minor_radius,
    make_reff_profile_grid,
    compute_profile_objectives_fast,
    compute_beta_max_fast,
)
from pyna.topo.island_healed_coords import (
    IslandHealedCoordMap,
    InnerFourierSection,
    XOCycAnchor,
    make_xcyc_anchors,
    build_from_trajectory_npz,
    fp_by_section_from_orbits,
    build_from_orbits,
    build_from_island_chain,
)


# Layer 0: Phase space and dynamics
from pyna.topo.dynamics import (
    PhaseSpace, MCF_2D, GC_4D,
    DynamicalSystem, ContinuousFlow, HamiltonianFlow,
    MagneticFieldLine, DiscreteMap, StandardMap, PoincareMap,
    MCFPoincareMap,
    GeneralPoincareMap,
)
# Layer 1: Invariant objects (InvariantTorus; manifolds in manifold_improve.py)
from pyna.topo.invariant import (
    InvariantObject,
    InvariantTorus,
)
# Layer 3: Sections
from pyna.topo.section import (
    Section, ToroidalSection, HyperplaneSection, ParametricSection,
    toroidal_sections, HAO_SECTIONS,
)
# Layer 2: Resonance
from pyna.topo.resonance import ResonanceNumber
__all__ = [
    # 3D trajectories (continuous flow)
    "ToroidalTrajectory3D",
    "trace_toroidal_trajectory",
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
    "group_fixed_points_by_orbit",
    # island topology
    "Island",
    "IslandChain",
    "ChainRole",
    "FluxSurface",
    "FluxSurfaceMap",
    "XPointOrbit",
    "detect_residual_islands",
    # rotational transform
    "rotational_transform_from_trajectory",
    "PoincareAccumulator",
    # monodromy / variational
    "evolve_DPm_along_cycle",
    "CycleVariationalData",
    "orbit_shift_under_perturbation",
    "monodromy_change_under_perturbation",
    "second_order_orbit_variation",
    "monodromy_matrix",
    # island chain connectivity
    "FixedPoint",
    # identity / bridge layer
    "ResonanceID",
    "TubeID",
    "IslandID",
    "SectionViewPoint",
    "SectionCorrespondence",
    "SectionView",
    "SectionViewBuilder",
    # continuous-time resonance geometry
    "TubeCutPoint",
    "Tube",
    "TubeChain",
    "Cycle",
    # generic island / island-chain plotting
    "plot_island",
    "island_section_points",
    "plot_island_chain",
    "island_chain_section_points",
    # island-constrained healed coordinates
    "IslandHealedCoordMap",
    "InnerFourierSection",
    "XOCycAnchor",
    "make_xcyc_anchors",
    "build_from_trajectory_npz",
    "fp_by_section_from_orbits",
    "build_from_orbits",
    "build_from_island_chain",
    "lcfs_effective_minor_radius",
    "make_reff_profile_grid",
    "compute_profile_objectives_fast",
    "compute_beta_max_fast",
    # dynamics / Poincaré map
    "PhaseSpace",
    "MCF_2D",
    "GC_4D",
    "DynamicalSystem",
    "ContinuousFlow",
    "HamiltonianFlow",
    "MagneticFieldLine",
    "DiscreteMap",
    "StandardMap",
    "PoincareMap",
    "MCFPoincareMap",
    "GeneralPoincareMap",
    # InvariantObject layer
    "InvariantObject",
    "InvariantTorus",
    # Sections
    "Section",
    "ToroidalSection",
    "HyperplaneSection",
    "ParametricSection",
    "toroidal_sections",
    "HAO_SECTIONS",
    # Resonance
    "ResonanceNumber",
]

# --- New invariant-object skeleton (backward-compatible) ---
try:
    from pyna.topo.invariants import (
        Stability,
        MonodromyData,
        InvariantObject,
        FixedPoint,
        Cycle,
        InvariantTorus,
        StableManifold,
        UnstableManifold,
        Island,
        IslandChain,
        Tube,
        TubeChain,
    )
except ImportError:
    pass
