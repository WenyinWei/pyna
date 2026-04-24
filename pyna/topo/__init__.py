"""pyna.topo �?topology analysis subpackage."""

from pyna.topo._base import (
    InvariantSet, InvariantSet, InvariantManifold, SectionCuttable,
)
from pyna.topo.toroidal_trajectory import (
    ToroidalTrajectory,
    trace_toroidal_trajectory,
)
from pyna.topo.core import (
    Stability,
    LinearStabilityData,
    SectionPoint,
    Trajectory,
    Orbit,
    PeriodicOrbit,
    Cycle,
)
from pyna.topo.toroidal import (
    ToroidalSectionPoint,
    ToroidalPeriodicOrbit,
    ToroidalCycle,
)

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
from pyna.topo.toroidal_island import Island as ToroidalIsland, IslandChain as ToroidalIslandChain
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
from pyna.topo.toroidal_section_view import (
    SectionViewPoint as ToroidalSectionViewPoint,
    SectionCorrespondence as ToroidalSectionCorrespondence,
    SectionView as ToroidalSectionView,
    SectionViewBuilder as ToroidalSectionViewBuilder,
)
from pyna.topo.toroidal_tube import TubeCutPoint, Tube as ToroidalTube, TubeChain as ToroidalTubeChain
from pyna.plot.island import plot_island, island_section_points
from pyna.plot.island_chain import plot_island_chain, island_chain_section_points
from pyna.topo.fast_metrics import (
    lcfs_effective_minor_radius,
    make_reff_profile_grid,
    compute_profile_objectives_fast,
    compute_beta_max_fast,
)
from pyna.topo.regularity import (
    spectral_regularity,
    spectral_regularity_single,
    classify_orbit,
    hessian_regularity,
    eigenvalue_evolution_from_sequence,
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
from pyna.topo.healed_scaffold_3d import (
    TransportedSection,
    SectionCorrespondence,
    FieldLineScaffold3D,
    trace_grid_to_phi,
    trace_section_curve_to_phi,
    trace_surface_family_to_sections,
)


# Layer 0: Phase space and dynamics
from pyna.topo.dynamics import (
    PhaseSpace, MCF_2D, GC_4D,
    DynamicalSystem, ContinuousFlow, HamiltonianFlow,
    MagneticFieldLine, DiscreteMap, StandardMap, PoincareMap,
    MCFPoincareMap,
    GeneralPoincareMap,
)
# Layer 1: Invariant objects
from pyna.topo._base import (
    GeometricObject,
    InvariantSet,
    InvariantManifold,
    SectionCuttable,
)
from pyna.topo.invariant import (
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
    # Base hierarchy
    "GeometricObject",
    "InvariantSet",
    "InvariantManifold",
    "SectionCuttable",
    "Trajectory",            # generic sampled continuous trajectory
    "Orbit",                 # generic sampled discrete orbit
    "SectionPoint",
    "Stability",
    "LinearStabilityData",
    "PeriodicOrbit",         # generic invariant periodic orbit of a map
    "Cycle",                 # generic invariant periodic orbit of a flow
    # toroidal sampled trajectories / specializations
    "ToroidalTrajectory",
    "ToroidalSectionPoint",
    "ToroidalPeriodicOrbit",
    "ToroidalCycle",
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
    # spectral regularity diagnostics
    "spectral_regularity",
    "spectral_regularity_single",
    "classify_orbit",
    "hessian_regularity",
    "eigenvalue_evolution_from_sequence",
    # high-level facade
    "analyse_topology",
    "TopologyReport",
    # fixed points
    "find_periodic_orbit",
    "classify_fixed_point",
    "classify_fixed_point_higher_order",
    "group_fixed_points_by_orbit",
    # generic island topology
    "Island",
    "IslandChain",
    "ChainRole",
    # toroidal island topology
    "ToroidalIsland",
    "ToroidalIslandChain",
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
    # toroidal / legacy island-chain connectivity
    "FixedPoint",
    "ToroidalPeriodicOrbitTrace",
    # identity / bridge layer
    "ResonanceID",
    "TubeID",
    "IslandID",
    "ToroidalSectionViewPoint",
    "ToroidalSectionCorrespondence",
    "ToroidalSectionView",
    "ToroidalSectionViewBuilder",
    # toroidal continuous-time resonance geometry
    "TubeCutPoint",
    "ToroidalTube",
    "ToroidalTubeChain",
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
    # traced 3D healed scaffold
    "TransportedSection",
    "SectionCorrespondence",
    "FieldLineScaffold3D",
    "trace_grid_to_phi",
    "trace_section_curve_to_phi",
    "trace_surface_family_to_sections",
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
    # invariant / geometry layer
    "GeometricObject",
    "InvariantSet",
    "InvariantManifold",
    "SectionCuttable",
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

# --- Toroidal specializations kept available explicitly ---
# Note: Island and IslandChain are intentionally NOT re-exported here to
# avoid overriding the full implementations in island.py.
try:
    from pyna.topo.toroidal_invariants import (
        MonodromyData,
        FixedPoint,
        InvariantTorus,
        StableManifold,
        UnstableManifold,
        # Tube and TubeChain come from tube.py (which uses toroidal invariants.Cycle)
    )
    from pyna.topo.toroidal_cycle import ToroidalPeriodicOrbitTrace
except ImportError:
    pass
