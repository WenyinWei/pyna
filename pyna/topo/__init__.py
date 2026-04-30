"""pyna.topo — topology analysis subpackage."""

from pyna.topo._base import (
    InvariantSet, InvariantManifold, SectionCuttable,
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
    PeriodicOrbit as CorePeriodicOrbit,
    Cycle as CoreCycle,
)
from pyna.topo.toroidal import (
    PeriodicOrbit,
    Cycle,
    FixedPoint,
    MonodromyData,
    Tube,
    TubeChain,
    TubeCutPoint,
    Island,
    IslandChain,
    ChainRole,
    StableManifold,
    UnstableManifold,
    ToroidalSectionPoint,
    ToroidalPeriodicOrbit,
    ToroidalCycle,
)
from pyna.topo.toroidal_island import (
    locate_rational_surface,
    locate_all_rational_surfaces,
    island_halfwidth,
    all_rational_q,
)

# Backward-compat aliases
ToroidalIsland = Island
ToroidalIslandChain = IslandChain
ToroidalTube = Tube
ToroidalTubeChain = TubeChain

from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import ScipyStableManifold, ScipyUnstableManifold, CynaStableManifold, CynaUnstableManifold
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
from pyna.topo.healed_flux_coords import (
    TransportedSection,
    SectionCorrespondence,
    BoundarySection,
    BoundaryConstraintSet,
    BoundaryFamily3D,
    FieldLineScaffold3D,
    trace_grid_to_phi,
    trace_section_curve_to_phi,
    trace_surface_family_to_sections,
    build_xo_sequence,
    xo_sequence_boundary_arcs,
    build_cxo_spline,
    correct_boundary_with_constraints,
    fit_ring_fourier,
    build_pchip_family,
    eval_fourier_family,
    build_section_scaffold_bundle,
)

# Layer 0: Phase space and dynamics
from pyna.topo.dynamics import (
    PhaseSpace, MCF_2D, GC_4D,
    DynamicalSystem, ContinuousFlow, HamiltonianFlow,
    MagneticFieldLine, DiscreteMap, StandardMap, PoincareMap,
    MCFPoincareMap,
    GeneralPoincareMap,
)
# Layer 0.5: Base hierarchy
from pyna.topo._base import (
    GeometricObject,
    InvariantSet,
    InvariantManifold,
    SectionCuttable,
)
from pyna.topo.invariant import InvariantTorus
# Layer 3: Sections
from pyna.topo.section import (
    Section, ToroidalSection, HyperplaneSection, ParametricSection,
    toroidal_sections, HAO_SECTIONS,
)
# Layer 2: Resonance
from pyna.topo.resonance import ResonanceNumber

# Field classes (new names + backward compat)
from pyna.topo.field import (
    ScalarField,
    VectorField,
    TensorField,
    VectorFieldCylind,
    VectorFieldCylindAxisym,
    ToroidalField,        # backward compat = VectorFieldCylind
    AxisymmetricField,    # backward compat = VectorFieldCylindAxisym
    Equilibrium,
    EquilibriumLike,
    compute_J_by_curl,
    MU0,
)

# --- Toroidal specializations kept available explicitly ---
try:
    from pyna.topo.toroidal_cycle import ToroidalPeriodicOrbitTrace
except ImportError:
    ToroidalPeriodicOrbitTrace = None

__all__ = [
    # Base hierarchy
    "GeometricObject",
    "InvariantSet",
    "InvariantManifold",
    "SectionCuttable",
    "Trajectory",
    "Orbit",
    "SectionPoint",
    "Stability",
    "LinearStabilityData",
    "PeriodicOrbit",
    "Cycle",
    # Toroidal sampled trajectories / specializations
    "ToroidalTrajectory",
    "ToroidalSectionPoint",
    "ToroidalPeriodicOrbit",
    "ToroidalCycle",
    "FixedPoint",
    "MonodromyData",
    "ToroidalPeriodicOrbitTrace",
    "trace_toroidal_trajectory",
    "PoincareMapVariationalEquations",
    "StableManifold",
    "UnstableManifold",
    "ScipyStableManifold",
    "ScipyUnstableManifold",
    "CynaStableManifold",
    "CynaUnstableManifold",
    # Chaos diagnostics
    "chirikov_overlap",
    "ftle_field",
    "chaotic_boundary_estimate",
    # Spectral regularity diagnostics
    "spectral_regularity",
    "spectral_regularity_single",
    "classify_orbit",
    "hessian_regularity",
    "eigenvalue_evolution_from_sequence",
    # High-level facade
    "analyse_topology",
    "TopologyReport",
    # Fixed points
    "find_periodic_orbit",
    "classify_fixed_point",
    "classify_fixed_point_higher_order",
    "group_fixed_points_by_orbit",
    # Island topology
    "Island",
    "IslandChain",
    "ChainRole",
    "ToroidalIsland",
    "ToroidalIslandChain",
    "FluxSurface",
    "FluxSurfaceMap",
    "XPointOrbit",
    "detect_residual_islands",
    # Rotational transform
    "rotational_transform_from_trajectory",
    "PoincareAccumulator",
    # Monodromy / variational
    "evolve_DPm_along_cycle",
    "CycleVariationalData",
    "orbit_shift_under_perturbation",
    "monodromy_change_under_perturbation",
    "second_order_orbit_variation",
    "monodromy_matrix",
    # Identity / bridge layer
    "ResonanceID",
    "TubeID",
    "IslandID",
    "ToroidalSectionViewPoint",
    "ToroidalSectionCorrespondence",
    "ToroidalSectionView",
    "ToroidalSectionViewBuilder",
    # Toroidal continuous-time resonance geometry
    "TubeCutPoint",
    "ToroidalTube",
    "ToroidalTubeChain",
    "Tube",
    "TubeChain",
    # Island / island-chain plotting
    "plot_island",
    "island_section_points",
    "plot_island_chain",
    "island_chain_section_points",
    # Rational surface / island width utilities
    "locate_rational_surface",
    "locate_all_rational_surfaces",
    "island_halfwidth",
    "all_rational_q",
    # Island-constrained healed coordinates
    "IslandHealedCoordMap",
    "InnerFourierSection",
    "XOCycAnchor",
    "make_xcyc_anchors",
    "build_from_trajectory_npz",
    "fp_by_section_from_orbits",
    "build_from_orbits",
    "build_from_island_chain",
    # Traced 3D healed scaffold
    "TransportedSection",
    "SectionCorrespondence",
    "BoundarySection",
    "BoundaryConstraintSet",
    "BoundaryFamily3D",
    "FieldLineScaffold3D",
    "trace_grid_to_phi",
    "trace_section_curve_to_phi",
    "trace_surface_family_to_sections",
    "build_xo_sequence",
    "xo_sequence_boundary_arcs",
    "build_cxo_spline",
    "correct_boundary_with_constraints",
    "fit_ring_fourier",
    "build_pchip_family",
    "eval_fourier_family",
    "build_section_scaffold_bundle",
    "lcfs_effective_minor_radius",
    "make_reff_profile_grid",
    "compute_profile_objectives_fast",
    "compute_beta_max_fast",
    # Dynamics / Poincaré map
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
    # Invariant / geometry layer
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
    # Field
    "ScalarField",
    "VectorField",
    "TensorField",
    "VectorFieldCylind",
    "VectorFieldCylindAxisym",
    "ToroidalField",
    "AxisymmetricField",
    "Equilibrium",
    "EquilibriumLike",
    "compute_J_by_curl",
    "MU0",
]
