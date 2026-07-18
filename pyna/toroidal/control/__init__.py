"""Toroidal topology and response-control tools.

The package facade keeps the established lightweight API eager and resolves
new, higher-level workflows lazily.  Domain modules therefore remain usable
without importing plotting stacks, optional external backends, or every
other control workflow.
"""

from importlib import import_module

from pyna.toroidal.control.gap_response import gap_response_matrix_fpt
from pyna.toroidal.control.island_control import (
    compute_resonant_amplitude,
    island_suppression_current,
    multi_mode_control,
    phase_control_current,
)
from pyna.toroidal.control.island_optimizer import (
    IslandOptimizer,
    OptimisationResult,
    UnperturbedSurfaceReconstructor,
    compute_surface_deformation,
    epsilon_eff_proxy,
)
from pyna.toroidal.control.qprofile_response import (
    build_qprofile_response,
    iota_response_matrix,
    q_by_fieldline_tracing,
    q_by_fieldline_winding,
    q_from_flux_surface_integral,
    q_response_matrix_analytic,
    q_response_matrix_fd,
)
from pyna.toroidal.control.wall import WallGeometry, make_east_like_wall


_LAZY_MODULES = (
    "boundary_topology_design",
    "boundary_topology_control",
    "boundary_optimization_workflow",
    "boundary_field_basis",
    "boundary_perturbation_candidates",
    "boundary_plasma_response",
    "boundary_nonlinear_validation",
    "boundary_topology_cases",
    "heat_distribution",
    "strike_heat",
    "topoquest_fpt",
    "fusionsc_heat",
)

_DIRECT_LAZY_EXPORTS = {
    "BoundaryTopologyHeatState": "heat_contracts",
    "BoundaryTopologyHeatForwardModel": "heat_contracts",
    "CallableBoundaryTopologyHeatForwardModel": "heat_contracts",
    "ReducedSpectralHeatModel": "reduced_heat",
}


def __getattr__(name):
    """Resolve opt-in workflows without coupling package import to them."""

    if name in _LAZY_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    module_name = _DIRECT_LAZY_EXPORTS.get(name)
    if module_name is not None:
        value = getattr(import_module(f"{__name__}.{module_name}"), name)
        globals()[name] = value
        return value

    for module_name in _LAZY_MODULES:
        module = import_module(f"{__name__}.{module_name}")
        if name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WallGeometry",
    "make_east_like_wall",
    "gap_response_matrix_fpt",
    "q_from_flux_surface_integral",
    "q_by_fieldline_tracing",
    "q_by_fieldline_winding",
    "q_response_matrix_analytic",
    "q_response_matrix_fd",
    "iota_response_matrix",
    "build_qprofile_response",
    "compute_resonant_amplitude",
    "island_suppression_current",
    "phase_control_current",
    "multi_mode_control",
    "IslandOptimizer",
    "OptimisationResult",
    "UnperturbedSurfaceReconstructor",
    "compute_surface_deformation",
    "epsilon_eff_proxy",
    *_LAZY_MODULES,
]
