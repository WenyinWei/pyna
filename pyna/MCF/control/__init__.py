"""pyna.MCF.control — thin package-level facade over :mod:`pyna.toroidal.control`.

The old per-module wrappers in ``pyna.MCF.control`` were removed. Import from
``pyna.toroidal.control`` directly for new code, or from this package root when
maintaining older notebooks.
"""

from importlib import import_module

from pyna.toroidal.control import (
    WallGeometry,
    make_east_like_wall,
    gap_response_matrix_fpt,
    q_from_flux_surface_integral,
    q_by_fieldline_tracing,
    q_by_fieldline_winding,
    q_response_matrix_analytic,
    q_response_matrix_fd,
    iota_response_matrix,
    build_qprofile_response,
    compute_resonant_amplitude,
    island_suppression_current,
    phase_control_current,
    multi_mode_control,
    IslandOptimizer,
    OptimisationResult,
    UnperturbedSurfaceReconstructor,
    compute_surface_deformation,
    epsilon_eff_proxy,
)

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
]


def __getattr__(name):
    return getattr(import_module("pyna.toroidal.control"), name)
