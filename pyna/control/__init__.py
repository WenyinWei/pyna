"""pyna.control — Generic multi-objective topology control via FPT.

This module is INDEPENDENT of fusion details — the same API works for any
area-preserving 2-D system.  For MCF-specific control (gap response, island
control, q-profile, wall geometry), see :mod:`pyna.toroidal.control`.
"""

from pyna.control.FPT import (
    A_matrix,
    DPm_axisymmetric,
    cycle_shift,
    delta_g_from_delta_B,
    DPm_change,
    delta_A_total,
    manifold_shift,
    flux_surface_deformation,
)

from pyna.control.topology_state import (
    SurfaceFate,
    XPointState,
    OPointState,
    TopologyState,
    compute_topology_state,
)

from pyna.control.response_matrix import build_response_matrix, build_full_response_matrix

from pyna.control._cache import memory as cache_memory, hash_eq_params, invalidate_cache

from pyna.control.optimizer import (
    ControlWeights,
    ControlConstraints,
    TopologyController,
)

from pyna.control.surface_fate import (
    Greene_residue,
    classify_surface_fate,
    scan_surface_fates,
)

__all__ = [
    "A_matrix", "DPm_axisymmetric", "cycle_shift", "delta_g_from_delta_B",
    "DPm_change", "delta_A_total", "manifold_shift", "flux_surface_deformation",
    "SurfaceFate", "XPointState", "OPointState", "TopologyState",
    "compute_topology_state",
    "build_response_matrix", "build_full_response_matrix",
    "cache_memory", "hash_eq_params", "invalidate_cache",
    "ControlWeights", "ControlConstraints", "TopologyController",
    "Greene_residue", "classify_surface_fate", "scan_surface_fates",
]
