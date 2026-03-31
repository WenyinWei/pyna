"""
pyna.MCF.flt — Field-line tracing utilities.

Public API wrapping the pyna._cyna C++ extension.  All callers (including
pyna.topo.*) should import from here instead of touching pyna._cyna directly.
"""

from .numba_poincare import (
    precompile_tracer,
    trace_poincare_batch,
    trace_poincare_multi_batch,
    trace_poincare_batch_twall,
    find_fixed_points_batch,
    trace_orbit_along_phi,
    field_arrays_from_interpolators,
)

__all__ = [
    "precompile_tracer",
    "trace_poincare_batch",
    "trace_poincare_multi_batch",
    "trace_poincare_batch_twall",
    "find_fixed_points_batch",
    "trace_orbit_along_phi",
    "field_arrays_from_interpolators",
]
