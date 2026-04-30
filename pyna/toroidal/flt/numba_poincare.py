"""Toroidal Poincaré tracing helpers backed by the cyna C++ extension.

Canonical toroidal ownership for the grid-backed batch tracing helpers.
"""
from __future__ import annotations

import numpy as np
from pyna._cyna import (
    is_available as _cyna_available,
    trace_poincare_batch as _cyna_trace_poincare_batch,
    trace_poincare_multi as _cyna_trace_poincare_multi,
    trace_poincare_batch_twall as _cyna_trace_poincare_batch_twall,
    find_fixed_points_batch as _cyna_find_fixed_points_batch,
    trace_orbit_along_phi as _cyna_trace_orbit_along_phi,
)



def field_arrays_from_interpolators(itp_BR, itp_BPhi, itp_BZ):
    """Extract contiguous field arrays from ``RegularGridInterpolator`` objects."""
    R_grid = np.ascontiguousarray(itp_BR.grid[0], dtype=np.float64)
    Z_grid = np.ascontiguousarray(itp_BR.grid[1], dtype=np.float64)
    Phi_grid = np.ascontiguousarray(itp_BR.grid[2], dtype=np.float64)
    nx, ny, nz = len(R_grid), len(Z_grid), len(Phi_grid)

    BR_flat = np.ascontiguousarray(itp_BR.values.ravel(), dtype=np.float64)
    BPhi_flat = np.ascontiguousarray(itp_BPhi.values.ravel(), dtype=np.float64)
    BZ_flat = np.ascontiguousarray(itp_BZ.values.ravel(), dtype=np.float64)
    return R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat, nx, ny, nz



def precompile_tracer(R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat):
    """Compatibility no-op retained for the historical numba-era API."""
    _ = (R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat)



def trace_poincare_batch(
    R_seeds,
    Z_seeds,
    phi_section,
    N_turns,
    DPhi,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BPhi_flat,
    BZ_flat,
    wall_R,
    wall_Z,
):
    """Trace field lines and record a single Poincaré section in batch."""
    if not _cyna_available():
        raise ImportError("pyna._cyna C++ extension is unavailable. Build cyna first.")
    return _cyna_trace_poincare_batch(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        float(phi_section),
        int(N_turns),
        float(DPhi),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        np.ascontiguousarray(wall_R, dtype=np.float64),
        np.ascontiguousarray(wall_Z, dtype=np.float64),
    )



def trace_poincare_multi_batch(
    R_seeds,
    Z_seeds,
    phi_sections_arr,
    N_turns,
    DPhi,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BPhi_flat,
    BZ_flat,
    wall_R,
    wall_Z,
):
    """Trace field lines and record multiple Poincaré sections in batch."""
    if not _cyna_available():
        raise ImportError("pyna._cyna C++ extension is unavailable. Build cyna first.")
    counts, pR, pZ = _cyna_trace_poincare_multi(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        np.ascontiguousarray(phi_sections_arr, dtype=np.float64),
        int(N_turns),
        float(DPhi),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        np.ascontiguousarray(wall_R, dtype=np.float64),
        np.ascontiguousarray(wall_Z, dtype=np.float64),
    )
    return counts.reshape(len(R_seeds), len(phi_sections_arr)), pR, pZ



def trace_poincare_batch_twall(
    R_seeds,
    Z_seeds,
    phi_section,
    N_turns,
    DPhi,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BPhi_flat,
    BZ_flat,
    wall_phi,
    wall_R_all,
    wall_Z_all,
):
    """Trace field lines against a toroidal 3-D wall and record a section."""
    if not _cyna_available() or _cyna_trace_poincare_batch_twall is None:
        raise ImportError("pyna._cyna.trace_poincare_batch_twall is unavailable. Build cyna first.")
    return _cyna_trace_poincare_batch_twall(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        float(phi_section),
        int(N_turns),
        float(DPhi),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        np.ascontiguousarray(wall_phi, dtype=np.float64),
        np.ascontiguousarray(wall_R_all, dtype=np.float64),
        np.ascontiguousarray(wall_Z_all, dtype=np.float64),
    )



def find_fixed_points_batch(
    R_seeds,
    Z_seeds,
    phi_start,
    period,
    N_periods,
    DPhi,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BPhi_flat,
    BZ_flat,
    **kwargs,
):
    """Batch search for Poincaré-map fixed points starting from seed points."""
    if not _cyna_available() or _cyna_find_fixed_points_batch is None:
        raise ImportError("pyna._cyna.find_fixed_points_batch is unavailable. Build cyna first.")
    return _cyna_find_fixed_points_batch(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        float(phi_start),
        int(period),
        int(N_periods),
        float(DPhi),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        **kwargs,
    )



def trace_orbit_along_phi(
    R0,
    Z0,
    phi_start,
    phi_end,
    DPhi,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BPhi_flat,
    BZ_flat,
):
    """Trace a single orbit from ``phi_start`` to ``phi_end``."""
    if not _cyna_available() or _cyna_trace_orbit_along_phi is None:
        raise ImportError("pyna._cyna.trace_orbit_along_phi is unavailable. Build cyna first.")
    return _cyna_trace_orbit_along_phi(
        float(R0),
        float(Z0),
        float(phi_start),
        float(phi_end),
        float(DPhi),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
    )


__all__ = [
    "field_arrays_from_interpolators",
    "precompile_tracer",
    "trace_poincare_batch",
    "trace_poincare_multi_batch",
    "trace_poincare_batch_twall",
    "find_fixed_points_batch",
    "trace_orbit_along_phi",
]
