"""Toroidal Poincaré tracing helpers backed by the cyna C++ extension.

Canonical toroidal ownership for the grid-backed batch tracing helpers.
"""
from __future__ import annotations

import numpy as np
from pyna.fields.cylindrical import CylindricalFieldArrays, as_vector_field_cylindrical
from pyna._cyna import (
    is_available as _cyna_available,
    trace_poincare_batch as _cyna_trace_poincare_batch,
    trace_poincare_multi as _cyna_trace_poincare_multi,
    trace_poincare_batch_twall as _cyna_trace_poincare_batch_twall,
    find_fixed_points_batch as _cyna_find_fixed_points_batch,
    trace_orbit_along_phi as _cyna_trace_orbit_along_phi,
)



def field_arrays_from_interpolators(itp_BR, itp_BZ, itp_BPhi):
    """Extract contiguous field arrays from ``RegularGridInterpolator`` objects."""
    R_grid = np.ascontiguousarray(itp_BR.grid[0], dtype=np.float64)
    Z_grid = np.ascontiguousarray(itp_BR.grid[1], dtype=np.float64)
    Phi_grid = np.ascontiguousarray(itp_BR.grid[2], dtype=np.float64)
    nx, ny, nz = len(R_grid), len(Z_grid), len(Phi_grid)

    BR_flat = np.ascontiguousarray(itp_BR.values.ravel(), dtype=np.float64)
    BZ_flat = np.ascontiguousarray(itp_BZ.values.ravel(), dtype=np.float64)
    BPhi_flat = np.ascontiguousarray(itp_BPhi.values.ravel(), dtype=np.float64)
    return R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat, nx, ny, nz


def field_arrays_from_field(field, *, extend_phi: bool = False) -> CylindricalFieldArrays:
    """Return a named cyna-array view from a cylindrical vector field object."""

    return as_vector_field_cylindrical(field).cyna_arrays(extend_phi=extend_phi)



def precompile_tracer(R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat):
    """Compatibility no-op retained for the historical numba-era API."""
    _ = (R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat)



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
    BZ_flat,
    BPhi_flat,
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
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        np.ascontiguousarray(wall_R, dtype=np.float64),
        np.ascontiguousarray(wall_Z, dtype=np.float64),
    )


def trace_poincare_batch_field(
    field,
    R_seeds,
    Z_seeds,
    phi_section,
    N_turns,
    DPhi,
    wall_R,
    wall_Z,
    *,
    extend_phi: bool = True,
):
    """Object-first Poincare tracing wrapper.

    ``field`` is a cylindrical vector field object; only this bridge unpacks
    it into the cyna ``BR, BZ, BPhi`` ABI.
    """

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return trace_poincare_batch(
        R_seeds,
        Z_seeds,
        phi_section,
        N_turns,
        DPhi,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        wall_R,
        wall_Z,
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
    BZ_flat,
    BPhi_flat,
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
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        np.ascontiguousarray(wall_R, dtype=np.float64),
        np.ascontiguousarray(wall_Z, dtype=np.float64),
    )
    return counts.reshape(len(R_seeds), len(phi_sections_arr)), pR, pZ


def trace_poincare_multi_batch_field(
    field,
    R_seeds,
    Z_seeds,
    phi_sections_arr,
    N_turns,
    DPhi,
    wall_R,
    wall_Z,
    *,
    extend_phi: bool = True,
):
    """Object-first multi-section Poincare tracing wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return trace_poincare_multi_batch(
        R_seeds,
        Z_seeds,
        phi_sections_arr,
        N_turns,
        DPhi,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        wall_R,
        wall_Z,
    )



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
    BZ_flat,
    BPhi_flat,
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
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        np.ascontiguousarray(wall_phi, dtype=np.float64),
        np.ascontiguousarray(wall_R_all, dtype=np.float64),
        np.ascontiguousarray(wall_Z_all, dtype=np.float64),
    )


def trace_poincare_batch_twall_field(
    field,
    R_seeds,
    Z_seeds,
    phi_section,
    N_turns,
    DPhi,
    wall_phi,
    wall_R_all,
    wall_Z_all,
    *,
    extend_phi: bool = True,
):
    """Object-first toroidal-wall Poincare tracing wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return trace_poincare_batch_twall(
        R_seeds,
        Z_seeds,
        phi_section,
        N_turns,
        DPhi,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        wall_phi,
        wall_R_all,
        wall_Z_all,
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
    BZ_flat,
    BPhi_flat,
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
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        **kwargs,
    )


def find_fixed_points_batch_field(
    field,
    R_seeds,
    Z_seeds,
    phi_start,
    period,
    N_periods,
    DPhi,
    *,
    extend_phi: bool = True,
    **kwargs,
):
    """Object-first fixed-point search wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return find_fixed_points_batch(
        R_seeds,
        Z_seeds,
        phi_start,
        period,
        N_periods,
        DPhi,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
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
    BZ_flat,
    BPhi_flat,
    *,
    dphi_out=None,
    m_turns_DPm: int = 0,
    fd_eps: float = 1e-4,
):
    """Trace a single orbit from ``phi_start`` to ``phi_end``.

    The public wrapper keeps the historical ``phi_start``/``phi_end`` form.
    The cyna backend now accepts ``phi_span`` and can optionally compute
    ``DPm(phi)`` for an ``m_turns_DPm``-turn map.  ``m_turns_DPm=0`` returns
    identity DPm blocks and is the compatibility default for plain tracing.
    """
    if not _cyna_available() or _cyna_trace_orbit_along_phi is None:
        raise ImportError("pyna._cyna.trace_orbit_along_phi is unavailable. Build cyna first.")
    phi_span = float(phi_end) - float(phi_start)
    if dphi_out is None:
        dphi_out = DPhi
    return _cyna_trace_orbit_along_phi(
        float(R0),
        float(Z0),
        float(phi_start),
        float(phi_span),
        float(dphi_out),
        int(m_turns_DPm),
        float(DPhi),
        float(fd_eps),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
    )


def trace_orbit_along_phi_field(
    field,
    R0,
    Z0,
    phi_start,
    phi_end,
    DPhi,
    *,
    extend_phi: bool = True,
    dphi_out=None,
    m_turns_DPm: int = 0,
    fd_eps: float = 1e-4,
):
    """Object-first single-orbit tracing wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return trace_orbit_along_phi(
        R0,
        Z0,
        phi_start,
        phi_end,
        DPhi,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        dphi_out=dphi_out,
        m_turns_DPm=m_turns_DPm,
        fd_eps=fd_eps,
    )


__all__ = [
    "field_arrays_from_interpolators",
    "field_arrays_from_field",
    "precompile_tracer",
    "trace_poincare_batch",
    "trace_poincare_batch_field",
    "trace_poincare_multi_batch",
    "trace_poincare_multi_batch_field",
    "trace_poincare_batch_twall",
    "trace_poincare_batch_twall_field",
    "find_fixed_points_batch",
    "find_fixed_points_batch_field",
    "trace_orbit_along_phi",
    "trace_orbit_along_phi_field",
]
