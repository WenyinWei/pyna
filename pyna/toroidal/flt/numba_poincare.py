"""Toroidal Poincaré tracing helpers backed by the cyna C++ extension.

Canonical toroidal ownership for the grid-backed batch tracing helpers.
"""
from __future__ import annotations

import numpy as np
from pyna.fields.cylindrical import (
    CylindricalFieldArrays,
    as_vector_field_cylindrical,
    close_periodic_phi_grid,
)
from pyna.toroidal.geometry import coerce_toroidal_surface_arrays
from pyna._cyna import (
    is_available as _cyna_available,
    VectorFieldCylind as _cyna_VectorFieldCylind,
    trace_poincare_batch as _cyna_trace_poincare_batch,
    trace_poincare_multi as _cyna_trace_poincare_multi,
    trace_poincare_batch_twall as _cyna_trace_poincare_batch_twall,
    trace_map_batch_span as _cyna_trace_map_batch_span,
    trace_connection_length_twall as _cyna_trace_connection_length_twall,
    trace_wall_hits_twall as _cyna_trace_wall_hits_twall,
    find_fixed_points_batch as _cyna_find_fixed_points_batch,
    find_fixed_points_batch_span as _cyna_find_fixed_points_batch_span,
    trace_orbit_along_phi as _cyna_trace_orbit_along_phi,
    progress_DX_pol_along_orbit as _cyna_progress_DX_pol_along_orbit,
    progress_delta_X_along_orbit as _cyna_progress_delta_X_along_orbit,
    evolve_delta_X_cycle_along_cycle as _cyna_evolve_delta_X_cycle_along_cycle,
    trace_poincare_dpk_growth as _cyna_trace_poincare_dpk_growth,
    trace_poincare_dpk_growth_twall as _cyna_trace_poincare_dpk_growth_twall,
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


def _close_raw_field_arrays_for_cyna(
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BZ_flat,
    BPhi_flat,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Close raw endpoint=False phi grids before passing arrays to cyna."""

    Rg = np.ascontiguousarray(R_grid, dtype=np.float64)
    Zg = np.ascontiguousarray(Z_grid, dtype=np.float64)
    Pg = np.ascontiguousarray(Phi_grid, dtype=np.float64)
    nR, nZ, nPhi = int(Rg.size), int(Zg.size), int(Pg.size)
    expected = nR * nZ * nPhi
    BR3 = np.ascontiguousarray(BR_flat, dtype=np.float64)
    BZ3 = np.ascontiguousarray(BZ_flat, dtype=np.float64)
    BP3 = np.ascontiguousarray(BPhi_flat, dtype=np.float64)
    if BR3.size != expected or BZ3.size != expected or BP3.size != expected:
        raise ValueError("BR_flat, BZ_flat and BPhi_flat must each have size len(R_grid)*len(Z_grid)*len(Phi_grid)")
    BR3 = BR3.reshape(nR, nZ, nPhi)
    BZ3 = BZ3.reshape(nR, nZ, nPhi)
    BP3 = BP3.reshape(nR, nZ, nPhi)
    Pg, BR3, BZ3, BP3 = close_periodic_phi_grid(Pg, BR3, BZ3, BP3)
    return (
        Rg,
        Zg,
        np.ascontiguousarray(Pg, dtype=np.float64),
        np.ascontiguousarray(BR3, dtype=np.float64).ravel(),
        np.ascontiguousarray(BZ3, dtype=np.float64).ravel(),
        np.ascontiguousarray(BP3, dtype=np.float64).ravel(),
    )


def vector_field_cylind_from_field(field, *, extend_phi: bool = True):
    """Build a cyna ``VectorFieldCylind`` handle from a pyna VectorFieldCylind."""

    if not _cyna_available() or _cyna_VectorFieldCylind is None:
        raise ImportError("pyna._cyna.VectorFieldCylind is unavailable. Build cyna first.")
    normalized = as_vector_field_cylindrical(field)
    arrays = normalized.cyna_arrays(extend_phi=extend_phi)
    try:
        return _cyna_VectorFieldCylind(
            arrays.R_grid,
            arrays.Z_grid,
            arrays.Phi_grid,
            arrays.BR_flat,
            arrays.BZ_flat,
            arrays.BPhi_flat,
            nfp=int(normalized.nfp),
        )
    except TypeError as exc:
        # Local editable installs can have a stale cyna extension before the
        # new nfp-aware binding is rebuilt.  The six-array constructor keeps
        # tracing usable; rebuilt wheels expose nfp on the handle.
        if "nfp" not in str(exc):
            raise
        return _cyna_VectorFieldCylind(
            arrays.R_grid,
            arrays.Z_grid,
            arrays.Phi_grid,
            arrays.BR_flat,
            arrays.BZ_flat,
            arrays.BPhi_flat,
        )



def precompile_tracer(R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat):
    """Compatibility no-op retained for the historical numba-era API."""
    _ = (R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat)


def _direction_sign(direction) -> int:
    if isinstance(direction, str):
        d = direction.strip().lower()
        if d in {"+", "plus", "forward", "fwd", "phi+", "increasing"}:
            return +1
        if d in {"-", "minus", "backward", "back", "reverse", "rev", "phi-", "decreasing"}:
            return -1
    try:
        return +1 if float(direction) >= 0.0 else -1
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("direction must be '+', '-', 'forward', or 'backward'") from exc


def _filter_directional_result(result: dict[str, np.ndarray], direction: str) -> dict[str, np.ndarray]:
    if direction == "both":
        return result
    keep_plus = direction == "+"
    keys = {
        "Lc_plus", "hit_plus", "term_plus",
    } if keep_plus else {
        "Lc_minus", "hit_minus", "term_minus",
    }
    return {k: v for k, v in result.items() if k in keys}



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
    *,
    direction="+",
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
        -1,
        _direction_sign(direction),
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
    direction="+",
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
        direction=direction,
    )


def trace_poincare_bidirectional_batch(
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
    """Trace the same Poincaré section in both φ directions.

    Returns ``{"forward": (...), "backward": (...)}``, where each value has
    the same ``(counts, R_flat, Z_flat)`` layout as :func:`trace_poincare_batch`.
    """

    return {
        "forward": trace_poincare_batch(
            R_seeds, Z_seeds, phi_section, N_turns, DPhi,
            R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat,
            wall_R, wall_Z, direction="+",
        ),
        "backward": trace_poincare_batch(
            R_seeds, Z_seeds, phi_section, N_turns, DPhi,
            R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat,
            wall_R, wall_Z, direction="-",
        ),
    }


def trace_poincare_bidirectional_batch_field(
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
    """Object-first bidirectional Poincaré tracing wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return trace_poincare_bidirectional_batch(
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
    *,
    direction="+",
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
        -1,
        _direction_sign(direction),
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
    direction="+",
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
        direction=direction,
    )


def trace_poincare_multi_bidirectional_batch(
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
    """Trace multiple Poincaré sections in both φ directions."""

    return {
        "forward": trace_poincare_multi_batch(
            R_seeds, Z_seeds, phi_sections_arr, N_turns, DPhi,
            R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat,
            wall_R, wall_Z, direction="+",
        ),
        "backward": trace_poincare_multi_batch(
            R_seeds, Z_seeds, phi_sections_arr, N_turns, DPhi,
            R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat,
            wall_R, wall_Z, direction="-",
        ),
    }


def trace_poincare_multi_bidirectional_batch_field(
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
    """Object-first bidirectional multi-section Poincaré wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return trace_poincare_multi_bidirectional_batch(
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
    *,
    direction="+",
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
        -1,
        _direction_sign(direction),
    )


def trace_poincare_batch_twall_field(
    field,
    R_seeds,
    Z_seeds,
    phi_section,
    N_turns,
    DPhi,
    wall_phi,
    wall_R_all=None,
    wall_Z_all=None,
    *,
    extend_phi: bool = True,
    direction="+",
):
    """Object-first toroidal-wall Poincare tracing wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
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
        direction=direction,
    )


def trace_poincare_bidirectional_batch_twall(
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
    """Trace one Poincaré section in both directions against a 3-D wall."""

    return {
        "forward": trace_poincare_batch_twall(
            R_seeds, Z_seeds, phi_section, N_turns, DPhi,
            R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat,
            wall_phi, wall_R_all, wall_Z_all, direction="+",
        ),
        "backward": trace_poincare_batch_twall(
            R_seeds, Z_seeds, phi_section, N_turns, DPhi,
            R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat,
            wall_phi, wall_R_all, wall_Z_all, direction="-",
        ),
    }


def trace_poincare_bidirectional_batch_twall_field(
    field,
    R_seeds,
    Z_seeds,
    phi_section,
    N_turns,
    DPhi,
    wall_phi,
    wall_R_all=None,
    wall_Z_all=None,
    *,
    extend_phi: bool = True,
):
    """Object-first bidirectional toroidal-wall Poincaré wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
    return trace_poincare_bidirectional_batch_twall(
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


def trace_connection_length_twall(
    R_seeds,
    Z_seeds,
    phi_start,
    max_turns,
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
    *,
    direction="both",
):
    """Compute forward/backward connection lengths against a toroidal wall."""

    if not _cyna_available() or _cyna_trace_connection_length_twall is None:
        raise ImportError("pyna._cyna.trace_connection_length_twall is unavailable. Build cyna first.")
    L_fwd, L_bwd = _cyna_trace_connection_length_twall(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        float(phi_start),
        int(max_turns),
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
    result = {
        "Lc_plus": np.asarray(L_fwd),
        "Lc_minus": np.asarray(L_bwd),
        "Lc_sum": np.asarray(L_fwd) + np.asarray(L_bwd),
        "Lc_max": np.maximum(L_fwd, L_bwd),
        "Lc_min": np.minimum(L_fwd, L_bwd),
    }
    if direction == "both":
        return result
    return _filter_directional_result(result, "+" if _direction_sign(direction) > 0 else "-")


def trace_connection_length_twall_field(
    field,
    R_seeds,
    Z_seeds,
    phi_start,
    max_turns,
    DPhi,
    wall_phi,
    wall_R_all=None,
    wall_Z_all=None,
    *,
    extend_phi: bool = True,
    direction="both",
):
    """Object-first wrapper for toroidal-wall connection lengths."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
    return trace_connection_length_twall(
        R_seeds,
        Z_seeds,
        phi_start,
        max_turns,
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
        direction=direction,
    )


def trace_wall_hits_twall(
    R_seeds,
    Z_seeds,
    phi_start,
    max_turns,
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
    *,
    direction="both",
):
    """Trace seeds to a toroidal wall and return hit points for both directions.

    ``term_plus`` / ``term_minus`` use cyna's convention:
    ``0`` no termination, ``1`` wall polygon, ``2`` field-grid exit,
    ``3`` non-finite field.
    """

    if not _cyna_available() or _cyna_trace_wall_hits_twall is None:
        raise ImportError("pyna._cyna.trace_wall_hits_twall is unavailable. Build cyna first.")
    (
        L_fwd,
        L_bwd,
        R_hf,
        Z_hf,
        phi_hf,
        R_hb,
        Z_hb,
        phi_hb,
        term_fwd,
        term_bwd,
    ) = _cyna_trace_wall_hits_twall(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        float(phi_start),
        int(max_turns),
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
    hit_plus = np.column_stack([R_hf, Z_hf, phi_hf])
    hit_minus = np.column_stack([R_hb, Z_hb, phi_hb])
    result = {
        "Lc_plus": np.asarray(L_fwd),
        "Lc_minus": np.asarray(L_bwd),
        "Lc_sum": np.asarray(L_fwd) + np.asarray(L_bwd),
        "Lc_max": np.maximum(L_fwd, L_bwd),
        "Lc_min": np.minimum(L_fwd, L_bwd),
        "hit_plus": hit_plus,
        "hit_minus": hit_minus,
        "term_plus": np.asarray(term_fwd),
        "term_minus": np.asarray(term_bwd),
    }
    if direction == "both":
        return result
    return _filter_directional_result(result, "+" if _direction_sign(direction) > 0 else "-")


def trace_wall_hits_twall_field(
    field,
    R_seeds,
    Z_seeds,
    phi_start,
    max_turns,
    DPhi,
    wall_phi,
    wall_R_all=None,
    wall_Z_all=None,
    *,
    extend_phi: bool = True,
    direction="both",
):
    """Object-first wrapper for toroidal-wall hit tracing."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
    return trace_wall_hits_twall(
        R_seeds,
        Z_seeds,
        phi_start,
        max_turns,
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
        direction=direction,
    )


def strike_line_from_wall_hits(wall_hits: dict, *, direction="+", wall_term: int = 1) -> dict[str, np.ndarray]:
    """Extract strike points from a ``trace_wall_hits_twall`` result.

    The returned arrays preserve seed order, which is usually the right ordering
    for a bundle launched from an ordered curve.
    """

    suffix = "plus" if _direction_sign(direction) > 0 else "minus"
    terms = np.asarray(wall_hits[f"term_{suffix}"])
    mask = terms == int(wall_term)
    hits = np.asarray(wall_hits[f"hit_{suffix}"])[mask]
    idx = np.nonzero(mask)[0]
    return {
        "R": hits[:, 0] if hits.size else np.empty(0, dtype=float),
        "Z": hits[:, 1] if hits.size else np.empty(0, dtype=float),
        "phi": hits[:, 2] if hits.size else np.empty(0, dtype=float),
        "seed_index": idx,
        "connection_length": np.asarray(wall_hits[f"Lc_{suffix}"])[mask],
        "term_type": terms[mask],
    }


def trace_strike_line_twall(
    R_seeds,
    Z_seeds,
    phi_start,
    max_turns,
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
    *,
    direction="+",
):
    """Trace an ordered seed bundle and return its toroidal-wall strike line."""

    hits = trace_wall_hits_twall(
        R_seeds,
        Z_seeds,
        phi_start,
        max_turns,
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
        direction="both",
    )
    return strike_line_from_wall_hits(hits, direction=direction)


def trace_strike_line_twall_field(
    field,
    R_seeds,
    Z_seeds,
    phi_start,
    max_turns,
    DPhi,
    wall_phi,
    wall_R_all=None,
    wall_Z_all=None,
    *,
    extend_phi: bool = True,
    direction="+",
):
    """Object-first wrapper for strike-line tracing from an ordered seed bundle."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
    return trace_strike_line_twall(
        R_seeds,
        Z_seeds,
        phi_start,
        max_turns,
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
        direction=direction,
    )



def find_fixed_points_batch(
    R_seeds,
    Z_seeds,
    phi_start,
    map_power,
    field_period_count,
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
    map_step_count = int(map_power) * int(field_period_count)
    if map_step_count <= 0:
        raise ValueError("map_power * field_period_count must be positive")
    fd_eps = float(kwargs.pop("fd_eps", 1.0e-4))
    max_iter = int(kwargs.pop("max_iter", 40))
    tol = float(kwargs.pop("tol", 1.0e-9))
    n_threads = int(kwargs.pop("n_threads", -1))
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unknown}")
    return _cyna_find_fixed_points_batch(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        float(phi_start),
        map_step_count,
        float(DPhi),
        fd_eps,
        max_iter,
        tol,
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        n_threads,
    )


def find_fixed_points_batch_field(
    field,
    R_seeds,
    Z_seeds,
    phi_start,
    map_power,
    field_period_count,
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
        map_power,
        field_period_count,
        DPhi,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        **kwargs,
    )


def find_fixed_points_batch_span(
    R_seeds,
    Z_seeds,
    phi_start,
    map_span,
    DPhi,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BZ_flat,
    BPhi_flat,
    **kwargs,
):
    """Batch search for fixed points of an arbitrary toroidal-span map."""
    if not _cyna_available() or _cyna_find_fixed_points_batch_span is None:
        raise ImportError("pyna._cyna.find_fixed_points_batch_span is unavailable. Build cyna first.")
    if not np.isfinite(float(map_span)) or abs(float(map_span)) <= 1.0e-14:
        raise ValueError("map_span must be a nonzero finite toroidal angle")
    if not np.isfinite(float(DPhi)) or abs(float(DPhi)) <= 1.0e-14:
        raise ValueError("DPhi must be a nonzero finite toroidal angle step")
    fd_eps = float(kwargs.pop("fd_eps", 1.0e-4))
    max_iter = int(kwargs.pop("max_iter", 40))
    tol = float(kwargs.pop("tol", 1.0e-9))
    n_threads = int(kwargs.pop("n_threads", -1))
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unknown}")
    R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat = _close_raw_field_arrays_for_cyna(
        R_grid,
        Z_grid,
        Phi_grid,
        BR_flat,
        BZ_flat,
        BPhi_flat,
    )
    return _cyna_find_fixed_points_batch_span(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        float(phi_start),
        float(map_span),
        float(DPhi),
        fd_eps,
        max_iter,
        tol,
        BR_flat,
        BZ_flat,
        BPhi_flat,
        R_grid,
        Z_grid,
        Phi_grid,
        n_threads,
    )


def find_fixed_points_batch_span_field(
    field,
    R_seeds,
    Z_seeds,
    phi_start,
    map_span,
    DPhi,
    *,
    extend_phi: bool = True,
    **kwargs,
):
    """Object-first arbitrary-span fixed-point search wrapper."""

    try:
        fd_eps = float(kwargs.pop("fd_eps", 1.0e-4))
        max_iter = int(kwargs.pop("max_iter", 40))
        tol = float(kwargs.pop("tol", 1.0e-9))
        n_threads = int(kwargs.pop("n_threads", -1))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")
        handle = vector_field_cylind_from_field(field, extend_phi=extend_phi)
        return handle.find_fixed_points_batch_span(
            np.ascontiguousarray(R_seeds, dtype=np.float64),
            np.ascontiguousarray(Z_seeds, dtype=np.float64),
            float(phi_start),
            float(map_span),
            float(DPhi),
            fd_eps,
            max_iter,
            tol,
            n_threads,
        )
    except ImportError:
        arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return find_fixed_points_batch_span(
        R_seeds,
        Z_seeds,
        phi_start,
        map_span,
        DPhi,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        **kwargs,
    )


def trace_map_batch_span(
    R_seeds,
    Z_seeds,
    phi_start,
    map_span,
    N_steps,
    DPhi,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BZ_flat,
    BPhi_flat,
    wall_R,
    wall_Z,
    *,
    n_threads: int = -1,
):
    """Batch trace iterates of an arbitrary toroidal-span map."""

    if not _cyna_available() or _cyna_trace_map_batch_span is None:
        raise ImportError("pyna._cyna.trace_map_batch_span is unavailable. Build cyna first.")
    if not np.isfinite(float(map_span)) or abs(float(map_span)) <= 1.0e-14:
        raise ValueError("map_span must be a nonzero finite toroidal angle")
    if not np.isfinite(float(DPhi)) or abs(float(DPhi)) <= 1.0e-14:
        raise ValueError("DPhi must be a nonzero finite toroidal angle step")
    if int(N_steps) <= 0:
        raise ValueError("N_steps must be positive")
    R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat = _close_raw_field_arrays_for_cyna(
        R_grid,
        Z_grid,
        Phi_grid,
        BR_flat,
        BZ_flat,
        BPhi_flat,
    )
    return _cyna_trace_map_batch_span(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        float(phi_start),
        float(map_span),
        int(N_steps),
        float(DPhi),
        BR_flat,
        BZ_flat,
        BPhi_flat,
        R_grid,
        Z_grid,
        Phi_grid,
        np.ascontiguousarray(wall_R, dtype=np.float64),
        np.ascontiguousarray(wall_Z, dtype=np.float64),
        int(n_threads),
    )


def trace_map_batch_span_field(
    field,
    R_seeds,
    Z_seeds,
    phi_start,
    map_span,
    N_steps,
    DPhi,
    wall_R=None,
    wall_Z=None,
    *,
    extend_phi: bool = True,
    n_threads: int = -1,
):
    """Object-first wrapper for arbitrary-span map iterates."""

    if wall_R is None or wall_Z is None:
        wall_R = np.empty(0, dtype=np.float64)
        wall_Z = np.empty(0, dtype=np.float64)
    handle = vector_field_cylind_from_field(field, extend_phi=extend_phi)
    return handle.trace_map_batch_span(
        np.ascontiguousarray(R_seeds, dtype=np.float64),
        np.ascontiguousarray(Z_seeds, dtype=np.float64),
        phi_start,
        map_span,
        N_steps,
        DPhi,
        np.ascontiguousarray(wall_R, dtype=np.float64),
        np.ascontiguousarray(wall_Z, dtype=np.float64),
        n_threads=n_threads,
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
    if phi_span != 0.0:
        dphi_out = np.copysign(abs(float(dphi_out)), phi_span)
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


def trace_orbit_bidirectional_along_phi(
    R0,
    Z0,
    phi_start,
    phi_span,
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
    """Trace one seed in both φ directions.

    ``phi_span`` is treated as an absolute span; the returned dict has
    ``"forward"`` and ``"backward"`` entries, each matching
    :func:`trace_orbit_along_phi` output.
    """

    span = abs(float(phi_span))
    out_step = None if dphi_out is None else abs(float(dphi_out))
    return {
        "forward": trace_orbit_along_phi(
            R0,
            Z0,
            phi_start,
            float(phi_start) + span,
            DPhi,
            R_grid,
            Z_grid,
            Phi_grid,
            BR_flat,
            BZ_flat,
            BPhi_flat,
            dphi_out=out_step,
            m_turns_DPm=m_turns_DPm,
            fd_eps=fd_eps,
        ),
        "backward": trace_orbit_along_phi(
            R0,
            Z0,
            phi_start,
            float(phi_start) - span,
            DPhi,
            R_grid,
            Z_grid,
            Phi_grid,
            BR_flat,
            BZ_flat,
            BPhi_flat,
            dphi_out=out_step,
            m_turns_DPm=m_turns_DPm,
            fd_eps=fd_eps,
        ),
    }


def trace_orbit_bidirectional_along_phi_field(
    field,
    R0,
    Z0,
    phi_start,
    phi_span,
    DPhi,
    *,
    extend_phi: bool = True,
    dphi_out=None,
    m_turns_DPm: int = 0,
    fd_eps: float = 1e-4,
):
    """Object-first bidirectional orbit tracing wrapper."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return trace_orbit_bidirectional_along_phi(
        R0,
        Z0,
        phi_start,
        phi_span,
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


def progress_DX_pol_along_orbit(
    R_traj,
    Z_traj,
    phi_traj,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BZ_flat,
    BPhi_flat,
    *,
    max_step: float = 0.005,
):
    """Progress ``DX_pol(phi_e, phi_s)`` along an already sampled orbit.

    The returned array has shape ``(N, 2, 2)`` and starts from identity at
    index 0.  The cyna backend uses the supplied orbit samples as the path;
    it does not retrace the field line.
    """
    if not _cyna_available() or _cyna_progress_DX_pol_along_orbit is None:
        raise ImportError("pyna._cyna.progress_DX_pol_along_orbit is unavailable. Build cyna first.")
    return _cyna_progress_DX_pol_along_orbit(
        np.ascontiguousarray(R_traj, dtype=np.float64),
        np.ascontiguousarray(Z_traj, dtype=np.float64),
        np.ascontiguousarray(phi_traj, dtype=np.float64),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        float(max_step),
    )


def progress_DX_pol_along_orbit_field(
    field,
    R_traj,
    Z_traj,
    phi_traj,
    *,
    extend_phi: bool = True,
    max_step: float = 0.005,
):
    """Object-first wrapper for ``progress_DX_pol_along_orbit``."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return progress_DX_pol_along_orbit(
        R_traj,
        Z_traj,
        phi_traj,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        max_step=max_step,
    )


def progress_delta_X_along_orbit(
    R_traj,
    Z_traj,
    phi_traj,
    delta_X0,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BZ_flat,
    BPhi_flat,
    dBR_flat,
    dBZ_flat,
    dBPhi_flat,
    *,
    max_step: float = 0.005,
):
    """Progress first-order ``delta_X(phi_s, phi_e)`` along a sampled orbit.

    ``progress`` means the source point ``phi_s`` is fixed and only ``phi_e``
    advances.  This is the correct verb for ``delta_X_pol`` and other open
    trajectory responses.  The returned array has shape ``(N, 2)``.
    """
    if not _cyna_available() or _cyna_progress_delta_X_along_orbit is None:
        raise ImportError("pyna._cyna.progress_delta_X_along_orbit is unavailable. Build cyna first.")
    return _cyna_progress_delta_X_along_orbit(
        np.ascontiguousarray(R_traj, dtype=np.float64),
        np.ascontiguousarray(Z_traj, dtype=np.float64),
        np.ascontiguousarray(phi_traj, dtype=np.float64),
        np.ascontiguousarray(delta_X0, dtype=np.float64),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(dBR_flat, dtype=np.float64),
        np.ascontiguousarray(dBZ_flat, dtype=np.float64),
        np.ascontiguousarray(dBPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        float(max_step),
    )


def evolve_delta_X_cycle_along_cycle(
    R_traj,
    Z_traj,
    phi_traj,
    delta_X_cyc0,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BZ_flat,
    BPhi_flat,
    dBR_flat,
    dBZ_flat,
    dBPhi_flat,
    *,
    max_step: float = 0.005,
):
    """Evolve periodic-cycle displacement ``delta_X_cyc(phi)`` along a cycle.

    ``evolve`` means ``phi_s`` and ``phi_e = phi_s + 2*pi*m`` move together
    along a periodic cycle.  This intentionally uses the same inhomogeneous
    response equation as :func:`progress_delta_X_along_orbit`; the difference is
    the interpretation and initial value.  ``delta_X_cyc0`` must already be the
    periodic initial displacement from the cycle closure solve, while
    ``progress_delta_X_along_orbit`` advances an open-trajectory response from a
    chosen source point.
    """
    if not _cyna_available() or _cyna_evolve_delta_X_cycle_along_cycle is None:
        raise ImportError("pyna._cyna.evolve_delta_X_cycle_along_cycle is unavailable. Build cyna first.")
    return _cyna_evolve_delta_X_cycle_along_cycle(
        np.ascontiguousarray(R_traj, dtype=np.float64),
        np.ascontiguousarray(Z_traj, dtype=np.float64),
        np.ascontiguousarray(phi_traj, dtype=np.float64),
        np.ascontiguousarray(delta_X_cyc0, dtype=np.float64),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(dBR_flat, dtype=np.float64),
        np.ascontiguousarray(dBZ_flat, dtype=np.float64),
        np.ascontiguousarray(dBPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
        float(max_step),
    )


def evolve_delta_X_cycle_along_orbit(*args, **kwargs):
    """Compatibility alias for :func:`evolve_delta_X_cycle_along_cycle`."""

    return evolve_delta_X_cycle_along_cycle(*args, **kwargs)


def progress_delta_X_along_orbit_field(
    field,
    delta_field,
    R_traj,
    Z_traj,
    phi_traj,
    delta_X0=(0.0, 0.0),
    *,
    extend_phi: bool = True,
    max_step: float = 0.005,
):
    """Object-first wrapper for ``progress_delta_X_along_orbit``."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    d_arrays = field_arrays_from_field(delta_field, extend_phi=extend_phi)
    return progress_delta_X_along_orbit(
        R_traj,
        Z_traj,
        phi_traj,
        delta_X0,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        d_arrays.BR_flat,
        d_arrays.BZ_flat,
        d_arrays.BPhi_flat,
        max_step=max_step,
    )


def evolve_delta_X_cycle_along_cycle_field(
    field,
    delta_field,
    R_traj,
    Z_traj,
    phi_traj,
    delta_X_cyc0,
    *,
    extend_phi: bool = True,
    max_step: float = 0.005,
):
    """Object-first wrapper for ``evolve_delta_X_cycle_along_cycle``."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    d_arrays = field_arrays_from_field(delta_field, extend_phi=extend_phi)
    return evolve_delta_X_cycle_along_cycle(
        R_traj,
        Z_traj,
        phi_traj,
        delta_X_cyc0,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        d_arrays.BR_flat,
        d_arrays.BZ_flat,
        d_arrays.BPhi_flat,
        max_step=max_step,
    )


def evolve_delta_X_cycle_along_orbit_field(*args, **kwargs):
    """Compatibility alias for :func:`evolve_delta_X_cycle_along_cycle_field`."""

    return evolve_delta_X_cycle_along_cycle_field(*args, **kwargs)


def trace_poincare_dpk_growth(
    R0,
    Z0,
    phi_start,
    max_returns,
    DPhi,
    R_grid,
    Z_grid,
    Phi_grid,
    BR_flat,
    BZ_flat,
    BPhi_flat,
    *,
    return_period: float = 2.0 * np.pi,
    record_stride: int = 1,
):
    """Trace one seed and record cumulative ``DP^k`` at Poincare returns.

    This calls the cyna variational-equation path once and records
    ``DX_pol(phi_start, phi_start + k * return_period)`` at each requested
    return.  It is the hot-path API for chaos/LCFS screening with k up to
    hundreds of returns.
    """
    if not _cyna_available() or _cyna_trace_poincare_dpk_growth is None:
        raise ImportError("pyna._cyna.trace_poincare_dpk_growth is unavailable. Build cyna first.")
    return _cyna_trace_poincare_dpk_growth(
        float(R0),
        float(Z0),
        float(phi_start),
        int(max_returns),
        float(return_period),
        int(record_stride),
        float(DPhi),
        np.ascontiguousarray(BR_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat, dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(R_grid, dtype=np.float64),
        np.ascontiguousarray(Z_grid, dtype=np.float64),
        np.ascontiguousarray(Phi_grid, dtype=np.float64),
    )


def trace_poincare_dpk_growth_field(
    field,
    R0,
    Z0,
    phi_start,
    max_returns,
    DPhi,
    *,
    extend_phi: bool = True,
    return_period: float = 2.0 * np.pi,
    record_stride: int = 1,
):
    """Object-first wrapper for cumulative Poincare ``DP^k`` growth."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    return trace_poincare_dpk_growth(
        R0,
        Z0,
        phi_start,
        max_returns,
        DPhi,
        arrays.R_grid,
        arrays.Z_grid,
        arrays.Phi_grid,
        arrays.BR_flat,
        arrays.BZ_flat,
        arrays.BPhi_flat,
        return_period=return_period,
        record_stride=record_stride,
    )


def trace_poincare_dpk_growth_twall(
    R0,
    Z0,
    phi_start,
    max_returns,
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
    *,
    return_period: float = 2.0 * np.pi,
    record_stride: int = 1,
    stop_at_wall: bool = True,
):
    """Wall-aware cumulative ``DP^k`` tracing.

    Returns the same trajectory data as :func:`trace_poincare_dpk_growth`,
    plus ``hit=[R, Z, phi, k_float]`` for the first wall crossing and
    ``term`` where 0 means no hit, 1 means wall hit, and 2 means grid/nonfinite
    termination.  If ``stop_at_wall`` is false, tracing continues after the
    first wall hit so topology can be inspected without wall truncation.
    """
    if not _cyna_available() or _cyna_trace_poincare_dpk_growth_twall is None:
        raise ImportError("pyna._cyna.trace_poincare_dpk_growth_twall is unavailable. Build cyna first.")
    return _cyna_trace_poincare_dpk_growth_twall(
        float(R0),
        float(Z0),
        float(phi_start),
        int(max_returns),
        float(return_period),
        int(record_stride),
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
        bool(stop_at_wall),
    )


def trace_poincare_dpk_growth_twall_field(
    field,
    R0,
    Z0,
    phi_start,
    max_returns,
    DPhi,
    wall_phi,
    wall_R_all=None,
    wall_Z_all=None,
    *,
    extend_phi: bool = True,
    return_period: float = 2.0 * np.pi,
    record_stride: int = 1,
    stop_at_wall: bool = True,
):
    """Object-first wrapper for wall-aware cumulative Poincare ``DP^k``."""

    arrays = field_arrays_from_field(field, extend_phi=extend_phi)
    wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
    return trace_poincare_dpk_growth_twall(
        R0,
        Z0,
        phi_start,
        max_returns,
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
        return_period=return_period,
        record_stride=record_stride,
        stop_at_wall=stop_at_wall,
    )


__all__ = [
    "field_arrays_from_interpolators",
    "field_arrays_from_field",
    "vector_field_cylind_from_field",
    "precompile_tracer",
    "trace_poincare_batch",
    "trace_poincare_batch_field",
    "trace_poincare_bidirectional_batch",
    "trace_poincare_bidirectional_batch_field",
    "trace_poincare_multi_batch",
    "trace_poincare_multi_batch_field",
    "trace_poincare_multi_bidirectional_batch",
    "trace_poincare_multi_bidirectional_batch_field",
    "trace_poincare_batch_twall",
    "trace_poincare_batch_twall_field",
    "trace_poincare_bidirectional_batch_twall",
    "trace_poincare_bidirectional_batch_twall_field",
    "trace_connection_length_twall",
    "trace_connection_length_twall_field",
    "trace_wall_hits_twall",
    "trace_wall_hits_twall_field",
    "strike_line_from_wall_hits",
    "trace_strike_line_twall",
    "trace_strike_line_twall_field",
    "find_fixed_points_batch",
    "find_fixed_points_batch_field",
    "find_fixed_points_batch_span",
    "find_fixed_points_batch_span_field",
    "trace_map_batch_span",
    "trace_map_batch_span_field",
    "trace_orbit_along_phi",
    "trace_orbit_along_phi_field",
    "trace_orbit_bidirectional_along_phi",
    "trace_orbit_bidirectional_along_phi_field",
    "progress_DX_pol_along_orbit",
    "progress_DX_pol_along_orbit_field",
    "progress_delta_X_along_orbit",
    "progress_delta_X_along_orbit_field",
    "evolve_delta_X_cycle_along_cycle",
    "evolve_delta_X_cycle_along_cycle_field",
    "evolve_delta_X_cycle_along_orbit",
    "evolve_delta_X_cycle_along_orbit_field",
    "trace_poincare_dpk_growth",
    "trace_poincare_dpk_growth_field",
    "trace_poincare_dpk_growth_twall",
    "trace_poincare_dpk_growth_twall_field",
]
