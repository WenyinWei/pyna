"""Cycle shifts under a cylindrical magnetic-field perturbation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pyna._cyna.utils import prepare_field_cache
from pyna.fields.periodicity import ToroidalPeriodicity, normalize_nfp
from pyna.topo.fpt import CyclePerturbationShift, compute_cycle_shift_from_cache


@dataclass(frozen=True)
class AxisCycleShift:
    """Shifted O-cycle samples and the underlying FPT cycle shift."""

    axis_R: NDArray[np.float64]
    axis_Z: NDArray[np.float64]
    cycle_shift: CyclePerturbationShift
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CyclePointShift:
    """Shifted periodic-cycle points sampled on requested toroidal sections."""

    sections: list[NDArray[np.float64]]
    cycle_shifts: tuple[CyclePerturbationShift, ...]
    diagnostics: dict[str, Any]


def field_period_cache_from_components(
    R: NDArray[np.float64],
    Z: NDArray[np.float64],
    Phi: NDArray[np.float64],
    *,
    BR: NDArray[np.float64],
    BZ: NDArray[np.float64],
    BPhi: NDArray[np.float64],
    nfp: int,
) -> dict[str, NDArray[np.float64]]:
    """Build a native one-field-period cyna cache with explicit ``nfp``."""

    phi = np.asarray(Phi, dtype=np.float64)
    n_periods = normalize_nfp(nfp)
    if phi.ndim != 1 or phi.size == 0:
        raise ValueError("Phi must be a non-empty 1-D grid")
    def _component(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(values, dtype=np.float64)
        expected = (np.asarray(R).size, np.asarray(Z).size, phi.size)
        if arr.shape != expected:
            raise ValueError(f"{name} shape {arr.shape} does not match grid shape {expected}")
        return arr

    return prepare_field_cache(
        {
            "R_grid": np.asarray(R, dtype=np.float64),
            "Z_grid": np.asarray(Z, dtype=np.float64),
            "Phi_grid": phi,
            "BR": _component("BR", BR),
            "BZ": _component("BZ", BZ),
            "BPhi": _component("BPhi", BPhi),
            "nfp": n_periods,
        },
        extend_phi=True,
    )


def cycle_shift_from_fields(
    R0: float,
    Z0: float,
    phi0: float,
    phi_span: float,
    base_field: Any,
    delta_field: Any,
    *,
    dphi_out: float = 0.01,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
) -> CyclePerturbationShift:
    """Return the first-order periodic-cycle shift for ``base_field + delta_field``."""

    return compute_cycle_shift_from_cache(
        float(R0),
        float(Z0),
        float(phi0),
        float(phi_span),
        base_field,
        delta_field,
        dphi_out=float(dphi_out),
        DPhi=float(DPhi),
        fd_eps=float(fd_eps),
    )


def axis_cycle_shift_from_fields(
    axis_R: NDArray[np.float64] | float,
    axis_Z: NDArray[np.float64] | float,
    phi_grid: NDArray[np.float64],
    base_field: Any,
    delta_field: Any,
    *,
    nfp: int,
    field_periods: float = 1.0,
    steps_per_field_period: int = 200,
    fd_eps: float = 1.0e-4,
) -> AxisCycleShift:
    """Shift an O-cycle represented by ``axis_R/Z(phi_grid)`` using δXcyc."""

    phi_native = np.asarray(phi_grid, dtype=np.float64)
    if phi_native.ndim != 1 or phi_native.size == 0:
        raise ValueError("phi_grid must be a non-empty 1-D grid")
    periodicity = ToroidalPeriodicity(nfp=nfp)
    period = periodicity.field_period
    span_periods = float(field_periods)
    if span_periods <= 0.0:
        raise ValueError("field_periods must be positive")
    steps_per_period = int(steps_per_field_period)
    if steps_per_period <= 0:
        raise ValueError("steps_per_field_period must be positive")

    axis_R_arr = np.broadcast_to(np.asarray(axis_R, dtype=np.float64), phi_native.shape)
    axis_Z_arr = np.broadcast_to(np.asarray(axis_Z, dtype=np.float64), phi_native.shape)
    phi_wrapped = periodicity.wrap(phi_native)
    seed_phi = float(phi_wrapped[0])
    seed_idx = int(np.argmin(np.abs(phi_wrapped - seed_phi)))
    phi_span = period * span_periods
    n_steps = max(int(round(steps_per_period * span_periods)), 16)
    dphi = phi_span / float(n_steps)
    cycle_shift = cycle_shift_from_fields(
        float(axis_R_arr[seed_idx]),
        float(axis_Z_arr[seed_idx]),
        seed_phi,
        phi_span,
        base_field,
        delta_field,
        dphi_out=dphi,
        DPhi=dphi,
        fd_eps=float(fd_eps),
    )
    delta_R, delta_Z, finite_fraction = _delta_cycle_shift_on_queries(
        cycle_shift,
        seed_phi,
        periodicity.wrap(phi_native - seed_phi),
    )
    shifted_R = axis_R_arr + delta_R
    shifted_Z = axis_Z_arr + delta_Z
    dXcyc = np.asarray(cycle_shift.delta_X_cyc, dtype=np.float64)
    dXcyc0 = np.asarray(cycle_shift.delta_X_cyc0, dtype=np.float64)
    periodic_residual = cycle_shift.periodic_residual
    diagnostics = {
        "method": "cyna_evolve_delta_X_cycle_along_orbit",
        "seed_phi": seed_phi,
        "field_period": float(period),
        "field_periods": span_periods,
        "steps_per_field_period": steps_per_period,
        "n_steps": int(n_steps),
        "fd_eps": float(fd_eps),
        "delta_X_cyc0_R_m": float(dXcyc0[0]),
        "delta_X_cyc0_Z_m": float(dXcyc0[1]),
        "delta_X_cyc0_norm_m": float(np.linalg.norm(dXcyc0)),
        "max_delta_X_cyc_norm_m": _safe_norm_max(dXcyc),
        "periodic_residual_R_m": float(periodic_residual[0]),
        "periodic_residual_Z_m": float(periodic_residual[1]),
        "periodic_residual_norm_m": float(np.linalg.norm(periodic_residual)),
        "alive_fraction": _alive_fraction(cycle_shift),
        "finite_fraction": finite_fraction,
        "axis_shift_R_mean_m": float(np.mean(shifted_R - axis_R_arr)),
        "axis_shift_Z_mean_m": float(np.mean(shifted_Z - axis_Z_arr)),
        "axis_shift_norm_max_m": float(np.max(np.hypot(shifted_R - axis_R_arr, shifted_Z - axis_Z_arr))),
    }
    return AxisCycleShift(
        axis_R=shifted_R,
        axis_Z=shifted_Z,
        cycle_shift=cycle_shift,
        diagnostics=diagnostics,
    )


def cycle_points_shift_from_fields(
    seed_points: NDArray[np.float64],
    phi_sections: Sequence[float],
    base_field: Any,
    delta_field: Any,
    *,
    nfp: int,
    field_periods: float = 1.0,
    steps_per_field_period: int = 200,
    fd_eps: float = 1.0e-4,
) -> CyclePointShift:
    """Shift periodic-cycle seed points and sample them at ``phi_sections``."""

    seeds = np.asarray(seed_points, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 2 or seeds.shape[0] == 0:
        raise ValueError("seed_points must have shape (n, 2)")
    phi_arr = np.asarray(phi_sections, dtype=np.float64)
    if phi_arr.ndim != 1 or phi_arr.size == 0:
        raise ValueError("phi_sections must be a non-empty 1-D sequence")
    periodicity = ToroidalPeriodicity(nfp=nfp)
    period = periodicity.field_period
    span_periods = float(field_periods)
    if span_periods <= 0.0:
        raise ValueError("field_periods must be positive")
    steps_per_period = int(steps_per_field_period)
    if steps_per_period <= 0:
        raise ValueError("steps_per_field_period must be positive")
    seed_phi = float(periodicity.wrap(phi_arr[0]))
    phi_span = period * span_periods
    n_steps = max(int(round(steps_per_period * span_periods)), 16)
    dphi = phi_span / float(n_steps)
    rel_queries = periodicity.wrap(phi_arr - seed_phi)
    sections = [np.empty((seeds.shape[0], 2), dtype=np.float64) for _ in phi_arr]
    cycle_shifts: list[CyclePerturbationShift] = []
    finite_fractions: list[float] = []
    cycle_rows: list[dict[str, Any]] = []

    for idx, (R0, Z0) in enumerate(seeds):
        cycle_shift = cycle_shift_from_fields(
            float(R0),
            float(Z0),
            seed_phi,
            phi_span,
            base_field,
            delta_field,
            dphi_out=dphi,
            DPhi=dphi,
            fd_eps=float(fd_eps),
        )
        shifted_R, shifted_Z, finite_fraction = _shifted_orbit_on_queries(
            cycle_shift,
            seed_phi,
            rel_queries,
        )
        for sec_idx in range(phi_arr.size):
            sections[sec_idx][idx, 0] = shifted_R[sec_idx]
            sections[sec_idx][idx, 1] = shifted_Z[sec_idx]
        cycle_shifts.append(cycle_shift)
        finite_fractions.append(finite_fraction)
        DP = np.asarray(cycle_shift.DP, dtype=np.float64)
        DP_end = DP[-1] if DP.ndim == 3 and DP.shape[1:] == (2, 2) and DP.shape[0] else np.full((2, 2), np.nan)
        dXpol = np.asarray(cycle_shift.delta_X_pol, dtype=np.float64)
        dXcyc = np.asarray(cycle_shift.delta_X_cyc, dtype=np.float64)
        residual = np.asarray(cycle_shift.periodic_residual, dtype=np.float64)
        cycle_rows.append({
            "cycle_index": int(idx),
            "seed_R": float(R0),
            "seed_Z": float(Z0),
            "delta_X_cyc0_R_m": float(cycle_shift.delta_X_cyc0[0]),
            "delta_X_cyc0_Z_m": float(cycle_shift.delta_X_cyc0[1]),
            "delta_X_cyc0_norm_m": float(np.linalg.norm(np.asarray(cycle_shift.delta_X_cyc0, dtype=np.float64))),
            "delta_X_pol_max_norm_m": _safe_norm_max(dXpol),
            "delta_X_cyc_max_norm_m": _safe_norm_max(dXcyc),
            "periodic_residual_R_m": float(residual[0]),
            "periodic_residual_Z_m": float(residual[1]),
            "periodic_residual_norm_m": float(np.linalg.norm(residual)),
            "alive_fraction": _alive_fraction(cycle_shift),
            "finite_fraction": float(finite_fraction),
            "I_minus_DPm_cond": _safe_matrix_stat(DP_end, "cond_i_minus"),
            "DPm_trace": _safe_matrix_stat(DP_end, "trace"),
            "DPm_det": _safe_matrix_stat(DP_end, "det"),
        })

    dX0_norms = [float(np.linalg.norm(np.asarray(r.delta_X_cyc0, dtype=np.float64))) for r in cycle_shifts]
    dXpol_norms = [_safe_norm_max(np.asarray(r.delta_X_pol, dtype=np.float64)) for r in cycle_shifts]
    dX_norms = [_safe_norm_max(np.asarray(r.delta_X_cyc, dtype=np.float64)) for r in cycle_shifts]
    residual_norms = [float(np.linalg.norm(np.asarray(r.periodic_residual, dtype=np.float64))) for r in cycle_shifts]
    cond_values = np.asarray([row["I_minus_DPm_cond"] for row in cycle_rows], dtype=np.float64)
    finite_cond = cond_values[np.isfinite(cond_values)]
    diagnostics = {
        "method": "cyna_evolve_delta_X_cycle_along_orbit",
        "n_cycles": int(seeds.shape[0]),
        "seed_phi": seed_phi,
        "field_period": float(period),
        "field_periods": span_periods,
        "steps_per_field_period": steps_per_period,
        "n_steps": int(n_steps),
        "fd_eps": float(fd_eps),
        "max_delta_X_cyc0_norm_m": float(np.nanmax(dX0_norms)) if dX0_norms else np.nan,
        "max_delta_X_pol_norm_m": float(np.nanmax(dXpol_norms)) if dXpol_norms else np.nan,
        "max_delta_X_cyc_norm_m": float(np.nanmax(dX_norms)) if dX_norms else np.nan,
        "max_periodic_residual_norm_m": float(np.nanmax(residual_norms)) if residual_norms else np.nan,
        "min_alive_fraction": float(np.nanmin([_alive_fraction(r) for r in cycle_shifts])) if cycle_shifts else np.nan,
        "min_finite_fraction": float(np.nanmin(finite_fractions)) if finite_fractions else np.nan,
        "max_I_minus_DPm_cond": float(np.max(finite_cond)) if finite_cond.size else float("inf"),
        "cycles": cycle_rows,
    }
    return CyclePointShift(
        sections=sections,
        cycle_shifts=tuple(cycle_shifts),
        diagnostics=diagnostics,
    )


def _shifted_orbit_on_queries(
    cycle_shift: CyclePerturbationShift,
    seed_phi: float,
    rel_queries: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    rel_phi = np.asarray(cycle_shift.phi, dtype=np.float64) - float(seed_phi)
    shifted_R = np.asarray(cycle_shift.R, dtype=np.float64) + np.asarray(cycle_shift.delta_X_cyc)[:, 0]
    shifted_Z = np.asarray(cycle_shift.Z, dtype=np.float64) + np.asarray(cycle_shift.delta_X_cyc)[:, 1]
    finite = np.isfinite(rel_phi) & np.isfinite(shifted_R) & np.isfinite(shifted_Z)
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError("cycle shift produced fewer than two finite samples")
    rel = rel_phi[finite]
    order = np.argsort(rel)
    rel = rel[order]
    r_vals = shifted_R[finite][order]
    z_vals = shifted_Z[finite][order]
    return (
        np.asarray([float(np.interp(q, rel, r_vals)) for q in rel_queries], dtype=np.float64),
        np.asarray([float(np.interp(q, rel, z_vals)) for q in rel_queries], dtype=np.float64),
        float(np.mean(finite)),
    )


def _delta_cycle_shift_on_queries(
    cycle_shift: CyclePerturbationShift,
    seed_phi: float,
    rel_queries: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    rel_phi = np.asarray(cycle_shift.phi, dtype=np.float64) - float(seed_phi)
    dX = np.asarray(cycle_shift.delta_X_cyc, dtype=np.float64)
    if dX.ndim != 2 or dX.shape[1] != 2:
        raise ValueError("delta_X_cyc must have shape (n, 2)")
    finite = np.isfinite(rel_phi) & np.isfinite(dX[:, 0]) & np.isfinite(dX[:, 1])
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError("cycle shift produced fewer than two finite delta samples")
    rel = rel_phi[finite]
    order = np.argsort(rel)
    rel = rel[order]
    dR_vals = dX[finite, 0][order]
    dZ_vals = dX[finite, 1][order]
    return (
        np.asarray([float(np.interp(q, rel, dR_vals)) for q in rel_queries], dtype=np.float64),
        np.asarray([float(np.interp(q, rel, dZ_vals)) for q in rel_queries], dtype=np.float64),
        float(np.mean(finite)),
    )


def _alive_fraction(cycle_shift: CyclePerturbationShift) -> float:
    alive = np.asarray(cycle_shift.alive, dtype=bool)
    return float(np.mean(alive)) if alive.size else 0.0


def _safe_norm_max(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    arr = np.reshape(arr, (-1, arr.shape[-1])) if arr.ndim > 1 else arr.reshape(-1, 1)
    finite = np.isfinite(arr).all(axis=1)
    if not np.any(finite):
        return float("nan")
    return float(np.max(np.linalg.norm(arr[finite], axis=1)))


def _safe_matrix_stat(matrix: Any, op: str) -> float:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (2, 2) or not np.isfinite(mat).all():
        return float("nan")
    if op == "cond_i_minus":
        try:
            return float(np.linalg.cond(np.eye(2) - mat))
        except np.linalg.LinAlgError:
            return float("inf")
    if op == "trace":
        return float(np.trace(mat))
    if op == "det":
        return float(np.linalg.det(mat))
    raise ValueError(f"Unknown matrix stat {op!r}")


__all__ = [
    "AxisCycleShift",
    "CyclePointShift",
    "axis_cycle_shift_from_fields",
    "cycle_points_shift_from_fields",
    "cycle_shift_from_fields",
    "field_period_cache_from_components",
]
