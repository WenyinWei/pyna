from __future__ import annotations

import math
from typing import Optional

import numpy as np

from pyna._cyna import (
    is_available as _cyna_available,
    summarize_profile_objectives as _cyna_summarize_profile_objectives,
    trace_surface_metrics_batch_twall as _cyna_trace_surface_metrics_batch_twall,
)
from pyna.topo.topology_eval import _FC, _find_wall_file, _load_wall, _make_virtual_wall_from_grid


def lcfs_effective_minor_radius(V_lcfs: float, R_axis: float) -> float:
    if not np.isfinite(V_lcfs) or not np.isfinite(R_axis) or V_lcfs <= 0.0 or R_axis <= 0.0:
        return float("nan")
    return float(np.sqrt(V_lcfs / (2.0 * np.pi ** 2 * R_axis)))


def make_reff_profile_grid(
    V_lcfs: float,
    R_axis: float,
    n_surfaces: int = 8,
    min_fraction: float = 0.15,
    max_fraction: float = 0.95,
) -> tuple[float, np.ndarray, np.ndarray]:
    a_eff = lcfs_effective_minor_radius(V_lcfs, R_axis)
    if not np.isfinite(a_eff):
        return a_eff, np.array([]), np.array([])
    n_surfaces = max(int(n_surfaces), 2)
    fracs = np.linspace(min_fraction, max_fraction, n_surfaces)
    fracs = np.clip(fracs, 1e-3, 1.0)
    return a_eff, a_eff * fracs, fracs


def _resolve_wall_arrays(fc: _FC, wall_file: Optional[str]):
    if wall_file:
        phi_c, wall_R, wall_Z, _ = _load_wall(wall_file)
        return phi_c, wall_R, wall_Z
    wall_file = _find_wall_file()
    if wall_file:
        phi_c, wall_R, wall_Z, _ = _load_wall(wall_file)
        return phi_c, wall_R, wall_Z
    phi_c, wall_R, wall_Z = _make_virtual_wall_from_grid(fc)
    if wall_R.ndim == 1:
        wall_R = wall_R[np.newaxis, :]
        wall_Z = wall_Z[np.newaxis, :]
    return phi_c, wall_R, wall_Z


def compute_profile_objectives_fast(
    field_cache: dict,
    V_lcfs: float,
    R_axis: float,
    Z_axis: float,
    wall_file: Optional[str] = None,
    n_surfaces: int = 8,
    n_turns: int = 20,
    DPhi: float = 0.05,
    min_fraction: float = 0.15,
    max_fraction: float = 0.95,
    fd_eps_R: float = 1e-4,
    fd_eps_Z: float = 1e-4,
    fd_eps_phi: float = 1e-4,
) -> dict:
    result = {
        "a_eff": float("nan"),
        "r_eff": np.array([]),
        "r_eff_fraction": np.array([]),
        "iota": np.array([]),
        "B_mean": np.array([]),
        "B2_mean": np.array([]),
        "B_min": np.array([]),
        "B_max": np.array([]),
        "JxB_mean": np.array([]),
        "turns": np.array([]),
        "alive": np.array([], dtype=int),
        "mean_iota_prime_reff": float("nan"),
        "mean_abs_iota_prime_reff": float("nan"),
        "epsilon_eff": float("nan"),
        "B_volume_avg": float("nan"),
        "beta_max_fast": float("nan"),
        "force_balance_proxy": float("nan"),
        "magnetic_pressure_avg": float("nan"),
        "n_valid_surfaces": 0,
        "D_Merc_proxy": float("nan"),
    }
    a_eff, r_eff, fracs = make_reff_profile_grid(
        V_lcfs, R_axis,
        n_surfaces=n_surfaces,
        min_fraction=min_fraction,
        max_fraction=max_fraction,
    )
    result["a_eff"] = a_eff
    result["r_eff"] = r_eff
    result["r_eff_fraction"] = fracs
    if not np.isfinite(a_eff) or a_eff <= 0.0 or len(r_eff) == 0:
        return result
    if not _cyna_available() or _cyna_trace_surface_metrics_batch_twall is None or _cyna_summarize_profile_objectives is None:
        return result

    fc = _FC(field_cache)
    phi_c, wall_R, wall_Z = _resolve_wall_arrays(fc, wall_file)
    R_seeds = np.ascontiguousarray(R_axis + r_eff, dtype=np.float64)
    Z_seeds = np.ascontiguousarray(np.full_like(r_eff, Z_axis), dtype=np.float64)

    iota, B_mean, B2_mean, B_min, B_max, JxB_mean, turns, alive = _cyna_trace_surface_metrics_batch_twall(
        R_seeds, Z_seeds,
        float(R_axis), float(Z_axis),
        0.0, int(n_turns), float(DPhi),
        fc.BR, fc.BPhi, fc.BZ,
        fc.Rg, fc.Zg, fc.Pg_ext,
        np.ascontiguousarray(phi_c, dtype=np.float64),
        np.ascontiguousarray(wall_R, dtype=np.float64),
        np.ascontiguousarray(wall_Z, dtype=np.float64),
        float(fd_eps_R), float(fd_eps_Z), float(fd_eps_phi),
    )
    result.update({
        "iota": np.asarray(iota, dtype=np.float64),
        "B_mean": np.asarray(B_mean, dtype=np.float64),
        "B2_mean": np.asarray(B2_mean, dtype=np.float64),
        "B_min": np.asarray(B_min, dtype=np.float64),
        "B_max": np.asarray(B_max, dtype=np.float64),
        "JxB_mean": np.asarray(JxB_mean, dtype=np.float64),
        "turns": np.asarray(turns, dtype=np.float64),
        "alive": np.asarray(alive, dtype=int),
    })
    mean_iota_prime, mean_abs_iota_prime, eps_eff, B_volume_avg, beta_max_fast, force_balance_proxy, magnetic_pressure_avg, n_valid, D_Merc_proxy = _cyna_summarize_profile_objectives(
        np.ascontiguousarray(r_eff, dtype=np.float64),
        np.ascontiguousarray(result["iota"], dtype=np.float64),
        np.ascontiguousarray(result["B_mean"], dtype=np.float64),
        np.ascontiguousarray(result["B2_mean"], dtype=np.float64),
        np.ascontiguousarray(result["B_min"], dtype=np.float64),
        np.ascontiguousarray(result["B_max"], dtype=np.float64),
        np.ascontiguousarray(result["JxB_mean"], dtype=np.float64),
        float(a_eff),
    )
    result.update({
        "mean_iota_prime_reff": float(mean_iota_prime),
        "mean_abs_iota_prime_reff": float(mean_abs_iota_prime),
        "epsilon_eff": float(eps_eff),
        "B_volume_avg": float(B_volume_avg),
        "beta_max_fast": float(beta_max_fast),
        "force_balance_proxy": float(force_balance_proxy),
        "magnetic_pressure_avg": float(magnetic_pressure_avg),
        "n_valid_surfaces": int(n_valid),
        "D_Merc_proxy": float(D_Merc_proxy),
    })
    return result


def compute_beta_max_fast(
    field_cache: dict,
    V_lcfs: float,
    R_axis: float,
    Z_axis: float,
    wall_file: Optional[str] = None,
    **kwargs,
) -> dict:
    metrics = compute_profile_objectives_fast(
        field_cache,
        V_lcfs=V_lcfs,
        R_axis=R_axis,
        Z_axis=Z_axis,
        wall_file=wall_file,
        **kwargs,
    )
    return {
        "beta_max_fast": metrics["beta_max_fast"],
        "force_balance_proxy": metrics["force_balance_proxy"],
        "magnetic_pressure_avg": metrics["magnetic_pressure_avg"],
        "a_eff": metrics["a_eff"],
        "n_valid_surfaces": metrics["n_valid_surfaces"],
    }
