#!/usr/bin/env python3
"""Plot finite-beta W7-X VMEC current streamlines on PEST surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyna.fields import VectorFieldCylind
from pyna.plot.j_streamlines import (
    GriddedPestVectorField,
    plot_j_streamline_seed_sections,
    plot_j_streamlines_on_pest_surface_plotly,
    trace_j_streamlines_on_pest,
)
from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates, smooth_pest_derivatives


DEFAULT_WOUT = Path("~/MCFdata/W7X_public/stagextender_beta1/wout_std_scp00_beta1.nc")
DEFAULT_OUT = Path("~/MCFdata/W7X_public/stagextender_beta1/j_streamlines_finite_beta")


def parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one float is required")
    return values


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _wout_metadata(path: Path) -> dict[str, object]:
    from netCDF4 import Dataset

    keys = ("nfp", "betatotal", "betapol", "betator", "betaxis", "ns", "mpol", "ntor")
    out: dict[str, object] = {"basename": path.name}
    with Dataset(str(path), "r") as ds:
        for key in keys:
            if key in ds.variables:
                value = np.asarray(ds.variables[key][...])
                out[key] = value.item() if value.shape == () else value.tolist()
    return out


def _load_desc_equilibrium(path: Path, *, device: str):
    import desc
    from desc.vmec import VMECIO

    desc.set_device(device)
    return VMECIO.load(str(path))


def _desc_grid(nodes: np.ndarray):
    from desc.grid import Grid

    return Grid(np.asarray(nodes, dtype=np.float64), sort=False)


def _map_pest_nodes(eq, nodes: np.ndarray, *, tol: float, maxiter: int) -> np.ndarray:
    return np.asarray(
        eq.map_coordinates(
            np.asarray(nodes, dtype=np.float64),
            ("rho", "theta_PEST", "zeta"),
            tol=float(tol),
            maxiter=int(maxiter),
        ),
        dtype=np.float64,
    )


def _pest_tangent_coefficients(
    pest: SmoothPestCoordinates,
    *,
    JR: np.ndarray,
    JPhi: np.ndarray,
    JZ: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    deriv = smooth_pest_derivatives(pest)
    R = np.asarray(pest.R_surf, dtype=np.float64)
    phi = np.asarray(pest.phi_vals, dtype=np.float64)[:, None, None]
    cp = np.cos(phi)
    sp = np.sin(phi)
    dR_dtheta = np.asarray(deriv[2], dtype=np.float64)
    dZ_dtheta = np.asarray(deriv[3], dtype=np.float64)
    dR_dphi = np.asarray(deriv[4], dtype=np.float64)
    dZ_dphi = np.asarray(deriv[5], dtype=np.float64)
    e_theta = np.stack([dR_dtheta * cp, dR_dtheta * sp, dZ_dtheta], axis=-1)
    e_phi = np.stack([dR_dphi * cp - R * sp, dR_dphi * sp + R * cp, dZ_dphi], axis=-1)
    j_cart = np.stack(
        [
            np.asarray(JR, dtype=np.float64) * cp - np.asarray(JPhi, dtype=np.float64) * sp,
            np.asarray(JR, dtype=np.float64) * sp + np.asarray(JPhi, dtype=np.float64) * cp,
            np.asarray(JZ, dtype=np.float64),
        ],
        axis=-1,
    )
    gtt = np.sum(e_theta * e_theta, axis=-1)
    gtp = np.sum(e_theta * e_phi, axis=-1)
    gpp = np.sum(e_phi * e_phi, axis=-1)
    rhs_t = np.sum(j_cart * e_theta, axis=-1)
    rhs_p = np.sum(j_cart * e_phi, axis=-1)
    det = gtt * gpp - gtp * gtp
    Jtheta = np.full(R.shape, np.nan, dtype=np.float64)
    Jphi = np.full(R.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(det) & (np.abs(det) > 1.0e-28)
    Jtheta[valid] = (rhs_t[valid] * gpp[valid] - rhs_p[valid] * gtp[valid]) / det[valid]
    Jphi[valid] = (gtt[valid] * rhs_p[valid] - gtp[valid] * rhs_t[valid]) / det[valid]
    return Jtheta, Jphi


def build_desc_pest_j_payload(
    eq,
    *,
    rho_vals: np.ndarray,
    n_phi: int,
    n_theta: int,
    current_mode: str,
    tol: float,
    maxiter: int,
) -> tuple[SmoothPestCoordinates, GriddedPestVectorField, dict[str, object]]:
    """Sample finite-beta DESC/VMEC geometry and current on a full-torus PEST mesh."""

    nfp = int(eq.NFP)
    field_period = 2.0 * np.pi / max(nfp, 1)
    phi_vals = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False, dtype=np.float64)
    theta_vals = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float64)
    rho_vals = np.asarray(rho_vals, dtype=np.float64)

    phi_grid, rho_grid, theta_grid = np.meshgrid(phi_vals, rho_vals, theta_vals, indexing="ij")
    nodes_pest = np.column_stack(
        [
            rho_grid.ravel(),
            np.mod(theta_grid.ravel(), 2.0 * np.pi),
            np.mod(phi_grid.ravel(), field_period),
        ]
    )
    nodes_desc = _map_pest_nodes(eq, nodes_pest, tol=tol, maxiter=maxiter)
    data = eq.compute(
        [
            "R",
            "Z",
            "J_R",
            "J_phi",
            "J_Z",
            "J^rho",
            "J^theta",
            "J^zeta",
            "e_rho",
            "e_theta",
            "e_zeta",
            "p",
        ],
        grid=_desc_grid(nodes_desc),
    )
    shape = (phi_vals.size, rho_vals.size, theta_vals.size)
    R_surf = np.asarray(data["R"], dtype=np.float64).reshape(shape)
    Z_surf = np.asarray(data["Z"], dtype=np.float64).reshape(shape)
    raw_JR = np.asarray(data["J_R"], dtype=np.float64)
    raw_JPhi = np.asarray(data["J_phi"], dtype=np.float64)
    raw_JZ = np.asarray(data["J_Z"], dtype=np.float64)
    raw_J = np.stack([raw_JR, raw_JPhi, raw_JZ], axis=1)
    tangent_J = (
        np.asarray(data["J^theta"], dtype=np.float64)[:, None] * np.asarray(data["e_theta"], dtype=np.float64)
        + np.asarray(data["J^zeta"], dtype=np.float64)[:, None] * np.asarray(data["e_zeta"], dtype=np.float64)
    )
    if current_mode == "full":
        sampled_J = raw_J
    elif current_mode == "tangent":
        sampled_J = tangent_J
    else:
        raise ValueError("current_mode must be 'tangent' or 'full'")
    JR = sampled_J[:, 0].reshape(shape)
    JPhi = sampled_J[:, 1].reshape(shape)
    JZ = sampled_J[:, 2].reshape(shape)
    pressure = np.asarray(data["p"], dtype=np.float64).reshape(shape)

    axis_nodes = np.column_stack(
        [
            np.zeros_like(phi_vals),
            np.zeros_like(phi_vals),
            np.mod(phi_vals, field_period),
        ]
    )
    axis_desc = _map_pest_nodes(eq, axis_nodes, tol=tol, maxiter=maxiter)
    axis = eq.compute(["R", "Z"], grid=_desc_grid(axis_desc))

    pest = SmoothPestCoordinates(
        R_surf=np.ascontiguousarray(R_surf),
        Z_surf=np.ascontiguousarray(Z_surf),
        rho_vals=np.ascontiguousarray(rho_vals),
        theta_vals=np.ascontiguousarray(theta_vals),
        phi_vals=np.ascontiguousarray(phi_vals),
        axis_R=np.ascontiguousarray(axis["R"], dtype=np.float64),
        axis_Z=np.ascontiguousarray(axis["Z"], dtype=np.float64),
        source="DESC VMECIO finite-beta theta_PEST mesh",
    )
    Jtheta = None
    Jphi = None
    if current_mode == "tangent":
        Jtheta, Jphi = _pest_tangent_coefficients(pest, JR=JR, JPhi=JPhi, JZ=JZ)
    field = GriddedPestVectorField.from_pest_coordinates(
        pest,
        JR=np.ascontiguousarray(JR),
        JZ=np.ascontiguousarray(JZ),
        JPhi=np.ascontiguousarray(JPhi),
        Jtheta=None if Jtheta is None else np.ascontiguousarray(Jtheta),
        Jphi=None if Jphi is None else np.ascontiguousarray(Jphi),
        nfp=nfp,
        source=f"DESC finite-beta VMEC {current_mode} J_R/J_phi/J_Z",
    )
    j_abs = np.sqrt(JR * JR + JZ * JZ + JPhi * JPhi)
    raw_j_abs = np.linalg.norm(raw_J, axis=1)
    radial_j_abs = np.abs(np.asarray(data["J^rho"], dtype=np.float64)) * np.linalg.norm(
        np.asarray(data["e_rho"], dtype=np.float64),
        axis=1,
    )
    diagnostics = {
        "current_mode": current_mode,
        "nfp": nfp,
        "field_period_rad": field_period,
        "rho_values": [float(x) for x in rho_vals],
        "n_phi": int(n_phi),
        "n_theta": int(n_theta),
        "J_abs_min": float(np.nanmin(j_abs)),
        "J_abs_median": float(np.nanmedian(j_abs)),
        "J_abs_p95": float(np.nanpercentile(j_abs, 95.0)),
        "J_abs_max": float(np.nanmax(j_abs)),
        "raw_J_abs_median": float(np.nanmedian(raw_j_abs)),
        "raw_J_abs_p95": float(np.nanpercentile(raw_j_abs, 95.0)),
        "raw_radial_J_abs_median": float(np.nanmedian(radial_j_abs)),
        "raw_radial_J_abs_p95": float(np.nanpercentile(radial_j_abs, 95.0)),
        "raw_radial_over_raw_J_median": float(np.nanmedian(radial_j_abs / np.maximum(raw_j_abs, 1.0e-30))),
        "raw_radial_over_raw_J_p95": float(np.nanpercentile(radial_j_abs / np.maximum(raw_j_abs, 1.0e-30), 95.0)),
        "pressure_min": float(np.nanmin(pressure)),
        "pressure_max": float(np.nanmax(pressure)),
    }
    return pest, field, diagnostics


def _map_real_space_nodes(
    eq,
    nodes: np.ndarray,
    *,
    nfp: int,
    guess: np.ndarray | None = None,
    tol: float,
    maxiter: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    field_period = 2.0 * np.pi / max(int(nfp), 1)
    mapped_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    iteration_parts: list[np.ndarray] = []
    coords = np.asarray(nodes, dtype=np.float64)
    guess_arr = None if guess is None else np.asarray(guess, dtype=np.float64)
    if guess_arr is not None and guess_arr.shape != coords.shape:
        raise ValueError("real-space coordinate guesses must match node shape")
    batch = max(int(batch_size), 1)
    for lo in range(0, coords.shape[0], batch):
        chunk = coords[lo : lo + batch]
        result = eq.map_coordinates(
            chunk,
            ("R", "phi", "Z"),
            outbasis=("rho", "theta", "zeta"),
            guess=None if guess_arr is None else guess_arr[lo : lo + batch],
            period=(np.inf, field_period, np.inf),
            tol=float(tol),
            maxiter=int(maxiter),
            full_output=True,
        )
        if isinstance(result, tuple) and len(result) == 2:
            mapped, info = result
            if isinstance(info, (tuple, list)) and len(info) >= 2:
                residual_parts.append(np.asarray(info[0], dtype=np.float64))
                iteration_parts.append(np.asarray(info[1], dtype=np.float64))
        else:
            mapped = result
        mapped_parts.append(np.asarray(mapped, dtype=np.float64))
    mapped_all = np.concatenate(mapped_parts, axis=0) if mapped_parts else np.empty((0, 3), dtype=np.float64)
    residual_norm = np.empty(0, dtype=np.float64)
    if residual_parts:
        residual = np.concatenate([np.reshape(part, (part.shape[0], -1)) for part in residual_parts], axis=0)
        residual_norm = np.linalg.norm(residual, axis=1)
    if residual_norm.size != coords.shape[0]:
        residual_norm = np.full((coords.shape[0],), np.nan, dtype=np.float64)
    iterations = np.concatenate([np.ravel(part) for part in iteration_parts]) if iteration_parts else np.empty(0, dtype=np.float64)
    finite_res = residual_norm[np.isfinite(residual_norm)]
    finite_it = iterations[np.isfinite(iterations)]
    diagnostics = {
        "map_residual_norm_median": float(np.nanmedian(finite_res)) if finite_res.size else float("nan"),
        "map_residual_norm_p95": float(np.nanpercentile(finite_res, 95.0)) if finite_res.size else float("nan"),
        "map_iterations_median": float(np.nanmedian(finite_it)) if finite_it.size else float("nan"),
        "map_iterations_p95": float(np.nanpercentile(finite_it, 95.0)) if finite_it.size else float("nan"),
    }
    return mapped_all, residual_norm, diagnostics


def _pest_ring_at_phi(values: np.ndarray, phi: float, *, phi_period: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n_phi = arr.shape[0]
    if n_phi < 2:
        return arr[0].copy()
    u = np.mod(float(phi), float(phi_period)) * (float(n_phi) / float(phi_period))
    f = np.floor(u)
    i0 = int(f) % n_phi
    i1 = (i0 + 1) % n_phi
    a = float(u - f)
    return (1.0 - a) * arr[i0] + a * arr[i1]


def _nearest_pest_coordinate_guess(
    pest: SmoothPestCoordinates,
    real_nodes: np.ndarray,
    *,
    nfp: int,
) -> np.ndarray:
    field_period = 2.0 * np.pi / max(int(nfp), 1)
    full_period = float(getattr(pest, "period", 2.0 * np.pi) or (2.0 * np.pi))
    nodes = np.asarray(real_nodes, dtype=np.float64)
    guess = np.full(nodes.shape, np.nan, dtype=np.float64)
    rho_vals = np.asarray(pest.rho_vals, dtype=np.float64)
    theta_vals = np.asarray(pest.theta_vals, dtype=np.float64)
    n_theta = theta_vals.size
    if rho_vals.size == 0 or theta_vals.size == 0:
        return guess
    phi_vals = np.unique(nodes[:, 1])
    for phi in phi_vals:
        selected = np.isclose(nodes[:, 1], float(phi), rtol=0.0, atol=1.0e-14)
        if not np.any(selected):
            continue
        R_ring = _pest_ring_at_phi(pest.R_surf, float(phi), phi_period=full_period).reshape(-1)
        Z_ring = _pest_ring_at_phi(pest.Z_surf, float(phi), phi_period=full_period).reshape(-1)
        valid_ring = np.isfinite(R_ring) & np.isfinite(Z_ring)
        if not np.any(valid_ring):
            continue
        R_flat = R_ring[valid_ring]
        Z_flat = Z_ring[valid_ring]
        flat_indices = np.flatnonzero(valid_ring)
        points = nodes[selected]
        dist2 = (points[:, 0, None] - R_flat[None, :]) ** 2 + (points[:, 2, None] - Z_flat[None, :]) ** 2
        nearest_flat = flat_indices[np.argmin(dist2, axis=1)]
        guess[selected, 0] = rho_vals[nearest_flat // n_theta]
        guess[selected, 1] = theta_vals[nearest_flat % n_theta]
        guess[selected, 2] = np.mod(float(phi), field_period)
    return guess


def _compute_desc_current_on_nodes(
    eq,
    nodes_desc: np.ndarray,
    *,
    current_mode: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    keys = ["J_R", "J_phi", "J_Z"]
    if current_mode == "tangent":
        keys += ["J^theta", "J^zeta", "e_theta", "e_zeta"]
    elif current_mode != "full":
        raise ValueError("current_mode must be 'tangent' or 'full'")

    nodes = np.asarray(nodes_desc, dtype=np.float64)
    out = np.full((nodes.shape[0], 3), np.nan, dtype=np.float64)
    raw_abs_parts: list[np.ndarray] = []
    sample_abs_parts: list[np.ndarray] = []
    batch = max(int(batch_size), 1)
    for lo in range(0, nodes.shape[0], batch):
        chunk = nodes[lo : lo + batch]
        data = eq.compute(keys, grid=_desc_grid(chunk))
        raw_J = np.stack(
            [
                np.asarray(data["J_R"], dtype=np.float64),
                np.asarray(data["J_phi"], dtype=np.float64),
                np.asarray(data["J_Z"], dtype=np.float64),
            ],
            axis=1,
        )
        if current_mode == "full":
            sampled_J = raw_J
        else:
            sampled_J = (
                np.asarray(data["J^theta"], dtype=np.float64)[:, None] * np.asarray(data["e_theta"], dtype=np.float64)
                + np.asarray(data["J^zeta"], dtype=np.float64)[:, None] * np.asarray(data["e_zeta"], dtype=np.float64)
            )
        out[lo : lo + chunk.shape[0]] = sampled_J
        raw_abs_parts.append(np.linalg.norm(raw_J, axis=1))
        sample_abs_parts.append(np.linalg.norm(sampled_J, axis=1))
    raw_abs = np.concatenate(raw_abs_parts) if raw_abs_parts else np.empty(0, dtype=np.float64)
    sample_abs = np.concatenate(sample_abs_parts) if sample_abs_parts else np.empty(0, dtype=np.float64)
    finite_sample = sample_abs[np.isfinite(sample_abs)]
    finite_raw = raw_abs[np.isfinite(raw_abs)]
    diagnostics = {
        "raw_J_abs_median": float(np.nanmedian(finite_raw)) if finite_raw.size else float("nan"),
        "raw_J_abs_p95": float(np.nanpercentile(finite_raw, 95.0)) if finite_raw.size else float("nan"),
        "sampled_J_abs_median": float(np.nanmedian(finite_sample)) if finite_sample.size else float("nan"),
        "sampled_J_abs_p95": float(np.nanpercentile(finite_sample, 95.0)) if finite_sample.size else float("nan"),
    }
    return out, diagnostics


def build_desc_cartesian_j_field(
    eq,
    pest: SmoothPestCoordinates,
    *,
    current_mode: str,
    n_R: int,
    n_Z: int,
    n_phi: int,
    margin: float,
    rho_limit: float,
    map_residual_limit: float,
    tol: float,
    maxiter: int,
    batch_size: int,
) -> tuple[VectorFieldCylind, dict[str, object]]:
    """Sample finite-beta current on a regular cylindrical volume for Cartesian tracing."""

    nfp = int(eq.NFP)
    field_period = 2.0 * np.pi / max(nfp, 1)
    R_surf = np.asarray(pest.R_surf, dtype=np.float64)
    Z_surf = np.asarray(pest.Z_surf, dtype=np.float64)
    pad = max(float(margin), 0.0)
    R_grid = np.linspace(float(np.nanmin(R_surf) - pad), float(np.nanmax(R_surf) + pad), max(int(n_R), 4))
    Z_grid = np.linspace(float(np.nanmin(Z_surf) - pad), float(np.nanmax(Z_surf) + pad), max(int(n_Z), 4))
    Phi_grid = np.linspace(0.0, field_period, max(int(n_phi), 4), endpoint=False, dtype=np.float64)
    RR, ZZ, PP = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")
    real_nodes = np.column_stack([RR.ravel(), PP.ravel(), ZZ.ravel()])
    initial_guess = _nearest_pest_coordinate_guess(pest, real_nodes, nfp=nfp)
    desc_nodes, map_residual_norm, map_diag = _map_real_space_nodes(
        eq,
        real_nodes,
        nfp=nfp,
        guess=initial_guess,
        tol=tol,
        maxiter=maxiter,
        batch_size=batch_size,
    )
    rho = desc_nodes[:, 0] if desc_nodes.size else np.empty(0, dtype=np.float64)
    residual_limit = float(map_residual_limit)
    residual_ok = np.isfinite(map_residual_norm) & (map_residual_norm <= residual_limit)
    if not np.isfinite(residual_limit) or residual_limit <= 0.0:
        residual_ok = np.ones_like(rho, dtype=bool)
    valid = np.isfinite(desc_nodes).all(axis=1) & residual_ok & (rho >= 0.0) & (rho <= float(rho_limit))
    flat_J = np.full((real_nodes.shape[0], 3), np.nan, dtype=np.float64)
    current_diag: dict[str, object] = {
        "raw_J_abs_median": float("nan"),
        "raw_J_abs_p95": float("nan"),
        "sampled_J_abs_median": float("nan"),
        "sampled_J_abs_p95": float("nan"),
    }
    if np.any(valid):
        sampled_J, current_diag = _compute_desc_current_on_nodes(
            eq,
            desc_nodes[valid],
            current_mode=current_mode,
            batch_size=batch_size,
        )
        flat_J[valid] = sampled_J
    shape = (R_grid.size, Z_grid.size, Phi_grid.size)
    JR = np.ascontiguousarray(flat_J[:, 0].reshape(shape))
    JPhi = np.ascontiguousarray(flat_J[:, 1].reshape(shape))
    JZ = np.ascontiguousarray(flat_J[:, 2].reshape(shape))
    field = VectorFieldCylind(
        R=R_grid,
        Z=Z_grid,
        Phi=Phi_grid,
        BR=JR,
        BZ=JZ,
        BPhi=JPhi,
        nfp=nfp,
        name=f"W7-X public finite-beta {current_mode} J_total",
        units="A/m^2",
    )
    finite_field = np.isfinite(JR) & np.isfinite(JPhi) & np.isfinite(JZ)
    finite_rho = rho[np.isfinite(rho)]
    j_abs = np.sqrt(JR * JR + JPhi * JPhi + JZ * JZ)
    finite_j = j_abs[np.isfinite(j_abs)]
    diagnostics = {
        "current_mode": current_mode,
        "trace_field_type": "VectorFieldCylind",
        "trace_space": "cartesian",
        "nfp": nfp,
        "field_period_rad": field_period,
        "n_R": int(R_grid.size),
        "n_Z": int(Z_grid.size),
        "n_phi": int(Phi_grid.size),
        "R_min": float(R_grid[0]),
        "R_max": float(R_grid[-1]),
        "Z_min": float(Z_grid[0]),
        "Z_max": float(Z_grid[-1]),
        "margin": float(pad),
        "rho_limit": float(rho_limit),
        "map_residual_limit": float(map_residual_limit),
        "mapped_rho_min": float(np.nanmin(finite_rho)) if finite_rho.size else float("nan"),
        "mapped_rho_max": float(np.nanmax(finite_rho)) if finite_rho.size else float("nan"),
        "residual_gate_fraction": float(np.count_nonzero(residual_ok) / max(residual_ok.size, 1)),
        "valid_desc_coordinate_fraction": float(np.count_nonzero(valid) / max(valid.size, 1)),
        "finite_field_fraction": float(np.count_nonzero(finite_field) / max(finite_field.size, 1)),
        "J_abs_median": float(np.nanmedian(finite_j)) if finite_j.size else float("nan"),
        "J_abs_p95": float(np.nanpercentile(finite_j, 95.0)) if finite_j.size else float("nan"),
        **map_diag,
        **current_diag,
    }
    return field, diagnostics


def write_npz_payload(path: Path, pest: SmoothPestCoordinates, field: GriddedPestVectorField) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        R_surf=pest.R_surf,
        Z_surf=pest.Z_surf,
        rho_vals=pest.rho_vals,
        theta_vals=pest.theta_vals,
        phi_vals=pest.phi_vals,
        axis_R=pest.axis_R,
        axis_Z=pest.axis_Z,
        JR=field.JR,
        JZ=field.JZ,
        JPhi=field.JPhi,
        Jtheta=field.Jtheta if field.Jtheta is not None else np.empty(0, dtype=np.float64),
        Jphi=field.Jphi if field.Jphi is not None else np.empty(0, dtype=np.float64),
        nfp=np.asarray(field.nfp, dtype=np.int64),
    )


def write_vector_field_npz(path: Path, field: VectorFieldCylind) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field.to_npz(str(path))


def write_streamlines_npz(path: Path, streamlines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        R=streamlines.R,
        Z=streamlines.Z,
        phi=streamlines.phi,
        theta=streamlines.theta,
        x=streamlines.x,
        y=streamlines.y,
        z=streamlines.z,
        seed_R=streamlines.seed_R,
        seed_Z=streamlines.seed_Z,
        seed_phi=streamlines.seed_phi,
        seed_rho=streamlines.seed_rho,
        seed_theta=streamlines.seed_theta,
        seed_surface_index=streamlines.seed_surface_index,
        seed_phi_index=streamlines.seed_phi_index,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=DEFAULT_WOUT, help="Public finite-beta VMEC wout file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory outside the repo.")
    parser.add_argument("--desc-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--rho-values", type=parse_floats, default=(0.35, 0.55, 0.75, 0.9))
    parser.add_argument("--stream-rho", type=float, default=0.75)
    parser.add_argument("--n-phi", type=int, default=80)
    parser.add_argument("--n-theta", type=int, default=160)
    parser.add_argument("--current-mode", choices=("tangent", "full"), default="tangent", help="Use the local surface-tangent finite-beta current by default; 'full' keeps DESC raw J.")
    parser.add_argument("--trace-space", choices=("cartesian", "pest"), default="cartesian", help="Trace in physical Cartesian space by default; 'pest' keeps the surface-constrained PEST diagnostic.")
    parser.add_argument("--cartesian-field-source", choices=("pest-surface", "desc-grid"), default="pest-surface", help="For Cartesian tracing, use PEST-surface projected J by default; 'desc-grid' builds a regular VectorFieldCylind body grid through DESC inverse mapping.")
    parser.add_argument("--phi-indices", type=parse_ints, default=None, help="Seed section indices, comma-separated.")
    parser.add_argument("--seed-count", type=int, default=3, help="Seeds per selected toroidal section.")
    parser.add_argument("--n-turns", type=float, default=1.35, help="Trace half-length in seed-surface perimeter units used by pyna.plot.")
    parser.add_argument("--steps-per-turn", type=int, default=1200)
    parser.add_argument("--surface-downsample", type=int, default=2)
    parser.add_argument("--cartesian-n-r", dest="cartesian_n_r", type=int, default=64, help="R grid points for Cartesian-space J tracing.")
    parser.add_argument("--cartesian-n-z", dest="cartesian_n_z", type=int, default=64, help="Z grid points for Cartesian-space J tracing.")
    parser.add_argument("--cartesian-n-phi", type=int, default=40, help="Toroidal grid points over one field period for Cartesian-space J tracing.")
    parser.add_argument("--cartesian-margin", type=float, default=0.04, help="R/Z padding around the PEST surface envelope for Cartesian J sampling.")
    parser.add_argument("--cartesian-rho-limit", type=float, default=1.02, help="Maximum DESC rho retained in the Cartesian J grid.")
    parser.add_argument("--cartesian-map-residual-limit", type=float, default=1.0e-4, help="Reject Cartesian J grid points whose DESC inverse residual exceeds this value; use <=0 to disable.")
    parser.add_argument("--cartesian-batch-size", type=int, default=4096)
    parser.add_argument("--projection-distance", type=float, default=None, help="Maximum R/Z distance from the seed PEST surface when projecting Cartesian lines back to seed sections.")
    parser.add_argument("--max-projection-step", type=float, default=None, help="Break projected 2-D lines when adjacent projected points jump farther than this distance.")
    parser.add_argument("--tol", type=float, default=1.0e-9, help="DESC theta_PEST coordinate-map tolerance.")
    parser.add_argument("--maxiter", type=int, default=40, help="DESC theta_PEST coordinate-map Newton iterations.")
    args = parser.parse_args(argv)

    wout = args.wout.expanduser()
    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    meta = _wout_metadata(wout)
    eq = _load_desc_equilibrium(wout, device=args.desc_device)
    rho_vals = np.asarray(args.rho_values, dtype=np.float64)
    if np.any(rho_vals <= 0.0) or np.any(rho_vals >= 1.0):
        raise ValueError("--rho-values must lie strictly inside (0, 1)")

    pest, pest_field, diagnostics = build_desc_pest_j_payload(
        eq,
        rho_vals=rho_vals,
        n_phi=args.n_phi,
        n_theta=args.n_theta,
        current_mode=args.current_mode,
        tol=args.tol,
        maxiter=args.maxiter,
    )
    stream_surface = int(np.argmin(np.abs(pest.rho_vals - float(args.stream_rho))))
    cartesian_field = None
    cartesian_diagnostics: dict[str, object] | None = {
        "trace_field_source": "pest_surface_projected",
        "trace_space": "cartesian",
        "note": "Cartesian RK4 uses PEST projection only to evaluate J on the seed surface; it does not step in PEST coordinates.",
    } if args.trace_space == "cartesian" else None
    if args.trace_space == "cartesian" and args.cartesian_field_source == "desc-grid":
        cartesian_field, cartesian_diagnostics = build_desc_cartesian_j_field(
            eq,
            pest,
            current_mode=args.current_mode,
            n_R=args.cartesian_n_r,
            n_Z=args.cartesian_n_z,
            n_phi=args.cartesian_n_phi,
            margin=args.cartesian_margin,
            rho_limit=args.cartesian_rho_limit,
            map_residual_limit=args.cartesian_map_residual_limit,
            tol=args.tol,
            maxiter=args.maxiter,
            batch_size=args.cartesian_batch_size,
        )
    trace_field = cartesian_field if cartesian_field is not None else pest_field
    streamlines = trace_j_streamlines_on_pest(
        trace_field,
        pest,
        surface_index=stream_surface,
        phi_indices=args.phi_indices,
        seed_count=args.seed_count,
        n_turns=args.n_turns,
        steps_per_turn=args.steps_per_turn,
        constrain_to_surface=args.trace_space == "pest",
        max_surface_distance=args.projection_distance,
    )

    coords_path = out / "w7x_public_finite_beta_pest_j_payload.npz"
    cartesian_field_path = out / "w7x_public_finite_beta_cartesian_j_field.npz"
    lines_path = out / "w7x_public_finite_beta_j_streamlines.npz"
    write_npz_payload(coords_path, pest, pest_field)
    if cartesian_field is not None:
        write_vector_field_npz(cartesian_field_path, cartesian_field)
    write_streamlines_npz(lines_path, streamlines)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, _axes = plot_j_streamline_seed_sections(
        streamlines,
        pest,
        title="W7-X public finite-beta VMEC J streamlines on PEST sections",
        line_width=1.0,
        alpha=0.88,
        max_projection_step=args.max_projection_step,
        max_surface_distance=args.projection_distance,
    )
    png_path = out / "w7x_public_finite_beta_j_streamline_seed_sections.png"
    fig.savefig(png_path, dpi=210, bbox_inches="tight")
    plt.close(fig)

    html_path = out / "w7x_public_finite_beta_j_streamlines_3d.html"
    plot_j_streamlines_on_pest_surface_plotly(
        streamlines,
        pest,
        surface_index=stream_surface,
        html_path=html_path,
        surface_downsample=args.surface_downsample,
        surface_opacity=0.26,
        line_width=4.2,
        title="W7-X public finite-beta VMEC J streamlines",
    )

    summary = {
        "schema": "w7x_public_finite_beta_j_streamlines_v1",
        "source": meta,
        "desc": {
            "device": args.desc_device,
            "vmecio_note": "DESC VMECIO warned this wout is VMEC 8.47; geometry/J diagnostics should be treated as benchmark-grade, not final validation.",
        },
        "pest_j_payload": diagnostics,
        "cartesian_j_field": cartesian_diagnostics,
        "stream_surface_index": stream_surface,
        "stream_surface_rho": float(pest.rho_vals[stream_surface]),
        "streamlines": streamlines.metadata,
        "outputs": {
            "coords_npz": str(coords_path),
            "cartesian_field_npz": str(cartesian_field_path) if cartesian_field is not None else "",
            "streamlines_npz": str(lines_path),
            "sections_png": str(png_path),
            "plotly_html": str(html_path),
        },
    }
    summary_path = out / "w7x_public_finite_beta_j_streamlines_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    for key, value in summary["outputs"].items():
        print(f"{key}: {value}")
    print(f"summary_json: {summary_path}")
    print(f"betatotal: {meta.get('betatotal')}")
    print(f"stream_surface_rho: {float(pest.rho_vals[stream_surface]):.6f}")
    print(f"n_seed_lines: {streamlines.n_lines}")
    print(f"n_points: {streamlines.n_points}")
    print(f"normal_leakage_p95: {streamlines.metadata.get('normal_leakage_abs_over_norm_p95')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
