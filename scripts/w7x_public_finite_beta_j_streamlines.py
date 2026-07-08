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
    VmecCurrentFourier,
    field_period_phi_range,
    pest_tangent_components_to_cylindrical,
    plot_j_streamline_seed_sections,
    plot_j_streamlines_on_pest_surface_plotly,
    plotly_streamline_style,
    trace_j_streamlines_on_pest,
    vmec_current_fourier_to_pest_field,
)
from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates


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


def parse_pair(text: str) -> tuple[float, float]:
    values = parse_floats(text)
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated floats")
    return float(values[0]), float(values[1])


def _option_supplied(argv: list[str], *names: str) -> bool:
    for token in argv:
        for name in names:
            if token == name or token.startswith(f"{name}="):
                return True
    return False


def _set_default_unless_supplied(args, argv: list[str], attr: str, value, *options: str) -> None:
    if not _option_supplied(argv, *options):
        setattr(args, attr, value)


def _apply_plot_preset(args, argv: list[str], *, nfp: int) -> None:
    key = str(args.plot_preset).strip().lower().replace("_", "-")
    if key == "manual":
        return
    if key not in {"stellarator-j-b", "one-period-dense"}:
        raise ValueError("--plot-preset must be 'manual', 'stellarator-j-b', or 'one-period-dense'")

    style = plotly_streamline_style("one-period-dense" if key == "one-period-dense" else "stellarator-j-b")
    style_kwargs = style.to_plotly_kwargs()
    option_map = {
        "surface_opacity": ("--surface-opacity",),
        "line_width": ("--j-line-width",),
        "companion_line_width": ("--b-line-width",),
        "j_color": ("--j-color",),
        "companion_color": ("--b-color",),
        "line_opacity": ("--j-line-opacity",),
        "companion_line_opacity": ("--b-line-opacity",),
        "arrow_count_per_line": ("--arrow-count-per-line",),
        "companion_arrow_count_per_line": ("--b-arrow-count-per-line",),
        "arrow_line_stride": ("--arrow-line-stride",),
        "companion_arrow_line_stride": ("--b-arrow-line-stride",),
        "arrow_size": ("--arrow-size",),
        "companion_arrow_size": ("--b-arrow-size",),
        "j_arrow_color": ("--j-arrow-color",),
        "companion_arrow_color": ("--b-arrow-color",),
    }
    attr_map = {
        "line_width": "j_line_width",
        "companion_line_width": "b_line_width",
        "line_opacity": "j_line_opacity",
        "companion_line_opacity": "b_line_opacity",
        "companion_color": "b_color",
        "companion_arrow_count_per_line": "b_arrow_count_per_line",
        "companion_arrow_line_stride": "b_arrow_line_stride",
        "companion_arrow_size": "b_arrow_size",
        "companion_arrow_color": "b_arrow_color",
    }
    for key_name, options in option_map.items():
        attr = attr_map.get(key_name, key_name)
        _set_default_unless_supplied(args, argv, attr, style_kwargs[key_name], *options)

    if key == "one-period-dense":
        dense_rho = (0.42, 0.52, 0.62, 0.72, 0.82)
        _set_default_unless_supplied(args, argv, "rho_values", dense_rho, "--rho-values")
        _set_default_unless_supplied(args, argv, "stream_rho_values", dense_rho, "--stream-rho-values")
        _set_default_unless_supplied(
            args,
            argv,
            "phi_range",
            field_period_phi_range(int(nfp), period_index=int(args.field_period_index)),
            "--phi-range",
        )
        _set_default_unless_supplied(args, argv, "phi_seed_count", 11, "--phi-seed-count")
        _set_default_unless_supplied(args, argv, "seed_count", 2, "--seed-count")
        _set_default_unless_supplied(args, argv, "b_seed_count", 1, "--b-seed-count")
        _set_default_unless_supplied(args, argv, "n_turns", 1.6, "--n-turns")
        _set_default_unless_supplied(args, argv, "b_n_turns", 0.9, "--b-n-turns")
        _set_default_unless_supplied(args, argv, "steps_per_turn", 1500, "--steps-per-turn")
        _set_default_unless_supplied(args, argv, "surface_phi_samples", 72, "--surface-phi-samples")


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


def _load_vmec_current_profiles(path: Path, rho_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from netCDF4 import Dataset

    with Dataset(str(path), "r") as ds:
        if "jcuru" not in ds.variables or "jcurv" not in ds.variables:
            raise KeyError("wout file must provide jcuru and jcurv for VMEC profile current tracing")
        jcuru = np.asarray(ds.variables["jcuru"][...], dtype=np.float64)
        jcurv = np.asarray(ds.variables["jcurv"][...], dtype=np.float64)
        ns = int(np.asarray(ds.variables["ns"][...]).item()) if "ns" in ds.variables else int(jcuru.size)
    s_grid = np.linspace(0.0, 1.0, int(ns), dtype=np.float64)
    if jcuru.size != s_grid.size or jcurv.size != s_grid.size:
        s_grid = np.linspace(0.0, 1.0, int(jcuru.size), dtype=np.float64)
    s_eval = np.asarray(rho_vals, dtype=np.float64) ** 2
    ju = np.interp(s_eval, s_grid, jcuru)
    jv = np.interp(s_eval, s_grid, jcurv)
    ratio = np.abs(jv) / np.maximum(np.abs(ju), 1.0e-300)
    diagnostics = {
        "vmec_jcuru_min": float(np.nanmin(jcuru)),
        "vmec_jcuru_median": float(np.nanmedian(jcuru)),
        "vmec_jcuru_max": float(np.nanmax(jcuru)),
        "vmec_jcurv_min": float(np.nanmin(jcurv)),
        "vmec_jcurv_median": float(np.nanmedian(jcurv)),
        "vmec_jcurv_max": float(np.nanmax(jcurv)),
        "vmec_abs_jcurv_over_jcuru_median": float(np.nanmedian(np.abs(jcurv) / np.maximum(np.abs(jcuru), 1.0e-300))),
        "sample_abs_jcurv_over_jcuru_median": float(np.nanmedian(ratio)),
        "sample_abs_jcurv_over_jcuru_p95": float(np.nanpercentile(ratio, 95.0)),
    }
    return ju, jv, diagnostics


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


def _desc_rpz_vector_to_cartesian_components(values: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Convert DESC default vector components from ``(Rhat, phihat, Zhat)`` to XYZ."""

    vec = np.asarray(values, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    if vec.shape != (phi_arr.size, 3):
        raise ValueError("DESC rpz vector values must have shape (len(phi), 3)")
    cp = np.cos(phi_arr)
    sp = np.sin(phi_arr)
    return np.stack(
        [
            vec[:, 0] * cp - vec[:, 1] * sp,
            vec[:, 0] * sp + vec[:, 1] * cp,
            vec[:, 2],
        ],
        axis=1,
    )


def build_desc_pest_j_payload(
    eq,
    *,
    rho_vals: np.ndarray,
    n_phi: int,
    n_theta: int,
    current_source: str,
    current_mode: str,
    vmec_jcuru: np.ndarray | None = None,
    vmec_jcurv: np.ndarray | None = None,
    vmec_current: VmecCurrentFourier | None = None,
    tol: float,
    maxiter: int,
) -> tuple[SmoothPestCoordinates, GriddedPestVectorField, GriddedPestVectorField, dict[str, object]]:
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
            "J^theta_PEST",
            "J^zeta",
            "B_R",
            "B_phi",
            "B_Z",
            "e_rho",
            "e_theta",
            "e_zeta",
            "theta_PEST_t",
            "theta_PEST_z",
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
    raw_BR = np.asarray(data["B_R"], dtype=np.float64)
    raw_BPhi = np.asarray(data["B_phi"], dtype=np.float64)
    raw_BZ = np.asarray(data["B_Z"], dtype=np.float64)
    raw_B = np.stack([raw_BR, raw_BPhi, raw_BZ], axis=1)
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
    desc_Jtheta_pest = np.asarray(data["J^theta_PEST"], dtype=np.float64).reshape(shape)
    desc_Jzeta = np.asarray(data["J^zeta"], dtype=np.float64).reshape(shape)
    if current_source == "vmec-profile":
        if vmec_jcuru is None or vmec_jcurv is None:
            raise ValueError("vmec-profile current source requires vmec_jcuru and vmec_jcurv")
        Jtheta = np.broadcast_to(np.asarray(vmec_jcuru, dtype=np.float64)[None, :, None], shape).copy()
        Jphi = np.broadcast_to(np.asarray(vmec_jcurv, dtype=np.float64)[None, :, None], shape).copy()
        JR, JPhi, JZ = pest_tangent_components_to_cylindrical(pest, Jtheta=Jtheta, Jphi=Jphi)
        sampled_J = np.stack([JR.ravel(), JPhi.ravel(), JZ.ravel()], axis=1)
    elif current_source == "vmec-fourier":
        if vmec_current is None:
            raise ValueError("vmec-fourier current source requires vmec_current")
        theta_desc = np.asarray(nodes_desc[:, 1], dtype=np.float64).reshape(shape)
        zeta_desc = np.asarray(nodes_desc[:, 2], dtype=np.float64).reshape(shape)
        fourier_field = vmec_current_fourier_to_pest_field(
            pest,
            vmec_current,
            theta_vmec=np.mod(-theta_desc, 2.0 * np.pi),
            zeta=zeta_desc,
            theta_pest_t=np.asarray(data["theta_PEST_t"], dtype=np.float64).reshape(shape),
            theta_pest_z=np.asarray(data["theta_PEST_z"], dtype=np.float64).reshape(shape),
            vmec_to_desc_theta_sign=-1.0,
            source="W7-X public finite-beta VMEC current harmonics",
        )
        JR = np.asarray(fourier_field.JR, dtype=np.float64)
        JZ = np.asarray(fourier_field.JZ, dtype=np.float64)
        JPhi = np.asarray(fourier_field.JPhi, dtype=np.float64)
        assert fourier_field.Jtheta is not None and fourier_field.Jphi is not None
        Jtheta = np.asarray(fourier_field.Jtheta, dtype=np.float64)
        Jphi = np.asarray(fourier_field.Jphi, dtype=np.float64)
        sampled_J = np.stack([JR.ravel(), JPhi.ravel(), JZ.ravel()], axis=1)
    elif current_source == "desc-local":
        if current_mode == "full":
            sampled_J = raw_J
            Jtheta = None
            Jphi = None
        elif current_mode == "tangent":
            Jtheta = desc_Jtheta_pest
            Jphi = desc_Jzeta
            JR, JPhi, JZ = pest_tangent_components_to_cylindrical(pest, Jtheta=Jtheta, Jphi=Jphi)
            sampled_J = np.stack([JR.ravel(), JPhi.ravel(), JZ.ravel()], axis=1)
        else:
            raise ValueError("current_mode must be 'tangent' or 'full'")
    else:
        raise ValueError("current_source must be 'vmec-profile', 'vmec-fourier', or 'desc-local'")
    if Jtheta is None:
        JR = sampled_J[:, 0].reshape(shape)
        JPhi = sampled_J[:, 1].reshape(shape)
        JZ = sampled_J[:, 2].reshape(shape)
    sampled_J_cartesian = _desc_rpz_vector_to_cartesian_components(sampled_J, phi_grid.ravel())
    Jx = sampled_J_cartesian[:, 0].reshape(shape)
    Jy = sampled_J_cartesian[:, 1].reshape(shape)
    Jz_cart = sampled_J_cartesian[:, 2].reshape(shape)
    field = GriddedPestVectorField.from_pest_coordinates(
        pest,
        JR=np.ascontiguousarray(JR),
        JZ=np.ascontiguousarray(JZ),
        JPhi=np.ascontiguousarray(JPhi),
        Jx=np.ascontiguousarray(Jx),
        Jy=np.ascontiguousarray(Jy),
        Jz=np.ascontiguousarray(Jz_cart),
        Jtheta=None if Jtheta is None else np.ascontiguousarray(Jtheta),
        Jphi=None if Jphi is None else np.ascontiguousarray(Jphi),
        nfp=nfp,
        source=f"W7-X public finite-beta {current_source} {current_mode} current",
    )
    sampled_B_cartesian = _desc_rpz_vector_to_cartesian_components(raw_B, phi_grid.ravel())
    BR = raw_BR.reshape(shape)
    BZ = raw_BZ.reshape(shape)
    BPhi = raw_BPhi.reshape(shape)
    b_field = GriddedPestVectorField.from_pest_coordinates(
        pest,
        JR=np.ascontiguousarray(BR),
        JZ=np.ascontiguousarray(BZ),
        JPhi=np.ascontiguousarray(BPhi),
        Jx=np.ascontiguousarray(sampled_B_cartesian[:, 0].reshape(shape)),
        Jy=np.ascontiguousarray(sampled_B_cartesian[:, 1].reshape(shape)),
        Jz=np.ascontiguousarray(sampled_B_cartesian[:, 2].reshape(shape)),
        nfp=nfp,
        source="W7-X public finite-beta DESC magnetic field",
    )
    j_abs = np.sqrt(JR * JR + JZ * JZ + JPhi * JPhi)
    b_abs = np.sqrt(BR * BR + BZ * BZ + BPhi * BPhi)
    raw_j_abs = np.linalg.norm(raw_J, axis=1)
    tangent_j_abs = np.linalg.norm(sampled_J, axis=1)
    radial_j_abs = np.abs(np.asarray(data["J^rho"], dtype=np.float64)) * np.linalg.norm(
        np.asarray(data["e_rho"], dtype=np.float64),
        axis=1,
    )
    diagnostics = {
        "current_mode": current_mode,
        "current_source": current_source,
        "nfp": nfp,
        "field_period_rad": field_period,
        "rho_values": [float(x) for x in rho_vals],
        "n_phi": int(n_phi),
        "n_theta": int(n_theta),
        "J_abs_min": float(np.nanmin(j_abs)),
        "J_abs_median": float(np.nanmedian(j_abs)),
        "J_abs_p95": float(np.nanpercentile(j_abs, 95.0)),
        "J_abs_max": float(np.nanmax(j_abs)),
        "B_abs_min": float(np.nanmin(b_abs)),
        "B_abs_median": float(np.nanmedian(b_abs)),
        "B_abs_p95": float(np.nanpercentile(b_abs, 95.0)),
        "B_abs_max": float(np.nanmax(b_abs)),
        "raw_J_abs_median": float(np.nanmedian(raw_j_abs)),
        "raw_J_abs_p95": float(np.nanpercentile(raw_j_abs, 95.0)),
        "tangent_J_abs_median": float(np.nanmedian(tangent_j_abs)),
        "tangent_J_abs_p95": float(np.nanpercentile(tangent_j_abs, 95.0)),
        "raw_radial_J_abs_median": float(np.nanmedian(radial_j_abs)),
        "raw_radial_J_abs_p95": float(np.nanpercentile(radial_j_abs, 95.0)),
        "raw_radial_over_raw_J_median": float(np.nanmedian(radial_j_abs / np.maximum(raw_j_abs, 1.0e-30))),
        "raw_radial_over_raw_J_p95": float(np.nanpercentile(radial_j_abs / np.maximum(raw_j_abs, 1.0e-30), 95.0)),
        "desc_local_abs_Jzeta_over_Jtheta_PEST_median": float(np.nanmedian(np.abs(desc_Jzeta) / np.maximum(np.abs(desc_Jtheta_pest), 1.0e-300))),
        "desc_local_abs_Jzeta_over_Jtheta_PEST_p95": float(np.nanpercentile(np.abs(desc_Jzeta) / np.maximum(np.abs(desc_Jtheta_pest), 1.0e-300), 95.0)),
        "sample_abs_Jphi_over_Jtheta_median": float(np.nanmedian(np.abs(Jphi) / np.maximum(np.abs(Jtheta), 1.0e-300))) if Jtheta is not None and Jphi is not None else float("nan"),
        "sample_abs_Jphi_over_Jtheta_p95": float(np.nanpercentile(np.abs(Jphi) / np.maximum(np.abs(Jtheta), 1.0e-300), 95.0)) if Jtheta is not None and Jphi is not None else float("nan"),
        "pressure_min": float(np.nanmin(pressure)),
        "pressure_max": float(np.nanmax(pressure)),
    }
    return pest, field, b_field, diagnostics


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
        Jx=field.Jx if field.Jx is not None else np.empty(0, dtype=np.float64),
        Jy=field.Jy if field.Jy is not None else np.empty(0, dtype=np.float64),
        Jz=field.Jz if field.Jz is not None else np.empty(0, dtype=np.float64),
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


def _phi_span_diagnostics(streamlines, *, limit_turns: float) -> dict[str, object]:
    spans = []
    for line_idx in range(streamlines.n_lines):
        keep = (
            np.isfinite(streamlines.phi[line_idx])
            & np.isfinite(streamlines.x[line_idx])
            & np.isfinite(streamlines.y[line_idx])
            & np.isfinite(streamlines.z[line_idx])
        )
        if np.count_nonzero(keep) < 2:
            spans.append(float("nan"))
            continue
        phi = np.unwrap(streamlines.phi[line_idx, keep])
        spans.append(float((np.nanmax(phi) - np.nanmin(phi)) / (2.0 * np.pi)))
    finite = np.asarray([x for x in spans if np.isfinite(x)], dtype=np.float64)
    limit = float(limit_turns)
    return {
        "phi_span_turns": spans,
        "phi_span_limit_turns": limit,
        "phi_span_max_turns": float(np.nanmax(finite)) if finite.size else float("nan"),
        "phi_span_p95_turns": float(np.nanpercentile(finite, 95.0)) if finite.size else float("nan"),
        "phi_span_all_ok": bool(finite.size > 0 and np.all(finite <= limit * (1.0 + 1.0e-12))),
    }


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=DEFAULT_WOUT, help="Public finite-beta VMEC wout file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory outside the repo.")
    parser.add_argument("--plot-preset", choices=("manual", "stellarator-j-b", "one-period-dense"), default="manual", help="Optional plotting/tracing preset. 'one-period-dense' uses one full field period and dense J-line seeding.")
    parser.add_argument("--field-period-index", type=int, default=0, help="Field-period index used by --plot-preset one-period-dense when --phi-range is omitted.")
    parser.add_argument("--desc-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--rho-values", type=parse_floats, default=(0.35, 0.55, 0.75, 0.9))
    parser.add_argument("--stream-rho", type=float, default=0.75)
    parser.add_argument("--stream-rho-values", type=parse_floats, default=None, help="One or more rho values for J-line magnetic surfaces. Overrides --stream-rho.")
    parser.add_argument("--n-phi", type=int, default=80)
    parser.add_argument("--n-theta", type=int, default=160)
    parser.add_argument("--current-source", choices=("vmec-profile", "vmec-fourier", "desc-local"), default="vmec-profile", help="Use VMEC jcuru/jcurv profile current by default; vmec-fourier uses wout currumn/currvmn harmonics; desc-local uses DESC local curl(B)-derived current diagnostics.")
    parser.add_argument("--current-mode", choices=("tangent", "full"), default="tangent", help="For desc-local, use surface-tangent current by default; 'full' keeps DESC raw J. Ignored by vmec-profile.")
    parser.add_argument("--trace-space", choices=("pest", "cartesian"), default="pest", help="Trace on the PEST surface by default. Use 'cartesian' only for Cartesian field diagnostic tracing.")
    parser.add_argument("--cartesian-field-source", choices=("pest-surface", "desc-grid"), default="pest-surface", help="For Cartesian diagnostic tracing, use PEST-surface projected J by default; 'desc-grid' builds a regular VectorFieldCylind body grid through DESC inverse mapping.")
    parser.add_argument("--phi-indices", type=parse_ints, default=None, help="Seed section indices, comma-separated.")
    parser.add_argument("--phi-range", type=parse_pair, default=None, help="Optional toroidal plotting/tracing sector as start,end radians.")
    parser.add_argument("--phi-seed-count", type=int, default=None, help="Number of toroidal seed sections inside --phi-range when --phi-indices is omitted.")
    parser.add_argument("--no-clip-phi-range", action="store_true", help="Use --phi-range only for seed selection/Plotly masking, not as a tracing stop condition.")
    parser.add_argument("--seed-count", type=int, default=3, help="Seeds per selected toroidal seed value and magnetic surface.")
    parser.add_argument("--seed-spacing", choices=("arclength", "theta"), default="arclength", help="Distribute seeds uniformly in cross-section arclength by default.")
    parser.add_argument("--no-b-streamlines", action="store_true", help="Do not draw magnetic-field companion streamlines in the Plotly view.")
    parser.add_argument("--b-seed-count", type=int, default=None, help="Seeds per toroidal seed value and surface for B companion lines. Defaults to --seed-count.")
    parser.add_argument("--b-n-turns", type=float, default=None, help="Trace half-length for B companion lines. Defaults to --n-turns.")
    parser.add_argument("--j-line-width", type=float, default=3.4, help="Plotly line width for J streamlines.")
    parser.add_argument("--b-line-width", type=float, default=1.7, help="Plotly line width for B companion streamlines.")
    parser.add_argument("--j-line-opacity", type=float, default=0.94, help="Plotly opacity for J streamlines.")
    parser.add_argument("--b-line-opacity", type=float, default=0.54, help="Plotly opacity for B companion streamlines.")
    parser.add_argument("--j-color", default="rgba(136, 28, 25, 0.92)", help="Plotly color for J streamlines.")
    parser.add_argument("--b-color", default="rgba(14, 116, 235, 0.50)", help="Plotly color for B companion streamlines.")
    parser.add_argument("--j-arrow-color", default="#f97316", help="Plotly cone color for J direction arrows.")
    parser.add_argument("--b-arrow-color", default="#0284c7", help="Plotly cone color for B direction arrows.")
    parser.add_argument("--no-direction-arrows", action="store_true", help="Disable 3-D direction arrows in the Plotly view.")
    parser.add_argument("--arrow-count-per-line", type=int, default=1, help="Direction arrows per sampled J line.")
    parser.add_argument("--b-arrow-count-per-line", type=int, default=1, help="Direction arrows per sampled B line.")
    parser.add_argument("--arrow-line-stride", type=int, default=1, help="Draw J direction arrows on every Nth visible J line.")
    parser.add_argument("--b-arrow-line-stride", type=int, default=1, help="Draw B direction arrows on every Nth visible B line.")
    parser.add_argument("--arrow-size", type=float, default=0.16, help="Absolute Plotly cone size for J direction arrows.")
    parser.add_argument("--b-arrow-size", type=float, default=0.18, help="Absolute Plotly cone size for B direction arrows.")
    parser.add_argument("--n-turns", type=float, default=1.35, help="Trace half-length in seed-surface perimeter units used by pyna.plot.")
    parser.add_argument("--steps-per-turn", type=int, default=1200)
    parser.add_argument("--surface-downsample", type=int, default=2)
    parser.add_argument("--surface-phi-samples", type=int, default=None, help="Optional Plotly surface phi samples, useful for narrow --phi-range sectors.")
    parser.add_argument("--surface-opacity", type=float, default=0.22, help="Plotly opacity for PEST magnetic surfaces.")
    parser.add_argument("--cartesian-n-r", dest="cartesian_n_r", type=int, default=64, help="R grid points for Cartesian-space J tracing.")
    parser.add_argument("--cartesian-n-z", dest="cartesian_n_z", type=int, default=64, help="Z grid points for Cartesian-space J tracing.")
    parser.add_argument("--cartesian-n-phi", type=int, default=40, help="Toroidal grid points over one field period for Cartesian-space J tracing.")
    parser.add_argument("--cartesian-margin", type=float, default=0.04, help="R/Z padding around the PEST surface envelope for Cartesian J sampling.")
    parser.add_argument("--cartesian-rho-limit", type=float, default=1.02, help="Maximum DESC rho retained in the Cartesian J grid.")
    parser.add_argument("--cartesian-map-residual-limit", type=float, default=1.0e-4, help="Reject Cartesian J grid points whose DESC inverse residual exceeds this value; use <=0 to disable.")
    parser.add_argument("--cartesian-batch-size", type=int, default=4096)
    parser.add_argument("--projection-distance", type=float, default=None, help="Maximum R/Z distance from the seed PEST surface when projecting Cartesian lines back to seed sections.")
    parser.add_argument("--max-projection-step", type=float, default=None, help="Break projected 2-D lines when adjacent projected points jump farther than this distance.")
    parser.add_argument("--max-3d-segment-angle", type=float, default=45.0, help="Break Plotly 3-D line traces when adjacent segment directions turn more than this angle in degrees.")
    parser.add_argument("--phi-span-limit-turns", type=float, default=1.0 / 15.0, help="Maximum allowed azimuthal span in toroidal turns for trusted W7-X J-line plots.")
    parser.add_argument("--allow-phi-span-fail", action="store_true", help="Write outputs even if the J-line phi-span gate fails.")
    parser.add_argument("--min-current-fraction", type=float, default=1.0e-3, help="Stop streamlines where |J| falls below this fraction of the sampled median |J|.")
    parser.add_argument("--min-current-abs", type=float, default=None, help="Absolute |J| floor for tracing; overrides --min-current-fraction when set.")
    parser.add_argument("--no-cartesian-surface-snap", action="store_true", help="Disable per-step projection of Cartesian surface-projected traces back to the seed PEST surface.")
    parser.add_argument("--tol", type=float, default=1.0e-9, help="DESC theta_PEST coordinate-map tolerance.")
    parser.add_argument("--maxiter", type=int, default=40, help="DESC theta_PEST coordinate-map Newton iterations.")
    args = parser.parse_args(argv)

    wout = args.wout.expanduser()
    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    meta = _wout_metadata(wout)
    eq = _load_desc_equilibrium(wout, device=args.desc_device)
    _apply_plot_preset(args, raw_argv, nfp=int(eq.NFP))
    rho_vals = np.asarray(args.rho_values, dtype=np.float64)
    if np.any(rho_vals <= 0.0) or np.any(rho_vals >= 1.0):
        raise ValueError("--rho-values must lie strictly inside (0, 1)")

    vmec_jcuru = None
    vmec_jcurv = None
    vmec_current = None
    vmec_current_diag: dict[str, object] = {}
    if args.current_source == "vmec-profile":
        vmec_jcuru, vmec_jcurv, vmec_current_diag = _load_vmec_current_profiles(wout, rho_vals)
    elif args.current_source == "vmec-fourier":
        vmec_current = VmecCurrentFourier.from_wout(wout)
        vmec_current_diag = {
            "vmec_current_harmonics_source": str(wout),
            "vmec_current_harmonics_modes": int(vmec_current.xm.size),
        }

    pest, pest_field, b_field, diagnostics = build_desc_pest_j_payload(
        eq,
        rho_vals=rho_vals,
        n_phi=args.n_phi,
        n_theta=args.n_theta,
        current_source=args.current_source,
        current_mode=args.current_mode,
        vmec_jcuru=vmec_jcuru,
        vmec_jcurv=vmec_jcurv,
        vmec_current=vmec_current,
        tol=args.tol,
        maxiter=args.maxiter,
    )
    diagnostics.update(vmec_current_diag)
    stream_rhos = np.asarray(args.stream_rho_values if args.stream_rho_values is not None else (args.stream_rho,), dtype=np.float64)
    if np.any(stream_rhos <= 0.0) or np.any(stream_rhos >= 1.0):
        raise ValueError("stream rho values must lie strictly inside (0, 1)")
    stream_surfaces = np.asarray(
        [int(np.argmin(np.abs(pest.rho_vals - float(rho)))) for rho in stream_rhos],
        dtype=np.int64,
    )
    stream_surfaces = np.unique(stream_surfaces)
    stream_surface_arg = int(stream_surfaces[0]) if stream_surfaces.size == 1 else [int(i) for i in stream_surfaces]
    cartesian_field = None
    cartesian_diagnostics: dict[str, object] | None = {
        "trace_field_source": "pest_surface_projected",
        "trace_space": "cartesian",
        "note": "Cartesian RK4 uses nearest-surface projection to evaluate and snap J on the seed surface. Prefer --trace-space pest for trusted PEST-surface current-line plots.",
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
    if args.min_current_abs is not None:
        min_current_norm = float(args.min_current_abs)
    else:
        median_current = float(diagnostics.get("J_abs_median", 0.0))
        min_current_norm = max(float(args.min_current_fraction), 0.0) * median_current
    streamlines = trace_j_streamlines_on_pest(
        trace_field,
        pest,
        surface_index=stream_surface_arg,
        phi_indices=args.phi_indices,
        phi_range=args.phi_range,
        phi_seed_count=args.phi_seed_count,
        clip_phi_range=not bool(args.no_clip_phi_range),
        seed_count=args.seed_count,
        seed_spacing=args.seed_spacing,
        n_turns=args.n_turns,
        steps_per_turn=args.steps_per_turn,
        min_field_norm=min_current_norm,
        constrain_to_surface=args.trace_space == "pest",
        max_surface_distance=args.projection_distance,
        snap_cartesian_to_surface=not bool(args.no_cartesian_surface_snap),
    )
    b_streamlines = None
    if not bool(args.no_b_streamlines):
        median_b = float(diagnostics.get("B_abs_median", 0.0))
        min_b_norm = max(float(args.min_current_fraction), 0.0) * median_b
        b_streamlines = trace_j_streamlines_on_pest(
            b_field,
            pest,
            surface_index=stream_surface_arg,
            phi_indices=args.phi_indices,
            phi_range=args.phi_range,
            phi_seed_count=args.phi_seed_count,
            clip_phi_range=not bool(args.no_clip_phi_range),
            seed_count=int(args.b_seed_count) if args.b_seed_count is not None else int(args.seed_count),
            seed_spacing=args.seed_spacing,
            n_turns=float(args.b_n_turns) if args.b_n_turns is not None else float(args.n_turns),
            steps_per_turn=args.steps_per_turn,
            min_field_norm=min_b_norm,
            constrain_to_surface=args.trace_space == "pest",
            max_surface_distance=args.projection_distance,
            snap_cartesian_to_surface=not bool(args.no_cartesian_surface_snap),
        )
    phi_span_diag = _phi_span_diagnostics(streamlines, limit_turns=float(args.phi_span_limit_turns))

    coords_path = out / "w7x_public_finite_beta_pest_j_payload.npz"
    cartesian_field_path = out / "w7x_public_finite_beta_cartesian_j_field.npz"
    lines_path = out / "w7x_public_finite_beta_j_streamlines.npz"
    b_lines_path = out / "w7x_public_finite_beta_b_streamlines.npz"
    write_npz_payload(coords_path, pest, pest_field)
    if cartesian_field is not None:
        write_vector_field_npz(cartesian_field_path, cartesian_field)
    write_streamlines_npz(lines_path, streamlines)
    if b_streamlines is not None:
        write_streamlines_npz(b_lines_path, b_streamlines)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    source_label = {
        "vmec-profile": "VMEC profile current",
        "vmec-fourier": "VMEC current-harmonic",
        "desc-local": "DESC local-current diagnostic",
    }[args.current_source]
    gate_label = "phi-span gate OK" if bool(phi_span_diag["phi_span_all_ok"]) else "phi-span gate FAILED"
    fig, _axes = plot_j_streamline_seed_sections(
        streamlines,
        pest,
        title=f"W7-X public finite-beta {source_label} streamlines on PEST sections ({gate_label})",
        line_width=1.0,
        alpha=0.88,
        max_projection_step=args.max_projection_step,
        max_surface_distance=args.projection_distance,
    )
    png_path = out / "w7x_public_finite_beta_j_streamline_seed_sections.png"
    fig.savefig(png_path, dpi=210, bbox_inches="tight")
    plt.close(fig)

    html_path = out / (
        "w7x_public_finite_beta_j_b_streamlines_3d.html"
        if b_streamlines is not None
        else "w7x_public_finite_beta_j_streamlines_3d.html"
    )
    plot_j_streamlines_on_pest_surface_plotly(
        streamlines,
        pest,
        surface_index=stream_surface_arg,
        phi_range=args.phi_range,
        companion_streamlines=b_streamlines,
        companion_name="B",
        html_path=html_path,
        surface_downsample=args.surface_downsample,
        surface_phi_samples=args.surface_phi_samples,
        surface_opacity=args.surface_opacity,
        line_width=args.j_line_width,
        companion_line_width=args.b_line_width,
        j_color=args.j_color,
        companion_color=args.b_color,
        line_opacity=args.j_line_opacity,
        companion_line_opacity=args.b_line_opacity,
        show_arrows=not bool(args.no_direction_arrows),
        arrow_count_per_line=args.arrow_count_per_line,
        companion_arrow_count_per_line=args.b_arrow_count_per_line,
        arrow_line_stride=args.arrow_line_stride,
        companion_arrow_line_stride=args.b_arrow_line_stride,
        arrow_size=args.arrow_size,
        companion_arrow_size=args.b_arrow_size,
        j_arrow_color=args.j_arrow_color,
        companion_arrow_color=args.b_arrow_color,
        max_segment_angle_deg=args.max_3d_segment_angle,
        title=f"W7-X public finite-beta {source_label} J/B streamlines ({gate_label})",
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
        "stream_surface_indices": [int(i) for i in stream_surfaces],
        "stream_surface_rho": [float(pest.rho_vals[int(i)]) for i in stream_surfaces],
        "streamlines": streamlines.metadata,
        "b_streamlines": None if b_streamlines is None else b_streamlines.metadata,
        "phi_span_gate": phi_span_diag,
        "trace_controls": {
            "plot_preset": str(args.plot_preset),
            "field_period_index": int(args.field_period_index),
            "min_current_norm": float(min_current_norm),
            "min_current_fraction": float(args.min_current_fraction),
            "min_current_abs": None if args.min_current_abs is None else float(args.min_current_abs),
            "snap_cartesian_to_surface": not bool(args.no_cartesian_surface_snap),
            "seed_spacing": str(args.seed_spacing),
            "phi_range": None if args.phi_range is None else [float(args.phi_range[0]), float(args.phi_range[1])],
            "clip_phi_range": not bool(args.no_clip_phi_range),
            "b_streamlines": b_streamlines is not None,
            "b_seed_count": None if args.b_seed_count is None else int(args.b_seed_count),
            "b_n_turns": None if args.b_n_turns is None else float(args.b_n_turns),
            "surface_phi_samples": None if args.surface_phi_samples is None else int(args.surface_phi_samples),
            "direction_arrows": not bool(args.no_direction_arrows),
            "j_color": str(args.j_color),
            "b_color": str(args.b_color),
            "j_line_width": float(args.j_line_width),
            "b_line_width": float(args.b_line_width),
            "j_line_opacity": float(args.j_line_opacity),
            "b_line_opacity": float(args.b_line_opacity),
            "arrow_count_per_line": int(args.arrow_count_per_line),
            "b_arrow_count_per_line": int(args.b_arrow_count_per_line),
            "arrow_line_stride": int(args.arrow_line_stride),
            "b_arrow_line_stride": int(args.b_arrow_line_stride),
            "arrow_size": float(args.arrow_size),
            "b_arrow_size": float(args.b_arrow_size),
            "surface_opacity": float(args.surface_opacity),
        },
        "outputs": {
            "coords_npz": str(coords_path),
            "cartesian_field_npz": str(cartesian_field_path) if cartesian_field is not None else "",
            "streamlines_npz": str(lines_path),
            "b_streamlines_npz": str(b_lines_path) if b_streamlines is not None else "",
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
    print(f"stream_surface_rho: {', '.join(f'{float(pest.rho_vals[int(i)]):.6f}' for i in stream_surfaces)}")
    print(f"n_seed_lines: {streamlines.n_lines}")
    print(f"n_points: {streamlines.n_points}")
    print(f"normal_leakage_p95: {streamlines.metadata.get('normal_leakage_abs_over_norm_p95')}")
    if b_streamlines is not None:
        print(f"b_n_seed_lines: {b_streamlines.n_lines}")
        print(f"b_normal_leakage_p95: {b_streamlines.metadata.get('normal_leakage_abs_over_norm_p95')}")
    print(f"phi_span_max_turns: {phi_span_diag.get('phi_span_max_turns')}")
    print(f"phi_span_all_ok: {phi_span_diag.get('phi_span_all_ok')}")
    if not bool(phi_span_diag["phi_span_all_ok"]) and not bool(args.allow_phi_span_fail):
        raise SystemExit(
            "J-line phi-span gate failed; use --allow-phi-span-fail only for explicit diagnostics."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
