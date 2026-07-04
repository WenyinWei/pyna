"""Rebuild NCSX PEST/Boozer-like surface coordinates from fresh traces.

This script is intentionally self-contained enough to run when the local cyna
extension is not built.  It traces field lines through the NCSX vacuum mgrid
with a small Scipy/RK4 backend, stitches the crossings to a straight-theta grid,
then builds a Boozer-like remap and writes static/Plotly diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


TWOPI = 2.0 * np.pi
DEFAULT_NCSX_DIR = Path(os.environ.get("PYNA_NCSX_ROOT", "data/NCSX"))
DEFAULT_TOPOQUEST = Path(os.environ.get("TOPOQUEST_ROOT", "../topoquest"))
DEFAULT_PYNA = Path(os.environ.get("PYNA_ROOT", "."))
DEFAULT_OUT = DEFAULT_NCSX_DIR / "ncsx_redo_fieldline_coords_20260620_v1"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_local_coordinate_modules(pyna_root: Path) -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """Load the local toroidal coordinate modules without importing pyna.__init__."""

    pyna_pkg = sys.modules.setdefault("pyna", types.ModuleType("pyna"))
    pyna_pkg.__path__ = [str(pyna_root / "pyna")]
    toroidal_pkg = sys.modules.setdefault("pyna.toroidal", types.ModuleType("pyna.toroidal"))
    toroidal_pkg.__path__ = [str(pyna_root / "pyna" / "toroidal")]
    base = pyna_root / "pyna" / "toroidal"
    periodic = _load_module("pyna.toroidal._periodic_grid", base / "_periodic_grid.py")
    pest = _load_module("pyna.toroidal.pest_coords", base / "pest_coords.py")
    boozer = _load_module("pyna.toroidal.boozer_coords", base / "boozer_coords.py")
    return periodic, pest, boozer


@dataclass(frozen=True)
class TraceResult:
    R: np.ndarray
    Z: np.ndarray
    turn: np.ndarray


def periodic_rgi(axis_R: np.ndarray, axis_Z: np.ndarray, axis_phi: np.ndarray, values: np.ndarray) -> RegularGridInterpolator:
    phi_ext = np.concatenate([axis_phi, [axis_phi[0] + TWOPI]])
    vals_ext = np.concatenate([values, values[:, :, :1]], axis=2)
    return RegularGridInterpolator(
        (axis_R, axis_Z, phi_ext),
        vals_ext,
        bounds_error=False,
        fill_value=np.nan,
    )


def sample_field(interp_BR, interp_BZ, interp_Bphi, R: float, Z: float, phi: float) -> tuple[float, float, float]:
    point = np.array([[R, Z, np.mod(phi, TWOPI)]], dtype=np.float64)
    BR = float(interp_BR(point)[0])
    BZ = float(interp_BZ(point)[0])
    Bphi = float(interp_Bphi(point)[0])
    return BR, BZ, Bphi


def rhs(interp_BR, interp_BZ, interp_Bphi, R: float, Z: float, phi: float) -> tuple[float, float]:
    BR, BZ, Bphi = sample_field(interp_BR, interp_BZ, interp_Bphi, R, Z, phi)
    if not np.isfinite(Bphi) or abs(Bphi) < 1.0e-14:
        return float("nan"), float("nan")
    return R * BR / Bphi, R * BZ / Bphi


def rk4_step(interp_BR, interp_BZ, interp_Bphi, R: float, Z: float, phi: float, h: float) -> tuple[float, float]:
    k1R, k1Z = rhs(interp_BR, interp_BZ, interp_Bphi, R, Z, phi)
    k2R, k2Z = rhs(interp_BR, interp_BZ, interp_Bphi, R + 0.5 * h * k1R, Z + 0.5 * h * k1Z, phi + 0.5 * h)
    k3R, k3Z = rhs(interp_BR, interp_BZ, interp_Bphi, R + 0.5 * h * k2R, Z + 0.5 * h * k2Z, phi + 0.5 * h)
    k4R, k4Z = rhs(interp_BR, interp_BZ, interp_Bphi, R + h * k3R, Z + h * k3Z, phi + h)
    R_next = R + h * (k1R + 2.0 * k2R + 2.0 * k3R + k4R) / 6.0
    Z_next = Z + h * (k1Z + 2.0 * k2Z + 2.0 * k3Z + k4Z) / 6.0
    return R_next, Z_next


def trace_sections(
    interp_BR,
    interp_BZ,
    interp_Bphi,
    R0: float,
    Z0: float,
    *,
    n_phi: int,
    n_turns: int,
    steps_per_section: int,
) -> TraceResult:
    R = float(R0)
    Z = float(Z0)
    phi = 0.0
    h = TWOPI / float(n_phi * steps_per_section)
    out_R = np.full((n_phi, n_turns), np.nan, dtype=np.float64)
    out_Z = np.full_like(out_R, np.nan)
    out_turn = np.broadcast_to(np.arange(n_turns, dtype=np.int64), out_R.shape).copy()

    for iturn in range(n_turns):
        for iphi in range(n_phi):
            out_R[iphi, iturn] = R
            out_Z[iphi, iturn] = Z
            for _ in range(steps_per_section):
                R, Z = rk4_step(interp_BR, interp_BZ, interp_Bphi, R, Z, phi, h)
                phi += h
            if not np.isfinite(R) or not np.isfinite(Z):
                return TraceResult(out_R, out_Z, out_turn)
    return TraceResult(out_R, out_Z, out_turn)


def periodic_angle_interp(angle_src: np.ndarray, values: np.ndarray, angle_dst: np.ndarray) -> np.ndarray:
    src = np.mod(np.asarray(angle_src, dtype=np.float64), TWOPI)
    vals = np.asarray(values, dtype=np.float64)
    dst = np.mod(np.asarray(angle_dst, dtype=np.float64), TWOPI)
    order = np.argsort(src)
    src = src[order]
    vals = vals[order]
    keep = np.isfinite(src) & np.isfinite(vals)
    src = src[keep]
    vals = vals[keep]
    _, unique_idx = np.unique(np.round(src, 12), return_index=True)
    src = src[unique_idx]
    vals = vals[unique_idx]
    src_ext = np.concatenate([src[-1:] - TWOPI, src, src[:1] + TWOPI])
    vals_ext = np.concatenate([vals[-1:], vals, vals[:1]])
    return np.interp(dst, src_ext, vals_ext)


def build_seed_points(axis_R0: float, axis_Z0: float, lcfs_R: np.ndarray, lcfs_Z: np.ndarray, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    geom = np.mod(np.arctan2(lcfs_Z - axis_Z0, lcfs_R - axis_R0), TWOPI)
    lfs_angle = 0.0
    R_edge = float(periodic_angle_interp(geom, lcfs_R, np.array([lfs_angle]))[0])
    Z_edge = float(periodic_angle_interp(geom, lcfs_Z, np.array([lfs_angle]))[0])
    return axis_R0 + fractions * (R_edge - axis_R0), axis_Z0 + fractions * (Z_edge - axis_Z0)


def build_pest_surfaces(
    traces: list[TraceResult],
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    axis_R: np.ndarray,
    axis_Z: np.ndarray,
    pest,
    *,
    max_iota: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_phi = phi_vals.size
    n_r = len(traces)
    n_theta = theta_vals.size
    R_surf = np.full((n_phi, n_r, n_theta), np.nan, dtype=np.float64)
    Z_surf = np.full_like(R_surf, np.nan)
    iota = np.full(n_r, np.nan, dtype=np.float64)
    iota_rms = np.full(n_r, np.nan, dtype=np.float64)

    for ir, trace in enumerate(traces):
        ref_phase = np.mod(np.arctan2(trace.Z[0] - axis_Z[0], trace.R[0] - axis_R[0]), TWOPI)
        iota[ir], iota_rms[ir] = pest.circle_map_lift_iota(ref_phase, max_iota=max_iota, min_points=32)
        if not np.isfinite(iota[ir]) or iota[ir] <= 0.0:
            iota[ir] = 0.35
        for iphi, phi in enumerate(phi_vals):
            theta_sample = np.mod(TWOPI * iota[ir] * trace.turn[iphi] + iota[ir] * phi, TWOPI)
            R_surf[iphi, ir] = pest.stitch_periodic(theta_sample, trace.R[iphi], theta_vals, min_points=24)
            Z_surf[iphi, ir] = pest.stitch_periodic(theta_sample, trace.Z[iphi], theta_vals, min_points=24)

    return R_surf, Z_surf, iota, iota_rms


def sample_B_abs(interp_BR, interp_BZ, interp_Bphi, R_surf: np.ndarray, Z_surf: np.ndarray, phi_vals: np.ndarray) -> np.ndarray:
    B_abs = np.full_like(R_surf, np.nan)
    for iphi, phi in enumerate(phi_vals):
        pts = np.column_stack(
            [
                R_surf[iphi].ravel(),
                Z_surf[iphi].ravel(),
                np.full(R_surf.shape[1] * R_surf.shape[2], phi),
            ]
        )
        BR = interp_BR(pts)
        BZ = interp_BZ(pts)
        Bphi = interp_Bphi(pts)
        B_abs[iphi] = np.sqrt(BR * BR + BZ * BZ + Bphi * Bphi).reshape(R_surf.shape[1:])
    return B_abs


def remap_phi_B_grid(phi_vals, theta_vals, theta_B, theta_B_of_theta, iota, periodic) -> np.ndarray:
    raw_phi_B = (
        phi_vals[:, np.newaxis, np.newaxis]
        + (theta_B_of_theta - theta_vals[np.newaxis, np.newaxis, :])
        / np.maximum(iota[np.newaxis, :, np.newaxis], 1.0e-8)
    )
    out = np.empty((phi_vals.size, iota.size, theta_B.size), dtype=np.float64)
    for iphi in range(phi_vals.size):
        for ir in range(iota.size):
            c = periodic.periodic_interp(theta_B_of_theta[iphi, ir], np.cos(raw_phi_B[iphi, ir]), theta_B, TWOPI)
            s = periodic.periodic_interp(theta_B_of_theta[iphi, ir], np.sin(raw_phi_B[iphi, ir]), theta_B, TWOPI)
            out[iphi, ir] = np.mod(np.arctan2(s, c), TWOPI)
    return out


def plot_sections(path: Path, R_surf: np.ndarray, Z_surf: np.ndarray, phi_vals: np.ndarray, title: str) -> None:
    section_idx = np.linspace(0, phi_vals.size, 4, endpoint=False, dtype=int)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for ax, idx in zip(axes, section_idx):
        for ir in range(R_surf.shape[1]):
            ax.plot(
                np.r_[R_surf[idx, ir], R_surf[idx, ir, 0]],
                np.r_[Z_surf[idx, ir], Z_surf[idx, ir, 0]],
                lw=0.9,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"phi={np.degrees(phi_vals[idx]):.0f} deg")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def cartesian(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return R * np.cos(phi), R * np.sin(phi), Z


def interp_equal_phi_B_line(R: np.ndarray, Z: np.ndarray, phi_cyl: np.ndarray, phi_B: np.ndarray, target: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.unwrap(phi_B)
    base = src[0]
    tgt = base + np.mod(target - base, TWOPI)
    src_ext = np.concatenate([src, src[:1] + TWOPI])
    phi_ext = np.concatenate([phi_cyl, phi_cyl[:1] + TWOPI])
    R_ext = np.concatenate([R, R[:1]])
    Z_ext = np.concatenate([Z, Z[:1]])
    order = np.argsort(src_ext)
    src_ext = src_ext[order]
    phi_ext = phi_ext[order]
    R_ext = R_ext[order]
    Z_ext = Z_ext[order]
    return (
        np.interp(tgt, src_ext, R_ext),
        np.interp(tgt, src_ext, Z_ext),
        np.mod(np.interp(tgt, src_ext, phi_ext), TWOPI),
    )


def write_plotly_html(
    path: Path,
    R: np.ndarray,
    Z: np.ndarray,
    phi_cyl: np.ndarray,
    theta_B: np.ndarray,
    phi_B_grid: np.ndarray,
    radial_labels: np.ndarray,
) -> None:
    traces: list[dict] = []
    phi_closed = np.concatenate([phi_cyl, phi_cyl[:1] + TWOPI])
    theta_step = max(1, theta_B.size // 72)
    phi_step = max(1, phi_cyl.size // 36)
    theta_idx = np.arange(0, theta_B.size, theta_step)

    for ir in range(R.shape[1]):
        R_theta = np.concatenate([R[:, ir, ::theta_step], R[:, ir, :1]], axis=1)
        Z_theta = np.concatenate([Z[:, ir, ::theta_step], Z[:, ir, :1]], axis=1)
        R2 = np.vstack([R_theta, R_theta[:1]])
        Z2 = np.vstack([Z_theta, Z_theta[:1]])
        P2 = np.repeat(phi_closed[:, np.newaxis], R2.shape[1], axis=1)
        X, Y, Zc = cartesian(R2, Z2, P2)
        traces.append(
            {
                "type": "surface",
                "x": X.tolist(),
                "y": Y.tolist(),
                "z": Zc.tolist(),
                "opacity": 0.13,
                "showscale": False,
                "colorscale": "Viridis",
                "name": f"rho={radial_labels[ir]:.2f}",
                "hoverinfo": "skip",
            }
        )

    line_color_theta = "#173f5f"
    line_color_phi = "#d1495b"
    radial_for_grid = list(range(R.shape[1]))
    for ir in radial_for_grid:
        for it in np.arange(0, theta_B.size, max(1, theta_B.size // 12)):
            Rc = np.concatenate([R[:, ir, it], R[:1, ir, it]])
            Zc = np.concatenate([Z[:, ir, it], Z[:1, ir, it]])
            X, Y, Zline = cartesian(Rc, Zc, phi_closed)
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "x": X.tolist(),
                    "y": Y.tolist(),
                    "z": Zline.tolist(),
                    "line": {"color": line_color_theta, "width": 3},
                    "name": "theta_B const",
                    "showlegend": False,
                }
            )
        for target in np.linspace(0.0, TWOPI, 12, endpoint=False):
            pts_R = []
            pts_Z = []
            pts_phi = []
            for it in theta_idx:
                r_val, z_val, p_val = interp_equal_phi_B_line(
                    R[:, ir, it],
                    Z[:, ir, it],
                    phi_cyl,
                    phi_B_grid[:, ir, it],
                    target,
                )
                pts_R.append(r_val)
                pts_Z.append(z_val)
                pts_phi.append(p_val)
            pts_R.append(pts_R[0])
            pts_Z.append(pts_Z[0])
            pts_phi.append(pts_phi[0])
            X, Y, Zline = cartesian(np.asarray(pts_R), np.asarray(pts_Z), np.asarray(pts_phi))
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "x": X.tolist(),
                    "y": Y.tolist(),
                    "z": Zline.tolist(),
                    "line": {"color": line_color_phi, "width": 3},
                    "name": "phi_B const",
                    "showlegend": False,
                }
            )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <title>NCSX Boozer-like grid</title>
  <style>html,body,#plot{{width:100%;height:100%;margin:0;}}</style>
</head>
<body>
<div id="plot"></div>
<script>
const data = {json.dumps(traces)};
const layout = {{
  title: "NCSX Boozer-like coordinate grid",
  scene: {{
    xaxis: {{title: "X"}},
    yaxis: {{title: "Y"}},
    zaxis: {{title: "Z"}},
    aspectmode: "data"
  }},
  margin: {{l: 0, r: 0, t: 45, b: 0}}
}};
Plotly.newPlot("plot", data, layout, {{responsive: true}});
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def parse_fractions(text: str) -> np.ndarray:
    vals = np.array([float(part) for part in text.split(",") if part.strip()], dtype=np.float64)
    if vals.ndim != 1 or vals.size < 2 or np.any(np.diff(vals) <= 0.0):
        raise argparse.ArgumentTypeError("fractions must be a strictly increasing comma list")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgrid", type=Path, default=DEFAULT_NCSX_DIR / "mgrid_c09r00.nc")
    parser.add_argument("--wout", type=Path, default=DEFAULT_NCSX_DIR / "wout_ncsx_c09r00_free.nc")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pyna-root", type=Path, default=DEFAULT_PYNA)
    parser.add_argument("--topoquest-root", type=Path, default=DEFAULT_TOPOQUEST)
    parser.add_argument("--n-phi", type=int, default=36)
    parser.add_argument("--n-theta", type=int, default=96)
    parser.add_argument("--n-turns", type=int, default=220)
    parser.add_argument("--steps-per-section", type=int, default=4)
    parser.add_argument("--max-iota", type=float, default=0.6)
    parser.add_argument("--fractions", type=parse_fractions, default=parse_fractions("0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.topoquest_root))
    from topoquest.analysis.stellarator_mgrid import (
        load_vmec_mgrid_vacuum,
        vmec_axis_from_wout,
        vmec_lcfs_from_wout,
    )

    periodic, pest, boozer_mod = load_local_coordinate_modules(args.pyna_root)
    field, meta = load_vmec_mgrid_vacuum(str(args.mgrid), wout_path=str(args.wout))
    interp_BR = periodic_rgi(field.R_arr, field.Z_arr, field.Phi, field.BR)
    interp_BZ = periodic_rgi(field.R_arr, field.Z_arr, field.Phi, field.BZ)
    interp_Bphi = periodic_rgi(field.R_arr, field.Z_arr, field.Phi, field.BPhi)

    phi_vals = np.linspace(0.0, TWOPI, args.n_phi, endpoint=False, dtype=np.float64)
    theta_vals = np.linspace(0.0, TWOPI, args.n_theta, endpoint=False, dtype=np.float64)
    axis_R, axis_Z, nfp = vmec_axis_from_wout(str(args.wout), phi_vals)
    lcfs_R, lcfs_Z = vmec_lcfs_from_wout(str(args.wout), phi=0.0, ntheta=720)
    seed_R, seed_Z = build_seed_points(float(axis_R[0]), float(axis_Z[0]), lcfs_R, lcfs_Z, args.fractions)

    traces = []
    for ir, (R0, Z0) in enumerate(zip(seed_R, seed_Z)):
        print(f"trace {ir + 1}/{seed_R.size}: R0={R0:.6f} Z0={Z0:.6f}", flush=True)
        traces.append(
            trace_sections(
                interp_BR,
                interp_BZ,
                interp_Bphi,
                float(R0),
                float(Z0),
                n_phi=args.n_phi,
                n_turns=args.n_turns,
                steps_per_section=args.steps_per_section,
            )
        )

    R_pest, Z_pest, iota, iota_rms = build_pest_surfaces(
        traces,
        phi_vals,
        theta_vals,
        axis_R,
        axis_Z,
        pest,
        max_iota=float(args.max_iota),
    )
    B_abs = sample_B_abs(interp_BR, interp_BZ, interp_Bphi, R_pest, Z_pest, phi_vals)
    boozer = boozer_mod.build_Boozer_coordinates(
        R_pest,
        Z_pest,
        phi_vals,
        theta_vals,
        radial_labels=args.fractions,
        B_abs=B_abs,
        n_theta=args.n_theta,
    )
    phi_B_grid = remap_phi_B_grid(
        phi_vals,
        theta_vals,
        boozer.theta_B,
        boozer.theta_B_of_theta,
        iota,
        periodic,
    )

    np.savez_compressed(
        args.out_dir / "ncsx_redo_pest_coords.npz",
        R_surf=R_pest,
        Z_surf=Z_pest,
        phi_vals=phi_vals,
        theta_vals=theta_vals,
        radial_labels=args.fractions,
        axis_R=axis_R,
        axis_Z=axis_Z,
        seed_R=seed_R,
        seed_Z=seed_Z,
        iota=iota,
        iota_rms=iota_rms,
        B_abs=B_abs,
    )
    np.savez_compressed(
        args.out_dir / "ncsx_redo_boozer_coords.npz",
        R_surf=boozer.R_surf,
        Z_surf=boozer.Z_surf,
        theta_B=boozer.theta_B,
        phi_cyl=phi_vals,
        phi_B_grid=phi_B_grid,
        radial_labels=boozer.radial_labels,
        theta_B_of_theta=boozer.theta_B_of_theta,
        lambda_B=boozer.lambda_B,
        jacobian=boozer.jacobian,
        jacobian_B=boozer.jacobian_B,
        B2_jacobian_B=boozer.B2_jacobian_B,
        iota=iota,
        iota_rms=iota_rms,
    )
    plot_sections(args.out_dir / "ncsx_redo_pest_four_sections.png", R_pest, Z_pest, phi_vals, "NCSX rebuilt PEST-like sections")
    plot_sections(args.out_dir / "ncsx_redo_boozer_four_sections.png", boozer.R_surf, boozer.Z_surf, phi_vals, "NCSX rebuilt Boozer-like sections")
    write_plotly_html(
        args.out_dir / "ncsx_redo_boozer_grid_3d.html",
        boozer.R_surf,
        boozer.Z_surf,
        phi_vals,
        boozer.theta_B,
        phi_B_grid,
        boozer.radial_labels,
    )

    B2_jac_abs = np.abs(boozer.B2_jacobian_B)
    spread = np.nanstd(B2_jac_abs, axis=2) / np.maximum(
        np.nanmean(B2_jac_abs, axis=2),
        1.0e-30,
    )
    summary = {
        "mgrid": str(args.mgrid),
        "wout": str(args.wout),
        "nfp": int(nfp),
        "mgrid_mode": meta.mode,
        "n_phi": args.n_phi,
        "n_theta": args.n_theta,
        "n_turns": args.n_turns,
        "steps_per_section": args.steps_per_section,
        "max_iota": args.max_iota,
        "radial_fractions": args.fractions.tolist(),
        "iota": iota.tolist(),
        "iota_rms": iota_rms.tolist(),
        "B2_jacobian_B_theta_rel_std_median": float(np.nanmedian(spread)),
        "B2_jacobian_B_theta_rel_std_max": float(np.nanmax(spread)),
        "tracer": "pure Python RK4 over mgrid with phi as independent variable",
        "phi_B_grid_note": "Approximate Boozer toroidal angle induced from theta_B shift by theta_B ~= theta_P + iota*(phi_B-phi_cyl).",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
