"""Plot helpers for mgrid current-density and smooth PEST diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from pyna.io.mgrid import MGridCurrent, mgrid_toroidal_index
from pyna.toroidal.diagnostics.mgrid import (
    PestCurrentComponents,
    SmoothPestCoordinates,
    periodic_phi_slice,
)


def _robust_symmetric_scale(values: Sequence[np.ndarray], *, percentile: float = 99.0, floor: float = 1.0e-30) -> float:
    finite = [np.asarray(v, dtype=np.float64)[np.isfinite(v)] for v in values]
    finite = [v for v in finite if v.size]
    if not finite:
        return 1.0
    return max(float(np.nanpercentile(np.abs(np.concatenate(finite)), percentile)), floor)


def _triangles(nrho: int, ntheta: int) -> np.ndarray:
    out = []
    for irho in range(nrho - 1):
        for itheta in range(ntheta):
            a = irho * ntheta + itheta
            b = irho * ntheta + (itheta + 1) % ntheta
            c = (irho + 1) * ntheta + itheta
            d = (irho + 1) * ntheta + (itheta + 1) % ntheta
            out.append((a, c, d))
            out.append((a, d, b))
    return np.asarray(out, dtype=np.int64)


def draw_smooth_pest_grid(
    ax,
    R: np.ndarray,
    Z: np.ndarray,
    *,
    theta_stride: Optional[int] = None,
    rho_color: str = "0.86",
    theta_color: str = "0.89",
    lcfs_color: str = "0.16",
):
    """Overlay a smooth PEST ``rho,theta`` grid on a section axis."""

    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    for irho in range(1, R.shape[0]):
        is_edge = irho == R.shape[0] - 1
        ax.plot(
            np.r_[R[irho], R[irho, 0]],
            np.r_[Z[irho], Z[irho, 0]],
            color=lcfs_color if is_edge else rho_color,
            lw=0.9 if is_edge else 0.3,
            zorder=4,
        )
    stride = int(theta_stride or max(1, R.shape[1] // 12))
    for itheta in range(0, R.shape[1], stride):
        ax.plot(R[:, itheta], Z[:, itheta], color=theta_color, lw=0.25, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.93", lw=0.3)


def plot_mgrid_current_cylindrical_components(
    items: Sequence[tuple[str, MGridCurrent, Optional[SmoothPestCoordinates]]],
    sections_deg: Sequence[float],
    *,
    out_path: Optional[Union[str, Path]] = None,
    percentile: float = 99.0,
):
    """Plot ``J_R,J_phi,J_Z`` distributions for multiple mgrid currents."""

    components = ("JR", "JPhi", "JZ")
    labels = {"JR": "JR", "JPhi": "Jphi", "JZ": "JZ"}
    scales = {}
    for comp in components:
        vals = []
        for _label, current, _coords in items:
            for sec in sections_deg:
                vals.append(getattr(current, comp)[mgrid_toroidal_index(current, np.deg2rad(float(sec)))] / 1.0e3)
        scales[comp] = _robust_symmetric_scale(vals, percentile=percentile)

    nrows = len(items) * len(components)
    ncols = len(sections_deg)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.8, 2.1 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(nrows, ncols)
    for item_idx, (label, current, coords) in enumerate(items):
        for comp_idx, comp in enumerate(components):
            row = item_idx * len(components) + comp_idx
            vmax = scales[comp]
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            image = None
            for sec_idx, section_deg in enumerate(sections_deg):
                ax = axes[row, sec_idx]
                phi = np.deg2rad(float(section_deg))
                iphi = mgrid_toroidal_index(current, phi)
                values = getattr(current, comp)[iphi] / 1.0e3
                if coords is not None:
                    Rlcfs = periodic_phi_slice(coords.R_surf, phi)[-1]
                    Zlcfs = periodic_phi_slice(coords.Z_surf, phi)[-1]
                    RR, ZZ = np.meshgrid(current.R, current.Z)
                    path = MplPath(np.column_stack([np.r_[Rlcfs, Rlcfs[0]], np.r_[Zlcfs, Zlcfs[0]]]))
                    inside = path.contains_points(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(RR.shape)
                    values = np.where(inside, values, np.nan)
                    image = ax.pcolormesh(current.R, current.Z, values, shading="auto", cmap="RdBu_r", norm=norm)
                    ax.plot(np.r_[Rlcfs, Rlcfs[0]], np.r_[Zlcfs, Zlcfs[0]], color="0.1", lw=0.85)
                else:
                    image = ax.pcolormesh(current.R, current.Z, values, shading="auto", cmap="RdBu_r", norm=norm)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, color="0.9", lw=0.35)
                if row == 0:
                    ax.set_title(f"phi={float(section_deg):.0f} deg")
                if sec_idx == 0:
                    ax.set_ylabel(f"{label}\n{labels[comp]} [kA/m$^2$]\nZ [m]")
                if row == len(items) * len(components) - 1:
                    ax.set_xlabel("R [m]")
            fig.colorbar(image, ax=axes[row, :].ravel().tolist(), shrink=0.78, pad=0.006, label=f"{labels[comp]} [kA/m$^2$]")
    fig.suptitle("Current density cylindrical components from mgrid: J = curl(B)/mu0")
    if out_path is not None:
        fig.savefig(out_path, dpi=220)
    return fig, axes


def plot_pest_current_components(
    diagnostics: Sequence[PestCurrentComponents],
    *,
    out_path: Optional[Union[str, Path]] = None,
    ratio_clip: float = 20.0,
    percentile: float = 99.0,
):
    """Plot ``J^rho,J^theta,J^phi`` and ``J^theta/J^phi`` on PEST sections."""

    components = ("Jrho", "Jtheta", "Jphi", "Jtheta_over_Jphi")
    labels = {
        "Jrho": r"$J^\rho$",
        "Jtheta": r"$J^\theta$",
        "Jphi": r"$J^\phi$",
        "Jtheta_over_Jphi": r"$J^\theta/J^\phi$",
    }
    scales = {}
    for comp in components:
        vals = []
        for diag in diagnostics:
            for section in diag.sections:
                arr = getattr(section, comp)[1:]
                vals.append(np.clip(arr, -ratio_clip, ratio_clip) if comp == "Jtheta_over_Jphi" else arr)
        scales[comp] = min(float(ratio_clip), _robust_symmetric_scale(vals, percentile=98.0, floor=1.0)) if comp == "Jtheta_over_Jphi" else _robust_symmetric_scale(vals, percentile=percentile)

    nrows = len(diagnostics) * len(components)
    ncols = len(diagnostics[0].sections)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.8, 4.5 * len(diagnostics) * len(components) / 2.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(nrows, ncols)
    tri_cache: dict[tuple[int, int], np.ndarray] = {}
    for diag_idx, diag in enumerate(diagnostics):
        for comp_idx, comp in enumerate(components):
            row = diag_idx * len(components) + comp_idx
            vmax = scales[comp]
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            image = None
            for sec_idx, section in enumerate(diag.sections):
                ax = axes[row, sec_idx]
                R = section.R[1:]
                Z = section.Z[1:]
                values = getattr(section, comp)[1:]
                if comp == "Jtheta_over_Jphi":
                    values = np.clip(values, -vmax, vmax)
                key = (R.shape[0], R.shape[1])
                if key not in tri_cache:
                    tri_cache[key] = _triangles(*key)
                finite = np.isfinite(values.ravel())
                tri = mtri.Triangulation(R.ravel(), Z.ravel(), tri_cache[key])
                tri.set_mask(np.any(~finite[tri.triangles], axis=1))
                image = ax.tripcolor(tri, values.ravel(), shading="gouraud", cmap="RdBu_r", norm=norm)
                draw_smooth_pest_grid(ax, section.R, section.Z)
                ax.set_xlim(np.nanmin(section.R[-1]) - 0.08, np.nanmax(section.R[-1]) + 0.08)
                ax.set_ylim(np.nanmin(section.Z[-1]) - 0.08, np.nanmax(section.Z[-1]) + 0.08)
                if row == 0:
                    ax.set_title(f"phi={section.section_deg:.0f} deg")
                if sec_idx == 0:
                    unit = "" if comp == "Jtheta_over_Jphi" else " [A/m$^3$]"
                    ax.set_ylabel(f"{diag.label}\n{labels[comp]}{unit}\nZ [m]")
                if row == len(diagnostics) * len(components) - 1:
                    ax.set_xlabel("R [m]")
            unit_label = labels[comp] if comp == "Jtheta_over_Jphi" else labels[comp] + " [A/m$^3$]"
            fig.colorbar(image, ax=axes[row, :].ravel().tolist(), shrink=0.78, pad=0.006, label=unit_label)
    fig.suptitle("Smooth-PEST current-density components; ratio row shows sign reversal")
    if out_path is not None:
        fig.savefig(out_path, dpi=220)
    return fig, axes


def plot_surface_fourier_ripple_summary(
    spectra_by_label: Sequence[tuple[str, Sequence[dict[str, object]]]],
    *,
    out_path: Optional[Union[str, Path]] = None,
):
    """Plot mean high-m ripple fraction versus ``rho`` for surface spectra."""

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for label, rows in spectra_by_label:
        rho_values = sorted({float(row["rho"]) for row in rows})
        mean_fraction = []
        for rho in rho_values:
            vals = [float(row["high_rms_fraction"]) for row in rows if np.isclose(float(row["rho"]), rho)]
            mean_fraction.append(float(np.nanmean(vals)) * 100.0)
        ax.plot(rho_values, mean_fraction, marker="o", label=label)
    ax.set_xlabel("rho")
    ax.set_ylabel("high-m RMS / m=1..mode_max RMS [%]")
    ax.set_title("Smooth PEST surface high-m ripple fraction")
    ax.grid(True, color="0.9")
    ax.legend(frameon=False)
    if out_path is not None:
        fig.savefig(out_path, dpi=220)
    return fig, ax


__all__ = [
    "draw_smooth_pest_grid",
    "plot_mgrid_current_cylindrical_components",
    "plot_pest_current_components",
    "plot_surface_fourier_ripple_summary",
]
