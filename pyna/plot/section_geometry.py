"""Composable plotting primitives for toroidal section geometry."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SECTION_CYCLE_COLORS = (
    "#c62828",
    "#1565c0",
    "#2e7d32",
    "#6a1b9a",
    "#ef6c00",
    "#00838f",
    "#ad1457",
    "#455a64",
)


def create_section_grid(
    section_phis: Sequence[float],
    *,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    data_limits: tuple[float, float, float, float] | None = None,
    panel_height: float = 3.15,
    compact: bool = True,
    share_axes: bool = True,
    aspect_ratio: float = 1.0,
):
    """Create a compact grid for toroidal section plots."""

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    phi = np.asarray(section_phis, dtype=float).ravel()
    if phi.size == 0:
        raise ValueError("section_phis must not be empty")
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(phi.size / ncols))
    if figsize is None:
        if data_limits is None:
            panel_width = float(panel_height)
        else:
            xspan = max(float(data_limits[1]) - float(data_limits[0]), 1.0e-12)
            yspan = max(float(data_limits[3]) - float(data_limits[2]), 1.0e-12)
            panel_width = float(panel_height) * xspan / (yspan * max(float(aspect_ratio), 1.0e-12))
        figsize = (panel_width * ncols, float(panel_height) * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=bool(share_axes),
        sharey=bool(share_axes),
    )
    if compact:
        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.055, top=0.965, wspace=0.0, hspace=0.0)
    for ax in axes.ravel():
        ax.set_aspect(float(aspect_ratio), adjustable="box")
    return fig, axes


def format_section_axis(
    ax,
    *,
    section_phi: float | None = None,
    title: str | None = None,
    title_inside: bool = False,
    aspect_ratio: float = 1.0,
    grid: bool = True,
):
    """Apply common formatting to one section axis."""

    if title is None and section_phi is not None:
        title = rf"$\phi/\pi={float(section_phi) / np.pi:.2f}$"
    if title:
        if title_inside:
            from matplotlib import patheffects as pe

            ax.set_title("")
            ax.text(
                0.02,
                0.98,
                title,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                zorder=20,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )
        else:
            ax.set_title(title, fontsize=9, pad=2.0)
    ax.set_aspect(float(aspect_ratio), adjustable="box")
    if grid:
        ax.grid(True, lw=0.28, color="0.85", alpha=0.58)
    return ax


def trim_compact_tick_labels(axes, n_section: int, *, ncols: int) -> None:
    """Hide inner tick labels for a compact shared-axis grid."""

    axes_arr = np.asarray(axes, dtype=object)
    nrows = axes_arr.shape[0]
    for flat_idx, ax in enumerate(axes_arr.ravel()):
        if flat_idx >= int(n_section):
            ax.set_visible(False)
            continue
        if flat_idx % int(ncols) != 0:
            ax.tick_params(labelleft=False)
        if flat_idx < (nrows - 1) * int(ncols):
            ax.tick_params(labelbottom=False)


def save_figure(fig, out_path: str | Path | None, *, dpi: int = 180):
    """Save ``fig`` if an output path is provided."""

    if out_path is None:
        return None
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi))
    return out


def draw_wall_section(
    ax,
    wall_R,
    wall_Z,
    *,
    color: str = "0.34",
    lw: float = 1.0,
    alpha: float = 0.82,
    zorder: int = 1,
    **kwargs,
):
    """Draw one R/Z wall polygon or wall section curve."""

    return ax.plot(
        np.asarray(wall_R, dtype=float),
        np.asarray(wall_Z, dtype=float),
        color=color,
        lw=float(lw),
        alpha=float(alpha),
        zorder=int(zorder),
        **kwargs,
    )


def draw_axis_point(
    ax,
    R: float,
    Z: float,
    *,
    marker: str = "+",
    color: str = "0.05",
    ms: float = 5.0,
    mew: float = 1.0,
    zorder: int = 7,
    **kwargs,
):
    """Draw one magnetic-axis or reference point."""

    return ax.plot(
        float(R),
        float(Z),
        marker=marker,
        color=color,
        ms=float(ms),
        mew=float(mew),
        ls="None",
        zorder=int(zorder),
        **kwargs,
    )


def draw_poincare_points(
    ax,
    R,
    Z,
    seed_index=None,
    *,
    values=None,
    cmap: str = "tab20",
    point_size: float = 4.0,
    alpha: float = 0.30,
    color=None,
    zorder: int = 2,
    rasterized: bool = True,
    **kwargs,
):
    """Draw generic Poincare points for core, mid-radius, or edge traces."""

    import matplotlib.pyplot as plt

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if R_arr.size == 0 or Z_arr.size == 0:
        return None
    finite = np.isfinite(R_arr) & np.isfinite(Z_arr)
    if values is not None:
        c = np.asarray(values, dtype=float).ravel()
        if c.shape == R_arr.shape:
            finite &= np.isfinite(c)
            color_arg = c[finite]
        else:
            color_arg = color
    elif seed_index is not None:
        seed = np.asarray(seed_index).ravel()
        if seed.shape == R_arr.shape:
            color_arg = plt.get_cmap(cmap)(np.mod(seed[finite], plt.get_cmap(cmap).N))
        else:
            color_arg = color
    else:
        color_arg = color
    return ax.scatter(
        R_arr[finite],
        Z_arr[finite],
        s=float(point_size),
        c=color_arg,
        cmap=None if values is None else cmap,
        alpha=float(alpha),
        linewidths=0,
        rasterized=bool(rasterized),
        zorder=int(zorder),
        **kwargs,
    )


def draw_poincare_background(
    ax,
    background,
    section_index: int,
    *,
    point_size: float = 4.0,
    alpha: float = 0.30,
    cmap: str = "tab20",
    zorder: int = 2,
    **kwargs,
):
    """Draw any object exposing ``section_points(section_index)``."""

    R, Z, seed_index = background.section_points(int(section_index))
    return draw_poincare_points(
        ax,
        R,
        Z,
        seed_index,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
        zorder=zorder,
        **kwargs,
    )


def cycle_list(value) -> list:
    """Normalize a cycle, chain, or sequence of cycles to a list."""

    if value is None:
        return []
    if hasattr(value, "cycles"):
        return list(value.cycles)
    if hasattr(value, "points"):
        return [value]
    return list(value)


def cycles_for_section(section_cycles, phi: float, section_index: int) -> list:
    """Return cycles for one section from a mapping or indexable sequence."""

    if section_cycles is None:
        return []
    if isinstance(section_cycles, Mapping):
        if phi in section_cycles:
            return cycle_list(section_cycles[phi])
        if section_cycles:
            key = min(section_cycles, key=lambda p: abs(float(p) - float(phi)))
            return cycle_list(section_cycles[key])
        return []
    if len(section_cycles) == 0:
        return []
    first = section_cycles[0]
    if hasattr(first, "cycles") or hasattr(first, "points"):
        return cycle_list(section_cycles)
    return cycle_list(section_cycles[int(section_index)])


def cycle_identity(cycle, fallback: int = 0) -> tuple[str, int]:
    """Return a stable plot identity for one fixed-point cycle."""

    metadata = getattr(cycle, "metadata", {}) or {}
    key = metadata.get("same_cycle_key")
    cycle_id = getattr(cycle, "cycle_id", None)
    if key is None:
        key = f"cycle={cycle_id if cycle_id is not None else fallback}:kind={getattr(cycle, 'kind', '')}"
    if cycle_id is None:
        cycle_id = fallback
    return str(key), int(cycle_id)


def cycle_point_arrays(cycle) -> tuple[np.ndarray, np.ndarray]:
    """Return R/Z arrays for one fixed-point cycle-like object."""

    points = tuple(getattr(cycle, "points", ()) or ())
    return (
        np.asarray([float(fp.R) for fp in points], dtype=float),
        np.asarray([float(fp.Z) for fp in points], dtype=float),
    )


def draw_cycle_points(
    ax,
    cycles,
    *,
    identity_to_color: dict[str, str] | None = None,
    colors: Sequence[str] = SECTION_CYCLE_COLORS,
    marker_size: float = 76.0,
    label_cycle_ids: bool = False,
    zorder: int = 8,
):
    """Draw X/O cycle intersection points without assuming boundary topology."""

    from matplotlib import patheffects as pe

    if identity_to_color is None:
        identity_to_color = {}
    artists = []
    for cycle_index, cycle in enumerate(cycle_list(cycles)):
        identity, cycle_id = cycle_identity(cycle, cycle_index)
        if identity not in identity_to_color:
            identity_to_color[identity] = colors[len(identity_to_color) % len(colors)]
        color = identity_to_color[identity]
        marker = "X" if str(getattr(cycle, "kind", "")).upper() == "X" else "o"
        R, Z = cycle_point_arrays(cycle)
        if R.size == 0:
            continue
        artists.append(ax.scatter(
            R,
            Z,
            s=float(marker_size),
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=1.1,
            zorder=int(zorder),
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
        ))
        if label_cycle_ids:
            for fp in getattr(cycle, "points", ()):
                metadata = getattr(fp, "metadata", {})
                idx = metadata.get("orbit_point_index", metadata.get("point_index", ""))
                artists.append(ax.text(
                    float(fp.R),
                    float(fp.Z),
                    f"{cycle_id}:{idx}",
                    color=color,
                    fontsize=5.6,
                    ha="left",
                    va="bottom",
                    zorder=int(zorder) + 1,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
                ))
    return artists


def is_manifold_payload(value) -> bool:
    """Return whether ``value`` looks like a traced manifold payload."""

    return isinstance(value, Mapping) and (
        "u_R" in value or "u_Z" in value or "s_R" in value or "s_Z" in value
    )


def manifold_list(value) -> list:
    """Normalize manifold payloads to a list."""

    if value is None:
        return []
    if is_manifold_payload(value):
        return [value]
    return list(value)


def manifolds_for_section(manifolds_by_section, phi: float, section_index: int) -> list:
    """Return manifold payloads for one section."""

    if manifolds_by_section is None:
        return []
    if isinstance(manifolds_by_section, Mapping) and not is_manifold_payload(manifolds_by_section):
        if phi in manifolds_by_section:
            return manifold_list(manifolds_by_section[phi])
        if manifolds_by_section:
            key = min(manifolds_by_section, key=lambda p: abs(float(p) - float(phi)))
            return manifold_list(manifolds_by_section[key])
        return []
    if len(manifolds_by_section) == 0:
        return []
    first = manifolds_by_section[0]
    if is_manifold_payload(first):
        return manifold_list(manifolds_by_section)
    return manifold_list(manifolds_by_section[int(section_index)])


def manifold_lpol_max(manifolds_by_section, phi_sections: Sequence[float]) -> float | None:
    """Return the maximum finite arclength color value across sections."""

    values: list[np.ndarray] = []
    for i, phi in enumerate(phi_sections):
        for manifold in manifolds_for_section(manifolds_by_section, float(phi), i):
            for key in ("u_lpol", "s_lpol"):
                arr = np.asarray(manifold.get(key, []), dtype=float).ravel()
                if arr.size:
                    values.append(arr)
    if not values:
        return None
    all_values = np.concatenate(values)
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def draw_manifold_points(
    ax,
    manifolds,
    *,
    point_size: float = 5.0,
    alpha: float = 0.78,
    cmap: str = "viridis",
    vmax: float | None = None,
    unstable_color: str = "#43a047",
    stable_color: str = "#e64a19",
    zorder: int = 5,
):
    """Draw stable/unstable manifold point clouds for one section."""

    artists = []
    for manifold in manifold_list(manifolds):
        for prefix in ("u", "s"):
            R = np.asarray(manifold.get(f"{prefix}_R", []), dtype=float).ravel()
            Z = np.asarray(manifold.get(f"{prefix}_Z", []), dtype=float).ravel()
            if R.size == 0 or Z.size == 0:
                continue
            lpol = np.asarray(manifold.get(f"{prefix}_lpol", []), dtype=float).ravel()
            finite = np.isfinite(R) & np.isfinite(Z)
            if lpol.shape == R.shape:
                finite &= np.isfinite(lpol)
                artists.append(ax.scatter(
                    R[finite],
                    Z[finite],
                    s=float(point_size),
                    c=lpol[finite],
                    cmap=cmap,
                    vmin=0.0,
                    vmax=vmax,
                    alpha=float(alpha),
                    linewidths=0,
                    rasterized=True,
                    zorder=int(zorder),
                ))
            else:
                color = unstable_color if prefix == "u" else stable_color
                artists.append(ax.scatter(
                    R[finite],
                    Z[finite],
                    s=float(point_size),
                    c=color,
                    alpha=float(alpha),
                    linewidths=0,
                    rasterized=True,
                    zorder=int(zorder),
                ))
    return artists


def section_data_limits(
    *,
    section_phis: Sequence[float],
    background=None,
    section_cycles=None,
    manifolds_by_section=None,
    walls=None,
    pad_fraction: float = 0.035,
) -> tuple[float, float, float, float] | None:
    """Compute shared R/Z limits from common section geometry payloads."""

    R_parts: list[np.ndarray] = []
    Z_parts: list[np.ndarray] = []
    for i, phi in enumerate(np.asarray(section_phis, dtype=float).ravel()):
        if background is not None:
            Rb, Zb, _seed_idx = background.section_points(i)
            if np.asarray(Rb).size:
                R_parts.append(np.asarray(Rb, dtype=float).ravel())
                Z_parts.append(np.asarray(Zb, dtype=float).ravel())
        for cycle in cycles_for_section(section_cycles, float(phi), i):
            R, Z = cycle_point_arrays(cycle)
            if R.size:
                R_parts.append(R)
                Z_parts.append(Z)
        for manifold in manifolds_for_section(manifolds_by_section, float(phi), i):
            for prefix in ("u", "s"):
                Rm = np.asarray(manifold.get(f"{prefix}_R", []), dtype=float).ravel()
                Zm = np.asarray(manifold.get(f"{prefix}_Z", []), dtype=float).ravel()
                if Rm.size and Zm.size:
                    R_parts.append(Rm)
                    Z_parts.append(Zm)
        if walls is not None:
            wall_R, wall_Z = walls[i]
            R_parts.append(np.asarray(wall_R, dtype=float).ravel())
            Z_parts.append(np.asarray(wall_Z, dtype=float).ravel())
    if not R_parts:
        return None
    R_all = np.concatenate(R_parts)
    Z_all = np.concatenate(Z_parts)
    finite = np.isfinite(R_all) & np.isfinite(Z_all)
    if not np.any(finite):
        return None
    R_all = R_all[finite]
    Z_all = Z_all[finite]
    rmin = float(np.min(R_all))
    rmax = float(np.max(R_all))
    zmin = float(np.min(Z_all))
    zmax = float(np.max(Z_all))
    rpad = max(1.0e-9, float(pad_fraction) * max(rmax - rmin, 1.0e-9))
    zpad = max(1.0e-9, float(pad_fraction) * max(zmax - zmin, 1.0e-9))
    return rmin - rpad, rmax + rpad, zmin - zpad, zmax + zpad


def apply_section_limits(axes, limits: tuple[float, float, float, float] | None) -> None:
    """Apply shared R/Z limits to all visible axes."""

    if limits is None:
        return
    for ax in np.asarray(axes, dtype=object).ravel():
        if not ax.get_visible():
            continue
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])


__all__ = [
    "SECTION_CYCLE_COLORS",
    "apply_section_limits",
    "create_section_grid",
    "cycle_identity",
    "cycle_list",
    "cycle_point_arrays",
    "cycles_for_section",
    "draw_axis_point",
    "draw_cycle_points",
    "draw_manifold_points",
    "draw_poincare_background",
    "draw_poincare_points",
    "draw_wall_section",
    "format_section_axis",
    "is_manifold_payload",
    "manifold_list",
    "manifold_lpol_max",
    "manifolds_for_section",
    "save_figure",
    "section_data_limits",
    "trim_compact_tick_labels",
]
