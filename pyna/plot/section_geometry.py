"""Composable plotting primitives for toroidal section geometry."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SECTION_ORBIT_COLORS = (
    "#c62828",
    "#1565c0",
    "#2e7d32",
    "#6a1b9a",
    "#ef6c00",
    "#00838f",
    "#ad1457",
    "#455a64",
)

_MAP_POWER_METADATA_KEYS = (
    "map_order_index",
    "orbit_point_index",
    "point_index",
    "poincare_map_power",
    "map_power",
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


def orbit_list(value) -> list:
    """Normalize a orbit, chain, or sequence of orbits to a list."""

    if value is None:
        return []
    if hasattr(value, "orbits"):
        return list(value.orbits)
    if hasattr(value, "points"):
        return [value]
    return list(value)


def orbits_for_section(section_orbits, phi: float, section_index: int) -> list:
    """Return orbits for one section from a mapping or indexable sequence."""

    if section_orbits is None:
        return []
    if isinstance(section_orbits, Mapping):
        if phi in section_orbits:
            return orbit_list(section_orbits[phi])
        if section_orbits:
            key = min(section_orbits, key=lambda p: abs(float(p) - float(phi)))
            return orbit_list(section_orbits[key])
        return []
    if len(section_orbits) == 0:
        return []
    first = section_orbits[0]
    if hasattr(first, "orbits") or hasattr(first, "points"):
        return orbit_list(section_orbits)
    return orbit_list(section_orbits[int(section_index)])


def orbit_identity(orbit, fallback: int = 0) -> tuple[str, int]:
    """Return a stable plot identity for one fixed-point orbit."""

    metadata = getattr(orbit, "metadata", {}) or {}
    key = metadata.get("same_orbit_key")
    orbit_id = getattr(orbit, "orbit_id", None)
    if key is None:
        key = f"orbit={orbit_id if orbit_id is not None else fallback}:kind={getattr(orbit, 'kind', '')}"
    if orbit_id is None:
        orbit_id = fallback
    return str(key), int(orbit_id)


def orbit_point_arrays(orbit) -> tuple[np.ndarray, np.ndarray]:
    """Return R/Z arrays for one fixed-point orbit-like object."""

    points = tuple(getattr(orbit, "points", ()) or ())
    return (
        np.asarray([float(fp.R) for fp in points], dtype=float),
        np.asarray([float(fp.Z) for fp in points], dtype=float),
    )


def _first_metadata_value(metadata: Mapping, keys, default=""):
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return default


def draw_orbit_points(
    ax,
    orbits,
    *,
    identity_to_color: dict[str, str] | None = None,
    colors: Sequence[str] = SECTION_ORBIT_COLORS,
    marker_size: float = 76.0,
    label_orbit_ids: bool = False,
    label_map_power_key: str | Sequence[str] = _MAP_POWER_METADATA_KEYS,
    label_template: str = "{orbit_id}:P{map_power}",
    zorder: int = 8,
):
    """Draw X/O orbit intersection points without assuming boundary topology."""

    from matplotlib import patheffects as pe

    if identity_to_color is None:
        identity_to_color = {}
    artists = []
    for orbit_index, orbit in enumerate(orbit_list(orbits)):
        identity, orbit_id = orbit_identity(orbit, orbit_index)
        if identity not in identity_to_color:
            identity_to_color[identity] = colors[len(identity_to_color) % len(colors)]
        color = identity_to_color[identity]
        marker = "X" if str(getattr(orbit, "kind", "")).upper() == "X" else "o"
        R, Z = orbit_point_arrays(orbit)
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
        if label_orbit_ids:
            for fp in getattr(orbit, "points", ()):
                metadata = getattr(fp, "metadata", {})
                map_power = _first_metadata_value(metadata, label_map_power_key)
                label = label_template.format(
                    orbit_id=orbit_id,
                    index=map_power,
                    map_power=map_power,
                    poincare_map_power=map_power,
                    kind=str(getattr(orbit, "kind", "")),
                    identity=identity,
                )
                artists.append(ax.text(
                    float(fp.R),
                    float(fp.Z),
                    label,
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
    max_generation: int | None = None,
    max_arclength: float | None = None,
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
            generation = np.asarray(manifold.get(f"{prefix}_generation", []), dtype=float).ravel()
            if max_generation is not None and generation.shape == R.shape:
                finite &= generation <= float(max_generation)
            if lpol.shape == R.shape:
                finite &= np.isfinite(lpol)
                if max_arclength is not None:
                    finite &= lpol <= float(max_arclength)
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


def draw_manifold_lines(
    ax,
    manifolds,
    *,
    lw: float = 0.72,
    alpha: float = 0.86,
    cmap: str = "viridis",
    vmax: float | None = None,
    max_generation: int | None = None,
    max_arclength: float | None = None,
    max_segment_length: float | None = None,
    zorder: int = 5,
):
    """Draw stable/unstable manifolds as arclength-colored line segments."""

    from matplotlib.collections import LineCollection
    from matplotlib import colors as mcolors

    artists = []
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    for manifold in manifold_list(manifolds):
        for prefix in ("u", "s"):
            R = np.asarray(manifold.get(f"{prefix}_R", []), dtype=float).ravel()
            Z = np.asarray(manifold.get(f"{prefix}_Z", []), dtype=float).ravel()
            lpol = np.asarray(manifold.get(f"{prefix}_lpol", []), dtype=float).ravel()
            if R.size < 2 or Z.size < 2 or lpol.shape != R.shape:
                continue
            finite = np.isfinite(R) & np.isfinite(Z) & np.isfinite(lpol)
            generation = np.asarray(manifold.get(f"{prefix}_generation", []), dtype=float).ravel()
            if max_generation is not None and generation.shape == R.shape:
                finite &= generation <= float(max_generation)
            if max_arclength is not None:
                finite &= lpol <= float(max_arclength)
            if np.count_nonzero(finite) < 2:
                continue
            points = np.column_stack([R, Z])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            segment_finite = finite[:-1] & finite[1:]
            dl = np.diff(lpol)
            segment_finite &= dl >= -1.0e-12
            point_side = np.asarray(manifold.get(f"{prefix}_point_side", []), dtype=float).ravel()
            if point_side.shape == R.shape:
                segment_finite &= np.isfinite(point_side[:-1]) & np.isfinite(point_side[1:])
                segment_finite &= point_side[:-1] == point_side[1:]
            if max_segment_length is not None:
                length = np.hypot(np.diff(R), np.diff(Z))
                segment_finite &= length <= float(max_segment_length)
            if not np.any(segment_finite):
                continue
            values = 0.5 * (lpol[:-1] + lpol[1:])
            lc = LineCollection(
                segments[segment_finite],
                linewidths=float(lw),
                alpha=float(alpha),
                cmap=cmap,
                norm=norm,
                zorder=int(zorder),
            )
            lc.set_array(values[segment_finite])
            ax.add_collection(lc)
            artists.append(lc)
    return artists


def draw_manifold_origins(
    ax,
    manifolds,
    *,
    show_labels: bool = False,
    label_template: str = "{kind}{orbit_id}:P{map_power}",
    marker_size: float = 44.0,
    origin_edge_color: str = "0.08",
    origin_face_color: str = "white",
    unstable_color: str = "#2e7d32",
    stable_color: str = "#ef6c00",
    draw_branch_anchors: bool = True,
    zorder: int = 9,
):
    """Draw manifold origins and short branch anchors back to their X points."""

    from matplotlib import patheffects as pe

    artists = []
    for manifold in manifold_list(manifolds):
        if "origin_R" not in manifold or "origin_Z" not in manifold:
            continue
        R0 = float(manifold["origin_R"])
        Z0 = float(manifold["origin_Z"])
        if not (np.isfinite(R0) and np.isfinite(Z0)):
            continue
        if draw_branch_anchors:
            for prefix, color in (("u", unstable_color), ("s", stable_color)):
                R = np.asarray(manifold.get(f"{prefix}_R", []), dtype=float).ravel()
                Z = np.asarray(manifold.get(f"{prefix}_Z", []), dtype=float).ravel()
                finite = np.isfinite(R) & np.isfinite(Z)
                if not np.any(finite):
                    continue
                R = R[finite]
                Z = Z[finite]
                lpol = np.asarray(manifold.get(f"{prefix}_lpol", []), dtype=float).ravel()
                if lpol.shape == finite.shape:
                    lpol = lpol[finite]
                    order = np.argsort(np.where(np.isfinite(lpol), lpol, np.inf))
                    pick = int(order[0]) if order.size else 0
                else:
                    pick = 0
                artists.extend(ax.plot(
                    [R0, float(R[pick])],
                    [Z0, float(Z[pick])],
                    color=color,
                    lw=0.72,
                    alpha=0.88,
                    zorder=int(zorder) - 1,
                ))
        artists.append(ax.scatter(
            [R0],
            [Z0],
            s=float(marker_size),
            marker="o",
            facecolors=origin_face_color,
            edgecolors=origin_edge_color,
            linewidths=0.8,
            alpha=0.96,
            zorder=int(zorder),
        ))
        if show_labels:
            orbit_id = manifold.get("orbit_id", "?")
            map_power = _first_metadata_value(manifold, _MAP_POWER_METADATA_KEYS, default="?")
            label = label_template.format(
                orbit_id=orbit_id,
                map_power=map_power,
                poincare_map_power=map_power,
                index=map_power,
                kind=str(manifold.get("kind", "X")),
                same_orbit_key=manifold.get("same_orbit_key", ""),
            )
            artists.append(ax.text(
                R0,
                Z0,
                label,
                color=origin_edge_color,
                fontsize=5.4,
                ha="right",
                va="top",
                zorder=int(zorder) + 1,
                path_effects=[pe.withStroke(linewidth=1.4, foreground="white")],
            ))
    return artists


def section_data_limits(
    *,
    section_phis: Sequence[float],
    background=None,
    section_orbits=None,
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
        for orbit in orbits_for_section(section_orbits, float(phi), i):
            R, Z = orbit_point_arrays(orbit)
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
    "SECTION_ORBIT_COLORS",
    "apply_section_limits",
    "create_section_grid",
    "orbit_identity",
    "orbit_list",
    "orbit_point_arrays",
    "orbits_for_section",
    "draw_axis_point",
    "draw_orbit_points",
    "draw_manifold_lines",
    "draw_manifold_origins",
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
