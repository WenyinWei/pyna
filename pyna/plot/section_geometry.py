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
    "physical_map_order_index",
    "fieldline_map_order_index",
    "map_order_index",
    "orbit_point_index",
    "point_index",
    "poincare_map_power",
    "map_power",
)


def _validate_aspect_ratio(aspect_ratio: float) -> float:
    value = float(aspect_ratio)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("aspect_ratio must be positive and finite")
    return value


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
    aspect = _validate_aspect_ratio(aspect_ratio)
    nrows = int(np.ceil(phi.size / ncols))
    if figsize is None:
        if data_limits is None:
            panel_width = float(panel_height)
        else:
            xspan = max(float(data_limits[1]) - float(data_limits[0]), 1.0e-12)
            yspan = max(float(data_limits[3]) - float(data_limits[2]), 1.0e-12)
            panel_width = float(panel_height) * xspan / (yspan * aspect)
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
        ax.set_aspect(aspect, adjustable="box")
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
    ax.set_aspect(_validate_aspect_ratio(aspect_ratio), adjustable="box")
    if grid:
        ax.grid(True, lw=0.28, color="0.85", alpha=0.58)
    return ax


def trim_compact_tick_labels(axes, n_section: int, *, ncols: int) -> None:
    """Hide inner tick labels for a compact shared-axis grid."""

    axes_arr = np.asarray(axes, dtype=object)
    nrows = axes_arr.shape[0]
    ncols_eff = axes_arr.shape[1] if axes_arr.ndim >= 2 else max(1, int(ncols))
    for flat_idx, ax in enumerate(axes_arr.ravel()):
        if flat_idx >= int(n_section):
            ax.set_visible(False)
            continue
        if flat_idx % int(ncols_eff) != 0:
            ax.tick_params(labelleft=False)
        if flat_idx < (nrows - 1) * int(ncols_eff):
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


def _first_metadata_or_attr(obj, metadata: Mapping, keys, default=""):
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
        value = getattr(obj, key, None)
        if value is not None:
            return value
    return default


def map_order_value(obj, metadata: Mapping | None = None, keys=None, default=""):
    """Return the preferred Poincare-map order value for labels."""

    if metadata is None:
        metadata = getattr(obj, "metadata", {}) or {}
    if keys is None:
        keys = _MAP_POWER_METADATA_KEYS
    return _first_metadata_or_attr(obj, metadata, keys, default=default)


def _fixed_point_kind(orbit, point=None) -> str:
    kind = getattr(point, "kind", None) if point is not None else None
    if kind is None:
        kind = getattr(orbit, "kind", "")
    return str(kind).upper()


def draw_fixed_point_orbits(
    ax,
    orbits,
    *,
    identity_to_color: dict[str, str] | None = None,
    colors: Sequence[str] = SECTION_ORBIT_COLORS,
    x_color: str | None = None,
    o_color: str | None = None,
    other_color: str = "0.24",
    x_marker: str = "X",
    o_marker: str = "o",
    other_marker: str = "D",
    x_marker_size: float = 62.0,
    o_marker_size: float = 34.0,
    other_marker_size: float = 34.0,
    edge_color: str = "white",
    linewidth: float = 1.05,
    stroke_width: float = 2.0,
    show_labels: bool = False,
    label_template: str = "{kind}{orbit_id}:P{map_order}",
    label_map_order_key: str | Sequence[str] = _MAP_POWER_METADATA_KEYS,
    label_fontsize: float = 5.4,
    label_color: str | None = None,
    label_offset: tuple[float, float] | float = (0.006, 0.006),
    label_offset_mode: str = "fixed",
    label_clip_on: bool = False,
    label_only: set[int] | Sequence[int] | None = None,
    label_stride: int = 1,
    zorder: int = 9,
):
    """Draw fixed-point orbit intersections with stable cycle/order labels."""

    from matplotlib import patheffects as pe

    if identity_to_color is None:
        identity_to_color = {}
    if isinstance(label_offset, (int, float)):
        dx = float(label_offset)
        dy = float(label_offset)
    else:
        dx = float(label_offset[0])
        dy = float(label_offset[1])
    label_only_set = None if label_only is None else {int(v) for v in label_only}
    label_stride = max(1, int(label_stride))

    artists = []
    for orbit_index, orbit in enumerate(orbit_list(orbits)):
        identity, orbit_id = orbit_identity(orbit, orbit_index)
        orbit_kind = _fixed_point_kind(orbit)
        if orbit_kind == "X" and x_color is not None:
            color = x_color
        elif orbit_kind == "O" and o_color is not None:
            color = o_color
        else:
            if identity not in identity_to_color:
                identity_to_color[identity] = colors[len(identity_to_color) % len(colors)]
            color = identity_to_color.get(identity, other_color)
        if orbit_kind == "X":
            marker = x_marker
            marker_size = x_marker_size
        elif orbit_kind == "O":
            marker = o_marker
            marker_size = o_marker_size
        else:
            marker = other_marker
            marker_size = other_marker_size

        R, Z = orbit_point_arrays(orbit)
        if R.size == 0:
            continue
        R_mid = float(np.nanmedian(R)) if np.any(np.isfinite(R)) else 0.0
        Z_mid = float(np.nanmedian(Z)) if np.any(np.isfinite(Z)) else 0.0
        artists.append(ax.scatter(
            R,
            Z,
            s=float(marker_size),
            marker=marker,
            c=color,
            edgecolors=edge_color,
            linewidths=float(linewidth),
            zorder=int(zorder),
            path_effects=[pe.withStroke(linewidth=float(stroke_width), foreground=edge_color)],
        ))

        if not show_labels:
            continue
        for fp in getattr(orbit, "points", ()):
            metadata = getattr(fp, "metadata", {}) or {}
            map_order = map_order_value(fp, metadata, label_map_order_key, default="")
            try:
                map_order_int = int(map_order)
            except (TypeError, ValueError):
                map_order_int = None
            if label_only_set is not None and map_order_int not in label_only_set:
                continue
            if map_order_int is not None and map_order_int % label_stride != 0:
                continue
            point_kind = _fixed_point_kind(orbit, fp)
            label = label_template.format(
                orbit_id=orbit_id,
                index=map_order,
                map_order=map_order,
                map_power=map_order,
                poincare_map_power=map_order,
                kind=point_kind,
                identity=identity,
            )
            if str(label_offset_mode).lower() in {"median", "radial", "radial_median"}:
                xoff = dx if float(fp.R) >= R_mid else -dx
                yoff = dy if float(fp.Z) >= Z_mid else -dy
                ha = "left" if xoff >= 0.0 else "right"
                va = "bottom" if yoff >= 0.0 else "top"
            else:
                xoff = dx
                yoff = dy
                ha = "left"
                va = "bottom"
            artists.append(ax.text(
                float(fp.R) + xoff,
                float(fp.Z) + yoff,
                label,
                color=color if label_color is None else label_color,
                fontsize=float(label_fontsize),
                ha=ha,
                va=va,
                clip_on=bool(label_clip_on),
                zorder=int(zorder) + 1,
                path_effects=[pe.withStroke(linewidth=1.7, foreground="white")],
            ))
    return artists


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
                map_power = map_order_value(fp, metadata, label_map_power_key)
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
        "u_R" in value
        or "u_Z" in value
        or "s_R" in value
        or "s_Z" in value
        or ("R" in value and "Z" in value)
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
    if is_manifold_payload(manifolds_by_section):
        return manifold_list(manifolds_by_section)
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


def _iter_payloads(value):
    if value is None:
        return
    if is_manifold_payload(value):
        yield value
        return
    if isinstance(value, Mapping):
        for subvalue in value.values():
            yield from _iter_payloads(subvalue)
        return
    try:
        iterator = iter(value)
    except TypeError:
        return
    for subvalue in iterator:
        yield from _iter_payloads(subvalue)


def branch_payload_smax(payloads, *, s_key: str = "s") -> float | None:
    """Return the maximum finite arclength for generic branch payloads."""

    values: list[np.ndarray] = []
    for payload in _iter_payloads(payloads):
        arr = np.asarray(payload.get(s_key, []), dtype=float).ravel()
        if arr.size:
            values.append(arr)
    if not values:
        return None
    all_values = np.concatenate(values)
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _branch_kind_from_payload(payload: Mapping, *, kind_key: str) -> str:
    raw = str(payload.get(kind_key, payload.get("kind", ""))).strip().lower()
    if raw in {"u", "wu", "w^u", "unstable"} or "unstable" in raw:
        return "unstable"
    if raw in {"s", "ws", "w^s", "stable"} or "stable" in raw:
        return "stable"
    branch_id = str(payload.get("branch_id", "")).strip().lower()
    if branch_id.startswith("wu"):
        return "unstable"
    if branch_id.startswith("ws"):
        return "stable"
    return raw or "unknown"


def _branch_side_from_payload(payload: Mapping, *, side_key: str):
    if side_key in payload:
        return payload.get(side_key)
    branch_id = str(payload.get("branch_id", ""))
    if branch_id.endswith("+"):
        return "+"
    if branch_id.endswith("-"):
        return "-"
    return None


def _side_linestyle(side):
    raw = str(side).strip().lower()
    if raw in {"-1", "-", "minus", "negative", "neg"}:
        return (0.0, (3.4, 2.0))
    return "solid"


def draw_branch_manifold_lines(
    ax,
    payloads,
    *,
    R_key: str = "R",
    Z_key: str = "Z",
    s_key: str = "s",
    kind_key: str = "branch_kind",
    side_key: str = "side",
    unstable_cmap: str = "YlOrRd",
    stable_cmap: str = "Blues",
    unknown_cmap: str = "viridis",
    smax: float | None = None,
    lw: float = 1.05,
    alpha_start: float = 0.92,
    alpha_end: float = 0.58,
    cmap_min: float = 0.32,
    cmap_max: float = 0.94,
    linestyle_by_side: bool = True,
    max_arclength: float | None = None,
    max_segment_length: float | None = None,
    min_path_length: float = 0.0,
    point_size: float = 0.0,
    include_scalar_mappables: bool = False,
    zorder: int = 6,
):
    """Draw generic stable/unstable branch payloads as arclength-colored lines."""

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    payload_list = list(_iter_payloads(payloads))
    if smax is None:
        smax = branch_payload_smax(payload_list, s_key=s_key)
    if smax is None or not np.isfinite(float(smax)) or float(smax) <= 0.0:
        smax = 1.0
    smax = float(smax)
    alpha_start = float(alpha_start)
    alpha_end = float(alpha_end)
    cmap_min = float(cmap_min)
    cmap_max = float(cmap_max)

    artists = []
    used_kinds: set[str] = set()
    cmap_by_kind: dict[str, str] = {}
    for payload in payload_list:
        R = np.asarray(payload.get(R_key, []), dtype=float).ravel()
        Z = np.asarray(payload.get(Z_key, []), dtype=float).ravel()
        s = np.asarray(payload.get(s_key, []), dtype=float).ravel()
        if R.size < 2 or Z.size != R.size or s.shape != R.shape:
            continue
        finite = np.isfinite(R) & np.isfinite(Z) & np.isfinite(s)
        if max_arclength is not None:
            finite &= s <= float(max_arclength)
        if np.count_nonzero(finite) < 2:
            continue
        points = np.column_stack([R, Z])
        segment_finite = finite[:-1] & finite[1:]
        ds = np.diff(s)
        segment_finite &= ds >= -1.0e-12
        length = np.hypot(np.diff(R), np.diff(Z))
        if max_segment_length is not None:
            segment_finite &= length <= float(max_segment_length)
        if min_path_length > 0.0:
            if float(np.sum(length[segment_finite])) < float(min_path_length):
                continue
        if not np.any(segment_finite):
            continue

        kind = _branch_kind_from_payload(payload, kind_key=kind_key)
        cmap_name = stable_cmap if kind == "stable" else unstable_cmap if kind == "unstable" else unknown_cmap
        cmap_by_kind[kind] = cmap_name
        cmap = plt.get_cmap(cmap_name)
        values = 0.5 * (s[:-1] + s[1:])
        t = np.clip(values / smax, 0.0, 1.0)
        rgba = cmap(cmap_min + (cmap_max - cmap_min) * t)
        rgba[:, 3] = np.clip(alpha_start + (alpha_end - alpha_start) * t, 0.0, 1.0)
        segments = np.stack([points[:-1], points[1:]], axis=1)
        linestyle = _side_linestyle(_branch_side_from_payload(payload, side_key=side_key)) if linestyle_by_side else "solid"
        lc = LineCollection(
            segments[segment_finite],
            colors=rgba[segment_finite],
            linewidths=float(lw),
            linestyles=linestyle,
            zorder=int(zorder),
        )
        ax.add_collection(lc)
        artists.append(lc)
        if float(point_size) > 0.0:
            point_t = np.clip(s / smax, 0.0, 1.0)
            point_rgba = cmap(cmap_min + (cmap_max - cmap_min) * point_t)
            point_rgba[:, 3] = np.clip(0.34 + 0.18 * (alpha_end - alpha_start) * point_t, 0.08, 0.34)
            artists.append(
                ax.scatter(
                    R[finite],
                    Z[finite],
                    s=float(point_size),
                    c=point_rgba[finite],
                    linewidths=0,
                    rasterized=True,
                    zorder=int(zorder) + 1,
                )
            )
        used_kinds.add(kind)
    if include_scalar_mappables and artists:
        from matplotlib import colors as mcolors

        norm = mcolors.Normalize(vmin=0.0, vmax=smax)
        for kind in ("stable", "unstable", "unknown"):
            if kind not in used_kinds:
                continue
            sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_by_kind.get(kind, unknown_cmap)))
            sm.set_array([])
            artists.insert(0, sm)
    return artists


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
            point_side = np.asarray(manifold.get(f"{prefix}_point_side", []), dtype=float).ravel()
            if point_side.shape != R.shape:
                continue
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
            if "R" in manifold and "Z" in manifold:
                Rm = np.asarray(manifold.get("R", []), dtype=float).ravel()
                Zm = np.asarray(manifold.get("Z", []), dtype=float).ravel()
                if Rm.size and Zm.size:
                    R_parts.append(Rm)
                    Z_parts.append(Zm)
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
    "branch_payload_smax",
    "create_section_grid",
    "orbit_identity",
    "orbit_list",
    "orbit_point_arrays",
    "orbits_for_section",
    "draw_axis_point",
    "draw_branch_manifold_lines",
    "draw_fixed_point_orbits",
    "draw_orbit_points",
    "draw_manifold_lines",
    "draw_manifold_origins",
    "draw_manifold_points",
    "draw_poincare_background",
    "draw_poincare_points",
    "draw_wall_section",
    "format_section_axis",
    "is_manifold_payload",
    "map_order_value",
    "manifold_list",
    "manifold_lpol_max",
    "manifolds_for_section",
    "save_figure",
    "section_data_limits",
    "trim_compact_tick_labels",
]
