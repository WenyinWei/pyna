"""Compact multi-section plots for boundary island cycles."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


_CYCLE_COLORS = (
    "#c62828",
    "#1565c0",
    "#2e7d32",
    "#6a1b9a",
    "#ef6c00",
    "#00838f",
    "#ad1457",
    "#455a64",
)


def _cycle_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "cycles"):
        return list(value.cycles)
    if hasattr(value, "points"):
        return [value]
    return list(value)


def _section_cycles_for_section(section_cycles, phi: float, section_index: int) -> list:
    if section_cycles is None:
        return []
    if isinstance(section_cycles, Mapping):
        if phi in section_cycles:
            return _cycle_list(section_cycles[phi])
        if section_cycles:
            key = min(section_cycles, key=lambda p: abs(float(p) - float(phi)))
            return _cycle_list(section_cycles[key])
        return []
    if len(section_cycles) == 0:
        return []
    first = section_cycles[0]
    if hasattr(first, "cycles") or hasattr(first, "points"):
        return _cycle_list(section_cycles)
    return _cycle_list(section_cycles[int(section_index)])


def _is_manifold_payload(value) -> bool:
    return isinstance(value, Mapping) and (
        "u_R" in value or "u_Z" in value or "s_R" in value or "s_Z" in value
    )


def _manifold_list(value) -> list:
    if value is None:
        return []
    if _is_manifold_payload(value):
        return [value]
    return list(value)


def _section_manifolds_for_section(manifolds_by_section, phi: float, section_index: int) -> list:
    if manifolds_by_section is None:
        return []
    if isinstance(manifolds_by_section, Mapping) and not _is_manifold_payload(manifolds_by_section):
        if phi in manifolds_by_section:
            return _manifold_list(manifolds_by_section[phi])
        if manifolds_by_section:
            key = min(manifolds_by_section, key=lambda p: abs(float(p) - float(phi)))
            return _manifold_list(manifolds_by_section[key])
        return []
    if len(manifolds_by_section) == 0:
        return []
    first = manifolds_by_section[0]
    if _is_manifold_payload(first):
        return _manifold_list(manifolds_by_section)
    return _manifold_list(manifolds_by_section[int(section_index)])


def _cycle_identity(cycle, fallback: int) -> tuple[str, int]:
    metadata = getattr(cycle, "metadata", {}) or {}
    key = metadata.get("same_cycle_key")
    cycle_id = getattr(cycle, "cycle_id", None)
    if key is None:
        key = f"cycle={cycle_id if cycle_id is not None else fallback}:kind={getattr(cycle, 'kind', '')}"
    if cycle_id is None:
        cycle_id = fallback
    return str(key), int(cycle_id)


def _point_arrays(cycle) -> tuple[np.ndarray, np.ndarray]:
    points = tuple(getattr(cycle, "points", ()) or ())
    return (
        np.asarray([float(fp.R) for fp in points], dtype=float),
        np.asarray([float(fp.Z) for fp in points], dtype=float),
    )


def _limits_from_artists(
    *,
    phi_sections: Sequence[float],
    background,
    section_cycles,
    manifolds_by_section,
    walls,
    pad_fraction: float,
) -> tuple[float, float, float, float] | None:
    R_parts: list[np.ndarray] = []
    Z_parts: list[np.ndarray] = []
    for i, phi in enumerate(phi_sections):
        if background is not None:
            Rb, Zb, _seed_idx = background.section_points(i)
            if Rb.size:
                R_parts.append(np.asarray(Rb, dtype=float))
                Z_parts.append(np.asarray(Zb, dtype=float))
        for cycle in _section_cycles_for_section(section_cycles, float(phi), i):
            R, Z = _point_arrays(cycle)
            if R.size:
                R_parts.append(R)
                Z_parts.append(Z)
        for manifold in _section_manifolds_for_section(manifolds_by_section, float(phi), i):
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


def _manifold_lpol_max(manifolds_by_section, phi_sections: Sequence[float]) -> float | None:
    values: list[np.ndarray] = []
    for i, phi in enumerate(phi_sections):
        for manifold in _section_manifolds_for_section(manifolds_by_section, float(phi), i):
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


def plot_boundary_island_sections(
    section_phis: Sequence[float],
    *,
    background=None,
    section_cycles=None,
    manifolds_by_section=None,
    walls: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    axis_by_section: Sequence[tuple[float, float]] | None = None,
    out_path: str | Path | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    dpi: int = 180,
    compact: bool = True,
    share_axes: bool = True,
    aspect_ratio: float = 1.0,
    point_size: float = 4.0,
    point_alpha: float = 0.30,
    cycle_marker_size: float = 76.0,
    label_cycle_ids: bool = False,
    manifold_size: float = 5.0,
    manifold_alpha: float = 0.78,
    manifold_cmap: str = "viridis",
    manifold_vmax: float | None = None,
    axis_limits: tuple[float, float, float, float] | None = None,
    pad_fraction: float = 0.035,
):
    """Plot same-orbit Poincare backgrounds and boundary island cycles.

    ``section_cycles`` may be a mapping ``phi -> BoundaryIslandChain`` or
    ``phi -> sequence[BoundaryIslandCycle]``.  Points belonging to the same
    deduplicated cycle keep the same color across every section.  The plotted
    marker identity is derived from ``same_cycle_key``/``cycle_id`` metadata
    when present.  ``manifolds_by_section`` may provide the payload returned by
    ``trace_fixed_point_manifolds_multi_section_field``; ``u_lpol``/``s_lpol``
    values are used as colors when present.
    """

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe

    phi = np.asarray(section_phis, dtype=float).ravel()
    if phi.size == 0:
        raise ValueError("section_phis must not be empty")
    if walls is not None and len(walls) != phi.size:
        raise ValueError("walls must match section_phis length")
    if axis_by_section is not None and len(axis_by_section) != phi.size:
        raise ValueError("axis_by_section must match section_phis length")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(phi.size / ncols))
    if figsize is None:
        figsize = (3.15 * ncols, 3.15 * nrows)
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

    global_limits = axis_limits
    if global_limits is None and share_axes:
        global_limits = _limits_from_artists(
            phi_sections=phi,
            background=background,
            section_cycles=section_cycles,
            manifolds_by_section=manifolds_by_section,
            walls=walls,
            pad_fraction=pad_fraction,
        )

    seed_cmap = plt.get_cmap("tab20")
    manifold_vmax = (
        _manifold_lpol_max(manifolds_by_section, phi)
        if manifold_vmax is None
        else float(manifold_vmax)
    )
    identity_to_color: dict[str, str] = {}

    for flat_idx, ax in enumerate(axes.ravel()):
        if flat_idx >= phi.size:
            ax.set_visible(False)
            continue
        section_phi = float(phi[flat_idx])
        if walls is not None:
            wall_R, wall_Z = walls[flat_idx]
            ax.plot(np.asarray(wall_R, dtype=float), np.asarray(wall_Z, dtype=float),
                    color="0.34", lw=1.0, alpha=0.82, zorder=1)

        if background is not None:
            Rb, Zb, seed_idx = background.section_points(flat_idx)
            if Rb.size:
                ax.scatter(
                    Rb,
                    Zb,
                    s=float(point_size),
                    c=seed_cmap(np.mod(seed_idx, seed_cmap.N)),
                    alpha=float(point_alpha),
                    linewidths=0,
                    rasterized=True,
                    zorder=2,
                )

        for manifold in _section_manifolds_for_section(manifolds_by_section, section_phi, flat_idx):
            for prefix in ("u", "s"):
                Rm = np.asarray(manifold.get(f"{prefix}_R", []), dtype=float).ravel()
                Zm = np.asarray(manifold.get(f"{prefix}_Z", []), dtype=float).ravel()
                if Rm.size == 0 or Zm.size == 0:
                    continue
                lpol = np.asarray(manifold.get(f"{prefix}_lpol", []), dtype=float).ravel()
                finite = np.isfinite(Rm) & np.isfinite(Zm)
                if lpol.shape == Rm.shape:
                    finite &= np.isfinite(lpol)
                    ax.scatter(
                        Rm[finite],
                        Zm[finite],
                        s=float(manifold_size),
                        c=lpol[finite],
                        cmap=manifold_cmap,
                        vmin=0.0,
                        vmax=manifold_vmax,
                        alpha=float(manifold_alpha),
                        linewidths=0,
                        rasterized=True,
                        zorder=5,
                    )
                else:
                    color = "#43a047" if prefix == "u" else "#e64a19"
                    ax.scatter(
                        Rm[finite],
                        Zm[finite],
                        s=float(manifold_size),
                        c=color,
                        alpha=float(manifold_alpha),
                        linewidths=0,
                        rasterized=True,
                        zorder=5,
                    )

        for cycle_index, cycle in enumerate(_section_cycles_for_section(section_cycles, section_phi, flat_idx)):
            identity, cycle_id = _cycle_identity(cycle, cycle_index)
            if identity not in identity_to_color:
                identity_to_color[identity] = _CYCLE_COLORS[len(identity_to_color) % len(_CYCLE_COLORS)]
            color = identity_to_color[identity]
            marker = "X" if str(getattr(cycle, "kind", "")).upper() == "X" else "o"
            R, Z = _point_arrays(cycle)
            if R.size == 0:
                continue
            ax.scatter(
                R,
                Z,
                s=float(cycle_marker_size),
                marker=marker,
                c=color,
                edgecolors="white",
                linewidths=1.1,
                zorder=8,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            )
            if label_cycle_ids:
                for fp in getattr(cycle, "points", ()):
                    idx = getattr(fp, "metadata", {}).get("orbit_point_index", getattr(fp, "metadata", {}).get("point_index", ""))
                    ax.text(
                        float(fp.R),
                        float(fp.Z),
                        f"{cycle_id}:{idx}",
                        color=color,
                        fontsize=5.6,
                        ha="left",
                        va="bottom",
                        zorder=9,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")],
                    )

        if axis_by_section is not None:
            ax.plot(float(axis_by_section[flat_idx][0]), float(axis_by_section[flat_idx][1]),
                    marker="+", color="0.05", ms=5.0, mew=1.0, ls="None", zorder=7)

        ax.set_title(rf"$\phi/\pi={section_phi / np.pi:.2f}$", fontsize=9, pad=2.0)
        ax.set_aspect(float(aspect_ratio), adjustable="box")
        ax.grid(True, lw=0.28, color="0.85", alpha=0.58)
        if global_limits is not None:
            ax.set_xlim(global_limits[0], global_limits[1])
            ax.set_ylim(global_limits[2], global_limits[3])
        if compact:
            if flat_idx % ncols != 0:
                ax.tick_params(labelleft=False)
            if flat_idx < (nrows - 1) * ncols:
                ax.tick_params(labelbottom=False)
    if out_path is not None:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=int(dpi))
    return fig, axes


__all__ = ["plot_boundary_island_sections"]
