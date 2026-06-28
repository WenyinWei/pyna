"""Convenience composition for boundary-island section plots."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from pyna.plot.section_geometry import (
    apply_section_limits,
    create_section_grid,
    orbits_for_section,
    draw_axis_point,
    draw_orbit_points,
    draw_manifold_points,
    draw_poincare_background,
    draw_wall_section,
    format_section_axis,
    manifold_lpol_max,
    manifolds_for_section,
    save_figure,
    section_data_limits,
    trim_compact_tick_labels,
)


def plot_boundary_island_sections(
    section_phis: Sequence[float],
    *,
    background=None,
    section_orbits=None,
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
    orbit_marker_size: float = 76.0,
    label_orbit_ids: bool = False,
    manifold_size: float = 5.0,
    manifold_alpha: float = 0.78,
    manifold_cmap: str = "viridis",
    manifold_vmax: float | None = None,
    axis_limits: tuple[float, float, float, float] | None = None,
    pad_fraction: float = 0.035,
):
    """Plot a boundary-island overlay using generic section primitives.

    This wrapper is intentionally thin.  Callers that need a different plot
    composition can use the primitives in :mod:`pyna.plot.section_geometry`
    directly for core surfaces, edge traces, wall overlays, fixed-point orbits,
    and stable/unstable manifolds.
    """

    phi = np.asarray(section_phis, dtype=float).ravel()
    if phi.size == 0:
        raise ValueError("section_phis must not be empty")
    if walls is not None and len(walls) != phi.size:
        raise ValueError("walls must match section_phis length")
    if axis_by_section is not None and len(axis_by_section) != phi.size:
        raise ValueError("axis_by_section must match section_phis length")

    limits = axis_limits
    if limits is None and share_axes:
        limits = section_data_limits(
            section_phis=phi,
            background=background,
            section_orbits=section_orbits,
            manifolds_by_section=manifolds_by_section,
            walls=walls,
            pad_fraction=pad_fraction,
        )
    fig, axes = create_section_grid(
        phi,
        ncols=ncols,
        figsize=figsize,
        data_limits=limits,
        compact=compact,
        share_axes=share_axes,
        aspect_ratio=aspect_ratio,
    )
    manifold_vmax = (
        manifold_lpol_max(manifolds_by_section, phi)
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
            draw_wall_section(ax, walls[flat_idx][0], walls[flat_idx][1])

        if background is not None:
            draw_poincare_background(
                ax,
                background,
                flat_idx,
                point_size=point_size,
                alpha=point_alpha,
            )

        draw_manifold_points(
            ax,
            manifolds_for_section(manifolds_by_section, section_phi, flat_idx),
            point_size=manifold_size,
            alpha=manifold_alpha,
            cmap=manifold_cmap,
            vmax=manifold_vmax,
        )

        draw_orbit_points(
            ax,
            orbits_for_section(section_orbits, section_phi, flat_idx),
            identity_to_color=identity_to_color,
            marker_size=orbit_marker_size,
            label_orbit_ids=label_orbit_ids,
        )

        if axis_by_section is not None:
            draw_axis_point(ax, axis_by_section[flat_idx][0], axis_by_section[flat_idx][1])

        format_section_axis(
            ax,
            section_phi=section_phi,
            title_inside=bool(compact),
            aspect_ratio=aspect_ratio,
        )

    apply_section_limits(axes, limits)
    if compact:
        trim_compact_tick_labels(axes, int(phi.size), ncols=ncols)
    save_figure(fig, out_path, dpi=dpi)
    return fig, axes


__all__ = ["plot_boundary_island_sections"]
