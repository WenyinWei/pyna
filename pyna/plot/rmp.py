"""Plot helpers for analytic RMP resonance tutorials.

The functions in this module consume plain arrays plus the lightweight
``ResonantComponent`` protocol from :mod:`pyna.toroidal.visual.RMP_spectrum`.
They are intentionally plotting-only: field-line tracing and Fourier analysis
remain in the toroidal modules, while section layout and overlays live in
``pyna.plot``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from pyna.plot.section_geometry import (
    create_section_grid,
    draw_poincare_points,
    format_section_axis,
    trim_compact_tick_labels,
)
from pyna.plot.xo_points import draw_xo_points


DEFAULT_RMP_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
)


def _component_color(component, index: int, colors: Sequence[str] | None = None) -> str:
    palette = tuple(colors or DEFAULT_RMP_COLORS)
    if not palette:
        return "#2563eb"
    order = int(getattr(component, "harmonic_order", index + 1)) - 1
    return str(palette[order % len(palette)])


def _fixed_point_angles(component, phi: float) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(component, "fixed_points"):
        pts = component.fixed_points(float(phi))
    else:
        from pyna.toroidal.visual.RMP_spectrum import island_fixed_points

        pts = island_fixed_points(
            int(component.m),
            int(component.n),
            complex(component.b_mn),
            float(phi),
            int(getattr(component, "q_prime_sign", 1)),
        )
    return (
        np.asarray(pts["theta_O"][0], dtype=float),
        np.asarray(pts["theta_X"][0], dtype=float),
    )


def _point(R0: float, radius: float, theta: float, *, stable=True):
    tangent = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    radial = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    return SimpleNamespace(
        R=float(R0 + radius * np.cos(theta)),
        Z=float(radius * np.sin(theta)),
        stable_eigenvec=tangent if stable else None,
        unstable_eigenvec=radial,
    )


def draw_pest_grid(
    ax,
    eq,
    *,
    radial_values: Sequence[float] | None = None,
    theta_values: Sequence[float] | None = None,
    color: str = "0.42",
    lw: float = 0.55,
    alpha: float = 0.28,
    zorder: int = 1,
):
    """Overlay a PEST-style ``(S, theta*)`` grid on a circular analytic section.

    For the simple analytic stellarator used in the public RMP tutorial,
    ``S = sqrt(psi)`` labels circular flux surfaces and the straight-field-line
    poloidal angle is represented by the geometric angle on the section.
    """

    R0 = float(eq.R0)
    r0 = float(eq.r0)
    radial = np.asarray(
        radial_values if radial_values is not None else np.linspace(0.2, 0.95, 5),
        dtype=float,
    )
    theta = np.asarray(
        theta_values
        if theta_values is not None
        else np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
        dtype=float,
    )
    loop = np.linspace(0.0, 2.0 * np.pi, 320)
    artists = []

    for S in radial:
        radius = np.clip(float(S), 0.0, 1.05) * r0
        artists.extend(
            ax.plot(
                R0 + radius * np.cos(loop),
                radius * np.sin(loop),
                color=color,
                lw=lw,
                alpha=alpha,
                zorder=zorder,
            )
        )

    for th in theta:
        s = np.linspace(0.05, 0.98, 80)
        artists.extend(
            ax.plot(
                R0 + r0 * s * np.cos(float(th)),
                r0 * s * np.sin(float(th)),
                color=color,
                lw=lw,
                alpha=alpha,
                zorder=zorder,
            )
        )
    return artists


def draw_rmp_fixed_points(
    ax,
    components,
    eq,
    phi: float,
    *,
    colors: Sequence[str] | None = None,
    show_arrows: bool = True,
):
    """Draw analytic RMP O-points, X-points and local eigendirections."""

    R0 = float(eq.R0)
    r0 = float(eq.r0)
    all_x = []
    all_o = []

    for idx, component in enumerate(components):
        color = _component_color(component, idx, colors)
        r_res = np.sqrt(float(component.psi_res)) * r0
        theta_o, theta_x = _fixed_point_angles(component, phi)
        opts = [_point(R0, r_res, th, stable=False) for th in theta_o]
        xpts = [_point(R0, r_res, th, stable=True) for th in theta_x]
        all_o.extend(opts)
        all_x.extend(xpts)
        draw_xo_points(
            ax,
            xpts,
            opts,
            {
                "xpt_color": color,
                "xpt_ms": 6.6,
                "xpt_mew": 1.7,
                "opt_color": color,
                "opt_ms": 5.2,
                "opt_mew": 1.0,
                "arrow_len": 0.014,
                "arrow_lw": 1.0,
                "arrow_stable": "#0f766e",
                "arrow_unstable": "#b45309",
            },
            show_arrows=show_arrows,
        )
    return {"x_points": all_x, "o_points": all_o}


def draw_reduced_stable_manifolds(
    ax,
    components,
    eq,
    phi: float,
    *,
    colors: Sequence[str] | None = None,
    theta_samples: int = 96,
    lw: float = 1.15,
    alpha: float = 0.72,
    zorder: int = 8,
):
    """Draw reduced-island separatrix branches through the analytic X-points.

    The curves are the local island-Hamiltonian separatrix approximation.  They
    start and end at neighbouring X-points and reach maximum radial excursion at
    the O-point; for the tutorial this is the stable/unstable manifold skeleton
    predicted by the resonant component.
    """

    R0 = float(eq.R0)
    r0 = float(eq.r0)
    artists = []
    for idx, component in enumerate(components):
        color = _component_color(component, idx, colors)
        m = max(1, int(component.m))
        r_res = np.sqrt(float(component.psi_res)) * r0
        half_width = float(getattr(component, "half_width_r", 0.0))
        if not np.isfinite(half_width) or half_width <= 0.0:
            continue
        theta_o, _theta_x = _fixed_point_angles(component, phi)
        local = np.linspace(-np.pi / m, np.pi / m, int(theta_samples))
        envelope = np.clip(np.cos(0.5 * m * local), 0.0, None)
        for center in theta_o:
            theta = float(center) + local
            for sign in (-1.0, 1.0):
                radius = np.maximum(0.002 * r0, r_res + sign * half_width * envelope)
                artists.extend(
                    ax.plot(
                        R0 + radius * np.cos(theta),
                        radius * np.sin(theta),
                        color=color,
                        lw=lw,
                        alpha=alpha,
                        zorder=zorder,
                    )
                )
    return artists


def draw_resonant_surfaces(
    ax,
    components,
    eq,
    *,
    colors: Sequence[str] | None = None,
    lw: float = 0.75,
    alpha: float = 0.45,
    zorder: int = 4,
):
    """Draw resonant flux-surface guide curves for RMP components."""

    R0 = float(eq.R0)
    r0 = float(eq.r0)
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    artists = []
    for idx, component in enumerate(components):
        color = _component_color(component, idx, colors)
        radius = np.sqrt(float(component.psi_res)) * r0
        artists.extend(
            ax.plot(
                R0 + radius * np.cos(theta),
                radius * np.sin(theta),
                color=color,
                lw=lw,
                ls="--",
                alpha=alpha,
                zorder=zorder,
            )
        )
    return artists


def draw_rmp_resonance_section(
    ax,
    R,
    Z,
    *,
    eq,
    components,
    phi: float = 0.0,
    colors: Sequence[str] | None = None,
    title: str | None = None,
    show_pest_grid: bool = True,
    show_resonant_surfaces: bool = True,
    show_manifolds: bool = True,
    show_xo: bool = True,
    point_size: float = 2.0,
    point_alpha: float = 0.45,
    cmap: str = "viridis",
):
    """Draw one modern RMP Poincare section with topology overlays."""

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if show_pest_grid:
        draw_pest_grid(ax, eq)
    values = None
    if R_arr.size and Z_arr.size:
        values = np.clip(
            ((R_arr - float(eq.R0)) ** 2 + Z_arr**2) / float(eq.r0) ** 2,
            0.0,
            1.0,
        )
        draw_poincare_points(
            ax,
            R_arr,
            Z_arr,
            values=values,
            cmap=cmap,
            point_size=point_size,
            alpha=point_alpha,
            rasterized=False,
            zorder=3,
        )
    if show_resonant_surfaces:
        draw_resonant_surfaces(ax, components, eq, colors=colors)
    if show_manifolds:
        draw_reduced_stable_manifolds(ax, components, eq, phi, colors=colors)
    markers = {"x_points": [], "o_points": []}
    if show_xo:
        markers = draw_rmp_fixed_points(ax, components, eq, phi, colors=colors)

    lim = 1.15 * float(eq.r0)
    ax.set_xlim(float(eq.R0) - lim, float(eq.R0) + lim)
    ax.set_ylim(-lim, lim)
    format_section_axis(ax, section_phi=phi, title=title, grid=False)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return {"psi_values": values, **markers}


def plot_rmp_resonance_sections(
    sections_data: Sequence[Mapping[str, Sequence[float]]],
    phi_sections: Sequence[float],
    *,
    eq,
    components,
    colors: Sequence[str] | None = None,
    ncols: int = 3,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    point_size: float = 1.7,
    point_alpha: float = 0.42,
    cmap: str = "viridis",
):
    """Draw a compact multi-section RMP resonance figure."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    lim = 1.15 * float(eq.r0)
    data_limits = (float(eq.R0) - lim, float(eq.R0) + lim, -lim, lim)
    fig, axes = create_section_grid(
        phi_sections,
        ncols=ncols,
        figsize=figsize,
        data_limits=data_limits,
        panel_height=3.05,
        compact=False,
        share_axes=True,
    )
    axes_flat = axes.ravel()
    for idx, phi in enumerate(phi_sections):
        ax = axes_flat[idx]
        section = sections_data[idx] if idx < len(sections_data) else {}
        draw_rmp_resonance_section(
            ax,
            section.get("R", []),
            section.get("Z", []),
            eq=eq,
            components=components,
            phi=float(phi),
            colors=colors,
            title=rf"$\phi={np.degrees(float(phi)):.0f}^\circ$",
            point_size=point_size,
            point_alpha=point_alpha,
            cmap=cmap,
        )
        row, col = divmod(idx, int(ncols))
        nrows = axes.shape[0]
        if row < nrows - 1:
            ax.set_xlabel("")
        if col != 0:
            ax.set_ylabel("")
    trim_compact_tick_labels(axes, len(phi_sections), ncols=ncols)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0.0, 1.0))
    sm.set_array([])
    fig.subplots_adjust(bottom=0.16, top=0.88, right=0.88, wspace=0.08, hspace=0.28)
    cax = fig.add_axes([0.902, 0.24, 0.018, 0.52])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("normalized flux label")

    handles = []
    for idx, component in enumerate(components):
        color = _component_color(component, idx, colors)
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=1.5,
                label=f"({int(component.m)},{int(component.n)}) q={float(component.q_res):.2f}",
            )
        )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.35",
                markeredgecolor="k",
                markersize=5.5,
                label="O-point",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="0.35",
                markersize=6.5,
                markeredgewidth=1.5,
                linestyle="None",
                label="X-point",
            ),
            Line2D(
                [0],
                [0],
                color="0.35",
                lw=0.75,
                linestyle="--",
                label="resonant surface",
            ),
            Line2D(
                [0],
                [0],
                color=_component_color(components[0], 0, colors) if components else "#2563eb",
                lw=1.2,
                label="local stable branch",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 5),
        frameon=False,
        fontsize=8,
    )
    if title:
        fig.suptitle(title, fontsize=12, y=0.99)
    return fig, axes


__all__ = [
    "draw_pest_grid",
    "draw_reduced_stable_manifolds",
    "draw_resonant_surfaces",
    "draw_rmp_fixed_points",
    "draw_rmp_resonance_section",
    "plot_rmp_resonance_sections",
]
