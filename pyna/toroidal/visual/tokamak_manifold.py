"""Publication-quality tokamak manifold visualization for pyna.

Provides standalone, composable plotting functions for EAST-tokamak
research figures:

* :func:`plot_equilibrium_cross_section` — Bpol streamplot + ψ contours
* :func:`plot_poincare_orbits`           — Poincaré orbits coloured by ψ_norm
* :func:`plot_xcycle_marker`             — X/O fixed-point markers
* :func:`plot_manifold_1d`              — 1-D Poincaré section of a manifold
                                          coloured by poloidal arc length *s*
                                          (warm = unstable, cool = stable)
* :func:`plot_manifold_bundle`          — full manifold bundle in (R,Z) plane

Colour convention (matches the author's prior notebooks):
  * Unstable manifold (eigenvalue > 1, field-line diverges from X-cycle torus):
    warm colormap — default ``'Oranges'`` or ``'hot_r'``.
  * Stable manifold (0 < eigenvalue < 1, converges to X-cycle torus):
    cool colormap  — default ``'GnBu'`` or ``'winter'``.
  * Hue progression along either manifold arm encodes poloidal arc length *s*
    measured from the X-cycle (s = 0).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    Normalize, PowerNorm, LogNorm, ListedColormap
)
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Sequence, Union

from pyna.topo.manifold import accumulate_s_from_RZ_arr

# ── default style constants ────────────────────────────────────────────────
UNSTABLE_CMAPS = ('Oranges', 'YlOrRd', 'hot_r')   # warm palette family
STABLE_CMAPS   = ('GnBu',   'Blues',   'winter')   # cool palette family
EQUIL_CMAP     = 'magma'
DEFAULT_DPI    = 150

# rcParams suitable for single-column PRL/NF figure
PUBLICATION_RC = {
    'font.family'     : 'serif',
    'font.size'       : 9,
    'axes.labelsize'  : 10,
    'axes.titlesize'  : 10,
    'axes.linewidth'  : 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth' : 1.0,
    'text.usetex'     : False,
    'figure.dpi'      : DEFAULT_DPI,
    'figure.facecolor': 'white',
    'axes.facecolor'  : 'white',
    'savefig.dpi'     : 300,
    'savefig.bbox'    : 'tight',
}


# ────────────────────────────────────────────────────────────────────────────
# 1.  Equilibrium background
# ────────────────────────────────────────────────────────────────────────────

def plot_equilibrium_cross_section(
    eq,
    ax: Optional[plt.Axes] = None,
    *,
    n_surfaces: int = 20,
    psi_min: float = 0.04,
    psi_max: float = 0.96,
    n_grid: int = 300,
    cmap: str = EQUIL_CMAP,
    linewidth_max: float = 1.4,
    streamplot_density: float = 1.8,
    show_colorbar: bool = True,
    show_lcfs: bool = True,
    show_axis: bool = True,
    divertor_RZ: Optional[np.ndarray] = None,
    divertor_kw: Optional[dict] = None,
    limiter_RZ: Optional[np.ndarray] = None,
    limiter_kw: Optional[dict] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw Bpol streamplot + ψ flux-surface contours for an axisymmetric
    equilibrium.

    Parameters
    ----------
    eq:
        Object exposing ``BR(R,Z)``, ``BZ(R,Z)``, ``Bphi(R)`` or
        ``BR_BZ(R,Z)``, ``psi(R,Z)``, ``R0``, ``a``, ``kappa``.
        Compatible with :class:`~pyna.toroidal.equilibrium.Solovev.EquilibriumSolovev`
        and with EFIT-based :class:`EquilibriumAxisym` subclasses.
    ax:
        Target axes.  Created if *None*.
    n_surfaces, psi_min, psi_max:
        Number and ψ_norm range of flux-surface contours.
    n_grid:
        Resolution of the evaluation grid.
    show_colorbar:
        Attach a Bpol colourbar.
    divertor_RZ, limiter_RZ:
        Optional (N,2) arrays with first-wall / divertor coordinates.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5), dpi=DEFAULT_DPI)
    else:
        fig = ax.get_figure()

    R0     = getattr(eq, 'R0', 1.86)
    a      = getattr(eq, 'a',  0.60)
    kappa  = getattr(eq, 'kappa', 1.7)

    R_lo = R0 - 1.25 * a
    R_hi = R0 + 1.25 * a
    Z_lm = 1.25 * a * kappa

    R_arr = np.linspace(R_lo, R_hi, n_grid)
    Z_arr = np.linspace(-Z_lm, Z_lm, n_grid)
    RR, ZZ = np.meshgrid(R_arr, Z_arr)

    # --- vector field -------------------------------------------------------
    if hasattr(eq, 'BR_BZ'):
        BR_ZZ, BZ_ZZ = np.vectorize(eq.BR_BZ, otypes=[float, float])(RR, ZZ)
        if isinstance(BR_ZZ, tuple):
            BR_ZZ, BZ_ZZ = np.array(BR_ZZ), np.array(BZ_ZZ)
    elif hasattr(eq, 'BR') and hasattr(eq, 'BZ'):
        BR_ZZ = np.vectorize(eq.BR)(RR, ZZ)
        BZ_ZZ = np.vectorize(eq.BZ)(RR, ZZ)
    else:
        raise AttributeError("eq must provide BR/BZ or BR_BZ methods")

    Bpol = np.sqrt(BR_ZZ**2 + BZ_ZZ**2)
    finite_mask = np.isfinite(Bpol)
    vmax = float(np.percentile(Bpol[finite_mask], 99.5)) if finite_mask.any() else 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    lw   = np.clip(linewidth_max * Bpol / (vmax + 1e-30), 0.15, linewidth_max)

    strm = ax.streamplot(
        RR, ZZ, BR_ZZ, BZ_ZZ,
        linewidth=lw, color=Bpol, cmap=cmap, norm=norm,
        density=streamplot_density, arrowsize=0.0,
        broken_streamlines=False,
    )

    # --- flux-surface contours ----------------------------------------------
    if hasattr(eq, 'psi'):
        PSI = np.vectorize(eq.psi)(RR, ZZ)
        if hasattr(eq, 'find_opoint'):
            try:
                oR, oZ = eq.find_opoint()
                psi_axis = float(eq.psi(oR, oZ))
            except Exception:
                psi_axis = float(np.nanmin(PSI))
        else:
            psi_axis = float(np.nanmin(PSI))

        if hasattr(eq, 'psi_lcfs'):
            psi_lcfs = float(eq.psi_lcfs())
        else:
            psi_lcfs = float(np.nanpercentile(PSI, 99))

        levels_norm = np.linspace(psi_min, psi_max, n_surfaces)
        levels = psi_axis + levels_norm * (psi_lcfs - psi_axis)
        ax.contour(RR, ZZ, PSI, levels=levels,
                   colors='white', linewidths=0.25, alpha=0.35)

        if show_lcfs:
            ax.contour(RR, ZZ, PSI, levels=[psi_lcfs],
                       colors='white', linewidths=0.9, alpha=0.7,
                       linestyles='--')

    # --- magnetic axis marker -----------------------------------------------
    if show_axis and hasattr(eq, 'find_opoint'):
        try:
            oR, oZ = eq.find_opoint()
            ax.plot(oR, oZ, '+', color='white', ms=9, markeredgewidth=1.5,
                    zorder=10, label='O-point (axis)')
        except Exception:
            pass

    # --- first-wall geometry ------------------------------------------------
    _dkw = dict(color='#607D8B', lw=1.2, zorder=8)
    if divertor_kw:
        _dkw.update(divertor_kw)
    if divertor_RZ is not None:
        ax.plot(divertor_RZ[:, 0], divertor_RZ[:, 1], **_dkw)

    _lkw = dict(color='#455A64', lw=1.0, zorder=8)
    if limiter_kw:
        _lkw.update(limiter_kw)
    if limiter_RZ is not None:
        ax.plot(limiter_RZ[:, 0], limiter_RZ[:, 1], **_lkw)

    # --- colourbar ----------------------------------------------------------
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = fig.colorbar(strm.lines, cax=cax)
        cb.set_label(r'$B_{\rm pol}\ (\mathrm{T})$', fontsize=10)
        cb.ax.tick_params(labelsize=8)

    # --- axis labels --------------------------------------------------------
    ax.set_xlabel(r'$R\ (\mathrm{m})$', fontsize=10)
    ax.set_ylabel(r'$Z\ (\mathrm{m})$', fontsize=10)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=8)
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(R_lo, R_hi)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(-Z_lm, Z_lm)

    return fig, ax


# ────────────────────────────────────────────────────────────────────────────
# 2.  Poincaré orbits
# ────────────────────────────────────────────────────────────────────────────

def plot_poincare_orbits(
    trace_list: list,
    ax: plt.Axes,
    *,
    psi_values: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    s: float = 0.35,
    alpha: float = 0.75,
    rasterized: bool = True,
    zorder: int = 3,
    show_colorbar: bool = False,
    colorbar_label: str = r'$\psi_{\rm norm}$',
) -> ScalarMappable:
    """Scatter Poincaré section orbits coloured by ψ_norm.

    Parameters
    ----------
    trace_list:
        List of ``(N,3)`` arrays ``[R, Z, phi]`` — one per field-line.
        Output of :func:`pyna.topo.manifold._transect_initPhi0_Wivp_at_a_phi`
        or :func:`pyna.flt.load_Poincare_orbits`.
    ax:
        Target axes.
    psi_values:
        1-D array of ψ_norm for each trace (used for colour).  If *None*,
        traces are coloured by list index.
    """
    n = len(trace_list)
    if psi_values is None:
        psi_values = np.linspace(0.0, 1.0, n)

    cmap_obj = plt.colormaps.get_cmap(cmap)
    norm = Normalize(vmin=float(psi_values.min()), vmax=float(psi_values.max()))
    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])

    for trace, psi in zip(trace_list, psi_values):
        arr = np.asarray(trace)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        ax.scatter(arr[:, 0], arr[:, 1],
                   s=s, color=cmap_obj(norm(psi)),
                   alpha=alpha, linewidths=0,
                   rasterized=rasterized, zorder=zorder)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = ax.get_figure().colorbar(sm, cax=cax)
        cb.set_label(colorbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=8)

    return sm


# ────────────────────────────────────────────────────────────────────────────
# 3.  Fixed-point markers
# ────────────────────────────────────────────────────────────────────────────

def plot_xcycle_marker(
    RZ_arr: np.ndarray,
    stability: str = 'hyperbolic',
    ax: Optional[plt.Axes] = None,
    *,
    color_X: str = '#D32F2F',
    color_O: str = '#1565C0',
    ms_X: float = 7.0,
    ms_O: float = 5.5,
    zorder: int = 15,
    label: Optional[str] = None,
):
    """Mark X-points (hyperbolic) or O-points (elliptic) in the Poincaré plane.

    Parameters
    ----------
    RZ_arr:
        Shape ``(2,)`` or ``(N, 2)`` — coordinates of the fixed points.
    stability:
        ``'hyperbolic'`` for X-points, ``'elliptic'`` for O-points.
    """
    if ax is None:
        ax = plt.gca()
    pts = np.atleast_2d(RZ_arr)
    if stability == 'hyperbolic':
        ax.plot(pts[:, 0], pts[:, 1], 'x',
                color=color_X, ms=ms_X, markeredgewidth=1.8,
                zorder=zorder, label=label)
    else:
        ax.plot(pts[:, 0], pts[:, 1], 'o',
                color=color_O, ms=ms_O, markeredgewidth=0,
                zorder=zorder, label=label)


# ────────────────────────────────────────────────────────────────────────────
# 4.  Manifold arc-length coloured line — core building block
# ────────────────────────────────────────────────────────────────────────────

def _manifold_line_collection(
    W1d_RZ: np.ndarray,
    cmap: str,
    s_norm_gamma: float = 0.5,
    s_ref: Optional[float] = None,
    lw: float = 0.9,
    alpha: float = 0.9,
    zorder: int = 6,
) -> tuple[LineCollection, np.ndarray]:
    """Build a :class:`LineCollection` coloured by poloidal arc length *s*.

    The colour saturates toward the tip of the manifold arm
    (γ < 1 compresses dynamic range near s=0, making the near-X region
    visually distinct).

    Parameters
    ----------
    W1d_RZ:
        Shape ``(N, 2)`` — ordered (R, Z) points along one manifold arm.
    cmap:
        Colourmap name (warm for unstable, cool for stable).
    s_norm_gamma:
        Power-law exponent for the colour normalisation (default 0.5 → √s).
    s_ref:
        Max s used to normalise.  If *None*, uses ``W1d_RZ`` max.

    Returns
    -------
    lc : LineCollection
    s_arr : 1-D array of arc-length values
    """
    s_arr = accumulate_s_from_RZ_arr(W1d_RZ)
    s_max = float(s_arr.max()) if s_ref is None else float(s_ref)
    if s_max < 1e-15:
        s_max = 1.0

    norm = PowerNorm(gamma=s_norm_gamma, vmin=0.0, vmax=s_max)

    n = len(s_arr)
    segments = np.empty((n - 1, 2, 2))
    segments[:, 0, :] = W1d_RZ[:-1, :]
    segments[:, 1, :] = W1d_RZ[1:, :]

    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidth=lw, alpha=alpha, zorder=zorder,
                        capstyle='round', joinstyle='round')
    lc.set_array(s_arr)
    return lc, s_arr


# ────────────────────────────────────────────────────────────────────────────
# 5.  Plot a 1-D Poincaré section of the manifold
# ────────────────────────────────────────────────────────────────────────────

def plot_manifold_1d(
    fig: plt.Figure,
    ax: plt.Axes,
    W1d_RZ: np.ndarray,
    *,
    unstable: bool = True,
    cmap: Optional[str] = None,
    lw: float = 0.9,
    alpha: float = 0.90,
    s_norm_gamma: float = 0.5,
    s_ref: Optional[float] = None,
    zorder: int = 6,
    show_colorbar: bool = False,
    colorbar_label: Optional[str] = None,
) -> LineCollection:
    """Draw a single 1-D manifold arm (one Poincaré cross-section, one arm).

    Colour encodes poloidal arc length measured from the X-point cycle:
    * **Warm** palette (default ``'Oranges'``) for the **unstable** manifold.
    * **Cool** palette (default ``'GnBu'``)   for the **stable**   manifold.

    Parameters
    ----------
    W1d_RZ:
        Shape ``(N, 2)`` — ordered (R, Z) points of the manifold arm at a
        fixed toroidal angle φ.  Produced by
        :func:`pyna.topo.manifold._transect_initPhi0_Wivp_at_a_phi`.
    unstable:
        ``True``  → warm colour scale (unstable manifold).
        ``False`` → cool colour scale  (stable manifold).
    cmap:
        Override the default cmap choice.
    s_norm_gamma:
        Nonlinear colour compression exponent (γ = 0.5 → highlight near X).
    show_colorbar:
        Attach colourbar showing arc-length scale.
    """
    arr = np.asarray(W1d_RZ)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        raise ValueError("W1d_RZ must be shape (N≥2, 2+)")
    arr = arr[:, :2]  # only R, Z

    if cmap is None:
        cmap = UNSTABLE_CMAPS[0] if unstable else STABLE_CMAPS[0]

    lc, s_arr = _manifold_line_collection(
        arr, cmap, s_norm_gamma=s_norm_gamma, s_ref=s_ref,
        lw=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_collection(lc)
    # nudge axis limits to include new geometry
    ax.autoscale_view()

    if show_colorbar:
        _label = colorbar_label or (
            r'$s_{\rm unstable}\ (\mathrm{m})$' if unstable
            else r'$s_{\rm stable}\ (\mathrm{m})$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = fig.colorbar(lc, cax=cax)
        cb.set_label(_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    return lc


def _iter_manifold_segment_arrays(segments) -> list[np.ndarray]:
    """Normalize manifold segment containers to ordered R/Z arrays."""

    if segments is None:
        return []
    if isinstance(segments, dict):
        out: list[np.ndarray] = []
        for value in segments.values():
            out.extend(_iter_manifold_segment_arrays(value))
        return out

    try:
        arr = np.asarray(segments, dtype=float)
    except (TypeError, ValueError):
        arr = np.asarray(segments, dtype=object)

    if arr.dtype != object:
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return [arr[:, :2]] if arr.shape[0] >= 2 else []
        if arr.ndim == 3 and arr.shape[-1] >= 2:
            return [seg[:, :2] for seg in arr if seg.shape[0] >= 2]

    out = []
    for value in segments:
        out.extend(_iter_manifold_segment_arrays(value))
    return out


def draw_manifold_segments(
    ax: plt.Axes,
    segments,
    *,
    unstable: bool = True,
    fig: Optional[plt.Figure] = None,
    cmap: Optional[str] = None,
    lw: float = 0.9,
    alpha: float = 0.90,
    s_norm_gamma: float = 0.5,
    s_ref: Optional[float] = None,
    zorder: int = 6,
    show_colorbar: bool = False,
    colorbar_label: Optional[str] = None,
) -> list[LineCollection]:
    """Draw one or more ordered manifold sections on a Poincare plane.

    This is the public, composable facade around pyna's arc-length-coloured
    manifold line collection.  ``segments`` may be a single ``(N, 2+)`` array,
    a list of such arrays, or a nested dict/list produced by a workflow cell.
    """

    if fig is None:
        fig = ax.get_figure()
    arrays = _iter_manifold_segment_arrays(segments)
    if not arrays:
        return []

    shared_s_ref = s_ref
    if shared_s_ref is None and len(arrays) > 1:
        maxima = [float(accumulate_s_from_RZ_arr(arr).max()) for arr in arrays]
        shared_s_ref = max(maxima) if maxima else None

    collections = []
    for arr in arrays:
        collections.append(
            plot_manifold_1d(
                fig,
                ax,
                arr,
                unstable=unstable,
                cmap=cmap,
                lw=lw,
                alpha=alpha,
                s_norm_gamma=s_norm_gamma,
                s_ref=shared_s_ref,
                zorder=zorder,
                show_colorbar=False,
            )
        )

    if show_colorbar and collections:
        label = colorbar_label or (
            r'$s_{\rm unstable}\ (\mathrm{m})$'
            if unstable
            else r'$s_{\rm stable}\ (\mathrm{m})$'
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = fig.colorbar(collections[0], cax=cax)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    return collections


def _draw_poincare_payload(
    ax: plt.Axes,
    points,
    *,
    psi_values: Optional[np.ndarray] = None,
    color: str = '0.16',
    cmap: str = 'viridis',
    point_size: float = 0.55,
    alpha: float = 0.55,
    rasterized: bool = True,
    zorder: int = 3,
):
    if points is None:
        return None
    if isinstance(points, tuple) and len(points) >= 2:
        R = np.asarray(points[0], dtype=float).ravel()
        Z = np.asarray(points[1], dtype=float).ravel()
        good = np.isfinite(R) & np.isfinite(Z)
        return ax.scatter(
            R[good],
            Z[good],
            s=point_size,
            color=color,
            alpha=alpha,
            linewidths=0,
            rasterized=rasterized,
            zorder=zorder,
        )

    try:
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            good = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
            return ax.scatter(
                arr[good, 0],
                arr[good, 1],
                s=point_size,
                color=color,
                alpha=alpha,
                linewidths=0,
                rasterized=rasterized,
                zorder=zorder,
            )
    except (TypeError, ValueError):
        pass

    return plot_poincare_orbits(
        list(points),
        ax,
        psi_values=psi_values,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        rasterized=rasterized,
        zorder=zorder,
        show_colorbar=False,
    )


def plot_poincare_manifold_section(
    ax: plt.Axes,
    *,
    poincare_points=None,
    stable_segments=None,
    unstable_segments=None,
    x_points=None,
    o_points=None,
    psi_values: Optional[np.ndarray] = None,
    point_size: float = 0.55,
    point_alpha: float = 0.55,
    point_color: str = '0.16',
    poincare_cmap: str = 'viridis',
    unstable_cmap: str = UNSTABLE_CMAPS[0],
    stable_cmap: str = STABLE_CMAPS[0],
    manifold_lw: float = 0.9,
    manifold_alpha: float = 0.90,
    s_norm_gamma: float = 0.5,
    show_colorbar: bool = False,
    show_legend: bool = True,
    legend_loc: str = 'best',
    xlabel: str = r'$R\ (\mathrm{m})$',
    ylabel: str = r'$Z\ (\mathrm{m})$',
    title: Optional[str] = None,
) -> dict[str, object]:
    """Compose a compact Poincare/manifold section from optional plot layers.

    The function intentionally keeps each layer optional: tutorial notebooks can
    call one facade, while research scripts can still use
    :func:`draw_manifold_segments`, :func:`plot_poincare_orbits`, and
    :func:`plot_xcycle_marker` independently.
    """

    fig = ax.get_figure()
    artists: dict[str, object] = {}

    artists['poincare'] = _draw_poincare_payload(
        ax,
        poincare_points,
        psi_values=psi_values,
        color=point_color,
        cmap=poincare_cmap,
        point_size=point_size,
        alpha=point_alpha,
        zorder=3,
    )
    artists['stable'] = draw_manifold_segments(
        ax,
        stable_segments,
        unstable=False,
        fig=fig,
        cmap=stable_cmap,
        lw=manifold_lw,
        alpha=manifold_alpha,
        s_norm_gamma=s_norm_gamma,
        show_colorbar=False,
        zorder=6,
    )
    artists['unstable'] = draw_manifold_segments(
        ax,
        unstable_segments,
        unstable=True,
        fig=fig,
        cmap=unstable_cmap,
        lw=manifold_lw,
        alpha=manifold_alpha,
        s_norm_gamma=s_norm_gamma,
        show_colorbar=False,
        zorder=7,
    )

    if x_points is not None:
        artists['x_points'] = plot_xcycle_marker(x_points, 'hyperbolic', ax=ax, label='X-point')
    if o_points is not None:
        artists['o_points'] = plot_xcycle_marker(o_points, 'elliptic', ax=ax, label='O-point')

    if show_colorbar:
        ref = None
        if artists['unstable']:
            ref = artists['unstable'][0]
            label = r'$s_{\rm unstable}\ (\mathrm{m})$'
        elif artists['stable']:
            ref = artists['stable'][0]
            label = r'$s_{\rm stable}\ (\mathrm{m})$'
        else:
            label = r'$s\ (\mathrm{m})$'
        if ref is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4%', pad=0.05)
            cb = fig.colorbar(ref, cax=cax)
            cb.set_label(label, fontsize=9)
            cb.ax.tick_params(labelsize=8)

    if title:
        ax.set_title(title, fontsize=10, pad=2.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')

    if show_legend and (artists['stable'] or artists['unstable'] or x_points is not None or o_points is not None):
        handles = []
        cmap_u = plt.colormaps.get_cmap(unstable_cmap)
        cmap_s = plt.colormaps.get_cmap(stable_cmap)
        if artists['unstable']:
            handles.append(
                plt.Line2D([0], [0], color=cmap_u(0.75), lw=2.0, label=r'Unstable manifold $W^{\rm u}$')
            )
        if artists['stable']:
            handles.append(
                plt.Line2D([0], [0], color=cmap_s(0.75), lw=2.0, label=r'Stable manifold $W^{\rm s}$')
            )
        if x_points is not None:
            handles.append(
                plt.Line2D([0], [0], marker='x', color='#D32F2F', ms=7, lw=0, markeredgewidth=1.8, label='X-point')
            )
        if o_points is not None:
            handles.append(
                plt.Line2D([0], [0], marker='o', color='#1565C0', ms=6, lw=0, label='O-point')
            )
        ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=8)

    return artists


# ────────────────────────────────────────────────────────────────────────────
# 6.  Plot all four manifold arms from a single X-point
# ────────────────────────────────────────────────────────────────────────────

def plot_xcycle_all_manifolds(
    fig: plt.Figure,
    ax: plt.Axes,
    W_bundles: dict,
    phi: float = 0.0,
    mturn: int = 1,
    *,
    unstable_cmap: str = UNSTABLE_CMAPS[0],
    stable_cmap:   str = STABLE_CMAPS[0],
    lw: float = 0.9,
    alpha: float = 0.88,
    s_norm_gamma: float = 0.5,
    show_colorbar: bool = True,
) -> dict[str, LineCollection]:
    """Draw all four manifold arms (W^u+, W^u-, W^s+, W^s-) of one X-cycle.

    Parameters
    ----------
    W_bundles:
        Dict with keys ``'u+'``, ``'u-'``, ``'s+'``, ``'s-'`` (any subset),
        each mapping to an ``ivp_bundle`` returned by
        :func:`pyna.topo.manifold.grow_manifold_from_Xcycle_naive_init_segment`.
    phi:
        Toroidal angle of the Poincaré section (rad).
    mturn:
        Toroidal period of the X-cycle (= poloidal mode number m).

    Returns
    -------
    Dict of :class:`LineCollection` objects (same keys as *W_bundles*).
    """
    from pyna.topo.manifold import _transect_initPhi0_Wivp_at_a_phi

    result = {}
    for key, bundle in W_bundles.items():
        unstable = key.startswith('u')
        cmap = unstable_cmap if unstable else stable_cmap
        W1d = _transect_initPhi0_Wivp_at_a_phi(bundle, phi=phi, mturn=mturn)
        if W1d.shape[0] < 2:
            continue
        lc = plot_manifold_1d(
            fig, ax, W1d[:, :2],
            unstable=unstable, cmap=cmap,
            lw=lw, alpha=alpha,
            s_norm_gamma=s_norm_gamma,
            show_colorbar=False,
            zorder=6,
        )
        result[key] = lc

    if show_colorbar and result:
        # One shared colourbar for the arc-length scale
        lc_ref = next(iter(result.values()))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = fig.colorbar(lc_ref, cax=cax)
        cb.set_label(r'$s$ (poloidal arc length, m)', fontsize=9)
        cb.ax.tick_params(labelsize=8)

    return result


# ────────────────────────────────────────────────────────────────────────────
# 7.  Composite figure builder
# ────────────────────────────────────────────────────────────────────────────

def make_tokamak_overview_figure(
    eq,
    *,
    figsize: tuple = (10, 7),
    suptitle: str = 'EAST Tokamak Equilibrium & Manifolds',
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Create a two-panel (or three-panel) publication layout.

    Left panel:  full equilibrium cross-section (Bpol + flux surfaces).
    Right panel: zoom into divertor / island region with manifolds.

    Returns
    -------
    fig : Figure
    axes : dict with keys ``'overview'``, ``'detail'``
    """
    with plt.rc_context(PUBLICATION_RC):
        fig, (ax_ov, ax_det) = plt.subplots(
            1, 2, figsize=figsize, dpi=DEFAULT_DPI,
            gridspec_kw={'width_ratios': [1, 1.1]},
        )
        fig.suptitle(suptitle, fontsize=11, y=1.01)

    return fig, {'overview': ax_ov, 'detail': ax_det}


# ────────────────────────────────────────────────────────────────────────────
# 8.  Legend helper
# ────────────────────────────────────────────────────────────────────────────

def manifold_legend_handles(
    unstable_cmap: str = UNSTABLE_CMAPS[0],
    stable_cmap: str   = STABLE_CMAPS[0],
    xpoint_color: str  = '#D32F2F',
    opoint_color: str  = '#1565C0',
) -> list:
    """Return a list of legend handles for a manifold figure."""
    cmap_u = plt.colormaps.get_cmap(unstable_cmap)
    cmap_s = plt.colormaps.get_cmap(stable_cmap)
    handles = [
        plt.Line2D([0], [0], color=cmap_u(0.75), lw=2.0,
                   label=r'Unstable manifold $W^{\rm u}$ (warm $\leftrightarrow$ $s$ from X-cycle)'),
        plt.Line2D([0], [0], color=cmap_s(0.75), lw=2.0,
                   label=r'Stable manifold $W^{\rm s}$ (cool $\leftrightarrow$ $s$ from X-cycle)'),
        plt.Line2D([0], [0], marker='x', color=xpoint_color,
                   ms=7, lw=0, markeredgewidth=1.8,
                   label='X-point (hyperbolic fixed point)'),
        plt.Line2D([0], [0], marker='o', color=opoint_color,
                   ms=6, lw=0,
                   label='O-point (elliptic fixed point)'),
    ]
    return handles
