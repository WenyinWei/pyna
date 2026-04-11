"""Publication-quality equilibrium visualization for pyna MCF equilibria.

Supports StellaratorSimple (Poincaré-based) and EquilibriumSolovev (contour-based).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional

# Academic colormap choices
BPOL_CMAP = 'magma'
ISLAND_CMAPS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']


def plot_nested_flux_surfaces(
    eq,
    ax: Optional[plt.Axes] = None,
    n_surfaces: int = 25,
    psi_min: float = 0.05,
    psi_max: float = 0.92,
    n_grid: int = 250,
    cmap: str = BPOL_CMAP,
    linewidth_max: float = 1.4,
    density: float = 2.0,
    show_colorbar: bool = True,
    # Stellarator Poincaré params
    n_fieldlines: int = 20,
    n_turns: int = 200,
) -> tuple:
    """Draw nested flux surfaces.

    For stellarators (StellaratorSimple): uses Poincaré map scatter.
    For axisymmetric equilibria with .BR/.BZ: uses streamplot + psi contours.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 7), dpi=150)
    else:
        fig = ax.get_figure()

    # Detect stellarator vs axisymmetric
    is_stellarator = hasattr(eq, 'field_func') and not hasattr(eq, 'BR')

    if is_stellarator:
        _plot_stellarator_poincare(
            eq, ax, n_fieldlines=n_fieldlines, n_turns=n_turns,
            cmap=cmap, show_colorbar=show_colorbar,
        )
    else:
        _plot_axisym_streamplot(
            eq, ax, n_surfaces=n_surfaces, psi_min=psi_min, psi_max=psi_max,
            n_grid=n_grid, cmap=cmap, linewidth_max=linewidth_max,
            density=density, show_colorbar=show_colorbar,
        )

    ax.set_xlabel(r'$R\ (\mathrm{m})$', fontsize=12)
    ax.set_ylabel(r'$Z\ (\mathrm{m})$', fontsize=12)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=10)

    return fig, ax


def _plot_stellarator_poincare(eq, ax, n_fieldlines=20, n_turns=200,
                                cmap='magma', show_colorbar=True):
    """Draw Poincaré scatter for a StellaratorSimple."""
    from pyna.topo.poincare import poincare_from_fieldlines
    from pyna.topo.section import ToroidalSection

    R0 = eq.R0
    r0 = eq.r0

    # Start points from axis to near LCFS
    psi_starts = np.linspace(0.03, 0.93, n_fieldlines)
    r_starts = np.sqrt(psi_starts) * r0

    start_pts = np.zeros((n_fieldlines, 3))
    start_pts[:, 0] = R0 + r_starts
    start_pts[:, 1] = 0.0
    start_pts[:, 2] = 0.0

    # Toroidal section at phi=0
    sections = [ToroidalSection(phi0=0.0)]
    t_max = n_turns * 2 * np.pi * R0  # approximate arc length for n_turns

    pmap = poincare_from_fieldlines(
        eq.field_func,
        start_pts,
        sections=sections,
        t_max=t_max,
        dt=0.05,
    )

    crossings = pmap.crossing_array(0)  # shape (N, 3): R, Z, phi

    if len(crossings) == 0:
        ax.text(0.5, 0.5, 'No Poincaré crossings', transform=ax.transAxes,
                ha='center', va='center')
        return

    R_pts = crossings[:, 0]
    Z_pts = crossings[:, 1]

    # Color by normalized radial distance from axis
    psi_pts = ((R_pts - R0)**2 + Z_pts**2) / r0**2
    psi_pts = np.clip(psi_pts, 0, 1.05)

    scatter = ax.scatter(R_pts, Z_pts, c=psi_pts, cmap=cmap,
                         s=0.8, linewidths=0, alpha=0.85,
                         vmin=0, vmax=1.0, rasterized=True)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = ax.get_figure().colorbar(scatter, cax=cax)
        cb.set_label(r'$\psi_\mathrm{norm}$', fontsize=11)
        cb.ax.tick_params(labelsize=9)

    # Mark magnetic axis
    ax.plot(R0, 0.0, 'x', color='white', markersize=8, markeredgewidth=1.5, zorder=10)


def _plot_axisym_streamplot(eq, ax, n_surfaces=25, psi_min=0.05, psi_max=0.92,
                             n_grid=250, cmap='magma', linewidth_max=1.4,
                             density=2.0, show_colorbar=True):
    """Streamplot + flux surface contours for axisymmetric equilibria."""
    R0 = getattr(eq, 'R0', 1.86)
    a  = getattr(eq, 'a', 0.6)
    kappa = getattr(eq, 'kappa', 1.7)

    R_arr = np.linspace(R0 - 1.3*a, R0 + 1.3*a, n_grid)
    Z_arr = np.linspace(-1.35*a*kappa, 1.35*a*kappa, n_grid)
    RR, ZZ = np.meshgrid(R_arr, Z_arr)

    BR   = np.vectorize(eq.BR)(RR, ZZ)
    BZ   = np.vectorize(eq.BZ)(RR, ZZ)
    Bpol = np.sqrt(BR**2 + BZ**2)

    finite = Bpol[np.isfinite(Bpol)]
    vmax = np.percentile(finite, 99.5)
    norm = Normalize(vmin=0, vmax=vmax)
    lw   = np.clip(linewidth_max * Bpol / (vmax + 1e-30), 0.1, linewidth_max)

    strm = ax.streamplot(
        RR, ZZ, BR, BZ,
        linewidth=lw, color=Bpol, cmap=cmap, norm=norm,
        density=density, arrowsize=0.0, broken_streamlines=False,
    )

    if hasattr(eq, 'psi'):
        PSI = np.vectorize(eq.psi)(RR, ZZ)
        psi_axis = float(eq.psi(*eq.find_opoint())) if hasattr(eq, 'find_opoint') else float(np.nanmin(PSI))
        psi_lcfs = float(eq.psi_lcfs()) if hasattr(eq, 'psi_lcfs') else float(np.nanpercentile(PSI, 99))
        levels = psi_axis + np.linspace(psi_min, psi_max, n_surfaces) * (psi_lcfs - psi_axis)
        ax.contour(RR, ZZ, PSI, levels=levels, colors='white', linewidths=0.3, alpha=0.4)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = ax.get_figure().colorbar(strm.lines, cax=cax)
        cb.set_label(r'$B_\mathrm{pol}\ (\mathrm{T})$', fontsize=11)
        cb.ax.tick_params(labelsize=9)
