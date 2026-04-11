"""pyna.toroidal.visual — publication-quality toroidal equilibrium visualization."""
from .equilibrium import plot_nested_flux_surfaces, BPOL_CMAP, ISLAND_CMAPS
from .tokamak_manifold import (
    plot_equilibrium_cross_section,
    plot_poincare_orbits,
    plot_xcycle_marker,
    plot_manifold_1d,
    plot_xcycle_all_manifolds,
    make_tokamak_overview_figure,
    manifold_legend_handles,
    UNSTABLE_CMAPS,
    STABLE_CMAPS,
    PUBLICATION_RC,
)

__all__ = [
    "plot_nested_flux_surfaces",
    "BPOL_CMAP",
    "ISLAND_CMAPS",
    "plot_equilibrium_cross_section",
    "plot_poincare_orbits",
    "plot_xcycle_marker",
    "plot_manifold_1d",
    "plot_xcycle_all_manifolds",
    "make_tokamak_overview_figure",
    "manifold_legend_handles",
    "UNSTABLE_CMAPS",
    "STABLE_CMAPS",
    "PUBLICATION_RC",
]
