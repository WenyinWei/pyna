"""pyna.MCF.visual — legacy compatibility wrappers over ``pyna.toroidal.visual``."""
from pyna.toroidal.visual.equilibrium import plot_nested_flux_surfaces, BPOL_CMAP, ISLAND_CMAPS
from pyna.toroidal.visual.RMP_spectrum import (
    island_fixed_points,
    ResonantComponent,
    find_resonant_components_analytic,
    plot_island_width_bars,
    compute_mn_spectrum,
    plot_mn_heatmap,
    plot_mn_heatmap_radial,
)
from pyna.toroidal.visual.tokamak_manifold import (
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
    # equilibrium.py
    "plot_nested_flux_surfaces",
    "BPOL_CMAP",
    "ISLAND_CMAPS",
    # RMP_spectrum.py
    "island_fixed_points",
    "ResonantComponent",
    "find_resonant_components_analytic",
    "plot_island_width_bars",
    "compute_mn_spectrum",
    "plot_mn_heatmap",
    "plot_mn_heatmap_radial",
    # tokamak_manifold.py
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
