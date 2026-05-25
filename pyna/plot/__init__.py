"""Generic plotting helpers for pyna objects."""

from pyna.plot.island import plot_island, island_section_points
from pyna.plot.island_chain import plot_island_chain, island_chain_section_points
from pyna.plot.xo_points import draw_xo_points, XO_STYLE
from pyna.plot.poincare_beta import plot_poincare_beta_grid
from pyna.plot.poloidal import (
    B_pol_section,
    poloidal_field_section,
    robust_field_max,
    s_isolines,
    stream_field_from_components,
)
from pyna.plot.tube import (
    plot_tube_section,
    plot_tube_chain_section,
    plot_tube_chain_poincare,
    tube_legend_handles,
    TUBE_COLORS,
    TUBE_MARKERS_O,
    TUBE_MARKERS_X,
)

__all__ = [
    # Island / IslandChain
    "plot_island",
    "island_section_points",
    "plot_island_chain",
    "island_chain_section_points",
    "draw_xo_points",
    "XO_STYLE",
    "plot_poincare_beta_grid",
    "poloidal_field_section",
    "B_pol_section",
    "stream_field_from_components",
    "robust_field_max",
    "s_isolines",
    # Tube / TubeChain (first-class)
    "plot_tube_section",
    "plot_tube_chain_section",
    "plot_tube_chain_poincare",
    "tube_legend_handles",
    "TUBE_COLORS",
    "TUBE_MARKERS_O",
    "TUBE_MARKERS_X",
]
