"""Generic plotting helpers for pyna objects."""

from pyna.plot.island import plot_island, island_section_points
from pyna.plot.island_chain import plot_island_chain, island_chain_section_points
from pyna.plot.xo_points import draw_xo_points, XO_STYLE
from pyna.plot.poincare_beta import plot_poincare_beta_grid
from pyna.plot.boundary_island import plot_boundary_island_sections
from pyna.plot.section_geometry import (
    SECTION_CYCLE_COLORS,
    apply_section_limits,
    create_section_grid,
    cycle_identity,
    cycle_list,
    cycle_point_arrays,
    cycles_for_section,
    draw_axis_point,
    draw_cycle_points,
    draw_manifold_lines,
    draw_manifold_origins,
    draw_manifold_points,
    draw_poincare_background,
    draw_poincare_points,
    draw_wall_section,
    format_section_axis,
    manifold_lpol_max,
    manifolds_for_section,
    save_figure,
    section_data_limits,
    trim_compact_tick_labels,
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
    "plot_boundary_island_sections",
    "SECTION_CYCLE_COLORS",
    "apply_section_limits",
    "create_section_grid",
    "cycle_identity",
    "cycle_list",
    "cycle_point_arrays",
    "cycles_for_section",
    "draw_axis_point",
    "draw_cycle_points",
    "draw_manifold_lines",
    "draw_manifold_origins",
    "draw_manifold_points",
    "draw_poincare_background",
    "draw_poincare_points",
    "draw_wall_section",
    "format_section_axis",
    "manifold_lpol_max",
    "manifolds_for_section",
    "save_figure",
    "section_data_limits",
    "trim_compact_tick_labels",
    # Tube / TubeChain (first-class)
    "plot_tube_section",
    "plot_tube_chain_section",
    "plot_tube_chain_poincare",
    "tube_legend_handles",
    "TUBE_COLORS",
    "TUBE_MARKERS_O",
    "TUBE_MARKERS_X",
]
