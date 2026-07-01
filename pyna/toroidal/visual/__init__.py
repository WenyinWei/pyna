"""pyna.toroidal.visual — publication-quality toroidal equilibrium visualization."""
from .equilibrium import plot_nested_flux_surfaces, BPOL_CMAP, ISLAND_CMAPS
from .RMP_spectrum import (
    island_fixed_points,
    ResonantComponent,
    FixedPointPhaseComparison,
    DeformedFixedPointProjection,
    rmp_fixed_point_seeds,
    rmp_closure_map_span,
    compare_cyna_fixed_points_for_component,
    deformed_circular_section_rz,
    project_fixed_points_to_deformed_surface,
    find_resonant_components_analytic,
    plot_island_width_bars,
    compute_mn_spectrum,
    plot_mn_heatmap,
    plot_mn_heatmap_radial,
)
from .magnetic_spectrum import (
    SectionIslandBar,
    island_bars_on_section,
    plot_chirikov_overlaps,
    plot_island_chains_on_section,
    plot_island_phase_scan,
    plot_resonant_radial_profiles,
    plot_spectrum_heatmap,
)
try:
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
except ModuleNotFoundError as exc:
    _TOKAMAK_MANIFOLD_IMPORT_ERROR = exc

    def _tokamak_manifold_unavailable(*args, **kwargs):
        raise ModuleNotFoundError(
            "tokamak manifold plotting requires optional dependencies that are "
            "not available in this environment"
        ) from _TOKAMAK_MANIFOLD_IMPORT_ERROR

    plot_equilibrium_cross_section = _tokamak_manifold_unavailable
    plot_poincare_orbits = _tokamak_manifold_unavailable
    plot_xcycle_marker = _tokamak_manifold_unavailable
    plot_manifold_1d = _tokamak_manifold_unavailable
    plot_xcycle_all_manifolds = _tokamak_manifold_unavailable
    make_tokamak_overview_figure = _tokamak_manifold_unavailable
    manifold_legend_handles = _tokamak_manifold_unavailable
    UNSTABLE_CMAPS = []
    STABLE_CMAPS = []
    PUBLICATION_RC = {}

__all__ = [
    "plot_nested_flux_surfaces",
    "BPOL_CMAP",
    "ISLAND_CMAPS",
    "island_fixed_points",
    "ResonantComponent",
    "FixedPointPhaseComparison",
    "DeformedFixedPointProjection",
    "rmp_fixed_point_seeds",
    "rmp_closure_map_span",
    "compare_cyna_fixed_points_for_component",
    "deformed_circular_section_rz",
    "project_fixed_points_to_deformed_surface",
    "find_resonant_components_analytic",
    "plot_island_width_bars",
    "compute_mn_spectrum",
    "plot_mn_heatmap",
    "plot_mn_heatmap_radial",
    "SectionIslandBar",
    "island_bars_on_section",
    "plot_chirikov_overlaps",
    "plot_island_chains_on_section",
    "plot_island_phase_scan",
    "plot_resonant_radial_profiles",
    "plot_spectrum_heatmap",
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
