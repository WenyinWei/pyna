"""User-facing facade for Poincare boundary-topology figures."""
from __future__ import annotations

from pathlib import Path

from pyna.toroidal.visual.poincare_topology import (
    PoincareCurvedIslandBar,
    PoincareDPKClassification,
    PoincareFixedPointValidation,
    PoincareIslandSecondaryCoordinates,
    PoincareTopologyFigureStyle,
    PoincareTopologyReportPayload,
    PoincareTraceQuality,
    circular_flux_coordinate_island_bars,
    draw_poincare_curved_island_bars,
    draw_poincare_phase_response_arrows,
    draw_poincare_island_secondary_contours,
    draw_poincare_island_secondary_points,
    draw_poincare_island_secondary_trace_lines,
    fixed_point_phase_comparison_markers,
    fixed_point_centered_island_bars,
    plot_poincare_topology_map as _plot_poincare_topology_map,
    plot_poincare_topology_payload_map as _plot_poincare_topology_payload_map,
    plot_poincare_topology_payload_report as _plot_poincare_topology_payload_report,
    plot_poincare_topology_report as _plot_poincare_topology_report,
    poincare_dpk_classification,
    poincare_fixed_point_validation,
    poincare_island_secondary_coordinates,
    poincare_phase_response_radial_deltas,
    poincare_topology_report_payload,
    poincare_trace_index_from_counts,
    poincare_trace_quality,
    sample_fixed_s_theta_curve,
)


def save_poincare_topology_figure(fig, out_path: str | Path | None, *, dpi: int = 360, bbox_inches: str = "tight"):
    """Save a Poincare topology figure with publication-oriented defaults."""

    if out_path is None:
        return None
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches=bbox_inches)
    return out


def plot_poincare_topology_map(*args, out_path: str | Path | None = None, save_dpi: int | None = None, **kwargs):
    """Plot a Poincare topology map and optionally save it.

    The canonical implementation lives in ``pyna.toroidal.visual``.  This
    facade keeps the notebook/script entry point compact while preserving the
    full keyword surface of the domain-specific implementation.
    """

    fig, ax, classification, scatter = _plot_poincare_topology_map(*args, **kwargs)
    if out_path is not None:
        style = kwargs.get("style")
        dpi = save_dpi
        if dpi is None and isinstance(style, PoincareTopologyFigureStyle):
            dpi = style.dpi
        save_poincare_topology_figure(fig, out_path, dpi=360 if dpi is None else int(dpi))
    return fig, ax, classification, scatter


def plot_poincare_topology_report(*args, out_path: str | Path | None = None, save_dpi: int | None = None, **kwargs):
    """Plot a Poincare topology report and optionally save it."""

    fig, axes, classification = _plot_poincare_topology_report(*args, **kwargs)
    if out_path is not None:
        style = kwargs.get("style")
        dpi = save_dpi
        if dpi is None and isinstance(style, PoincareTopologyFigureStyle):
            dpi = style.dpi
        save_poincare_topology_figure(fig, out_path, dpi=360 if dpi is None else int(dpi))
    return fig, axes, classification


def plot_poincare_topology_payload_map(
    payload: PoincareTopologyReportPayload,
    *,
    out_path: str | Path | None = None,
    save_dpi: int | None = None,
    **kwargs,
):
    """Plot a Poincare topology map from a payload and optionally save it."""

    fig, ax, classification, scatter = _plot_poincare_topology_payload_map(payload, **kwargs)
    if out_path is not None:
        style = kwargs.get("style")
        dpi = save_dpi
        if dpi is None and isinstance(style, PoincareTopologyFigureStyle):
            dpi = style.dpi
        save_poincare_topology_figure(fig, out_path, dpi=360 if dpi is None else int(dpi))
    return fig, ax, classification, scatter


def plot_poincare_topology_payload_report(
    payload: PoincareTopologyReportPayload,
    *,
    out_path: str | Path | None = None,
    save_dpi: int | None = None,
    **kwargs,
):
    """Plot a Poincare topology report from a payload and optionally save it."""

    fig, axes, classification = _plot_poincare_topology_payload_report(payload, **kwargs)
    if out_path is not None:
        style = kwargs.get("style")
        dpi = save_dpi
        if dpi is None and isinstance(style, PoincareTopologyFigureStyle):
            dpi = style.dpi
        save_poincare_topology_figure(fig, out_path, dpi=360 if dpi is None else int(dpi))
    return fig, axes, classification


__all__ = [
    "PoincareCurvedIslandBar",
    "PoincareDPKClassification",
    "PoincareFixedPointValidation",
    "PoincareIslandSecondaryCoordinates",
    "PoincareTopologyFigureStyle",
    "PoincareTopologyReportPayload",
    "PoincareTraceQuality",
    "circular_flux_coordinate_island_bars",
    "draw_poincare_curved_island_bars",
    "draw_poincare_phase_response_arrows",
    "draw_poincare_island_secondary_contours",
    "draw_poincare_island_secondary_points",
    "draw_poincare_island_secondary_trace_lines",
    "fixed_point_phase_comparison_markers",
    "fixed_point_centered_island_bars",
    "plot_poincare_topology_map",
    "plot_poincare_topology_payload_map",
    "plot_poincare_topology_payload_report",
    "plot_poincare_topology_report",
    "poincare_dpk_classification",
    "poincare_fixed_point_validation",
    "poincare_island_secondary_coordinates",
    "poincare_phase_response_radial_deltas",
    "poincare_topology_report_payload",
    "poincare_trace_index_from_counts",
    "poincare_trace_quality",
    "sample_fixed_s_theta_curve",
    "save_poincare_topology_figure",
]
