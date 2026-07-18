"""User-facing facade for boundary-response optimization plots."""
from __future__ import annotations

from pathlib import Path

from pyna.toroidal.visual.boundary_optimization import (
    BoundaryResponseOptimizationObservableRow,
    BoundaryResponseOptimizationSummary,
    boundary_response_optimization_summary,
    plot_boundary_response_optimization_history as _plot_boundary_response_optimization_history,
)


def save_boundary_optimization_figure(fig, out_path: str | Path | None, *, dpi: int = 220, bbox_inches: str = "tight"):
    """Save a boundary optimization figure with report-oriented defaults."""

    if out_path is None:
        return None
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches=bbox_inches)
    return out


def plot_boundary_response_optimization_history(
    *args,
    out_path: str | Path | None = None,
    save_dpi: int = 220,
    **kwargs,
):
    """Plot a boundary optimization history and optionally save it."""

    fig, axes = _plot_boundary_response_optimization_history(*args, **kwargs)
    save_boundary_optimization_figure(fig, out_path, dpi=int(save_dpi))
    return fig, axes


__all__ = [
    "BoundaryResponseOptimizationObservableRow",
    "BoundaryResponseOptimizationSummary",
    "boundary_response_optimization_summary",
    "plot_boundary_response_optimization_history",
    "save_boundary_optimization_figure",
]
