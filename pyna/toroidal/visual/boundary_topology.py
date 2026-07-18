"""Boundary-topology visualization helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DPKRecurrenceProfile:
    """Radial profile of DP^k growth and spectral recurrence diagnostics."""

    radial_labels: np.ndarray
    eigenvalue_growth: np.ndarray
    spectral_recurrence_min: np.ndarray
    recurrent_surface_indicator: np.ndarray
    chaotic_mask: np.ndarray
    chaotic_intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class BoundaryTopologyComparisonSummary:
    """Before/after summary for boundary island chains and chaotic layers."""

    baseline_chain_count: int
    perturbed_chain_count: int
    baseline_total_half_width: float
    perturbed_total_half_width: float
    delta_total_half_width: float
    baseline_chaotic_width: float
    perturbed_chaotic_width: float
    delta_chaotic_width: float
    baseline_max_sigma: float
    perturbed_max_sigma: float
    strengthened_modes: tuple[tuple[int, int], ...]
    weakened_modes: tuple[tuple[int, int], ...]


def _metric_value(metric, name: str, default: float = 0.0) -> float:
    if isinstance(metric, Mapping):
        value = metric.get(name, default)
    else:
        value = getattr(metric, name, default)
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 0:
        raise ValueError(f"{name} metric values must be scalar")
    return float(arr)


def _has_metric(metric, name: str) -> bool:
    if isinstance(metric, Mapping):
        return name in metric
    return hasattr(metric, name)


def _radial_edges(radial: np.ndarray) -> np.ndarray:
    if radial.size == 0:
        return np.array([], dtype=float)
    if radial.size == 1:
        return np.array([radial[0] - 0.5, radial[0] + 0.5], dtype=float)
    mid = 0.5 * (radial[:-1] + radial[1:])
    first = radial[0] - 0.5 * (radial[1] - radial[0])
    last = radial[-1] + 0.5 * (radial[-1] - radial[-2])
    return np.concatenate([[first], mid, [last]]).astype(float)


def _mask_intervals(radial: np.ndarray, mask: np.ndarray) -> tuple[tuple[float, float], ...]:
    edges = _radial_edges(radial)
    intervals: list[tuple[float, float]] = []
    start = None
    for idx, enabled in enumerate(mask):
        if enabled and start is None:
            start = idx
        if start is not None and ((not enabled) or idx == mask.size - 1):
            end = idx if not enabled else idx + 1
            intervals.append((float(edges[start]), float(edges[end])))
            start = None
    return tuple(intervals)


def _chain_mode(chain) -> tuple[int, int]:
    return int(getattr(chain, "m")), int(getattr(chain, "n"))


def _chain_half_width(chain) -> float:
    return float(getattr(chain, "half_width", 0.0))


def _chain_radial_label(chain) -> float:
    return float(getattr(chain, "radial_label", np.nan))


def _interval_bounds(interval) -> tuple[float, float]:
    return float(getattr(interval, "inner")), float(getattr(interval, "outer"))


def _interval_width(interval) -> float:
    lo, hi = _interval_bounds(interval)
    return max(0.0, hi - lo)


def _interval_max_sigma(interval) -> float:
    return float(getattr(interval, "max_sigma", getattr(interval, "sigma", 0.0)))


def boundary_topology_comparison_summary(
    baseline_chains: Sequence[object],
    perturbed_chains: Sequence[object],
    *,
    baseline_intervals: Sequence[object] = (),
    perturbed_intervals: Sequence[object] = (),
    mode_tolerance: float = 0.0,
) -> BoundaryTopologyComparisonSummary:
    """Summarize island-chain and chaotic-layer changes after a perturbation."""

    before = tuple(baseline_chains)
    after = tuple(perturbed_chains)
    before_width = float(np.sum([_chain_half_width(chain) for chain in before]))
    after_width = float(np.sum([_chain_half_width(chain) for chain in after]))
    before_chaos = float(np.sum([_interval_width(interval) for interval in baseline_intervals]))
    after_chaos = float(np.sum([_interval_width(interval) for interval in perturbed_intervals]))
    before_sigma = float(max([_interval_max_sigma(interval) for interval in baseline_intervals], default=0.0))
    after_sigma = float(max([_interval_max_sigma(interval) for interval in perturbed_intervals], default=0.0))
    before_by_mode = {_chain_mode(chain): _chain_half_width(chain) for chain in before}
    after_by_mode = {_chain_mode(chain): _chain_half_width(chain) for chain in after}
    all_modes = sorted(set(before_by_mode) | set(after_by_mode))
    tol = max(0.0, float(mode_tolerance))
    strengthened = tuple(
        mode
        for mode in all_modes
        if after_by_mode.get(mode, 0.0) - before_by_mode.get(mode, 0.0) > tol
    )
    weakened = tuple(
        mode
        for mode in all_modes
        if before_by_mode.get(mode, 0.0) - after_by_mode.get(mode, 0.0) > tol
    )
    return BoundaryTopologyComparisonSummary(
        baseline_chain_count=len(before),
        perturbed_chain_count=len(after),
        baseline_total_half_width=before_width,
        perturbed_total_half_width=after_width,
        delta_total_half_width=after_width - before_width,
        baseline_chaotic_width=before_chaos,
        perturbed_chaotic_width=after_chaos,
        delta_chaotic_width=after_chaos - before_chaos,
        baseline_max_sigma=before_sigma,
        perturbed_max_sigma=after_sigma,
        strengthened_modes=strengthened,
        weakened_modes=weakened,
    )


def boundary_dpk_recurrence_profile(
    radial_labels: Sequence[float],
    metrics: Sequence[object],
    *,
    growth_threshold: float = 0.0,
    recurrence_threshold: float = 0.02,
    recurrent_surface_threshold: float = 0.5,
) -> DPKRecurrenceProfile:
    """Build a radial DP^k recurrence profile from per-seed metrics.

    The chaotic mask is true only where eigenvalue growth exceeds
    ``growth_threshold`` and the recurrent-surface indicator is below
    ``recurrent_surface_threshold``.  This keeps spectrally recurrent magnetic
    surfaces distinct from chaotic layers even when neighboring regions grow.
    """

    radial = np.asarray(radial_labels, dtype=float).ravel()
    metric_tuple = tuple(metrics)
    if radial.size != len(metric_tuple):
        raise ValueError("radial_labels length must match metrics length")
    if radial.size == 0:
        empty = np.array([], dtype=float)
        return DPKRecurrenceProfile(
            radial_labels=empty,
            eigenvalue_growth=empty,
            spectral_recurrence_min=empty,
            recurrent_surface_indicator=empty,
            chaotic_mask=np.array([], dtype=bool),
            chaotic_intervals=(),
        )
    order = np.argsort(radial)
    radial = radial[order]
    ordered = [metric_tuple[idx] for idx in order]
    growth = np.asarray(
        [_metric_value(metric, "eigenvalue_ftle", _metric_value(metric, "ftle", 0.0)) for metric in ordered],
        dtype=float,
    )
    recurrence = np.asarray(
        [_metric_value(metric, "spectral_recurrence_min", np.inf) for metric in ordered],
        dtype=float,
    )
    recurrent_values = []
    for metric, recurrence_value in zip(ordered, recurrence):
        if _has_metric(metric, "recurrent_surface_indicator"):
            recurrent_values.append(_metric_value(metric, "recurrent_surface_indicator", 0.0))
        else:
            recurrent_values.append(float(recurrence_value <= float(recurrence_threshold)))
    recurrent_surface = np.asarray(recurrent_values, dtype=float)
    chaotic = (
        np.isfinite(growth)
        & (growth > float(growth_threshold))
        & (recurrent_surface < float(recurrent_surface_threshold))
    )
    return DPKRecurrenceProfile(
        radial_labels=radial,
        eigenvalue_growth=growth,
        spectral_recurrence_min=recurrence,
        recurrent_surface_indicator=recurrent_surface,
        chaotic_mask=chaotic,
        chaotic_intervals=_mask_intervals(radial, chaotic),
    )


def plot_boundary_dpk_recurrence_profile(
    radial_labels: Sequence[float],
    metrics: Sequence[object],
    *,
    axes=None,
    growth_threshold: float = 0.0,
    recurrence_threshold: float = 0.02,
    recurrent_surface_threshold: float = 0.5,
    growth_color: str = "#8f1d5b",
    recurrence_color: str = "#2f7d6d",
    layer_color: str = "#f3c54a",
    title: str | None = None,
):
    """Plot radial DP^k growth and spectral-recurrence diagnostics.

    Returns ``(fig, axes, profile)``.  The highlighted radial bands are the
    regions where eigenvalue growth is above threshold and DP^k lacks a
    recurrent-surface signature.
    """

    import matplotlib.pyplot as plt

    profile = boundary_dpk_recurrence_profile(
        radial_labels,
        metrics,
        growth_threshold=growth_threshold,
        recurrence_threshold=recurrence_threshold,
        recurrent_surface_threshold=recurrent_surface_threshold,
    )
    if axes is None:
        fig, axes_arr = plt.subplots(1, 2, figsize=(7.4, 4.0), sharey=True, constrained_layout=True)
    else:
        axes_arr = np.asarray(axes, dtype=object).ravel()
        if axes_arr.size != 2:
            raise ValueError("axes must contain exactly two matplotlib axes")
        fig = axes_arr[0].figure
    ax_growth, ax_recur = axes_arr
    radial = profile.radial_labels
    for lo, hi in profile.chaotic_intervals:
        ax_growth.axhspan(lo, hi, color=layer_color, alpha=0.18, lw=0)
        ax_recur.axhspan(lo, hi, color=layer_color, alpha=0.18, lw=0)
    ax_growth.plot(profile.eigenvalue_growth, radial, color=growth_color, lw=1.8)
    ax_growth.axvline(float(growth_threshold), color="0.35", lw=0.9, ls="--")
    ax_growth.set_xlabel(r"$\lambda_{eig}$")
    ax_growth.set_ylabel("radial label")
    ax_growth.set_title("DP^k eigenvalue growth")
    ax_growth.grid(color="0.90", linewidth=0.45)

    ax_recur.plot(profile.spectral_recurrence_min, radial, color=recurrence_color, lw=1.8)
    ax_recur.axvline(float(recurrence_threshold), color="0.35", lw=0.9, ls="--")
    ax_recur.set_xlabel(r"$\delta_{min}$")
    ax_recur.set_title("spectral recurrence")
    ax_recur.grid(color="0.90", linewidth=0.45)
    if title:
        fig.suptitle(title)
    return fig, (ax_growth, ax_recur), profile


def _draw_intervals(ax, intervals: Sequence[object], y: float, *, color: str, label: str):
    artists = []
    for idx, interval in enumerate(intervals):
        lo, hi = _interval_bounds(interval)
        artists.append(
            ax.barh(
                y,
                max(0.0, hi - lo),
                left=lo,
                height=0.34,
                color=color,
                alpha=0.72,
                label=label if idx == 0 else None,
            )
        )
    return artists


def plot_boundary_topology_comparison(
    baseline_chains: Sequence[object],
    perturbed_chains: Sequence[object],
    *,
    baseline_intervals: Sequence[object] = (),
    perturbed_intervals: Sequence[object] = (),
    dpk_radial_labels: Sequence[float] | None = None,
    baseline_dpk_metrics: Sequence[object] | None = None,
    perturbed_dpk_metrics: Sequence[object] | None = None,
    growth_threshold: float = 0.0,
    recurrence_threshold: float = 0.02,
    recurrent_surface_threshold: float = 0.5,
    title: str | None = "Boundary topology response report",
):
    """Plot a compact before/after boundary-topology response report.

    The figure compares predicted island-chain half widths, merged chaotic-layer
    intervals, and optional ``DP^k`` growth/recurrence diagnostics.  It is meant
    for perturbation-screening reports before the more expensive nonlinear
    Poincare, fixed-point, manifold, and wall-strike validation.
    """

    import matplotlib.pyplot as plt

    summary = boundary_topology_comparison_summary(
        baseline_chains,
        perturbed_chains,
        baseline_intervals=baseline_intervals,
        perturbed_intervals=perturbed_intervals,
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), constrained_layout=True)
    ax_width, ax_layer, ax_growth, ax_recur = axes.ravel()

    before = tuple(baseline_chains)
    after = tuple(perturbed_chains)
    if before:
        ax_width.scatter(
            [_chain_radial_label(chain) for chain in before],
            [_chain_half_width(chain) for chain in before],
            s=42,
            color="#5d6974",
            alpha=0.82,
            label="baseline",
        )
    if after:
        ax_width.scatter(
            [_chain_radial_label(chain) for chain in after],
            [_chain_half_width(chain) for chain in after],
            s=46,
            color="#b23a67",
            alpha=0.86,
            label="perturbed",
        )
        for chain in sorted(after, key=_chain_half_width, reverse=True)[:8]:
            ax_width.annotate(
                f"({_chain_mode(chain)[0]},{_chain_mode(chain)[1]})",
                xy=(_chain_radial_label(chain), _chain_half_width(chain)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#8f1d5b",
            )
    ax_width.set_xlabel("radial label")
    ax_width.set_ylabel("island half-width")
    ax_width.set_title("resonant island-chain response")
    ax_width.grid(color="0.90", linewidth=0.45)
    ax_width.legend(loc="best", fontsize=8)

    _draw_intervals(ax_layer, baseline_intervals, 0.75, color="#5d6974", label="baseline")
    _draw_intervals(ax_layer, perturbed_intervals, 0.25, color="#d65076", label="perturbed")
    ax_layer.set_yticks([0.25, 0.75], ["perturbed", "baseline"])
    ax_layer.set_xlabel("radial label")
    ax_layer.set_title("merged Chirikov/chaotic-layer intervals")
    ax_layer.grid(color="0.90", axis="x", linewidth=0.45)
    ax_layer.legend(loc="best", fontsize=8)

    profiles = []
    if dpk_radial_labels is not None and baseline_dpk_metrics is not None:
        profiles.append(
            (
                "baseline",
                "#5d6974",
                boundary_dpk_recurrence_profile(
                    dpk_radial_labels,
                    baseline_dpk_metrics,
                    growth_threshold=growth_threshold,
                    recurrence_threshold=recurrence_threshold,
                    recurrent_surface_threshold=recurrent_surface_threshold,
                ),
            )
        )
    if dpk_radial_labels is not None and perturbed_dpk_metrics is not None:
        profiles.append(
            (
                "perturbed",
                "#b23a67",
                boundary_dpk_recurrence_profile(
                    dpk_radial_labels,
                    perturbed_dpk_metrics,
                    growth_threshold=growth_threshold,
                    recurrence_threshold=recurrence_threshold,
                    recurrent_surface_threshold=recurrent_surface_threshold,
                ),
            )
        )
    for label, color, profile in profiles:
        ax_growth.plot(profile.radial_labels, profile.eigenvalue_growth, color=color, lw=1.8, label=label)
        ax_recur.plot(profile.radial_labels, profile.spectral_recurrence_min, color=color, lw=1.8, label=label)
        for lo, hi in profile.chaotic_intervals:
            ax_growth.axvspan(lo, hi, color="#f3c54a", alpha=0.13, lw=0)
            ax_recur.axvspan(lo, hi, color="#f3c54a", alpha=0.13, lw=0)
    ax_growth.axhline(float(growth_threshold), color="0.35", lw=0.9, ls="--")
    ax_growth.set_xlabel("radial label")
    ax_growth.set_ylabel(r"$\lambda_{eig}$")
    ax_growth.set_title(r"DP$^k$ eigenvalue growth")
    ax_growth.grid(color="0.90", linewidth=0.45)
    ax_growth.legend(loc="best", fontsize=8)

    ax_recur.axhline(float(recurrence_threshold), color="0.35", lw=0.9, ls="--")
    ax_recur.set_xlabel("radial label")
    ax_recur.set_ylabel(r"$\delta_{min}$")
    ax_recur.set_title(r"DP$^k$ spectral recurrence")
    ax_recur.grid(color="0.90", linewidth=0.45)
    ax_recur.legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title)
    ax_layer.text(
        0.02,
        0.05,
        (
            f"delta island width={summary.delta_total_half_width:.3g}; "
            f"delta chaotic width={summary.delta_chaotic_width:.3g}; "
            f"max sigma {summary.baseline_max_sigma:.2g}->{summary.perturbed_max_sigma:.2g}"
        ),
        transform=ax_layer.transAxes,
        fontsize=8,
        color="0.32",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.88},
    )
    return fig, axes, summary


__all__ = [
    "BoundaryTopologyComparisonSummary",
    "DPKRecurrenceProfile",
    "boundary_dpk_recurrence_profile",
    "boundary_topology_comparison_summary",
    "plot_boundary_dpk_recurrence_profile",
    "plot_boundary_topology_comparison",
]
