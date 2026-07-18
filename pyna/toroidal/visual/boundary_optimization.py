"""Visualization helpers for boundary-response optimization histories."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BoundaryResponseOptimizationObservableRow:
    """One final observable residual row from a boundary optimization result."""

    label: str
    initial: float
    final: float
    target: float
    residual: float
    weight: float
    weighted_abs_residual: float


@dataclass(frozen=True)
class BoundaryResponseOptimizationSummary:
    """Compact audit summary for a boundary-response optimization result."""

    n_steps: int
    n_accepted: int
    n_rejected: int
    initial_residual_norm: float
    final_residual_norm: float
    residual_reduction: float
    residual_reduction_fraction: float
    final_controls_by_label: dict[str, float]
    final_active_lower_bounds: tuple[str, ...] = field(default_factory=tuple)
    final_active_upper_bounds: tuple[str, ...] = field(default_factory=tuple)
    max_condition_number: float = np.nan
    min_singular_value: float = np.nan
    max_column_correlation: float = np.nan
    observable_rows: tuple[BoundaryResponseOptimizationObservableRow, ...] = field(default_factory=tuple)


def _weighted_residual_norm(values, target, weights) -> float:
    return float(np.linalg.norm((np.asarray(values, dtype=float) - np.asarray(target, dtype=float)) * np.sqrt(weights)))


def _initial_values(result) -> np.ndarray:
    steps = tuple(result.steps)
    if steps:
        return np.asarray(steps[0].current_observables.values, dtype=float).ravel()
    return np.asarray(result.final_observables.values, dtype=float).ravel()


def _observable_rows(result, *, top_n: int | None = None) -> tuple[BoundaryResponseOptimizationObservableRow, ...]:
    labels = tuple(str(label) for label in result.final_observables.labels)
    initial = _initial_values(result)
    final = np.asarray(result.final_observables.values, dtype=float).ravel()
    target = np.asarray(result.target_observables.values, dtype=float).ravel()
    weights = np.asarray(result.final_observables.weights, dtype=float).ravel()
    if not (initial.size == final.size == target.size == weights.size == len(labels)):
        raise ValueError("optimization observable arrays must share the same length")
    residual = final - target
    weighted = np.abs(residual) * np.sqrt(weights)
    rows = [
        BoundaryResponseOptimizationObservableRow(
            label=label,
            initial=float(initial[idx]),
            final=float(final[idx]),
            target=float(target[idx]),
            residual=float(residual[idx]),
            weight=float(weights[idx]),
            weighted_abs_residual=float(weighted[idx]),
        )
        for idx, label in enumerate(labels)
    ]
    rows.sort(key=lambda row: row.weighted_abs_residual, reverse=True)
    if top_n is not None:
        rows = rows[: max(0, int(top_n))]
    return tuple(rows)


def _max_offdiag_correlation(correlation) -> float:
    corr = np.asarray(correlation, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = np.abs(corr[mask])
    finite = values[np.isfinite(values)]
    return float(np.nanmax(finite)) if finite.size else np.nan


def boundary_response_optimization_summary(result, *, top_n_observables: int | None = 8) -> BoundaryResponseOptimizationSummary:
    """Summarize controls, residuals, bounds, and matrix diagnostics for a run."""

    steps = tuple(result.steps)
    target = np.asarray(result.target_observables.values, dtype=float).ravel()
    final = np.asarray(result.final_observables.values, dtype=float).ravel()
    weights = np.asarray(result.final_observables.weights, dtype=float).ravel()
    if final.size != target.size or weights.size != final.size:
        raise ValueError("final and target observables must have matching values and weights")
    if steps:
        initial_norm = float(steps[0].current_residual_norm)
        final_norm = float(steps[-1].accepted_residual_norm)
        final_lower = tuple(steps[-1].solve.active_lower_bounds)
        final_upper = tuple(steps[-1].solve.active_upper_bounds)
    else:
        initial_norm = _weighted_residual_norm(final, target, weights)
        final_norm = initial_norm
        final_lower = ()
        final_upper = ()
    if not np.isfinite(final_norm) or abs(final_norm - _weighted_residual_norm(final, target, weights)) > 1.0e-8:
        final_norm = _weighted_residual_norm(final, target, weights)
    reduction = initial_norm - final_norm
    if initial_norm > 0.0 and np.isfinite(initial_norm):
        reduction_fraction = reduction / initial_norm
    else:
        reduction_fraction = 0.0

    condition_numbers = []
    singular_values = []
    correlations = []
    for step in steps:
        diagnostics = step.system.diagnostics
        condition_numbers.append(float(diagnostics.condition_number))
        singular_values.extend(float(value) for value in diagnostics.singular_values)
        correlations.append(_max_offdiag_correlation(diagnostics.column_correlation))
    finite_conditions = np.asarray([value for value in condition_numbers if np.isfinite(value)], dtype=float)
    finite_singular = np.asarray([value for value in singular_values if np.isfinite(value)], dtype=float)
    finite_corr = np.asarray([value for value in correlations if np.isfinite(value)], dtype=float)

    return BoundaryResponseOptimizationSummary(
        n_steps=len(steps),
        n_accepted=sum(bool(step.accepted) for step in steps),
        n_rejected=sum(not bool(step.accepted) for step in steps),
        initial_residual_norm=float(initial_norm),
        final_residual_norm=float(final_norm),
        residual_reduction=float(reduction),
        residual_reduction_fraction=float(reduction_fraction),
        final_controls_by_label={
            label: float(value) for label, value in zip(result.control_labels, result.final_controls)
        },
        final_active_lower_bounds=final_lower,
        final_active_upper_bounds=final_upper,
        max_condition_number=float(np.nanmax(finite_conditions)) if finite_conditions.size else np.nan,
        min_singular_value=float(np.nanmin(finite_singular)) if finite_singular.size else np.nan,
        max_column_correlation=float(np.nanmax(finite_corr)) if finite_corr.size else np.nan,
        observable_rows=_observable_rows(result, top_n=top_n_observables),
    )


def plot_boundary_response_optimization_history(
    result,
    *,
    axes=None,
    title: str | None = "Boundary response optimization history",
):
    """Plot nonlinear residual checks and control trajectory for an optimization run."""

    import matplotlib.pyplot as plt

    steps = tuple(result.steps)
    if not steps:
        raise ValueError("result must contain at least one optimization step")
    if axes is None:
        fig, axes_arr = plt.subplots(2, 2, figsize=(9.6, 6.8), constrained_layout=True)
    else:
        axes_arr = np.asarray(axes, dtype=object)
        if axes_arr.shape != (2, 2):
            raise ValueError("axes must have shape (2, 2)")
        fig = axes_arr.ravel()[0].figure

    iteration = np.array([step.iteration for step in steps], dtype=int)
    current_norm = np.array([step.current_residual_norm for step in steps], dtype=float)
    predicted_norm = np.array([step.predicted_residual_norm for step in steps], dtype=float)
    accepted_norm = np.array([step.accepted_residual_norm for step in steps], dtype=float)
    alpha = np.array([step.accepted_alpha for step in steps], dtype=float)

    ax_resid = axes_arr[0, 0]
    ax_resid.plot(iteration, current_norm, marker="o", lw=1.5, color="#5d6974", label="before")
    ax_resid.plot(iteration, predicted_norm, marker="o", lw=1.5, color="#8f1d5b", label="linear predicted")
    ax_resid.plot(iteration, accepted_norm, marker="o", lw=1.5, color="#2f7d6d", label="nonlinear accepted")
    ax_resid.set_title("weighted residual")
    ax_resid.set_xlabel("iteration")
    ax_resid.set_ylabel("norm")
    ax_resid.grid(True, alpha=0.25)
    ax_resid.legend(loc="best", fontsize=8, frameon=False)

    controls = np.vstack([steps[0].controls_before] + [step.controls_after for step in steps])
    control_iter = np.arange(controls.shape[0], dtype=int)
    ax_controls = axes_arr[0, 1]
    for idx, label in enumerate(result.control_labels):
        ax_controls.plot(control_iter, controls[:, idx], marker="o", lw=1.3, label=label)
    ax_controls.set_title("controls")
    ax_controls.set_xlabel("iteration boundary")
    ax_controls.set_ylabel("amplitude")
    ax_controls.grid(True, alpha=0.25)
    if len(result.control_labels) <= 6:
        ax_controls.legend(loc="best", fontsize=8, frameon=False)

    ax_alpha = axes_arr[1, 0]
    ax_alpha.bar(iteration, alpha, color="#3e5c76", alpha=0.82)
    ax_alpha.set_title("accepted line-search fraction")
    ax_alpha.set_xlabel("iteration")
    ax_alpha.set_ylabel("alpha")
    ax_alpha.set_ylim(0.0, 1.05)
    ax_alpha.grid(True, axis="y", alpha=0.25)

    ax_bounds = axes_arr[1, 1]
    lower_counts = np.array([len(step.solve.active_lower_bounds) for step in steps], dtype=float)
    upper_counts = np.array([len(step.solve.active_upper_bounds) for step in steps], dtype=float)
    width = 0.38
    ax_bounds.bar(iteration - width / 2.0, lower_counts, width=width, color="#748cab", label="lower")
    ax_bounds.bar(iteration + width / 2.0, upper_counts, width=width, color="#d65076", label="upper")
    ax_bounds.set_title("active control bounds")
    ax_bounds.set_xlabel("iteration")
    ax_bounds.set_ylabel("count")
    ax_bounds.grid(True, axis="y", alpha=0.25)
    ax_bounds.legend(loc="best", fontsize=8, frameon=False)
    if not np.any(lower_counts) and not np.any(upper_counts):
        ax_bounds.text(
            0.5,
            0.5,
            "no active bounds",
            transform=ax_bounds.transAxes,
            ha="center",
            va="center",
            color="0.45",
            fontsize=9,
        )
        ax_bounds.set_ylim(0.0, 1.0)

    if title:
        fig.suptitle(title)
    return fig, axes_arr


__all__ = [
    "BoundaryResponseOptimizationObservableRow",
    "BoundaryResponseOptimizationSummary",
    "boundary_response_optimization_summary",
    "plot_boundary_response_optimization_history",
]
