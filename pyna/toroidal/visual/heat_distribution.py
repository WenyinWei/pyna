"""Heat-distribution control visualization."""
from __future__ import annotations

import numpy as np


def plot_heat_distribution_control_result(
    result,
    *,
    axes=None,
    cmap: str = "inferno",
    residual_cmap: str = "coolwarm",
    title: str | None = "Heat-distribution control result",
):
    """Plot current, target, predicted, and residual heat distributions."""

    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes_arr = plt.subplots(2, 2, figsize=(8.4, 6.8), constrained_layout=True)
    else:
        axes_arr = np.asarray(axes, dtype=object)
        if axes_arr.shape != (2, 2):
            raise ValueError("axes must have shape (2, 2)")
        fig = axes_arr.ravel()[0].figure
    current = np.asarray(result.current_heat, dtype=float)
    target = np.asarray(result.target_heat, dtype=float)
    predicted = np.asarray(result.predicted_heat, dtype=float)
    residual = predicted - target
    vmax = float(np.nanmax([np.nanmax(current), np.nanmax(target), np.nanmax(predicted)]))
    vmax = max(vmax, 1.0e-300)
    panels = (
        ("current", current, cmap, 0.0, vmax),
        ("target", target, cmap, 0.0, vmax),
        ("predicted", predicted, cmap, 0.0, vmax),
        ("predicted - target", residual, residual_cmap, -float(np.nanmax(np.abs(residual))), float(np.nanmax(np.abs(residual)))),
    )
    for ax, (name, values, cmap_name, vmin, vmax_i) in zip(axes_arr.ravel(), panels):
        mesh = ax.imshow(values.T, origin="lower", aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax_i)
        ax.set_title(name)
        ax.set_xlabel("phi bin")
        ax.set_ylabel("wall arclength bin")
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    solve = result.solve
    axes_arr[1, 1].text(
        0.02,
        0.98,
        f"weighted residual={solve.weighted_residual_norm:.3g}\\nactive upper={len(solve.active_upper_bounds)}",
        transform=axes_arr[1, 1].transAxes,
        va="top",
        fontsize=8,
        color="0.18",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.88},
    )
    if title:
        fig.suptitle(title)
    return fig, axes_arr


def plot_heat_distribution_control_history(
    history,
    *,
    axes=None,
    cmap: str = "inferno",
    residual_cmap: str = "coolwarm",
    title: str | None = "Iterative heat-distribution control",
):
    """Plot heat-control convergence, final residual, and control trajectory."""

    import matplotlib.pyplot as plt

    steps = tuple(history)
    if not steps:
        raise ValueError("history must contain at least one heat-control step")
    if axes is None:
        fig, axes_arr = plt.subplots(2, 3, figsize=(11.2, 6.8), constrained_layout=True)
    else:
        axes_arr = np.asarray(axes, dtype=object)
        if axes_arr.shape != (2, 3):
            raise ValueError("axes must have shape (2, 3)")
        fig = axes_arr.ravel()[0].figure

    initial = np.asarray(steps[0].result.current_heat, dtype=float)
    target = np.asarray(steps[-1].result.target_heat, dtype=float)
    predicted = np.asarray(steps[-1].result.predicted_heat, dtype=float)
    residual = predicted - target
    vmax = float(np.nanmax([np.nanmax(initial), np.nanmax(target), np.nanmax(predicted)]))
    vmax = max(vmax, 1.0e-300)
    residual_abs = max(float(np.nanmax(np.abs(residual))), 1.0e-300)
    heat_panels = (
        (axes_arr[0, 0], "initial", initial, cmap, 0.0, vmax),
        (axes_arr[0, 1], "target", target, cmap, 0.0, vmax),
        (axes_arr[0, 2], "final predicted", predicted, cmap, 0.0, vmax),
        (axes_arr[1, 0], "final residual", residual, residual_cmap, -residual_abs, residual_abs),
    )
    for ax, name, values, cmap_name, vmin, vmax_i in heat_panels:
        mesh = ax.imshow(values.T, origin="lower", aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax_i)
        ax.set_title(name)
        ax.set_xlabel("phi bin")
        ax.set_ylabel("wall arclength bin")
        fig.colorbar(mesh, ax=ax, shrink=0.78)

    residual_norm = np.array([step.result.solve.weighted_residual_norm for step in steps], dtype=float)
    iteration = np.array([step.iteration for step in steps], dtype=int)
    ax_resid = axes_arr[1, 1]
    ax_resid.plot(iteration, residual_norm, marker="o", color="#2b6c9f", lw=1.7)
    ax_resid.set_title("weighted residual")
    ax_resid.set_xlabel("iteration")
    ax_resid.set_ylabel("norm")
    ax_resid.grid(True, alpha=0.25)

    controls = np.vstack([steps[0].controls_before] + [step.controls_after for step in steps])
    control_iter = np.arange(controls.shape[0], dtype=int)
    labels = steps[-1].result.solve.control_labels
    ax_ctrl = axes_arr[1, 2]
    for idx, label in enumerate(labels):
        ax_ctrl.plot(control_iter, controls[:, idx], marker="o", lw=1.4, label=label)
    ax_ctrl.set_title("controls")
    ax_ctrl.set_xlabel("iteration boundary")
    ax_ctrl.set_ylabel("amplitude")
    ax_ctrl.grid(True, alpha=0.25)
    if len(labels) <= 6:
        ax_ctrl.legend(loc="best", fontsize=8, frameon=False)

    if title:
        fig.suptitle(title)
    return fig, axes_arr


__all__ = ["plot_heat_distribution_control_history", "plot_heat_distribution_control_result"]
