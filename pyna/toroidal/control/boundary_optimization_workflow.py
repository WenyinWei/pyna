"""Iterative boundary-response optimization workflow helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from pyna.toroidal.control.boundary_topology_design import (
    BoundaryLinearResponseSystem,
    BoundaryResponseObservables,
    BoundaryResponseSolveResult,
    boundary_response_observables,
    finite_difference_boundary_response_system,
)


@dataclass(frozen=True)
class BoundaryResponseOptimizationStep:
    """One nonlinear validation step in an iterative boundary-control workflow."""

    iteration: int
    controls_before: np.ndarray
    proposed_step: np.ndarray
    accepted_step: np.ndarray
    controls_after: np.ndarray
    system: BoundaryLinearResponseSystem
    solve: BoundaryResponseSolveResult
    current_observables: BoundaryResponseObservables
    accepted_observables: BoundaryResponseObservables
    target_observables: BoundaryResponseObservables
    current_residual_norm: float
    predicted_residual_norm: float
    accepted_residual_norm: float
    accepted_alpha: float
    line_search_residuals: tuple[tuple[float, float], ...]
    accepted: bool
    line_search_failures: tuple[tuple[float, str], ...] = ()


@dataclass(frozen=True)
class BoundaryResponseOptimizationResult:
    """History and final state for callback-driven boundary optimization."""

    initial_controls: np.ndarray
    final_controls: np.ndarray
    control_labels: tuple[str, ...]
    steps: tuple[BoundaryResponseOptimizationStep, ...]
    final_observables: BoundaryResponseObservables
    target_observables: BoundaryResponseObservables

    @property
    def controls_by_label(self) -> dict[str, float]:
        """Return final controls keyed by label."""

        return {label: float(value) for label, value in zip(self.control_labels, self.final_controls)}


def _coerce_observables(obj, *, labels=None, weights=None) -> BoundaryResponseObservables:
    if isinstance(obj, BoundaryResponseObservables):
        return obj
    values = np.asarray(obj, dtype=float).ravel()
    if labels is None:
        labels = tuple(f"row.{idx}" for idx in range(values.size))
    return boundary_response_observables(labels, values, weights=weights)


def _target_values(
    target: Mapping[str, float] | Sequence[float],
    labels: tuple[str, ...],
    current: np.ndarray,
) -> np.ndarray:
    if isinstance(target, Mapping):
        out = np.asarray(current, dtype=float).ravel().copy()
        index = {label: idx for idx, label in enumerate(labels)}
        unknown = [str(label) for label in target if str(label) not in index]
        if unknown:
            raise ValueError(f"target labels are not present in observables: {unknown}")
        for label, value in target.items():
            arr = np.asarray(value, dtype=float)
            if arr.ndim != 0:
                raise ValueError("mapping target values must be scalar")
            out[index[str(label)]] = float(arr)
        return out
    out = np.asarray(target, dtype=float).ravel()
    if out.size != len(labels):
        raise ValueError("target length must match observable labels")
    return out


def boundary_response_residual_norm(
    observables: BoundaryResponseObservables,
    target: Mapping[str, float] | Sequence[float],
) -> float:
    """Return the weighted residual norm for observable rows and target values."""

    rows = _coerce_observables(observables)
    target_vec = _target_values(target, rows.labels, rows.values)
    return float(np.linalg.norm((rows.values - target_vec) * np.sqrt(rows.weights)))


def _coerce_bound_value(value) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        magnitude = abs(float(arr))
        return -magnitude, magnitude
    flat = arr.ravel()
    if flat.size != 2:
        raise ValueError("mapping bounds values must be scalar magnitudes or (lower, upper) pairs")
    return float(flat[0]), float(flat[1])


def _bounds_pair(bounds, labels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    n_controls = len(labels)
    if bounds is None:
        return np.full(n_controls, -np.inf, dtype=float), np.full(n_controls, np.inf, dtype=float)
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lo = np.broadcast_to(np.asarray(bounds.lb, dtype=float), (n_controls,)).copy()
        hi = np.broadcast_to(np.asarray(bounds.ub, dtype=float), (n_controls,)).copy()
    elif isinstance(bounds, Mapping):
        index = {label: idx for idx, label in enumerate(labels)}
        unknown = [str(label) for label in bounds if str(label) not in index]
        if unknown:
            raise ValueError(f"bounds labels are not present in controls: {unknown}")
        lo = np.full(n_controls, -np.inf, dtype=float)
        hi = np.full(n_controls, np.inf, dtype=float)
        for label, value in bounds.items():
            lo_i, hi_i = _coerce_bound_value(value)
            idx = index[str(label)]
            lo[idx] = lo_i
            hi[idx] = hi_i
    else:
        lower, upper = bounds
        lo = np.broadcast_to(np.asarray(lower, dtype=float), (n_controls,)).copy()
        hi = np.broadcast_to(np.asarray(upper, dtype=float), (n_controls,)).copy()
    if np.any(hi < lo):
        raise ValueError("upper bounds must be >= lower bounds")
    return lo, hi


def _increment_bounds(
    bounds,
    control_bounds,
    controls: np.ndarray,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _bounds_pair(bounds, labels)
    if control_bounds is None:
        return lo, hi
    abs_lo, abs_hi = _bounds_pair(control_bounds, labels)
    if np.any(controls < abs_lo - 1.0e-12) or np.any(controls > abs_hi + 1.0e-12):
        raise ValueError("current controls are outside control_bounds")
    lo = np.maximum(lo, abs_lo - controls)
    hi = np.minimum(hi, abs_hi - controls)
    if np.any(hi < lo):
        raise ValueError("bounds and control_bounds leave no feasible control step")
    return lo, hi


def _difference_steps(
    controls: np.ndarray,
    steps: np.ndarray,
    control_bounds,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    plus = np.abs(steps).astype(float)
    minus = np.abs(steps).astype(float)
    if control_bounds is not None:
        abs_lo, abs_hi = _bounds_pair(control_bounds, labels)
        plus = np.minimum(plus, np.maximum(0.0, abs_hi - controls))
        minus = np.minimum(minus, np.maximum(0.0, controls - abs_lo))
    return plus, minus


def iterative_boundary_response_optimization(
    evaluate_observables: Callable[[np.ndarray], BoundaryResponseObservables],
    initial_controls: Sequence[float],
    target: Mapping[str, float] | Sequence[float],
    *,
    control_labels: Sequence[str],
    steps=1.0,
    n_iterations: int = 4,
    bounds=None,
    control_bounds=None,
    regularization: float = 0.0,
    control_scale=None,
    line_search: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
    acceptance_tolerance: float = 1.0e-12,
    convergence_tolerance: float | None = None,
    stop_on_rejected: bool = True,
) -> BoundaryResponseOptimizationResult:
    """Iteratively solve, apply, and nonlinear-check boundary response updates.

    The callback should evaluate the full nonlinear workflow for a control
    vector and return named observables.  In real boundary-divertor design this
    callback is where spectrum projection, Poincare/DP^k tracing, fixed-point
    checks, manifold diagnostics, and wall heat rows are recomputed.
    """

    controls = np.asarray(initial_controls, dtype=float).ravel().copy()
    labels = tuple(str(label) for label in control_labels)
    if controls.size != len(labels):
        raise ValueError("initial_controls length must match control_labels")
    if len(set(labels)) != len(labels):
        raise ValueError("control_labels must be unique")
    step_arr = np.broadcast_to(np.asarray(steps, dtype=float), controls.shape).copy()
    if np.any(step_arr == 0.0) or not np.all(np.isfinite(step_arr)):
        raise ValueError("steps must be finite and nonzero")
    alphas = tuple(float(alpha) for alpha in line_search)
    if not alphas or any(alpha <= 0.0 or alpha > 1.0 or not np.isfinite(alpha) for alpha in alphas):
        raise ValueError("line_search entries must be finite and in (0, 1]")

    current_obs = _coerce_observables(evaluate_observables(controls))
    target_vec = _target_values(target, current_obs.labels, current_obs.values)
    target_obs = BoundaryResponseObservables(
        labels=current_obs.labels,
        values=target_vec,
        weights=current_obs.weights,
        metadata={"kind": "boundary_response_target"},
    )
    history: list[BoundaryResponseOptimizationStep] = []

    for iteration in range(int(n_iterations)):
        current_norm = boundary_response_residual_norm(current_obs, target_vec)
        if convergence_tolerance is not None and current_norm <= float(convergence_tolerance):
            break

        plus_steps, minus_steps = _difference_steps(controls, step_arr, control_bounds, labels)
        plus_rows = []
        minus_rows = []
        assembly_plus_steps = plus_steps.copy()
        assembly_minus_steps = minus_steps.copy()
        for idx in range(controls.size):
            plus_delta = np.zeros_like(controls)
            minus_delta = np.zeros_like(controls)
            plus_delta[idx] = plus_steps[idx]
            minus_delta[idx] = -minus_steps[idx]
            plus_rows.append(current_obs if plus_steps[idx] == 0.0 else _coerce_observables(evaluate_observables(controls + plus_delta)))
            minus_rows.append(current_obs if minus_steps[idx] == 0.0 else _coerce_observables(evaluate_observables(controls + minus_delta)))
            if plus_steps[idx] + minus_steps[idx] == 0.0:
                assembly_plus_steps[idx] = 0.5
                assembly_minus_steps[idx] = 0.5

        system = finite_difference_boundary_response_system(
            current_obs,
            plus_rows,
            minus_rows,
            steps=step_arr,
            plus_steps=assembly_plus_steps,
            minus_steps=assembly_minus_steps,
            control_labels=labels,
            metadata={"iteration": int(iteration)},
        )
        inc_bounds = _increment_bounds(bounds, control_bounds, controls, labels)
        solve = system.solve(
            target_vec,
            bounds=inc_bounds,
            regularization=regularization,
            control_scale=control_scale,
        )
        predicted_norm = float(np.linalg.norm((solve.predicted - solve.target) * np.sqrt(system.weights)))

        before = controls.copy()
        accepted = False
        accepted_alpha = 0.0
        accepted_step = np.zeros_like(controls)
        accepted_obs = current_obs
        accepted_norm = current_norm
        trials: list[tuple[float, float]] = []
        failures: list[tuple[float, str]] = []
        for alpha in alphas:
            trial_step = float(alpha) * solve.controls
            trial_controls = before + trial_step
            if control_bounds is not None:
                abs_lo, abs_hi = _bounds_pair(control_bounds, labels)
                trial_controls = np.minimum(np.maximum(trial_controls, abs_lo), abs_hi)
                trial_step = trial_controls - before
            try:
                trial_obs = _coerce_observables(evaluate_observables(trial_controls))
            except Exception as exc:
                trials.append((float(alpha), float("inf")))
                failures.append((float(alpha), f"{type(exc).__name__}: {exc}"))
                continue
            if trial_obs.labels != current_obs.labels:
                raise ValueError("observable labels must remain stable across iterations")
            trial_norm = boundary_response_residual_norm(trial_obs, target_vec)
            trials.append((float(alpha), float(trial_norm)))
            if np.isfinite(trial_norm) and trial_norm <= current_norm + float(acceptance_tolerance):
                accepted = True
                accepted_alpha = float(alpha)
                accepted_step = np.asarray(trial_step, dtype=float)
                controls = trial_controls
                accepted_obs = trial_obs
                accepted_norm = float(trial_norm)
                break

        step = BoundaryResponseOptimizationStep(
            iteration=int(iteration),
            controls_before=before,
            proposed_step=solve.controls,
            accepted_step=accepted_step,
            controls_after=controls.copy(),
            system=system,
            solve=solve,
            current_observables=current_obs,
            accepted_observables=accepted_obs,
            target_observables=target_obs,
            current_residual_norm=float(current_norm),
            predicted_residual_norm=predicted_norm,
            accepted_residual_norm=float(accepted_norm),
            accepted_alpha=float(accepted_alpha),
            line_search_residuals=tuple(trials),
            accepted=bool(accepted),
            line_search_failures=tuple(failures),
        )
        history.append(step)
        current_obs = accepted_obs
        if not accepted and stop_on_rejected:
            break

    return BoundaryResponseOptimizationResult(
        initial_controls=np.asarray(initial_controls, dtype=float).ravel(),
        final_controls=controls.copy(),
        control_labels=labels,
        steps=tuple(history),
        final_observables=current_obs,
        target_observables=target_obs,
    )


__all__ = [
    "BoundaryResponseOptimizationResult",
    "BoundaryResponseOptimizationStep",
    "boundary_response_residual_norm",
    "iterative_boundary_response_optimization",
]
