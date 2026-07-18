"""High-level boundary-topology control workflow."""
from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyna.toroidal.control.boundary_optimization_workflow import (
    BoundaryResponseOptimizationResult,
    iterative_boundary_response_optimization,
)
from pyna.toroidal.control.boundary_plasma_response import (
    PlasmaResponseObservableBuilder,
    finite_difference_plasma_response_system,
    plasma_response_observable_evaluator,
)
from pyna.toroidal.control.boundary_topology_design import (
    BoundaryLinearResponseSystem,
    BoundaryResponseObservables,
    boundary_response_observables,
)


@dataclass(frozen=True)
class BoundaryTopologyControlProblem:
    """Complete control specification for a boundary-topology optimization run.

    The problem combines a plasma-response backend, diagnostic observable
    builders, target rows, actuator bounds, and nonlinear validation settings.
    Core-preservation rows are conventionally zero-targeted through
    ``target_zero_prefixes=("core.",)``.
    """

    backend: Any
    initial_controls: SequenceABC[float]
    control_labels: SequenceABC[str]
    target: MappingABC[str, float] | SequenceABC[float]
    observable_builders: SequenceABC[PlasmaResponseObservableBuilder] = field(default_factory=tuple)
    core_reference: Any = None
    core_weights: MappingABC[str, float] | None = None
    core_prefix: str = "core"
    core_surface_scale: float = 1.0
    core_scalar_keys: SequenceABC[str] | None = None
    target_zero_prefixes: SequenceABC[str] = ("core.",)
    target_preserve_initial_prefixes: SequenceABC[str] = ()
    target_preserve_initial_labels: SequenceABC[str] = ()
    steps: Any = 1.0
    n_iterations: int = 4
    bounds: Any = None
    control_bounds: Any = None
    regularization: float = 0.0
    control_scale: Any = None
    line_search: SequenceABC[float] = (1.0, 0.5, 0.25, 0.125)
    acceptance_tolerance: float = 1.0e-12
    convergence_tolerance: float | None = None
    stop_on_rejected: bool = True
    baseline_equilibrium: Any = None
    baseline_field: Any = None
    vacuum_delta_field: Any = None
    metadata: MappingABC[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        controls = np.asarray(self.initial_controls, dtype=float).ravel()
        labels = tuple(str(label) for label in self.control_labels)
        if controls.size != len(labels):
            raise ValueError("initial_controls length must match control_labels")
        if len(set(labels)) != len(labels):
            raise ValueError("control_labels must be unique")
        target = _coerce_target(self.target)
        object.__setattr__(self, "initial_controls", controls)
        object.__setattr__(self, "control_labels", labels)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "observable_builders", tuple(self.observable_builders))
        object.__setattr__(self, "core_weights", None if self.core_weights is None else dict(self.core_weights))
        object.__setattr__(
            self,
            "core_scalar_keys",
            None if self.core_scalar_keys is None else tuple(str(key) for key in self.core_scalar_keys),
        )
        object.__setattr__(self, "target_zero_prefixes", tuple(str(prefix) for prefix in self.target_zero_prefixes))
        object.__setattr__(
            self,
            "target_preserve_initial_prefixes",
            tuple(str(prefix) for prefix in self.target_preserve_initial_prefixes),
        )
        object.__setattr__(
            self,
            "target_preserve_initial_labels",
            tuple(str(label) for label in self.target_preserve_initial_labels),
        )
        object.__setattr__(self, "line_search", tuple(float(alpha) for alpha in self.line_search))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def evaluator(self):
        """Return the nonlinear ``controls -> observables`` callback."""

        return boundary_topology_control_evaluator(self)

    def linearize(self, controls=None, *, plus_steps=None, minus_steps=None) -> BoundaryLinearResponseSystem:
        """Assemble a finite-difference response system around ``controls``."""

        return linearize_boundary_topology_control(self, controls, plus_steps=plus_steps, minus_steps=minus_steps)

    def solve(self) -> "BoundaryTopologyControlResult":
        """Run the bounded nonlinear-check optimization workflow."""

        return solve_boundary_topology_control(self)


@dataclass(frozen=True)
class BoundaryTopologyControlResidualRow:
    """One final residual row in a boundary-topology control audit."""

    label: str
    initial: float
    final: float
    target: float
    residual: float
    weight: float
    weighted_abs_residual: float


@dataclass(frozen=True)
class BoundaryTopologyControlValidation:
    """Nonlinear before/after validation summary for a control run."""

    initial_observables: BoundaryResponseObservables
    final_observables: BoundaryResponseObservables
    target_observables: BoundaryResponseObservables
    initial_weighted_residual_norm: float
    final_weighted_residual_norm: float
    residual_reduction_fraction: float
    group_weighted_residual_norms: dict[str, float]
    residual_rows: tuple[BoundaryTopologyControlResidualRow, ...]


@dataclass(frozen=True)
class BoundaryTopologyControlResult:
    """High-level result for one boundary-topology control problem."""

    problem: BoundaryTopologyControlProblem
    resolved_target: MappingABC[str, float] | np.ndarray
    optimization: BoundaryResponseOptimizationResult
    validation: BoundaryTopologyControlValidation

    @property
    def final_controls(self) -> np.ndarray:
        """Return final control amplitudes."""

        return self.optimization.final_controls

    @property
    def controls_by_label(self) -> dict[str, float]:
        """Return final control amplitudes keyed by label."""

        return self.optimization.controls_by_label

    @property
    def steps(self):
        """Return nonlinear optimization steps."""

        return self.optimization.steps

    @property
    def final_system(self) -> BoundaryLinearResponseSystem | None:
        """Return the last assembled linear system, if any iterations ran."""

        return None if not self.optimization.steps else self.optimization.steps[-1].system


def _coerce_target(target):
    if isinstance(target, MappingABC):
        out: dict[str, float] = {}
        for label, value in target.items():
            arr = np.asarray(value, dtype=float)
            if arr.ndim != 0:
                raise ValueError("mapping target values must be scalar")
            out[str(label)] = float(arr)
        return out
    return np.asarray(target, dtype=float).ravel()


def _matches_prefix(label: str, prefixes: SequenceABC[str]) -> bool:
    return any(str(label).startswith(str(prefix)) for prefix in prefixes)


def resolve_boundary_topology_control_target(
    target: MappingABC[str, float] | SequenceABC[float],
    observables: BoundaryResponseObservables,
    *,
    zero_prefixes: SequenceABC[str] = ("core.",),
    preserve_initial_prefixes: SequenceABC[str] = (),
    preserve_initial_labels: SequenceABC[str] = (),
) -> MappingABC[str, float] | np.ndarray:
    """Resolve target rows and freeze omitted mapping rows at their initial values."""

    rows = observables
    coerced = _coerce_target(target)
    if not isinstance(coerced, MappingABC):
        if coerced.size != rows.values.size:
            raise ValueError("target length must match observable labels")
        return coerced.copy()

    labels = tuple(rows.labels)
    index = {label: idx for idx, label in enumerate(labels)}
    unknown = [label for label in coerced if label not in index]
    if unknown:
        raise ValueError(f"target labels are not present in observables: {unknown}")
    resolved = dict(coerced)
    zero = tuple(str(prefix) for prefix in zero_prefixes)
    for idx, label in enumerate(labels):
        if label in resolved:
            continue
        if _matches_prefix(label, zero):
            resolved[label] = 0.0
        else:
            resolved[label] = float(rows.values[idx])
    return resolved


def boundary_topology_control_evaluator(problem: BoundaryTopologyControlProblem):
    """Return a nonlinear evaluator for a high-level topology-control problem."""

    return plasma_response_observable_evaluator(
        problem.backend,
        control_labels=problem.control_labels,
        observable_builders=problem.observable_builders,
        core_reference=problem.core_reference,
        core_weights=problem.core_weights,
        core_prefix=problem.core_prefix,
        core_surface_scale=problem.core_surface_scale,
        core_scalar_keys=problem.core_scalar_keys,
        baseline_equilibrium=problem.baseline_equilibrium,
        baseline_field=problem.baseline_field,
        vacuum_delta_field=problem.vacuum_delta_field,
        metadata=problem.metadata,
    )


def linearize_boundary_topology_control(
    problem: BoundaryTopologyControlProblem,
    controls=None,
    *,
    plus_steps=None,
    minus_steps=None,
) -> BoundaryLinearResponseSystem:
    """Assemble the finite-difference control matrix for a topology-control problem."""

    base = problem.initial_controls if controls is None else np.asarray(controls, dtype=float).ravel()
    return finite_difference_plasma_response_system(
        problem.backend,
        base,
        control_labels=problem.control_labels,
        steps=problem.steps,
        plus_steps=plus_steps,
        minus_steps=minus_steps,
        observable_builders=problem.observable_builders,
        core_reference=problem.core_reference,
        core_weights=problem.core_weights,
        core_prefix=problem.core_prefix,
        core_surface_scale=problem.core_surface_scale,
        core_scalar_keys=problem.core_scalar_keys,
        baseline_equilibrium=problem.baseline_equilibrium,
        baseline_field=problem.baseline_field,
        vacuum_delta_field=problem.vacuum_delta_field,
        metadata=problem.metadata,
    )


def _target_values(target, rows: BoundaryResponseObservables) -> np.ndarray:
    if isinstance(target, MappingABC):
        values = np.asarray(rows.values, dtype=float).ravel().copy()
        index = {label: idx for idx, label in enumerate(rows.labels)}
        unknown = [str(label) for label in target if str(label) not in index]
        if unknown:
            raise ValueError(f"target labels are not present in observables: {unknown}")
        for label, value in target.items():
            values[index[str(label)]] = float(value)
        return values
    arr = np.asarray(target, dtype=float).ravel()
    if arr.size != rows.values.size:
        raise ValueError("target length must match observable labels")
    return arr.copy()


def _weighted_norm(values, target, weights) -> float:
    return float(np.linalg.norm((np.asarray(values, dtype=float) - np.asarray(target, dtype=float)) * np.sqrt(weights)))


def _group_key(label: str) -> str:
    return str(label).split(".", 1)[0]


def validate_boundary_topology_control(
    initial_observables: BoundaryResponseObservables,
    final_observables: BoundaryResponseObservables,
    target_observables: BoundaryResponseObservables,
    *,
    top_n_residuals: int | None = None,
) -> BoundaryTopologyControlValidation:
    """Compute before/after residual diagnostics for a finished control run."""

    if initial_observables.labels != final_observables.labels or final_observables.labels != target_observables.labels:
        raise ValueError("validation observables must share labels")
    initial = np.asarray(initial_observables.values, dtype=float).ravel()
    final = np.asarray(final_observables.values, dtype=float).ravel()
    target = np.asarray(target_observables.values, dtype=float).ravel()
    weights = np.asarray(final_observables.weights, dtype=float).ravel()
    if not (initial.size == final.size == target.size == weights.size == len(final_observables.labels)):
        raise ValueError("validation observable arrays must share one length")
    initial_norm = _weighted_norm(initial, target, weights)
    final_norm = _weighted_norm(final, target, weights)
    reduction_fraction = 0.0 if initial_norm <= 0.0 or not np.isfinite(initial_norm) else (initial_norm - final_norm) / initial_norm
    weighted_residual = (final - target) * np.sqrt(weights)
    group_norms: dict[str, float] = {}
    for label, value in zip(final_observables.labels, weighted_residual):
        key = _group_key(label)
        group_norms[key] = group_norms.get(key, 0.0) + float(value * value)
    group_norms = {key: float(np.sqrt(value)) for key, value in sorted(group_norms.items())}
    rows = [
        BoundaryTopologyControlResidualRow(
            label=label,
            initial=float(initial[idx]),
            final=float(final[idx]),
            target=float(target[idx]),
            residual=float(final[idx] - target[idx]),
            weight=float(weights[idx]),
            weighted_abs_residual=float(abs(final[idx] - target[idx]) * np.sqrt(weights[idx])),
        )
        for idx, label in enumerate(final_observables.labels)
    ]
    rows.sort(key=lambda row: row.weighted_abs_residual, reverse=True)
    if top_n_residuals is not None:
        rows = rows[: max(0, int(top_n_residuals))]
    return BoundaryTopologyControlValidation(
        initial_observables=initial_observables,
        final_observables=final_observables,
        target_observables=target_observables,
        initial_weighted_residual_norm=float(initial_norm),
        final_weighted_residual_norm=float(final_norm),
        residual_reduction_fraction=float(reduction_fraction),
        group_weighted_residual_norms=group_norms,
        residual_rows=tuple(rows),
    )


def solve_boundary_topology_control(
    problem: BoundaryTopologyControlProblem,
    *,
    top_n_residuals: int | None = 12,
) -> BoundaryTopologyControlResult:
    """Run the complete nonlinear-check boundary-topology control workflow."""

    evaluator = boundary_topology_control_evaluator(problem)
    initial_observables = evaluator(problem.initial_controls)
    resolved_target = resolve_boundary_topology_control_target(
        problem.target,
        initial_observables,
        zero_prefixes=problem.target_zero_prefixes,
        preserve_initial_prefixes=problem.target_preserve_initial_prefixes,
        preserve_initial_labels=problem.target_preserve_initial_labels,
    )
    optimization = iterative_boundary_response_optimization(
        evaluator,
        problem.initial_controls,
        resolved_target,
        control_labels=problem.control_labels,
        steps=problem.steps,
        n_iterations=problem.n_iterations,
        bounds=problem.bounds,
        control_bounds=problem.control_bounds,
        regularization=problem.regularization,
        control_scale=problem.control_scale,
        line_search=problem.line_search,
        acceptance_tolerance=problem.acceptance_tolerance,
        convergence_tolerance=problem.convergence_tolerance,
        stop_on_rejected=problem.stop_on_rejected,
    )
    validation = validate_boundary_topology_control(
        initial_observables,
        optimization.final_observables,
        optimization.target_observables,
        top_n_residuals=top_n_residuals,
    )
    return BoundaryTopologyControlResult(
        problem=problem,
        resolved_target=resolved_target,
        optimization=optimization,
        validation=validation,
    )


def format_boundary_topology_control_summary(
    result: BoundaryTopologyControlResult,
    *,
    top_n_residuals: int = 8,
) -> str:
    """Return a compact text audit for a boundary-topology control result."""

    validation = result.validation
    lines = [
        "Boundary topology control report",
        "",
        f"optimization steps: {len(result.steps)}",
        f"accepted steps: {sum(bool(step.accepted) for step in result.steps)}",
        f"initial residual norm: {validation.initial_weighted_residual_norm:.6g}",
        f"final residual norm: {validation.final_weighted_residual_norm:.6g}",
        f"residual reduction fraction: {validation.residual_reduction_fraction:.6g}",
        "final controls:",
    ]
    for label, value in result.controls_by_label.items():
        lines.append(f"  {label}: {value:.6g}")
    if result.final_system is not None:
        diagnostics = result.final_system.diagnostics
        lines.extend(
            [
                "linear response diagnostics:",
                f"  rank: {diagnostics.rank}",
                f"  condition number: {diagnostics.condition_number:.6g}",
            ]
        )
    if validation.group_weighted_residual_norms:
        lines.append("final residual norm by group:")
        for label, value in validation.group_weighted_residual_norms.items():
            lines.append(f"  {label}: {value:.6g}")
    lines.append("largest final observable residual rows:")
    for row in validation.residual_rows[: max(0, int(top_n_residuals))]:
        lines.append(
            f"  {row.label}: final={row.final:.6g}, target={row.target:.6g}, "
            f"residual={row.residual:.6g}, weighted_abs={row.weighted_abs_residual:.6g}"
        )
    return "\n".join(lines)


__all__ = [
    "BoundaryTopologyControlProblem",
    "BoundaryTopologyControlResidualRow",
    "BoundaryTopologyControlResult",
    "BoundaryTopologyControlValidation",
    "boundary_topology_control_evaluator",
    "format_boundary_topology_control_summary",
    "linearize_boundary_topology_control",
    "resolve_boundary_topology_control_target",
    "solve_boundary_topology_control",
    "validate_boundary_topology_control",
]
