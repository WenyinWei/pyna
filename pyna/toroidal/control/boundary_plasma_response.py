"""Plasma-response interfaces for boundary topology design."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from pyna.toroidal.control.boundary_topology_design import (
    BoundaryLinearResponseSystem,
    BoundaryResponseObservables,
    boundary_response_observables,
    finite_difference_boundary_response_system,
    stack_boundary_response_observables,
)


@dataclass(frozen=True)
class CorePreservationSnapshot:
    """Core equilibrium geometry/profile data used as preservation constraints."""

    axis: np.ndarray | None = None
    radial_labels: np.ndarray | None = None
    surface_R: np.ndarray | None = None
    surface_Z: np.ndarray | None = None
    q_profile: np.ndarray | None = None
    iota_profile: np.ndarray | None = None
    scalars: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axis = None if self.axis is None else np.asarray(self.axis, dtype=float).ravel()
        if axis is not None and axis.size != 2:
            raise ValueError("axis must contain exactly two values (R, Z)")
        radial = None if self.radial_labels is None else np.asarray(self.radial_labels, dtype=float).ravel()
        surface_R = None if self.surface_R is None else np.asarray(self.surface_R, dtype=float)
        surface_Z = None if self.surface_Z is None else np.asarray(self.surface_Z, dtype=float)
        if (surface_R is None) != (surface_Z is None):
            raise ValueError("surface_R and surface_Z must be supplied together")
        if surface_R is not None and surface_R.shape != surface_Z.shape:
            raise ValueError("surface_R and surface_Z must have matching shapes")
        q_profile = None if self.q_profile is None else np.asarray(self.q_profile, dtype=float).ravel()
        iota_profile = None if self.iota_profile is None else np.asarray(self.iota_profile, dtype=float).ravel()
        if radial is not None:
            for name, values in (("q_profile", q_profile), ("iota_profile", iota_profile)):
                if values is not None and values.size != radial.size:
                    raise ValueError(f"{name} length must match radial_labels")
        scalars = {str(key): float(value) for key, value in dict(self.scalars or {}).items()}
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "radial_labels", radial)
        object.__setattr__(self, "surface_R", surface_R)
        object.__setattr__(self, "surface_Z", surface_Z)
        object.__setattr__(self, "q_profile", q_profile)
        object.__setattr__(self, "iota_profile", iota_profile)
        object.__setattr__(self, "scalars", scalars)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class BoundaryPlasmaResponseInput:
    """Input passed to a plasma-response backend for one control vector."""

    controls: np.ndarray
    control_labels: tuple[str, ...]
    baseline_equilibrium: Any = None
    baseline_field: Any = None
    vacuum_delta_field: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        controls = np.asarray(self.controls, dtype=float).ravel()
        labels = tuple(str(label) for label in self.control_labels)
        if controls.size != len(labels):
            raise ValueError("controls length must match control_labels")
        if len(set(labels)) != len(labels):
            raise ValueError("control_labels must be unique")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "control_labels", labels)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def controls_by_label(self) -> dict[str, float]:
        """Return control amplitudes keyed by label."""

        return {label: float(value) for label, value in zip(self.control_labels, self.controls)}


@dataclass(frozen=True)
class BoundaryPlasmaResponseSnapshot:
    """Self-consistent or surrogate plasma-response state for one control vector."""

    total_field: Any = None
    background_field: Any = None
    delta_field: Any = None
    equilibrium: Any = None
    core: CorePreservationSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        core = None if self.core is None else core_preservation_snapshot(self.core)
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def has_b0_delta_split(self) -> bool:
        """Whether this response carries an explicit ``B0``/``delta B`` split."""

        return self.background_field is not None and self.delta_field is not None


PlasmaResponseObservableBuilder = Callable[
    [BoundaryPlasmaResponseSnapshot, BoundaryPlasmaResponseInput],
    BoundaryResponseObservables,
]


@runtime_checkable
class BoundaryPlasmaResponseBackend(Protocol):
    """Protocol for vacuum, linear-MHD, or free-boundary plasma response backends."""

    def evaluate(self, request: BoundaryPlasmaResponseInput) -> BoundaryPlasmaResponseSnapshot: ...


@dataclass(frozen=True)
class CallableBoundaryPlasmaResponseBackend:
    """Wrap a callable as a boundary plasma-response backend."""

    evaluator: Callable[[BoundaryPlasmaResponseInput], Any]

    def evaluate(self, request: BoundaryPlasmaResponseInput) -> BoundaryPlasmaResponseSnapshot:
        """Evaluate the wrapped callable and coerce its result to a snapshot."""

        return boundary_plasma_response_snapshot(self.evaluator(request))


@dataclass(frozen=True)
class VacuumBoundaryPlasmaResponseBackend:
    """Vacuum-only backend that forwards the baseline field and vacuum perturbation."""

    core_reference: CorePreservationSnapshot | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def evaluate(self, request: BoundaryPlasmaResponseInput) -> BoundaryPlasmaResponseSnapshot:
        """Return a response snapshot with no plasma feedback."""

        md = {"response_model": "vacuum"}
        md.update(dict(self.metadata or {}))
        md.update(dict(request.metadata or {}))
        return BoundaryPlasmaResponseSnapshot(
            total_field=request.metadata.get("total_field"),
            background_field=request.baseline_field,
            delta_field=request.vacuum_delta_field,
            equilibrium=request.baseline_equilibrium,
            core=self.core_reference,
            metadata=md,
        )


def _mapping_value(obj, *names, default=None):
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def core_preservation_snapshot(obj=None, **kwargs) -> CorePreservationSnapshot:
    """Coerce mapping/object data to a :class:`CorePreservationSnapshot`."""

    if isinstance(obj, CorePreservationSnapshot) and not kwargs:
        return obj
    if obj is None:
        data = dict(kwargs)
    else:
        data = {
            "axis": _mapping_value(obj, "axis", "magnetic_axis", default=None),
            "radial_labels": _mapping_value(obj, "radial_labels", "rho", "s", default=None),
            "surface_R": _mapping_value(obj, "surface_R", "R_surf", "R", default=None),
            "surface_Z": _mapping_value(obj, "surface_Z", "Z_surf", "Z", default=None),
            "q_profile": _mapping_value(obj, "q_profile", "q", default=None),
            "iota_profile": _mapping_value(obj, "iota_profile", "iota", default=None),
            "scalars": _mapping_value(obj, "scalars", default={}),
            "metadata": _mapping_value(obj, "metadata", default={}),
        }
        data.update(kwargs)
    return CorePreservationSnapshot(**data)


def boundary_plasma_response_input(
    controls,
    *,
    control_labels: Sequence[str],
    baseline_equilibrium=None,
    baseline_field=None,
    vacuum_delta_field=None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryPlasmaResponseInput:
    """Build a validated plasma-response backend request."""

    return BoundaryPlasmaResponseInput(
        controls=controls,
        control_labels=tuple(control_labels),
        baseline_equilibrium=baseline_equilibrium,
        baseline_field=baseline_field,
        vacuum_delta_field=vacuum_delta_field,
        metadata={} if metadata is None else dict(metadata),
    )


def boundary_plasma_response_snapshot(obj=None, **kwargs) -> BoundaryPlasmaResponseSnapshot:
    """Coerce mapping/object data to a plasma-response snapshot."""

    if isinstance(obj, BoundaryPlasmaResponseSnapshot) and not kwargs:
        return obj
    if obj is None:
        data = dict(kwargs)
    else:
        data = {
            "total_field": _mapping_value(obj, "total_field", "field", default=None),
            "background_field": _mapping_value(obj, "background_field", "B0", default=None),
            "delta_field": _mapping_value(obj, "delta_field", "delta_B", default=None),
            "equilibrium": _mapping_value(obj, "equilibrium", "eq", default=None),
            "core": _mapping_value(obj, "core", "core_snapshot", default=None),
            "metadata": _mapping_value(obj, "metadata", default={}),
        }
        data.update(kwargs)
    return BoundaryPlasmaResponseSnapshot(**data)


def evaluate_boundary_plasma_response(
    backend,
    controls,
    *,
    control_labels: Sequence[str],
    baseline_equilibrium=None,
    baseline_field=None,
    vacuum_delta_field=None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryPlasmaResponseSnapshot:
    """Evaluate a backend or callable for one control vector."""

    request = boundary_plasma_response_input(
        controls,
        control_labels=control_labels,
        baseline_equilibrium=baseline_equilibrium,
        baseline_field=baseline_field,
        vacuum_delta_field=vacuum_delta_field,
        metadata=metadata,
    )
    return _evaluate_boundary_plasma_response_request(backend, request)


def _evaluate_boundary_plasma_response_request(
    backend,
    request: BoundaryPlasmaResponseInput,
) -> BoundaryPlasmaResponseSnapshot:
    if hasattr(backend, "evaluate"):
        result = backend.evaluate(request)
    else:
        result = backend(request)
    return boundary_plasma_response_snapshot(result)


def _core_observable_rows(
    snapshot: BoundaryPlasmaResponseSnapshot,
    core_reference,
    *,
    core_weights: Mapping[str, float] | None,
    core_prefix: str,
    core_surface_scale: float,
    core_scalar_keys: Sequence[str] | None,
) -> BoundaryResponseObservables | None:
    if core_reference is None:
        return None
    if snapshot.core is None:
        raise ValueError("plasma response snapshot must include core when core_reference is supplied")
    return core_preservation_observables(
        snapshot.core,
        core_reference,
        weights=core_weights,
        prefix=core_prefix,
        surface_scale=core_surface_scale,
        scalar_keys=core_scalar_keys,
    )


def plasma_response_observables(
    backend,
    controls,
    *,
    control_labels: Sequence[str],
    observable_builders: Sequence[PlasmaResponseObservableBuilder] = (),
    core_reference=None,
    core_weights: Mapping[str, float] | None = None,
    core_prefix: str = "core",
    core_surface_scale: float = 1.0,
    core_scalar_keys: Sequence[str] | None = None,
    baseline_equilibrium=None,
    baseline_field=None,
    vacuum_delta_field=None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryResponseObservables:
    """Evaluate a plasma-response backend and assemble boundary observable rows.

    ``observable_builders`` are called as ``builder(snapshot, request)``.  This
    keeps field, spectrum, heat, and fixed-point diagnostics outside the plasma
    backend while preserving one standard row interface for optimization.
    """

    request = boundary_plasma_response_input(
        controls,
        control_labels=control_labels,
        baseline_equilibrium=baseline_equilibrium,
        baseline_field=baseline_field,
        vacuum_delta_field=vacuum_delta_field,
        metadata=metadata,
    )
    snapshot = _evaluate_boundary_plasma_response_request(backend, request)
    groups: list[BoundaryResponseObservables] = []
    for builder in tuple(observable_builders):
        rows = builder(snapshot, request)
        if not isinstance(rows, BoundaryResponseObservables):
            raise TypeError("plasma response observable builders must return BoundaryResponseObservables")
        groups.append(rows)
    core_rows = _core_observable_rows(
        snapshot,
        core_reference,
        core_weights=core_weights,
        core_prefix=core_prefix,
        core_surface_scale=core_surface_scale,
        core_scalar_keys=core_scalar_keys,
    )
    if core_rows is not None:
        groups.append(core_rows)
    md = {
        "kind": "plasma_response_observables",
        "has_b0_delta_split": bool(snapshot.has_b0_delta_split),
        "n_observable_groups": len(groups),
    }
    if metadata:
        md.update(dict(metadata))
    return stack_boundary_response_observables(groups, metadata=md)


def plasma_response_observable_evaluator(
    backend,
    *,
    control_labels: Sequence[str],
    observable_builders: Sequence[PlasmaResponseObservableBuilder] = (),
    core_reference=None,
    core_weights: Mapping[str, float] | None = None,
    core_prefix: str = "core",
    core_surface_scale: float = 1.0,
    core_scalar_keys: Sequence[str] | None = None,
    baseline_equilibrium=None,
    baseline_field=None,
    vacuum_delta_field=None,
    metadata: Mapping[str, Any] | None = None,
) -> Callable[[np.ndarray], BoundaryResponseObservables]:
    """Return a ``controls -> observables`` callback for optimization loops."""

    labels = tuple(str(label) for label in control_labels)
    builders = tuple(observable_builders)
    md = {} if metadata is None else dict(metadata)

    def evaluate(controls) -> BoundaryResponseObservables:
        return plasma_response_observables(
            backend,
            controls,
            control_labels=labels,
            observable_builders=builders,
            core_reference=core_reference,
            core_weights=core_weights,
            core_prefix=core_prefix,
            core_surface_scale=core_surface_scale,
            core_scalar_keys=core_scalar_keys,
            baseline_equilibrium=baseline_equilibrium,
            baseline_field=baseline_field,
            vacuum_delta_field=vacuum_delta_field,
            metadata=md,
        )

    return evaluate


def _difference_step_arrays(steps, plus_steps, minus_steps, n_controls: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_arr = np.broadcast_to(np.asarray(steps, dtype=float), (n_controls,)).copy()
    if np.any(step_arr == 0.0) or not np.all(np.isfinite(step_arr)):
        raise ValueError("steps must be finite and nonzero")
    plus_arr = (
        np.abs(step_arr)
        if plus_steps is None
        else np.broadcast_to(np.asarray(plus_steps, dtype=float), (n_controls,)).copy()
    )
    minus_arr = (
        np.abs(step_arr)
        if minus_steps is None
        else np.broadcast_to(np.asarray(minus_steps, dtype=float), (n_controls,)).copy()
    )
    if np.any(plus_arr < 0.0) or np.any(minus_arr < 0.0):
        raise ValueError("plus_steps and minus_steps must be non-negative")
    if not np.all(np.isfinite(plus_arr)) or not np.all(np.isfinite(minus_arr)):
        raise ValueError("plus_steps and minus_steps must be finite")
    return step_arr, plus_arr, minus_arr


def finite_difference_plasma_response_system(
    backend,
    controls,
    *,
    control_labels: Sequence[str],
    steps=1.0,
    plus_steps=None,
    minus_steps=None,
    observable_builders: Sequence[PlasmaResponseObservableBuilder] = (),
    core_reference=None,
    core_weights: Mapping[str, float] | None = None,
    core_prefix: str = "core",
    core_surface_scale: float = 1.0,
    core_scalar_keys: Sequence[str] | None = None,
    baseline_equilibrium=None,
    baseline_field=None,
    vacuum_delta_field=None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryLinearResponseSystem:
    """Linearize plasma-response observable rows around one control vector."""

    controls_arr = np.asarray(controls, dtype=float).ravel()
    labels = tuple(str(label) for label in control_labels)
    if controls_arr.size != len(labels):
        raise ValueError("controls length must match control_labels")
    if len(set(labels)) != len(labels):
        raise ValueError("control_labels must be unique")
    step_arr, plus_arr, minus_arr = _difference_step_arrays(steps, plus_steps, minus_steps, controls_arr.size)
    evaluate = plasma_response_observable_evaluator(
        backend,
        control_labels=labels,
        observable_builders=observable_builders,
        core_reference=core_reference,
        core_weights=core_weights,
        core_prefix=core_prefix,
        core_surface_scale=core_surface_scale,
        core_scalar_keys=core_scalar_keys,
        baseline_equilibrium=baseline_equilibrium,
        baseline_field=baseline_field,
        vacuum_delta_field=vacuum_delta_field,
        metadata=metadata,
    )
    current = evaluate(controls_arr)
    plus_rows = []
    minus_rows = []
    for idx in range(controls_arr.size):
        plus_controls = controls_arr.copy()
        minus_controls = controls_arr.copy()
        plus_controls[idx] += plus_arr[idx]
        minus_controls[idx] -= minus_arr[idx]
        plus_rows.append(current if plus_arr[idx] == 0.0 else evaluate(plus_controls))
        minus_rows.append(current if minus_arr[idx] == 0.0 else evaluate(minus_controls))
    md = {"kind": "plasma_response_linear_system"}
    if metadata:
        md.update(dict(metadata))
    return finite_difference_boundary_response_system(
        current,
        plus_rows,
        minus_rows,
        steps=step_arr,
        plus_steps=plus_arr,
        minus_steps=minus_arr,
        control_labels=labels,
        metadata=md,
    )


def _rms_and_max_norm(delta_R: np.ndarray, delta_Z: np.ndarray, scale: float) -> tuple[float, float]:
    displacement = np.hypot(delta_R, delta_Z) / float(scale)
    finite = displacement[np.isfinite(displacement)]
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.sqrt(np.mean(finite * finite))), float(np.nanmax(finite))


def _profile_delta_rows(labels, values, weights, name: str, current, reference, *, weight: float) -> None:
    if current is None or reference is None:
        return
    cur = np.asarray(current, dtype=float).ravel()
    ref = np.asarray(reference, dtype=float).ravel()
    if cur.size != ref.size:
        raise ValueError(f"{name} profiles must have matching lengths")
    delta = cur - ref
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        rms = np.nan
        max_abs = np.nan
    else:
        rms = float(np.sqrt(np.mean(finite * finite)))
        max_abs = float(np.nanmax(np.abs(finite)))
    labels.extend((f"{name}.rms_delta", f"{name}.max_abs_delta"))
    values.extend((rms, max_abs))
    weights.extend((float(weight), float(weight)))


def core_preservation_observables(
    current,
    reference,
    *,
    weights: Mapping[str, float] | None = None,
    prefix: str = "core",
    surface_scale: float = 1.0,
    scalar_keys: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundaryResponseObservables:
    """Return observable rows measuring core deviation from a reference state.

    The rows are intended to be driven to zero or preserved during boundary
    topology optimization.
    """

    cur = core_preservation_snapshot(current)
    ref = core_preservation_snapshot(reference)
    w = dict(weights or {})
    labels: list[str] = []
    values: list[float] = []
    row_weights: list[float] = []

    if cur.axis is not None and ref.axis is not None:
        delta = cur.axis - ref.axis
        labels.extend(("axis.dR", "axis.dZ", "axis.displacement"))
        values.extend((float(delta[0]), float(delta[1]), float(np.linalg.norm(delta))))
        row_weights.extend(
            (
                float(w.get("axis.dR", w.get("axis", 1.0))),
                float(w.get("axis.dZ", w.get("axis", 1.0))),
                float(w.get("axis.displacement", w.get("axis", 1.0))),
            )
        )

    if cur.surface_R is not None and ref.surface_R is not None:
        if cur.surface_R.shape != ref.surface_R.shape or cur.surface_Z.shape != ref.surface_Z.shape:
            raise ValueError("current and reference core surfaces must have matching shapes")
        rms, max_disp = _rms_and_max_norm(cur.surface_R - ref.surface_R, cur.surface_Z - ref.surface_Z, surface_scale)
        labels.extend(("surface.rms_displacement", "surface.max_displacement"))
        values.extend((rms, max_disp))
        row_weights.extend(
            (
                float(w.get("surface.rms_displacement", w.get("surface", 1.0))),
                float(w.get("surface.max_displacement", w.get("surface", 1.0))),
            )
        )

    _profile_delta_rows(
        labels,
        values,
        row_weights,
        "q_profile",
        cur.q_profile,
        ref.q_profile,
        weight=float(w.get("q_profile", 1.0)),
    )
    _profile_delta_rows(
        labels,
        values,
        row_weights,
        "iota_profile",
        cur.iota_profile,
        ref.iota_profile,
        weight=float(w.get("iota_profile", 1.0)),
    )

    keys = tuple(scalar_keys) if scalar_keys is not None else tuple(sorted(set(cur.scalars) | set(ref.scalars)))
    for key in keys:
        if key not in cur.scalars or key not in ref.scalars:
            continue
        label = f"scalar.{key}.delta"
        labels.append(label)
        values.append(float(cur.scalars[key] - ref.scalars[key]))
        row_weights.append(float(w.get(label, w.get(f"scalar.{key}", w.get("scalar", 1.0)))))

    md = {"kind": "core_preservation", "surface_scale": float(surface_scale)}
    if metadata:
        md.update(dict(metadata))
    return boundary_response_observables(labels, values, weights=row_weights, prefix=prefix, metadata=md)


__all__ = [
    "BoundaryPlasmaResponseBackend",
    "BoundaryPlasmaResponseInput",
    "BoundaryPlasmaResponseSnapshot",
    "CallableBoundaryPlasmaResponseBackend",
    "CorePreservationSnapshot",
    "PlasmaResponseObservableBuilder",
    "VacuumBoundaryPlasmaResponseBackend",
    "boundary_plasma_response_input",
    "boundary_plasma_response_snapshot",
    "core_preservation_observables",
    "core_preservation_snapshot",
    "evaluate_boundary_plasma_response",
    "finite_difference_plasma_response_system",
    "plasma_response_observable_evaluator",
    "plasma_response_observables",
]
