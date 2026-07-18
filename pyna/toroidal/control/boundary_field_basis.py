"""Label-safe linear control bases for cylindrical perturbation fields."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyna.fields import VectorFieldCylind


def _as_candidate(value: Any):
    if hasattr(value, "B_at"):
        return value
    if isinstance(value, VectorFieldCylind) or hasattr(value, "interpolate_at"):
        return CylindricalGridFieldCandidate(value)
    raise TypeError("field candidate must provide B_at or interpolate_at")


@dataclass(frozen=True)
class CylindricalGridFieldCandidate:
    """Expose a cylindrical grid field through the perturbation ``B_at`` API."""

    grid_field: Any
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not hasattr(self.grid_field, "interpolate_at"):
            raise TypeError("grid_field must provide interpolate_at(R, Z, phi)")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def B_at(self, R, Z, phi):
        return self.grid_field.interpolate_at(R, Z, phi)


@dataclass(frozen=True)
class ScaledBoundaryFieldCandidate:
    """Scalar multiple of one perturbation field candidate."""

    candidate: Any
    factor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate", _as_candidate(self.candidate))
        factor = float(self.factor)
        if not np.isfinite(factor):
            raise ValueError("field scale must be finite")
        object.__setattr__(self, "factor", factor)

    def B_at(self, R, Z, phi):
        BR, BZ, BPhi = self.candidate.B_at(R, Z, phi)
        return self.factor * np.asarray(BR), self.factor * np.asarray(BZ), self.factor * np.asarray(BPhi)


@dataclass(frozen=True)
class BoundaryFieldSuperposition:
    """Linear superposition of arbitrary cylindrical perturbation candidates."""

    terms: tuple[ScaledBoundaryFieldCandidate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terms",
            tuple(term if isinstance(term, ScaledBoundaryFieldCandidate) else ScaledBoundaryFieldCandidate(*term) for term in self.terms),
        )

    def B_at(self, R, Z, phi):
        shape = np.broadcast_shapes(np.shape(R), np.shape(Z), np.shape(phi))
        BR = np.zeros(shape, dtype=float)
        BZ = np.zeros(shape, dtype=float)
        BPhi = np.zeros(shape, dtype=float)
        for term in self.terms:
            dBR, dBZ, dBPhi = term.B_at(R, Z, phi)
            BR += np.broadcast_to(np.asarray(dBR, dtype=float), shape)
            BZ += np.broadcast_to(np.asarray(dBZ, dtype=float), shape)
            BPhi += np.broadcast_to(np.asarray(dBPhi, dtype=float), shape)
        return BR, BZ, BPhi


@dataclass(frozen=True)
class BoundaryFieldActuatorSpec:
    """One independently commanded perturbation-field column."""

    label: str
    unit_field: Any
    lower_bound: float = -1.0
    upper_bound: float = 1.0
    control_scale: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        label = str(self.label)
        lower = float(self.lower_bound)
        upper = float(self.upper_bound)
        scale = float(self.control_scale)
        if not label:
            raise ValueError("actuator label must not be empty")
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            raise ValueError("actuator bounds must be finite with upper >= lower")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("control_scale must be positive and finite")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "unit_field", _as_candidate(self.unit_field))
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "control_scale", scale)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def field(self, command: float = 1.0) -> ScaledBoundaryFieldCandidate:
        return ScaledBoundaryFieldCandidate(self.unit_field, float(command))

    @property
    def grid_field(self) -> Any | None:
        return getattr(self.unit_field, "grid_field", None)


@dataclass(frozen=True)
class BoundaryFieldActuatorArray:
    """Label-safe collection of arbitrary linear perturbation-field columns."""

    actuators: tuple[BoundaryFieldActuatorSpec, ...]
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        actuators = tuple(self.actuators)
        if not actuators:
            raise ValueError("at least one field actuator is required")
        if not all(isinstance(item, BoundaryFieldActuatorSpec) for item in actuators):
            raise TypeError("actuators must contain BoundaryFieldActuatorSpec objects")
        labels = tuple(item.label for item in actuators)
        if len(labels) != len(set(labels)):
            raise ValueError("field actuator labels must be unique")
        object.__setattr__(self, "actuators", actuators)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def control_labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.actuators)

    @property
    def control_bounds(self) -> dict[str, tuple[float, float]]:
        return {item.label: (item.lower_bound, item.upper_bound) for item in self.actuators}

    @property
    def control_scale(self) -> np.ndarray:
        return np.asarray([item.control_scale for item in self.actuators], dtype=float)

    def actuator(self, label: str) -> BoundaryFieldActuatorSpec:
        key = str(label)
        for item in self.actuators:
            if item.label == key:
                return item
        raise KeyError(key)

    def unit_fields(self) -> tuple[ScaledBoundaryFieldCandidate, ...]:
        return tuple(item.field(1.0) for item in self.actuators)

    def field(self, controls: Sequence[float]) -> BoundaryFieldSuperposition:
        commands = np.asarray(controls, dtype=float).ravel()
        if commands.size != len(self.actuators):
            raise ValueError("controls length must match field actuator count")
        terms = tuple(
            ScaledBoundaryFieldCandidate(item.unit_field, float(command))
            for command, item in zip(commands, self.actuators)
            if float(command) != 0.0
        )
        return BoundaryFieldSuperposition(terms)

    def grid_field(self, controls: Sequence[float]) -> VectorFieldCylind:
        """Return the commanded perturbation on the shared cylindrical grid."""

        commands = np.asarray(controls, dtype=float).ravel()
        if commands.size != len(self.actuators):
            raise ValueError("controls length must match field actuator count")
        fields = [item.grid_field for item in self.actuators]
        if any(value is None for value in fields):
            raise TypeError("all actuator columns must wrap cylindrical grid fields")
        result = VectorFieldCylind.zero_like(fields[0], label="delta B")
        for command, grid_field in zip(commands, fields):
            if float(command) != 0.0:
                result = result + float(command) * grid_field
        return result


@dataclass(frozen=True)
class CylindricalGridFieldControlBasis:
    """Integrable background grid field plus independently commanded perturbations."""

    background_field: VectorFieldCylind
    actuators: BoundaryFieldActuatorArray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.background_field, VectorFieldCylind):
            raise TypeError("background_field must be VectorFieldCylind")
        for actuator in self.actuators.actuators:
            grid_field = actuator.grid_field
            if grid_field is None:
                raise TypeError("grid control basis requires grid-backed actuator fields")
            _validate_matching_grid(self.background_field, grid_field)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def control_labels(self) -> tuple[str, ...]:
        return self.actuators.control_labels

    @property
    def control_bounds(self) -> dict[str, tuple[float, float]]:
        return self.actuators.control_bounds

    def delta_field(self, controls: Sequence[float]) -> VectorFieldCylind:
        return self.actuators.grid_field(controls)

    def total_field(self, controls: Sequence[float]) -> VectorFieldCylind:
        return self.background_field + self.delta_field(controls)


def _validate_matching_grid(reference: VectorFieldCylind, candidate: VectorFieldCylind) -> None:
    for name in ("R", "Z", "Phi"):
        left = np.asarray(getattr(reference, name), dtype=float)
        right = np.asarray(getattr(candidate, name), dtype=float)
        if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"cylindrical field {name} grids differ")
    if int(reference.nfp) != int(candidate.nfp):
        raise ValueError("cylindrical field periods differ")


def cylindrical_vector_field_from_array(
    values,
    R,
    Z,
    Phi,
    *,
    component_order: Sequence[str] = ("BR", "BZ", "BPhi"),
    nfp: int = 1,
    name: str = "",
) -> VectorFieldCylind:
    """Build a field from ``(R, Z, Phi, component)`` data with explicit order."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("values must have shape (nR, nZ, nPhi, 3)")
    order = tuple(str(value).replace("B_", "B") for value in component_order)
    aliases = {"BR": "BR", "BZ": "BZ", "BPhi": "BPhi", "Bphi": "BPhi"}
    try:
        canonical = tuple(aliases[value] for value in order)
    except KeyError as exc:
        raise ValueError(f"unsupported cylindrical component {exc.args[0]!r}") from exc
    if set(canonical) != {"BR", "BZ", "BPhi"}:
        raise ValueError("component_order must contain BR, BZ, and BPhi exactly once")
    components = {key: array[..., index] for index, key in enumerate(canonical)}
    return VectorFieldCylind(
        R=np.asarray(R, dtype=float),
        Z=np.asarray(Z, dtype=float),
        Phi=np.asarray(Phi, dtype=float),
        BR=components["BR"],
        BZ=components["BZ"],
        BPhi=components["BPhi"],
        nfp=int(nfp),
        name=str(name),
        units="T",
    )


def load_cylindrical_vector_field_npz(path: str | Path, *, name: str = "") -> VectorFieldCylind:
    """Load a metadata-rich cylindrical field NPZ without guessing component order."""

    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as data:
        R = np.asarray(data["R"], dtype=float)
        Z = np.asarray(data["Z"], dtype=float)
        Phi = np.asarray(data["Phi"], dtype=float)
        periods = int(np.asarray(data["n_fp" if "n_fp" in data else "nfp"]).ravel()[0]) if ("n_fp" in data or "nfp" in data) else 1
        if "field" in data:
            values = np.asarray(data["field"], dtype=float)
            if "component_order" not in data:
                raise KeyError("field NPZ requires component_order when using a packed field array")
            order = tuple(str(value) for value in np.asarray(data["component_order"]).ravel())
            return cylindrical_vector_field_from_array(
                values, R, Z, Phi, component_order=order, nfp=periods, name=name
            )
        key_sets = (("BR", "BZ", "BPhi"), ("B_R", "B_Z", "B_Phi"), ("VR", "VZ", "VPhi"))
        for keys in key_sets:
            if all(key in data for key in keys):
                values = np.stack([np.asarray(data[key], dtype=float) for key in keys], axis=-1)
                return cylindrical_vector_field_from_array(
                    values, R, Z, Phi, component_order=("BR", "BZ", "BPhi"), nfp=periods, name=name
                )
    raise KeyError("field NPZ does not contain a supported cylindrical component set")


def boundary_field_actuator_array_from_grid_fields(
    fields: Sequence[VectorFieldCylind],
    *,
    labels: Sequence[str],
    bounds: Sequence[tuple[float, float]] | None = None,
    control_scales: Sequence[float] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryFieldActuatorArray:
    """Build a label-safe actuator array from compatible unit grid fields."""

    field_tuple = tuple(fields)
    label_tuple = tuple(str(value) for value in labels)
    if len(field_tuple) != len(label_tuple):
        raise ValueError("fields and labels must have the same length")
    if not field_tuple:
        raise ValueError("at least one grid field is required")
    for value in field_tuple[1:]:
        _validate_matching_grid(field_tuple[0], value)
    bound_tuple = tuple(bounds) if bounds is not None else tuple((-1.0, 1.0) for _ in field_tuple)
    scales = tuple(control_scales) if control_scales is not None else tuple(1.0 for _ in field_tuple)
    if len(bound_tuple) != len(field_tuple) or len(scales) != len(field_tuple):
        raise ValueError("bounds and control_scales must match the field count")
    return BoundaryFieldActuatorArray(
        actuators=tuple(
            BoundaryFieldActuatorSpec(
                label=label,
                unit_field=CylindricalGridFieldCandidate(value),
                lower_bound=float(bound[0]),
                upper_bound=float(bound[1]),
                control_scale=float(scale),
            )
            for value, label, bound, scale in zip(field_tuple, label_tuple, bound_tuple, scales)
        ),
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "BoundaryFieldActuatorArray",
    "BoundaryFieldActuatorSpec",
    "BoundaryFieldSuperposition",
    "CylindricalGridFieldCandidate",
    "CylindricalGridFieldControlBasis",
    "ScaledBoundaryFieldCandidate",
    "boundary_field_actuator_array_from_grid_fields",
    "cylindrical_vector_field_from_array",
    "load_cylindrical_vector_field_npz",
]
