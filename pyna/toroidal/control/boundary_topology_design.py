"""Boundary-topology design metrics for external perturbation workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import lsq_linear


_EDGE_KEYS = ("closed_core", "boundary_island", "open_loss", "chaotic_edge")
_DPK_UNAVAILABLE_PENALTY = 1.0


@dataclass(frozen=True)
class BoundaryTopologyMetrics:
    """Compact diagnostics extracted from a boundary-topology payload."""

    fixed_points: int
    x_points: int
    o_points: int
    edge_counts: dict[str, int] = field(default_factory=dict)
    edge_fractions: dict[str, float] = field(default_factory=dict)
    strike_centroid_xyz: tuple[float, float, float] | None = None
    strike_spread: float | None = None


@dataclass(frozen=True)
class BoundaryTopologyDesignTarget:
    """Desired boundary topology used to score candidate perturbations.

    Fraction targets may be exact floats or ``(low, high)`` acceptance ranges.
    Set ``preserve_strike_centroid`` or ``preserve_strike_spread`` with a
    reference payload to penalize strike-line drift while changing topology.
    """

    fixed_points: float | None = None
    x_points: float | None = None
    o_points: float | None = None
    boundary_island_fraction: float | tuple[float, float] | None = None
    chaotic_fraction: float | tuple[float, float] | None = None
    open_loss_fraction: float | tuple[float, float] | None = None
    strike_centroid_xyz: Sequence[float] | None = None
    strike_spread: float | None = None
    preserve_strike_centroid: bool = False
    preserve_strike_spread: bool = False
    strike_scale: float = 1.0
    missing_data_penalty: float = 1.0e6
    acceptance: float | None = None
    weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryTopologyDesignScore:
    """Scalar score and component breakdown for a topology-design candidate."""

    total: float
    components: dict[str, float]
    metrics: BoundaryTopologyMetrics
    diagnostics: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        acceptance = self.diagnostics.get("acceptance")
        return bool(acceptance is not None and self.total <= float(acceptance))


@dataclass(frozen=True)
class BoundaryResponseMatrixDiagnostics:
    """Linear controllability diagnostics for a boundary-control response matrix."""

    singular_values: tuple[float, ...]
    condition_number: float
    rank: int
    column_correlation: np.ndarray


@dataclass(frozen=True)
class BoundaryResponseSolveResult:
    """Result of a weighted bounded linear boundary-response solve."""

    controls: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    current: np.ndarray
    target: np.ndarray
    labels: tuple[str, ...]
    diagnostics: BoundaryResponseMatrixDiagnostics
    success: bool
    message: str
    control_labels: tuple[str, ...] = field(default_factory=tuple)
    weighted_residual_norm: float = 0.0
    weighted_control_norm: float = 0.0
    active_lower_bounds: tuple[str, ...] = field(default_factory=tuple)
    active_upper_bounds: tuple[str, ...] = field(default_factory=tuple)

    @property
    def controls_by_label(self) -> dict[str, float]:
        """Return solved control amplitudes keyed by control label."""

        return {label: float(value) for label, value in zip(self.control_labels, self.controls)}


@dataclass(frozen=True)
class BoundaryResponseObservables:
    """Named scalar observables used as rows in a boundary-control problem."""

    labels: tuple[str, ...]
    values: np.ndarray
    weights: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.labels)
        values = np.asarray(self.values, dtype=float).ravel()
        weights = np.asarray(self.weights, dtype=float).ravel()
        if len(labels) != values.size:
            raise ValueError("labels length must match values")
        if weights.size != values.size:
            raise ValueError("weights length must match values")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite and non-negative")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def with_prefix(self, prefix: str) -> "BoundaryResponseObservables":
        """Return a copy whose labels are prefixed with ``prefix``."""

        prefix_s = str(prefix).strip(".")
        if not prefix_s:
            return self
        return BoundaryResponseObservables(
            labels=tuple(f"{prefix_s}.{label}" for label in self.labels),
            values=self.values,
            weights=self.weights,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class BoundaryLinearResponseSystem:
    """Linearized boundary-control system assembled from observable rows."""

    response_matrix: np.ndarray
    current: np.ndarray
    labels: tuple[str, ...]
    weights: np.ndarray
    control_labels: tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.response_matrix, dtype=float)
        current = np.asarray(self.current, dtype=float).ravel()
        weights = np.asarray(self.weights, dtype=float).ravel()
        if matrix.ndim != 2:
            raise ValueError("response_matrix must be 2-D")
        if current.size != matrix.shape[0]:
            raise ValueError("current length must match response_matrix rows")
        if weights.size != matrix.shape[0]:
            raise ValueError("weights length must match response_matrix rows")
        labels = _labels_tuple(self.labels, matrix.shape[0])
        control_labels = tuple(str(label) for label in self.control_labels)
        if len(control_labels) != matrix.shape[1]:
            raise ValueError("control_labels length must match response_matrix columns")
        if len(set(control_labels)) != len(control_labels):
            raise ValueError("control_labels must be unique")
        object.__setattr__(self, "response_matrix", matrix)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "control_labels", control_labels)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def diagnostics(self) -> BoundaryResponseMatrixDiagnostics:
        """Return SVD and column-correlation diagnostics for the system matrix."""

        return boundary_response_matrix_diagnostics(self.response_matrix)

    @property
    def row_index(self) -> dict[str, int]:
        """Return observable row indices keyed by row label."""

        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def control_index(self) -> dict[str, int]:
        """Return control column indices keyed by control label."""

        return {label: idx for idx, label in enumerate(self.control_labels)}

    def solve(
        self,
        target: Mapping[str, float] | Sequence[float],
        *,
        bounds=None,
        regularization: float = 0.0,
        control_scale=None,
    ) -> BoundaryResponseSolveResult:
        """Solve this linearized system for a desired observable target.

        ``target`` may be a vector in row order or a mapping from row labels to
        desired values.  Mapping targets leave unspecified rows at their current
        values, so weighted rows can explicitly preserve observables while only a
        few labels are driven.
        """

        return solve_boundary_response_matrix(
            self.response_matrix,
            self.current,
            target,
            labels=self.labels,
            weights=self.weights,
            control_labels=self.control_labels,
            bounds=bounds,
            regularization=regularization,
            control_scale=control_scale,
        )


@dataclass(frozen=True)
class BoundaryDPKGrowthMetrics:
    """Chaos metrics extracted from cumulative Poincare ``DP^k`` records."""

    n_recorded: int
    k_last: int
    alive_fraction: float
    max_singular: float
    ftle: float
    eigenvalue_ftle: float
    growth_slope: float
    spectral_regularity: float
    svd_regularity: float
    det_error: float
    nonnormality: float
    spectral_recurrence_min: float
    spectral_recurrence_k: int
    spectral_recurrence_fraction: float
    svd_at_spectral_recurrence: float
    recurrent_surface_indicator: float
    term: int | None
    classification: str


def _kind_of(fp) -> str:
    if isinstance(fp, Mapping):
        kind = fp.get("kind", "")
    else:
        kind = getattr(fp, "kind", "")
    return str(kind).upper()


def _count_fixed_points(payload: Mapping) -> tuple[int, int, int]:
    fixed = payload.get("fixed_points")
    if fixed is not None:
        fps = list(fixed)
        x_count = sum(1 for fp in fps if _kind_of(fp) == "X")
        o_count = sum(1 for fp in fps if _kind_of(fp) == "O")
        return len(fps), x_count, o_count

    fp_by_sec = payload.get("fp_by_sec", {})
    total = x_count = o_count = 0
    if isinstance(fp_by_sec, Mapping):
        for section in fp_by_sec.values():
            if isinstance(section, Mapping):
                xpts = list(section.get("xpts", []))
                opts = list(section.get("opts", []))
                x_count += len(xpts)
                o_count += len(opts)
                total += len(xpts) + len(opts)
            else:
                fps = list(section)
                total += len(fps)
                x_count += sum(1 for fp in fps if _kind_of(fp) == "X")
                o_count += sum(1 for fp in fps if _kind_of(fp) == "O")
    return total, x_count, o_count


def _edge_counts(payload: Mapping) -> dict[str, int]:
    counts = {key: 0 for key in _EDGE_KEYS}
    for section in payload.get("edge_state_by_sec", []) or []:
        if not isinstance(section, Mapping):
            continue
        section_counts = section.get("counts")
        if isinstance(section_counts, Mapping):
            for key in _EDGE_KEYS:
                counts[key] += int(section_counts.get(key, 0))
            continue
        for key in _EDGE_KEYS:
            value = section.get(key, [])
            try:
                counts[key] += len(value)
            except TypeError:
                counts[key] += int(bool(value))
    return counts


def _extract_array(mapping: Mapping, *names: str) -> np.ndarray | None:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return np.asarray(value, dtype=float).ravel()
    return None


def _strike_source(payload) -> object | None:
    if payload is None:
        return None
    if hasattr(payload, "hit_R") and hasattr(payload, "hit_Z") and hasattr(payload, "hit_phi"):
        return payload
    if isinstance(payload, Mapping):
        for key in ("strike_points", "wall_strikes", "footprint"):
            source = payload.get(key)
            if source is not None:
                return source
        return payload
    return None


def _strike_moments(payload) -> tuple[tuple[float, float, float] | None, float | None]:
    source = _strike_source(payload)
    if source is None:
        return None, None
    if hasattr(source, "hit_R") and hasattr(source, "hit_Z") and hasattr(source, "hit_phi"):
        R = np.asarray(source.hit_R, dtype=float).ravel()
        Z = np.asarray(source.hit_Z, dtype=float).ravel()
        phi = np.asarray(source.hit_phi, dtype=float).ravel()
        weights = getattr(source, "hit_weight", None)
        w = None if weights is None else np.asarray(weights, dtype=float).ravel()
    elif isinstance(source, Mapping):
        R = _extract_array(source, "R", "hit_R")
        Z = _extract_array(source, "Z", "hit_Z")
        phi = _extract_array(source, "phi", "Phi", "hit_phi")
        w = _extract_array(source, "weight", "weights", "hit_weight")
        if R is None or Z is None or phi is None:
            return None, None
    else:
        return None, None

    R, Z, phi = np.broadcast_arrays(R, Z, phi)
    if w is None:
        w = np.ones(R.shape, dtype=float)
    else:
        w = np.broadcast_to(w, R.shape).astype(float)
    finite = np.isfinite(R) & np.isfinite(Z) & np.isfinite(phi) & np.isfinite(w) & (w > 0.0)
    if not np.any(finite):
        return None, None
    R = R[finite]
    Z = Z[finite]
    phi = phi[finite]
    w = w[finite]
    xyz = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
    wsum = float(np.sum(w))
    centroid = np.sum(xyz * w[:, None], axis=0) / wsum
    spread = float(np.sqrt(np.sum(w * np.sum((xyz - centroid[None, :]) ** 2, axis=1)) / wsum))
    return tuple(float(v) for v in centroid), spread


def boundary_topology_metrics(payload: Mapping) -> BoundaryTopologyMetrics:
    """Extract optimization-friendly topology metrics from a payload."""

    fixed_count, x_count, o_count = _count_fixed_points(payload)
    counts = _edge_counts(payload)
    total_edge = sum(counts.values())
    if total_edge > 0:
        fractions = {key: float(value) / float(total_edge) for key, value in counts.items()}
    else:
        fractions = {key: 0.0 for key in _EDGE_KEYS}
    centroid, spread = _strike_moments(payload)
    return BoundaryTopologyMetrics(
        fixed_points=int(fixed_count),
        x_points=int(x_count),
        o_points=int(o_count),
        edge_counts=counts,
        edge_fractions=fractions,
        strike_centroid_xyz=centroid,
        strike_spread=spread,
    )


def _target_penalty(value: float, target: float | tuple[float, float] | None) -> float | None:
    if target is None:
        return None
    if isinstance(target, tuple):
        if len(target) != 2:
            raise ValueError("range targets must be (low, high)")
        low, high = float(target[0]), float(target[1])
        if high < low:
            raise ValueError("range target high must be >= low")
        if value < low:
            delta = low - value
        elif value > high:
            delta = value - high
        else:
            return 0.0
        scale = max(high - low, abs(0.5 * (low + high)), 1.0)
        return float((delta / scale) ** 2)
    target_value = float(target)
    scale = max(abs(target_value), 1.0)
    return float(((float(value) - target_value) / scale) ** 2)


def _weight(target: BoundaryTopologyDesignTarget, name: str) -> float:
    defaults = {
        "fixed_points": 1.0,
        "x_points": 1.0,
        "o_points": 1.0,
        "boundary_island_fraction": 1.0,
        "chaotic_fraction": 1.0,
        "open_loss_fraction": 1.0,
        "strike_centroid": 1.0,
        "strike_spread": 0.25,
    }
    return float(target.weights.get(name, defaults[name]))


def _add_component(
    components: dict[str, float],
    *,
    target: BoundaryTopologyDesignTarget,
    name: str,
    penalty: float | None,
) -> None:
    if penalty is None:
        return
    components[name] = _weight(target, name) * float(penalty)


def score_boundary_topology_payload(
    payload: Mapping,
    target: BoundaryTopologyDesignTarget,
    *,
    reference_payload: Mapping | object | None = None,
) -> BoundaryTopologyDesignScore:
    """Score a candidate boundary topology against design targets.

    The score is dimensionless and additive, so it can be used directly as an
    objective term inside an external-perturbation optimizer.
    """

    metrics = boundary_topology_metrics(payload)
    components: dict[str, float] = {}
    _add_component(
        components,
        target=target,
        name="fixed_points",
        penalty=_target_penalty(metrics.fixed_points, target.fixed_points),
    )
    _add_component(
        components,
        target=target,
        name="x_points",
        penalty=_target_penalty(metrics.x_points, target.x_points),
    )
    _add_component(
        components,
        target=target,
        name="o_points",
        penalty=_target_penalty(metrics.o_points, target.o_points),
    )
    _add_component(
        components,
        target=target,
        name="boundary_island_fraction",
        penalty=_target_penalty(metrics.edge_fractions["boundary_island"], target.boundary_island_fraction),
    )
    _add_component(
        components,
        target=target,
        name="chaotic_fraction",
        penalty=_target_penalty(metrics.edge_fractions["chaotic_edge"], target.chaotic_fraction),
    )
    _add_component(
        components,
        target=target,
        name="open_loss_fraction",
        penalty=_target_penalty(metrics.edge_fractions["open_loss"], target.open_loss_fraction),
    )

    reference_metrics = boundary_topology_metrics(reference_payload) if reference_payload is not None else None
    centroid_target = target.strike_centroid_xyz
    if centroid_target is None and target.preserve_strike_centroid and reference_metrics is not None:
        centroid_target = reference_metrics.strike_centroid_xyz
    if centroid_target is not None or target.preserve_strike_centroid:
        if metrics.strike_centroid_xyz is None or centroid_target is None:
            components["strike_centroid"] = _weight(target, "strike_centroid") * float(target.missing_data_penalty)
        else:
            actual = np.asarray(metrics.strike_centroid_xyz, dtype=float)
            desired = np.asarray(centroid_target, dtype=float).reshape(3)
            scale = max(abs(float(target.strike_scale)), 1.0e-12)
            components["strike_centroid"] = _weight(target, "strike_centroid") * float(
                np.sum(((actual - desired) / scale) ** 2)
            )

    spread_target = target.strike_spread
    if spread_target is None and target.preserve_strike_spread and reference_metrics is not None:
        spread_target = reference_metrics.strike_spread
    if spread_target is not None or target.preserve_strike_spread:
        if metrics.strike_spread is None or spread_target is None:
            components["strike_spread"] = _weight(target, "strike_spread") * float(target.missing_data_penalty)
        else:
            scale = max(abs(float(spread_target)), abs(float(target.strike_scale)), 1.0e-12)
            components["strike_spread"] = _weight(target, "strike_spread") * float(
                ((float(metrics.strike_spread) - float(spread_target)) / scale) ** 2
            )

    total = float(sum(components.values()))
    diagnostics = {
        "acceptance": target.acceptance,
        "weights": dict(target.weights),
        "reference_used": reference_payload is not None,
    }
    return BoundaryTopologyDesignScore(
        total=total,
        components=components,
        metrics=metrics,
        diagnostics=diagnostics,
    )


def boundary_response_matrix_diagnostics(
    response_matrix: Sequence[Sequence[float]],
    *,
    rcond: float | None = None,
) -> BoundaryResponseMatrixDiagnostics:
    """Return SVD and column-correlation diagnostics for a response matrix."""

    matrix = np.asarray(response_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("response_matrix must be 2-D")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if rcond is None:
        tol = np.finfo(float).eps * max(matrix.shape) * (float(singular[0]) if singular.size else 0.0)
    else:
        tol = float(rcond) * (float(singular[0]) if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tol))
    if singular.size == 0:
        condition = float("inf")
    elif singular[-1] <= max(tol, 0.0):
        condition = float("inf")
    else:
        condition = float(singular[0] / singular[-1])

    n_col = matrix.shape[1]
    correlation = np.eye(n_col, dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    for i in range(n_col):
        for j in range(i + 1, n_col):
            denom = float(norms[i] * norms[j])
            value = 0.0 if denom <= 0.0 else float(np.dot(matrix[:, i], matrix[:, j]) / denom)
            correlation[i, j] = correlation[j, i] = value
    return BoundaryResponseMatrixDiagnostics(
        singular_values=tuple(float(v) for v in singular),
        condition_number=condition,
        rank=rank,
        column_correlation=correlation,
    )


def _labels_tuple(labels: Sequence[str] | None, n_obs: int) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"obs.{i}" for i in range(n_obs))
    out = tuple(str(label) for label in labels)
    if len(out) != n_obs:
        raise ValueError("labels length must match response_matrix rows")
    return out


def _weight_vector(weights, labels: tuple[str, ...]) -> np.ndarray:
    if weights is None:
        return np.ones(len(labels), dtype=float)
    if isinstance(weights, Mapping):
        return np.asarray([float(weights.get(label, 1.0)) for label in labels], dtype=float)
    arr = np.asarray(weights, dtype=float)
    if arr.ndim == 0:
        return np.full(len(labels), float(arr), dtype=float)
    arr = arr.ravel()
    if arr.size != len(labels):
        raise ValueError("weights must be scalar, mapping, or length matching labels")
    return arr


def _target_vector(
    target: Mapping[str, float] | Sequence[float],
    labels: tuple[str, ...],
    current: np.ndarray,
) -> np.ndarray:
    if isinstance(target, Mapping):
        label_index = {label: idx for idx, label in enumerate(labels)}
        unknown = [str(label) for label in target if str(label) not in label_index]
        if unknown:
            raise ValueError(f"target labels are not present in response rows: {unknown}")
        out = np.array(current, dtype=float, copy=True)
        for label, value in target.items():
            arr = np.asarray(value, dtype=float)
            if arr.ndim != 0:
                raise ValueError("mapping target values must be scalar")
            out[label_index[str(label)]] = float(arr)
        return out
    return np.asarray(target, dtype=float).ravel()


def _control_labels_tuple(control_labels: Sequence[str] | None, n_controls: int) -> tuple[str, ...]:
    if control_labels is None:
        return tuple(f"control.{idx}" for idx in range(n_controls))
    out = tuple(str(label) for label in control_labels)
    if len(out) != n_controls:
        raise ValueError("control_labels length must match response_matrix columns")
    if len(set(out)) != len(out):
        raise ValueError("control_labels must be unique")
    return out


def _control_scale_vector(control_scale, control_labels: tuple[str, ...]) -> np.ndarray:
    n_controls = len(control_labels)
    if control_scale is None:
        scale = np.ones(n_controls, dtype=float)
    elif isinstance(control_scale, Mapping):
        control_index = {label: idx for idx, label in enumerate(control_labels)}
        unknown = [str(label) for label in control_scale if str(label) not in control_index]
        if unknown:
            raise ValueError(f"control_scale labels are not present in response columns: {unknown}")
        scale = np.ones(n_controls, dtype=float)
        for label, value in control_scale.items():
            arr = np.asarray(value, dtype=float)
            if arr.ndim != 0:
                raise ValueError("mapping control_scale values must be scalar")
            scale[control_index[str(label)]] = float(arr)
    else:
        scale = np.broadcast_to(np.asarray(control_scale, dtype=float), (n_controls,)).copy()
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("control_scale must contain finite positive values")
    return scale


def _coerce_bound_value(value) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        magnitude = abs(float(arr))
        return -magnitude, magnitude
    flat = arr.ravel()
    if flat.size != 2:
        raise ValueError("mapping bounds values must be scalar magnitudes or (lower, upper) pairs")
    return float(flat[0]), float(flat[1])


def _bounds_pair(
    bounds,
    n_controls: int,
    control_labels: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        return np.full(n_controls, -np.inf), np.full(n_controls, np.inf)
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lower, upper = bounds.lb, bounds.ub
    elif isinstance(bounds, Mapping):
        if control_labels is None:
            control_labels = _control_labels_tuple(None, n_controls)
        control_index = {label: idx for idx, label in enumerate(control_labels)}
        unknown = [str(label) for label in bounds if str(label) not in control_index]
        if unknown:
            raise ValueError(f"bounds labels are not present in response columns: {unknown}")
        lo = np.full(n_controls, -np.inf, dtype=float)
        hi = np.full(n_controls, np.inf, dtype=float)
        for label, value in bounds.items():
            lower_i, upper_i = _coerce_bound_value(value)
            idx = control_index[str(label)]
            lo[idx] = lower_i
            hi[idx] = upper_i
        if np.any(hi < lo):
            raise ValueError("upper bounds must be >= lower bounds")
        return lo, hi
    else:
        lower, upper = bounds
    lo = np.broadcast_to(np.asarray(lower, dtype=float), (n_controls,)).copy()
    hi = np.broadcast_to(np.asarray(upper, dtype=float), (n_controls,)).copy()
    if np.any(hi < lo):
        raise ValueError("upper bounds must be >= lower bounds")
    return lo, hi


def boundary_response_observables(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    weights=None,
    prefix: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Create named scalar observables for boundary-response assembly."""

    labels_tuple = tuple(str(label) for label in labels)
    values_arr = np.asarray(values, dtype=float).ravel()
    if weights is None:
        weights_arr = np.ones(values_arr.size, dtype=float)
    else:
        weights_arr = np.broadcast_to(np.asarray(weights, dtype=float), values_arr.shape).copy()
    rows = BoundaryResponseObservables(
        labels=labels_tuple,
        values=values_arr,
        weights=weights_arr,
        metadata={} if metadata is None else dict(metadata),
    )
    return rows if prefix is None else rows.with_prefix(prefix)


def stack_boundary_response_observables(
    observables: Iterable[BoundaryResponseObservables],
    *,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Stack compatible observable groups into one row vector."""

    groups = tuple(observables)
    if not groups:
        return BoundaryResponseObservables(labels=(), values=np.array([]), weights=np.array([]), metadata=metadata or {})
    labels: list[str] = []
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for group in groups:
        labels.extend(group.labels)
        values.append(np.asarray(group.values, dtype=float))
        weights.append(np.asarray(group.weights, dtype=float))
    if len(set(labels)) != len(labels):
        raise ValueError("observable labels must be unique after stacking")
    md: dict[str, object] = {}
    for group in groups:
        md.update(group.metadata)
    if metadata:
        md.update(dict(metadata))
    return BoundaryResponseObservables(
        labels=tuple(labels),
        values=np.concatenate(values),
        weights=np.concatenate(weights),
        metadata=md,
    )


def _as_observables(obj, *, labels=None, weights=None) -> BoundaryResponseObservables:
    if isinstance(obj, BoundaryResponseObservables):
        return obj
    arr = np.asarray(obj, dtype=float).ravel()
    return boundary_response_observables(_labels_tuple(labels, arr.size), arr, weights=weights)


def finite_difference_boundary_response_system(
    current,
    plus,
    minus,
    *,
    steps=1.0,
    plus_steps=None,
    minus_steps=None,
    labels: Sequence[str] | None = None,
    weights=None,
    control_labels: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryLinearResponseSystem:
    """Assemble a centered finite-difference response system.

    ``plus`` and ``minus`` are sequences of observable vectors, one pair per
    actuator/control column.  By default each column is
    ``(plus - minus) / (2 * step)``.  ``plus_steps`` and ``minus_steps`` may be
    supplied for one-sided or asymmetric finite differences.
    """

    current_rows = _as_observables(current, labels=labels, weights=weights)
    plus_rows = tuple(_as_observables(row, labels=current_rows.labels, weights=current_rows.weights) for row in plus)
    minus_rows = tuple(_as_observables(row, labels=current_rows.labels, weights=current_rows.weights) for row in minus)
    if len(plus_rows) != len(minus_rows):
        raise ValueError("plus and minus must contain the same number of controls")
    n_controls = len(plus_rows)
    if n_controls == 0:
        matrix = np.empty((current_rows.values.size, 0), dtype=float)
    else:
        step_arr = np.broadcast_to(np.asarray(steps, dtype=float), (n_controls,)).copy()
        if np.any(step_arr == 0.0) or not np.all(np.isfinite(step_arr)):
            raise ValueError("steps must be finite and nonzero")
        plus_step_arr = (
            np.abs(step_arr)
            if plus_steps is None
            else np.broadcast_to(np.asarray(plus_steps, dtype=float), (n_controls,)).copy()
        )
        minus_step_arr = (
            np.abs(step_arr)
            if minus_steps is None
            else np.broadcast_to(np.asarray(minus_steps, dtype=float), (n_controls,)).copy()
        )
        if np.any(plus_step_arr < 0.0) or np.any(minus_step_arr < 0.0):
            raise ValueError("plus_steps and minus_steps must be non-negative")
        if not np.all(np.isfinite(plus_step_arr)) or not np.all(np.isfinite(minus_step_arr)):
            raise ValueError("plus_steps and minus_steps must be finite")
        denom = plus_step_arr + minus_step_arr
        if np.any(denom == 0.0):
            raise ValueError("each finite-difference column needs a nonzero plus/minus step")
        columns = []
        for idx, (plus_row, minus_row) in enumerate(zip(plus_rows, minus_rows)):
            if plus_row.labels != current_rows.labels or minus_row.labels != current_rows.labels:
                raise ValueError("all observable labels must match current labels")
            if plus_steps is None and minus_steps is None:
                column = (plus_row.values - minus_row.values) / (2.0 * float(step_arr[idx]))
            else:
                column = (plus_row.values - minus_row.values) / float(denom[idx])
            columns.append(column)
        matrix = np.column_stack(columns)
    if control_labels is None:
        control_tuple = tuple(f"control.{i}" for i in range(n_controls))
    else:
        control_tuple = tuple(str(label) for label in control_labels)
    return BoundaryLinearResponseSystem(
        response_matrix=matrix,
        current=current_rows.values,
        labels=current_rows.labels,
        weights=current_rows.weights,
        control_labels=control_tuple,
        metadata={} if metadata is None else dict(metadata),
    )


def stack_boundary_linear_response_systems(
    systems: Iterable[BoundaryLinearResponseSystem],
    *,
    control_labels: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryLinearResponseSystem:
    """Stack response systems while aligning columns by control label."""

    system_tuple = tuple(systems)
    if not system_tuple:
        raise ValueError("at least one response system is required")
    if control_labels is None:
        ordered_controls: list[str] = []
        for system in system_tuple:
            for label in system.control_labels:
                if label not in ordered_controls:
                    ordered_controls.append(label)
        control_tuple = tuple(ordered_controls)
    else:
        control_tuple = tuple(str(label) for label in control_labels)
    if len(set(control_tuple)) != len(control_tuple):
        raise ValueError("control_labels must be unique")
    control_index = {label: idx for idx, label in enumerate(control_tuple)}

    labels: list[str] = []
    current: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    for system in system_tuple:
        if any(label not in control_index for label in system.control_labels):
            missing = [label for label in system.control_labels if label not in control_index]
            raise ValueError(f"control_labels omitted existing controls: {missing}")
        row_matrix = np.zeros((system.response_matrix.shape[0], len(control_tuple)), dtype=float)
        for src, label in enumerate(system.control_labels):
            row_matrix[:, control_index[label]] = system.response_matrix[:, src]
        labels.extend(system.labels)
        current.append(system.current)
        weights.append(system.weights)
        matrices.append(row_matrix)
    if len(set(labels)) != len(labels):
        raise ValueError("response row labels must be unique after stacking")

    md: dict[str, object] = {}
    for system in system_tuple:
        md.update(system.metadata)
    if metadata:
        md.update(dict(metadata))
    return BoundaryLinearResponseSystem(
        response_matrix=np.vstack(matrices),
        current=np.concatenate(current),
        labels=tuple(labels),
        weights=np.concatenate(weights),
        control_labels=control_tuple,
        metadata=md,
    )


def wall_heat_region_observables(
    heat: Sequence[Sequence[float]],
    regions: Sequence[Sequence[Sequence[bool]]],
    labels: Sequence[str],
    *,
    weights=None,
    normalize: bool = False,
    prefix: str = "strike",
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Sum wall heat in named regions for strike-line control rows."""

    heat_arr = np.asarray(heat, dtype=float)
    label_tuple = tuple(str(label) for label in labels)
    if len(label_tuple) != len(regions):
        raise ValueError("labels length must match regions")
    total = float(np.nansum(heat_arr))
    values = []
    for region in regions:
        mask = np.asarray(region, dtype=bool)
        if mask.shape != heat_arr.shape:
            raise ValueError("each region mask must match heat shape")
        value = float(np.nansum(np.where(mask, heat_arr, 0.0)))
        if normalize:
            value = 0.0 if total == 0.0 else value / total
        values.append(value)
    return boundary_response_observables(
        label_tuple,
        values,
        weights=weights,
        prefix=prefix,
        metadata=metadata,
    )


def chaotic_layer_region_observables(
    intervals: Iterable[object],
    regions: Sequence[tuple[float, float]],
    labels: Sequence[str],
    *,
    weights=None,
    prefix: str = "chaos",
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Return fractional chaotic-layer coverage in named radial regions."""

    label_tuple = tuple(str(label) for label in labels)
    if len(label_tuple) != len(regions):
        raise ValueError("labels length must match regions")
    interval_bounds = [
        (float(getattr(interval, "inner")), float(getattr(interval, "outer")))
        for interval in intervals
        if np.isfinite(float(getattr(interval, "inner"))) and np.isfinite(float(getattr(interval, "outer")))
    ]
    values = []
    for low, high in regions:
        lo = float(low)
        hi = float(high)
        if hi < lo:
            raise ValueError("radial region high must be >= low")
        width = hi - lo
        covered = 0.0
        for inner, outer in interval_bounds:
            covered += max(0.0, min(hi, outer) - max(lo, inner))
        values.append(0.0 if width <= 0.0 else float(np.clip(covered / width, 0.0, 1.0)))
    return boundary_response_observables(
        label_tuple,
        values,
        weights=weights,
        prefix=prefix,
        metadata=metadata,
    )


def resonant_chain_observables(
    chains: Iterable[object],
    modes: Sequence[tuple[int, int]],
    *,
    quantities: Sequence[str] = ("b_res", "half_width"),
    weights=None,
    prefix: str = "island",
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Extract targetable scalar rows from resonant island-chain estimates.

    When several chains share the same physical ``(m, n)``, the largest
    half-width contributor is used.
    """

    selected: dict[tuple[int, int], object] = {}
    for chain in chains:
        key = (int(getattr(chain, "m")), int(getattr(chain, "n")))
        current = selected.get(key)
        if current is None or float(getattr(chain, "half_width", 0.0)) > float(getattr(current, "half_width", 0.0)):
            selected[key] = chain

    labels: list[str] = []
    values: list[float] = []
    for m_int, n_int in modes:
        key = (int(m_int), int(n_int))
        chain = selected.get(key)
        coefficient = 0.0 + 0.0j if chain is None else complex(getattr(chain, "coefficient", 0.0 + 0.0j))
        for quantity in quantities:
            q = str(quantity)
            labels.append(f"m{key[0]}.n{key[1]}.{q}")
            if chain is None:
                values.append(0.0)
            elif q == "radial_label":
                values.append(float(getattr(chain, "radial_label")))
            elif q == "b_res":
                values.append(float(getattr(chain, "b_res")))
            elif q == "half_width":
                values.append(float(getattr(chain, "half_width")))
            elif q == "coefficient_real":
                values.append(float(np.real(coefficient)))
            elif q == "coefficient_imag":
                values.append(float(np.imag(coefficient)))
            elif q == "phase_cos":
                values.append(float(np.cos(np.angle(coefficient))))
            elif q == "phase_sin":
                values.append(float(np.sin(np.angle(coefficient))))
            else:
                raise ValueError(
                    "quantities must contain radial_label, b_res, half_width, "
                    "coefficient_real, coefficient_imag, phase_cos, or phase_sin"
                )
    return boundary_response_observables(
        labels,
        values,
        weights=weights,
        prefix=prefix,
        metadata=metadata,
    )


def _dpk_payload_arrays(payload) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]:
    if isinstance(payload, Mapping):
        k = np.asarray(payload.get("k", payload.get("k_out", [])), dtype=float).ravel()
        DPk = np.asarray(payload.get("DPk", payload.get("DPk_out", [])), dtype=float)
        eig_abs = np.asarray(payload.get("eig_abs", payload.get("eig_abs_out", [])), dtype=float)
        alive = np.asarray(payload.get("alive", payload.get("alive_out", np.ones(k.shape))), dtype=int).ravel()
        term_value = payload.get("term", payload.get("term_type"))
    else:
        seq = tuple(payload)
        if len(seq) < 7:
            raise ValueError("DPk payload tuples must contain at least k, R, Z, phi, DPk, eig_abs, alive")
        k = np.asarray(seq[0], dtype=float).ravel()
        DPk = np.asarray(seq[4], dtype=float)
        eig_abs = np.asarray(seq[5], dtype=float)
        alive = np.asarray(seq[6], dtype=int).ravel()
        term_value = seq[8] if len(seq) >= 9 else None
    term = None
    if term_value is not None:
        term_arr = np.asarray(term_value).ravel()
        if term_arr.size:
            term = int(term_arr[0])
    return k, DPk, eig_abs, alive, term


def _dpk_matrix_stack(DPk: np.ndarray) -> np.ndarray:
    arr = np.asarray(DPk, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2, 2), dtype=float)
    if arr.ndim >= 2 and arr.shape[-2:] == (2, 2):
        return arr.reshape((-1, 2, 2))
    if arr.shape[-1] == 4:
        return arr.reshape((-1, 2, 2))
    if arr.ndim == 1 and arr.size % 4 == 0:
        return arr.reshape((-1, 2, 2))
    raise ValueError("DPk must be shaped as (n, 4), (n, 2, 2), or a flat multiple of 4")


def boundary_dpk_growth_metrics(
    payload,
    *,
    return_period: float = 2.0 * np.pi,
    regular_threshold: float = 0.01,
    weak_chaos_threshold: float = 0.3,
    recurrence_threshold: float = 0.02,
    recurrence_min_k: int = 1,
    recurrence_max_k: int | None = None,
) -> BoundaryDPKGrowthMetrics:
    """Extract chaos/regularity metrics from cyna cumulative ``DP^k`` output.

    ``spectral_recurrence_min`` measures whether some return has eigenvalues
    close to the unit circle, i.e. ``min_k max(abs(log(abs(eig(DP^k)))))``.
    This is useful for distinguishing nested/near-closed magnetic surfaces from
    chaotic layers.  It is reported together with the SVD at the same return so
    nonnormal shear is not mistaken for a regular surface.
    """

    k, DPk_raw, eig_abs_raw, alive_raw, term = _dpk_payload_arrays(payload)
    matrices = _dpk_matrix_stack(DPk_raw)
    n = min(k.size, matrices.shape[0], alive_raw.size)
    if eig_abs_raw.size:
        eig_abs = np.asarray(eig_abs_raw, dtype=float).reshape((-1, 2))[:n]
    else:
        eig_abs = np.empty((0, 2), dtype=float)
    k = k[:n]
    matrices = matrices[:n]
    alive = alive_raw[:n].astype(bool)
    finite_matrix = np.all(np.isfinite(matrices), axis=(1, 2))
    finite_k = np.isfinite(k) & (k > 0.0)
    keep = alive & finite_matrix & finite_k
    alive_fraction = 0.0 if n == 0 else float(np.count_nonzero(keep)) / float(n)
    if term == 1:
        classification = "open_loss"
    elif term == 2 or not np.any(keep):
        classification = "unknown"
    else:
        classification = ""

    if not np.any(keep):
        return BoundaryDPKGrowthMetrics(
            n_recorded=0,
            k_last=0,
            alive_fraction=alive_fraction,
            max_singular=0.0,
            ftle=_DPK_UNAVAILABLE_PENALTY,
            eigenvalue_ftle=_DPK_UNAVAILABLE_PENALTY,
            growth_slope=_DPK_UNAVAILABLE_PENALTY,
            spectral_regularity=_DPK_UNAVAILABLE_PENALTY,
            svd_regularity=_DPK_UNAVAILABLE_PENALTY,
            det_error=_DPK_UNAVAILABLE_PENALTY,
            nonnormality=_DPK_UNAVAILABLE_PENALTY,
            spectral_recurrence_min=_DPK_UNAVAILABLE_PENALTY,
            spectral_recurrence_k=0,
            spectral_recurrence_fraction=0.0,
            svd_at_spectral_recurrence=_DPK_UNAVAILABLE_PENALTY,
            recurrent_surface_indicator=0.0,
            term=term,
            classification=classification,
        )

    k_keep = k[keep]
    m_keep = matrices[keep]
    singular = np.linalg.svd(m_keep, compute_uv=False)
    sigma_max = np.max(singular, axis=1)
    log_sigma = np.log(np.clip(sigma_max, 1.0e-300, None))
    max_log_sigma = float(np.nanmax(log_sigma))
    span = np.abs(k_keep * float(return_period))
    if span.size >= 2 and np.ptp(span) > 0.0:
        growth_slope = float(max(0.0, np.polyfit(span, log_sigma, 1)[0]))
    else:
        growth_slope = 0.0
    k_last = int(k_keep[-1])
    denom = max(abs(float(k_last) * float(return_period)), 1.0e-300)
    ftle = float(max(0.0, max_log_sigma) / denom)
    svd_regularity = float(np.mean(np.abs(log_sigma)))

    if eig_abs.shape[0] >= n:
        eig_keep = eig_abs[keep]
    else:
        eig_keep = np.asarray([np.abs(np.linalg.eigvals(mat)) for mat in m_keep], dtype=float)
    eig_log = np.abs(np.log(np.clip(eig_keep, 1.0e-300, None)))
    spectral_regularity = float(np.mean(np.max(eig_log, axis=1))) if eig_log.size else 0.0
    max_log_eig = float(np.nanmax(eig_log)) if eig_log.size else 0.0
    eigenvalue_ftle = float(max(0.0, max_log_eig) / denom)
    det_error = float(np.nanmax(np.abs(np.linalg.det(m_keep) - 1.0)))
    nonnormality = float(max(0.0, max_log_sigma - max_log_eig))
    eig_recurrence = np.max(eig_log, axis=1) if eig_log.size else np.full(k_keep.shape, np.inf)
    recur_mask = k_keep >= max(1, int(recurrence_min_k))
    if recurrence_max_k is not None:
        recur_mask &= k_keep <= int(recurrence_max_k)
    if np.any(recur_mask):
        recur_scores = eig_recurrence[recur_mask]
        recur_indices = np.nonzero(recur_mask)[0]
        below_threshold = np.nonzero(recur_scores <= float(recurrence_threshold))[0]
        best_local = int(below_threshold[0]) if below_threshold.size else int(np.nanargmin(recur_scores))
        best_idx = int(recur_indices[best_local])
        spectral_recurrence_min = float(recur_scores[best_local])
        spectral_recurrence_k = int(k_keep[best_idx])
        spectral_recurrence_fraction = float(np.mean(recur_scores <= float(recurrence_threshold)))
        svd_at_spectral_recurrence = float(abs(log_sigma[best_idx]))
    else:
        spectral_recurrence_min = _DPK_UNAVAILABLE_PENALTY
        spectral_recurrence_k = 0
        spectral_recurrence_fraction = 0.0
        svd_at_spectral_recurrence = _DPK_UNAVAILABLE_PENALTY
    recurrent_surface_indicator = float(
        spectral_recurrence_min <= float(recurrence_threshold)
        and svd_at_spectral_recurrence <= float(regular_threshold)
    )

    if classification == "":
        score = max(spectral_regularity, svd_regularity)
        if score < float(regular_threshold):
            classification = "regular"
        elif score < float(weak_chaos_threshold):
            classification = "weakly_chaotic"
        else:
            classification = "strongly_chaotic"

    return BoundaryDPKGrowthMetrics(
        n_recorded=int(np.count_nonzero(keep)),
        k_last=k_last,
        alive_fraction=alive_fraction,
        max_singular=float(np.nanmax(sigma_max)),
        ftle=ftle,
        eigenvalue_ftle=eigenvalue_ftle,
        growth_slope=growth_slope,
        spectral_regularity=spectral_regularity,
        svd_regularity=svd_regularity,
        det_error=det_error,
        nonnormality=nonnormality,
        spectral_recurrence_min=spectral_recurrence_min,
        spectral_recurrence_k=spectral_recurrence_k,
        spectral_recurrence_fraction=spectral_recurrence_fraction,
        svd_at_spectral_recurrence=svd_at_spectral_recurrence,
        recurrent_surface_indicator=recurrent_surface_indicator,
        term=term,
        classification=classification,
    )


def dpk_growth_observables(
    payload,
    *,
    return_period: float = 2.0 * np.pi,
    regular_threshold: float = 0.01,
    weak_chaos_threshold: float = 0.3,
    recurrence_threshold: float = 0.02,
    recurrence_min_k: int = 1,
    recurrence_max_k: int | None = None,
    quantities: Sequence[str] | None = None,
    weights=None,
    prefix: str = "dpk",
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Convert cumulative ``DP^k`` output into selected chaos-control rows.

    By default all continuous metrics and classification flags are returned for
    compatibility.  Pass ``quantities`` to build an optimization system from
    continuous rows without imposing preservation targets on categorical flags.
    """

    metrics = (
        payload
        if isinstance(payload, BoundaryDPKGrowthMetrics)
        else boundary_dpk_growth_metrics(
            payload,
            return_period=return_period,
            regular_threshold=regular_threshold,
            weak_chaos_threshold=weak_chaos_threshold,
            recurrence_threshold=recurrence_threshold,
            recurrence_min_k=recurrence_min_k,
            recurrence_max_k=recurrence_max_k,
        )
    )
    class_flags = {
        "open_loss": 1.0 if metrics.classification == "open_loss" else 0.0,
        "unknown": 1.0 if metrics.classification == "unknown" else 0.0,
        "regular": 1.0 if metrics.classification == "regular" else 0.0,
        "weakly_chaotic": 1.0 if metrics.classification == "weakly_chaotic" else 0.0,
        "strongly_chaotic": 1.0 if metrics.classification == "strongly_chaotic" else 0.0,
    }
    available = {
        "ftle": metrics.ftle,
        "eigenvalue_ftle": metrics.eigenvalue_ftle,
        "growth_slope": metrics.growth_slope,
        "spectral_regularity": metrics.spectral_regularity,
        "svd_regularity": metrics.svd_regularity,
        "det_error": metrics.det_error,
        "nonnormality": metrics.nonnormality,
        "spectral_recurrence_min": metrics.spectral_recurrence_min,
        "spectral_recurrence_k": metrics.spectral_recurrence_k,
        "spectral_recurrence_fraction": metrics.spectral_recurrence_fraction,
        "svd_at_spectral_recurrence": metrics.svd_at_spectral_recurrence,
        "recurrent_surface": metrics.recurrent_surface_indicator,
        "alive_fraction": metrics.alive_fraction,
        **class_flags,
    }
    labels = tuple(available) if quantities is None else tuple(str(quantity) for quantity in quantities)
    unknown = [label for label in labels if label not in available]
    if unknown:
        raise ValueError(f"unknown DPk metric quantities: {unknown}")
    values = tuple(float(available[label]) for label in labels)
    md = {
        "classification": metrics.classification,
        "n_recorded": metrics.n_recorded,
        "k_last": metrics.k_last,
        "term": metrics.term,
        "spectral_recurrence_k": metrics.spectral_recurrence_k,
        "recurrence_threshold": float(recurrence_threshold),
    }
    if metadata:
        md.update(dict(metadata))
    return boundary_response_observables(labels, values, weights=weights, prefix=prefix, metadata=md)


def solve_boundary_response_matrix(
    response_matrix: Sequence[Sequence[float]],
    current: Sequence[float],
    target: Mapping[str, float] | Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    weights=None,
    control_labels: Sequence[str] | None = None,
    bounds=None,
    regularization: float = 0.0,
    control_scale=None,
) -> BoundaryResponseSolveResult:
    """Solve a weighted linearized boundary-control problem.

    The rows may mix spectral mode amplitudes, Chirikov/chaotic-layer metrics,
    fixed-point/manifold diagnostics, and strike-line observables.  The function
    only solves the linear algebra; callers remain responsible for nonlinear
    retracing and validation.
    """

    matrix = np.asarray(response_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("response_matrix must be 2-D")
    n_obs, n_controls = matrix.shape
    current_vec = np.asarray(current, dtype=float).ravel()
    if current_vec.size != n_obs:
        raise ValueError("current must match response_matrix rows")
    label_tuple = _labels_tuple(labels, n_obs)
    control_tuple = _control_labels_tuple(control_labels, n_controls)
    target_vec = _target_vector(target, label_tuple, current_vec)
    if target_vec.size != n_obs:
        raise ValueError("target must match response_matrix rows")
    weight_vec = _weight_vector(weights, label_tuple)
    if np.any(weight_vec < 0.0) or not np.all(np.isfinite(weight_vec)):
        raise ValueError("weights must be finite and non-negative")
    sqrt_w = np.sqrt(weight_vec)
    A = matrix * sqrt_w[:, None]
    b = (target_vec - current_vec) * sqrt_w

    reg = float(regularization)
    if reg < 0.0:
        raise ValueError("regularization must be non-negative")
    scale = _control_scale_vector(control_scale, control_tuple)
    if reg > 0.0:
        A = np.vstack([A, np.sqrt(reg) * np.diag(1.0 / scale)])
        b = np.concatenate([b, np.zeros(n_controls, dtype=float)])

    lower, upper = _bounds_pair(bounds, n_controls, control_tuple)
    fixed = lower == upper
    if np.any(fixed & ~np.isfinite(lower)):
        raise ValueError("fixed control bounds must be finite")
    free = ~fixed
    controls = np.zeros(n_controls, dtype=float)
    controls[fixed] = lower[fixed]

    if not np.any(free):
        success = True
        message = "all controls fixed by bounds"
    else:
        A_free = A[:, free]
        b_free = b - A[:, fixed] @ controls[fixed] if np.any(fixed) else b
        free_lower = lower[free]
        free_upper = upper[free]
        if np.all(np.isneginf(free_lower)) and np.all(np.isposinf(free_upper)):
            free_controls, *_rest = np.linalg.lstsq(A_free, b_free, rcond=None)
            success = True
            message = "unbounded least-squares solve"
        else:
            result = lsq_linear(
                A_free,
                b_free,
                bounds=(free_lower, free_upper),
                lsmr_tol="auto",
            )
            free_controls = np.asarray(result.x, dtype=float)
            success = bool(result.success)
            message = str(result.message)
        controls[free] = free_controls
        if np.any(fixed):
            message = f"{message}; fixed {int(np.count_nonzero(fixed))} bounded controls"

    predicted = current_vec + matrix @ controls
    residual = predicted - target_vec
    active_tol = 1.0e-10
    active_lower = tuple(
        label
        for label, value, lower_i in zip(control_tuple, controls, lower)
        if np.isfinite(lower_i) and value <= lower_i + active_tol * max(1.0, abs(lower_i))
    )
    active_upper = tuple(
        label
        for label, value, upper_i in zip(control_tuple, controls, upper)
        if np.isfinite(upper_i) and value >= upper_i - active_tol * max(1.0, abs(upper_i))
    )
    return BoundaryResponseSolveResult(
        controls=np.asarray(controls, dtype=float),
        predicted=np.asarray(predicted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        current=current_vec,
        target=target_vec,
        labels=label_tuple,
        diagnostics=boundary_response_matrix_diagnostics(matrix),
        success=success,
        message=message,
        control_labels=control_tuple,
        weighted_residual_norm=float(np.linalg.norm(residual * sqrt_w)),
        weighted_control_norm=float(np.linalg.norm(np.asarray(controls, dtype=float) / scale)),
        active_lower_bounds=active_lower,
        active_upper_bounds=active_upper,
    )


__all__ = [
    "BoundaryLinearResponseSystem",
    "BoundaryDPKGrowthMetrics",
    "BoundaryResponseObservables",
    "BoundaryResponseMatrixDiagnostics",
    "BoundaryResponseSolveResult",
    "BoundaryTopologyDesignScore",
    "BoundaryTopologyDesignTarget",
    "BoundaryTopologyMetrics",
    "boundary_response_matrix_diagnostics",
    "boundary_response_observables",
    "boundary_topology_metrics",
    "chaotic_layer_region_observables",
    "boundary_dpk_growth_metrics",
    "dpk_growth_observables",
    "finite_difference_boundary_response_system",
    "resonant_chain_observables",
    "score_boundary_topology_payload",
    "solve_boundary_response_matrix",
    "stack_boundary_linear_response_systems",
    "stack_boundary_response_observables",
    "wall_heat_region_observables",
]
