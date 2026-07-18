"""Bridge boundary topology seeds to wall strikes and heat observables.

This module deliberately reuses the public cyna toroidal-wall tracer and the
standard :class:`BoundaryTopologyHeatState`.  It adds no independent tracing,
transport, or optimization implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from pyna.toroidal.control.heat_contracts import BoundaryTopologyHeatState
from pyna.toroidal.flt.numba_poincare import (
    strike_line_from_wall_hits,
    trace_wall_hits_twall_field,
)
from pyna.toroidal.geometry import (
    coerce_toroidal_surface_arrays,
    project_points_to_toroidal_surface,
)


TWOPI = 2.0 * np.pi
_MISSING = object()


def _nonempty_label(value: object, name: str = "label") -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{name} must not be empty")
    return label


def _direction(value: object) -> Literal["+", "-"]:
    text = str(value).strip().lower()
    if text in {"+", "plus", "forward", "fwd", "phi+", "increasing"}:
        return "+"
    if text in {"-", "minus", "backward", "back", "reverse", "rev", "phi-", "decreasing"}:
        return "-"
    raise ValueError("direction must be '+' or '-'")


def _item_value(item: object, names: Sequence[str], default: object = _MISSING) -> object:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    if default is _MISSING:
        raise KeyError("item requires one of: " + ", ".join(names))
    return default


def _wall_arrays_and_period(wall: object, field_period: float | None = None):
    if all(hasattr(wall, name) for name in ("phi_values", "R", "Z")):
        phi = np.asarray(getattr(wall, "phi_values"), dtype=float).ravel()
        R = np.asarray(getattr(wall, "R"), dtype=float)
        Z = np.asarray(getattr(wall, "Z"), dtype=float)
    else:
        phi, R, Z = coerce_toroidal_surface_arrays(wall)
        phi = np.asarray(phi, dtype=float).ravel()
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
    if field_period is None:
        if hasattr(wall, "field_period"):
            period = float(getattr(wall, "field_period"))
        elif hasattr(wall, "toroidal_period"):
            period = float(getattr(wall, "toroidal_period"))
        else:
            period = TWOPI
    else:
        period = float(field_period)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("field_period must be positive and finite")
    return phi, R, Z, period


@dataclass(frozen=True)
class StrikeSeedBundle:
    """Ordered seeds for one physical strike-producing topology branch.

    ``relative`` weights describe only a profile and require a later total
    power.  ``power`` weights are already absolute powers.  Metadata always
    records whether the topology and weight provenance justify quantitative
    use; builders default to an explicit proxy classification.
    """

    label: str
    mode: Literal["chaotic_manifold", "regular_island"]
    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    direction: Literal["+", "-"]
    weights: np.ndarray
    weight_kind: Literal["relative", "power"]
    source_coordinate: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        mode = str(self.mode)
        if mode not in {"chaotic_manifold", "regular_island"}:
            raise ValueError("mode must be 'chaotic_manifold' or 'regular_island'")
        direction = _direction(self.direction)
        weight_kind = str(self.weight_kind).lower()
        if weight_kind not in {"relative", "power"}:
            raise ValueError("weight_kind must be 'relative' or 'power'")
        arrays = [
            np.asarray(value, dtype=float).ravel()
            for value in (self.R, self.Z, self.phi, self.weights, self.source_coordinate)
        ]
        R, Z, phi, weights, source_coordinate = arrays
        if R.size == 0 or any(value.size != R.size for value in arrays[1:]):
            raise ValueError("bundle arrays must be non-empty and have matching lengths")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("bundle arrays must be finite")
        if np.any(R <= 0.0):
            raise ValueError("bundle R must be positive")
        if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("bundle weights must be non-negative with positive sum")
        metadata = dict(self.metadata or {})
        metadata.setdefault("quantitative", False)
        metadata.setdefault("topology_provenance", "caller_supplied")
        metadata.setdefault("weight_provenance", "caller_supplied")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "weight_kind", weight_kind)
        object.__setattr__(self, "R", R.copy())
        object.__setattr__(self, "Z", Z.copy())
        object.__setattr__(self, "phi", phi.copy())
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "source_coordinate", source_coordinate.copy())
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class WallStrikeSamples:
    """Resolved first-wall strikes from one ordered seed bundle."""

    label: str
    mode: str
    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    weights: np.ndarray
    connection_length: np.ndarray
    seed_index: np.ndarray
    direction: Literal["+", "-"]
    unresolved_weight: float
    weight_kind: Literal["relative", "power"] = "relative"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        mode = str(self.mode)
        if mode not in {"chaotic_manifold", "regular_island"}:
            raise ValueError("invalid strike mode")
        direction = _direction(self.direction)
        weight_kind = str(self.weight_kind).lower()
        if weight_kind not in {"relative", "power"}:
            raise ValueError("weight_kind must be 'relative' or 'power'")
        arrays = [
            np.asarray(value, dtype=float).ravel()
            for value in (self.R, self.Z, self.phi, self.weights, self.connection_length)
        ]
        R, Z, phi, weights, connection_length = arrays
        seed_index = np.asarray(self.seed_index, dtype=int).ravel()
        if any(value.size != R.size for value in arrays[1:]) or seed_index.size != R.size:
            raise ValueError("wall-strike arrays must have matching lengths")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("wall-strike arrays must be finite")
        if R.size and np.any(R <= 0.0):
            raise ValueError("wall-strike R must be positive")
        if np.any(weights < 0.0) or np.any(connection_length < 0.0):
            raise ValueError("strike weights and connection lengths must be non-negative")
        if np.any(seed_index < 0) or np.unique(seed_index).size != seed_index.size:
            raise ValueError("seed_index must contain unique non-negative indices")
        unresolved = float(self.unresolved_weight)
        if not np.isfinite(unresolved) or unresolved < 0.0:
            raise ValueError("unresolved_weight must be finite and non-negative")
        if float(np.sum(weights)) + unresolved <= 0.0:
            raise ValueError("resolved and unresolved weights cannot both be zero")
        metadata = dict(self.metadata or {})
        metadata.setdefault("quantitative", False)
        metadata.setdefault("projection", "raw_toroidal_wall_collision")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "weight_kind", weight_kind)
        object.__setattr__(self, "R", R.copy())
        object.__setattr__(self, "Z", Z.copy())
        object.__setattr__(self, "phi", phi.copy())
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "connection_length", connection_length.copy())
        object.__setattr__(self, "seed_index", seed_index.copy())
        object.__setattr__(self, "unresolved_weight", unresolved)
        object.__setattr__(self, "metadata", metadata)

    @property
    def launched_weight(self) -> float:
        return float(np.sum(self.weights)) + float(self.unresolved_weight)


def _resolve_weights(
    default_weights: np.ndarray,
    context: Mapping[str, object],
    weight_builder: Callable[[Mapping[str, object]], object] | None,
    *,
    default_kind: str = "relative",
    default_quantitative: bool = False,
    default_provenance: str = "uniform_relative_seed_proxy",
) -> tuple[np.ndarray, str, bool, str, dict[str, object]]:
    result = None if weight_builder is None else weight_builder(context)
    metadata: dict[str, object] = {}
    if result is None:
        weights = np.asarray(default_weights, dtype=float).ravel()
        kind = str(default_kind).lower()
        quantitative = bool(default_quantitative)
        provenance = _nonempty_label(default_provenance, "weight provenance")
    elif isinstance(result, Mapping):
        if "weights" not in result:
            raise ValueError("weight_builder mapping must contain 'weights'")
        weights = np.asarray(result["weights"], dtype=float).ravel()
        kind = str(result.get("weight_kind", default_kind)).lower()
        quantitative = bool(result.get("quantitative", False))
        provenance = _nonempty_label(result.get("provenance", "weight_builder"), "weight provenance")
        metadata = dict(result.get("metadata", {}) or {})
    else:
        weights = np.asarray(result, dtype=float).ravel()
        kind = str(default_kind).lower()
        quantitative = False
        provenance = "weight_builder_array_proxy"
    if weights.size != np.asarray(default_weights).size:
        raise ValueError("weight_builder output must match the branch seed count")
    if kind not in {"relative", "power"}:
        raise ValueError("weight_kind must be 'relative' or 'power'")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    return weights, kind, quantitative, provenance, metadata


def manifold_strike_seed_bundles(
    manifold_payloads: Sequence[Mapping[str, object]],
    *,
    weight_builder: Callable[[Mapping[str, object]], object] | None = None,
) -> tuple[StrikeSeedBundle, ...]:
    """Adapt public X-point manifold payload seeds into four side branches.

    For a positive signed return-map span, unstable seeds trace in ``+phi``
    and stable seeds in ``-phi``.  A negative span reverses both physical
    directions.  Seed arrays are filtered by side without sorting, preserving
    the tracer's branch order and ``*_seed_order`` provenance.
    """

    bundles: list[StrikeSeedBundle] = []
    for payload_index, payload in enumerate(manifold_payloads):
        if not isinstance(payload, Mapping):
            raise TypeError("manifold payloads must be mappings")
        map_span = float(payload.get("manifold_field_period", np.nan))
        if not np.isfinite(map_span) or map_span == 0.0:
            raise ValueError("manifold_field_period must be finite and non-zero")
        origin_phi = float(payload.get("origin_phi", np.nan))
        if not np.isfinite(origin_phi):
            raise ValueError("manifold origin_phi must be finite")
        origin_label = payload.get("manifold_origin_label")
        if origin_label is None:
            origin_label = f"manifold{payload_index}"
        origin_label = _nonempty_label(origin_label)
        for code, stability in (("u", "unstable"), ("s", "stable")):
            R = np.asarray(payload.get(f"{code}_seed_R", []), dtype=float).ravel()
            Z = np.asarray(payload.get(f"{code}_seed_Z", []), dtype=float).ravel()
            distance = np.asarray(payload.get(f"{code}_seed_distance", []), dtype=float).ravel()
            side = np.asarray(payload.get(f"{code}_seed_side", []), dtype=float).ravel()
            order = np.asarray(payload.get(f"{code}_seed_order", []), dtype=int).ravel()
            if R.size == 0:
                continue
            if any(value.size != R.size for value in (Z, distance, side, order)):
                raise ValueError(f"{code}_seed_* arrays must have matching lengths")
            if not all(np.all(np.isfinite(value)) for value in (R, Z, distance, side)):
                raise ValueError(f"{code}_seed_* arrays must be finite")
            side_sign = np.sign(side).astype(int)
            if np.any(~np.isin(side_sign, (-1, 1))):
                raise ValueError(f"{code}_seed_side must contain only -1 or +1")
            physical_sign = np.sign(map_span) * (1.0 if stability == "unstable" else -1.0)
            trace_direction: Literal["+", "-"] = "+" if physical_sign > 0.0 else "-"
            for branch_side in (-1, 1):
                indices = np.flatnonzero(side_sign == branch_side)
                if indices.size == 0:
                    continue
                branch_R = R[indices]
                branch_Z = Z[indices]
                branch_distance = distance[indices]
                context = {
                    "payload": payload,
                    "payload_index": payload_index,
                    "stability": stability,
                    "side": branch_side,
                    "indices": indices.copy(),
                    "R": branch_R.copy(),
                    "Z": branch_Z.copy(),
                    "source_coordinate": branch_distance.copy(),
                }
                weights, kind, quantitative, provenance, weight_metadata = _resolve_weights(
                    np.ones(indices.size, dtype=float),
                    context,
                    weight_builder,
                )
                side_label = "minus" if branch_side < 0 else "plus"
                metadata = {
                    "quantitative": bool(quantitative),
                    "topology_provenance": "trace_fixed_point_manifolds_field.raw_local_seeds",
                    "weight_provenance": provenance,
                    "stability": stability,
                    "side": branch_side,
                    "signed_map_span": map_span,
                    "map_span_source": payload.get("manifold_field_period_source"),
                    "seed_order": order[indices].copy(),
                    "seed_indices": indices.copy(),
                    "chain_id": payload.get("chain_id"),
                    "orbit_id": payload.get("orbit_id"),
                    "point_index": payload.get("point_index"),
                }
                metadata.update(weight_metadata)
                bundles.append(
                    StrikeSeedBundle(
                        label=f"{origin_label}.{stability}.{side_label}",
                        mode="chaotic_manifold",
                        R=branch_R,
                        Z=branch_Z,
                        phi=np.full(indices.size, origin_phi, dtype=float),
                        direction=trace_direction,
                        weights=weights,
                        weight_kind=kind,
                        source_coordinate=branch_distance,
                        metadata=metadata,
                    )
                )
    return tuple(bundles)


def island_strike_seed_bundles(
    contours: Sequence[object],
    *,
    weight_builder: Callable[[Mapping[str, object]], object] | None = None,
) -> tuple[StrikeSeedBundle, ...]:
    """Build ordered regular-island seed bundles from generic contours.

    A contour may be a mapping or object exposing ``R``/``Z`` (or
    ``points_RZ_m``), ``phi``/``section_phi``, and optional ``directions``.
    Both tracing directions are used by default, with each seed's weight split
    evenly so absolute power is not duplicated.  Quantitative contours must
    explicitly declare ``closed=True`` and ``dpk_regular=True``; all other
    contours remain visible geometric proxies.
    """

    bundles: list[StrikeSeedBundle] = []
    for contour_index, contour in enumerate(contours):
        points = _item_value(contour, ("points_RZ_m", "points"), default=None)
        if points is not None:
            points_RZ = np.asarray(points, dtype=float)
            if points_RZ.ndim != 2 or points_RZ.shape[1] != 2:
                raise ValueError("regular-island points must have shape (n, 2)")
            R = points_RZ[:, 0]
            Z = points_RZ[:, 1]
        else:
            R = np.asarray(_item_value(contour, ("R",)), dtype=float).ravel()
            Z = np.asarray(_item_value(contour, ("Z",)), dtype=float).ravel()
        if R.size < 2 or Z.size != R.size:
            raise ValueError("regular-island R/Z must have matching lengths >= 2")
        phi_raw = _item_value(contour, ("phi", "section_phi"))
        phi_arr = np.asarray(phi_raw, dtype=float)
        if phi_arr.ndim == 0:
            phi = np.full(R.size, float(phi_arr), dtype=float)
        else:
            phi = phi_arr.ravel()
            if phi.size != R.size:
                raise ValueError("regular-island phi must be scalar or match R/Z")
        label = _nonempty_label(
            _item_value(contour, ("label",), default=f"regular_island{contour_index}")
        )
        source_raw = _item_value(contour, ("source_coordinate",), default=None)
        if source_raw is None:
            source_coordinate = np.concatenate(
                ([0.0], np.cumsum(np.hypot(np.diff(R), np.diff(Z))))
            )
        else:
            source_coordinate = np.asarray(source_raw, dtype=float).ravel()
            if source_coordinate.size != R.size:
                raise ValueError("source_coordinate must match regular-island seeds")

        closed = bool(_item_value(contour, ("closed",), default=False))
        dpk_regular = bool(
            _item_value(contour, ("dpk_regular", "regularity_passed"), default=False)
        )
        quantitative_requested = bool(
            _item_value(contour, ("quantitative",), default=False)
        )
        closure_error = _item_value(contour, ("closure_error",), default=None)
        closure_tolerance = _item_value(contour, ("closure_tolerance",), default=None)
        if quantitative_requested and not (closed and dpk_regular):
            raise ValueError(
                "quantitative regular-island contours require closed=True and dpk_regular=True"
            )
        if quantitative_requested and closure_error is not None and closure_tolerance is not None:
            if float(closure_error) > float(closure_tolerance):
                raise ValueError("regular-island closure error exceeds its tolerance")

        directions_raw = _item_value(contour, ("directions",), default=None)
        if directions_raw is None:
            one_direction = _item_value(contour, ("direction",), default=None)
            directions_raw = ("+", "-") if one_direction is None else (one_direction,)
        if isinstance(directions_raw, str):
            directions_raw = (directions_raw,)
        directions = tuple(_direction(value) for value in directions_raw)
        if not directions or len(set(directions)) != len(directions):
            raise ValueError("regular-island directions must be a non-empty unique sequence")

        supplied_weights = _item_value(contour, ("weights",), default=None)
        default_weights = (
            np.ones(R.size, dtype=float)
            if supplied_weights is None
            else np.asarray(supplied_weights, dtype=float).ravel()
        )
        if default_weights.size != R.size:
            raise ValueError("regular-island weights must match its seeds")
        default_kind = str(_item_value(contour, ("weight_kind",), default="relative")).lower()
        default_weight_provenance = str(
            _item_value(
                contour,
                ("weight_provenance",),
                default=("uniform_relative_seed_proxy" if supplied_weights is None else "contour_supplied"),
            )
        )
        context = {
            "contour": contour,
            "contour_index": contour_index,
            "R": R.copy(),
            "Z": Z.copy(),
            "phi": phi.copy(),
            "source_coordinate": source_coordinate.copy(),
        }
        weights, kind, weight_quantitative, provenance, weight_metadata = _resolve_weights(
            default_weights,
            context,
            weight_builder,
            default_kind=default_kind,
            default_quantitative=quantitative_requested and supplied_weights is not None,
            default_provenance=default_weight_provenance,
        )
        per_direction_weights = weights / float(len(directions))
        topology_quantitative = quantitative_requested and closed and dpk_regular
        quantitative = topology_quantitative and weight_quantitative
        for trace_direction in directions:
            metadata = {
                "quantitative": bool(quantitative),
                "topology_provenance": (
                    "validated_regular_island_contour"
                    if topology_quantitative
                    else "unvalidated_regular_island_geometry_proxy"
                ),
                "weight_provenance": provenance,
                "closed": closed,
                "dpk_regular": dpk_regular,
                "closure_error": closure_error,
                "closure_tolerance": closure_tolerance,
                "bidirectional_weight_split": float(1.0 / len(directions)),
                "contour_index": contour_index,
            }
            metadata.update(weight_metadata)
            bundles.append(
                StrikeSeedBundle(
                    label=f"{label}.{trace_direction}",
                    mode="regular_island",
                    R=R,
                    Z=Z,
                    phi=phi,
                    direction=trace_direction,
                    weights=per_direction_weights,
                    weight_kind=kind,
                    source_coordinate=source_coordinate,
                    metadata=metadata,
                )
            )
    return tuple(bundles)


def trace_wall_strikes_field(
    field: object,
    bundles: Sequence[StrikeSeedBundle],
    wall: object,
    *,
    max_turns: int,
    DPhi: float,
    extend_phi: bool = True,
    cache_dir: object | None = None,
    trace_function: Callable[..., Mapping[str, np.ndarray]] | None = None,
) -> tuple[WallStrikeSamples, ...]:
    """Trace ordered bundles to first wall hits with the existing cyna bridge."""

    turns = int(max_turns)
    if turns != max_turns or turns <= 0:
        raise ValueError("max_turns must be a positive integer")
    step = float(DPhi)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("DPhi must be positive and finite")
    if cache_dir is not None:
        raise ValueError("cache_dir is not implemented here; cache the existing tracer result upstream")
    wall_phi, wall_R, wall_Z, _period = _wall_arrays_and_period(wall)
    tracer = trace_wall_hits_twall_field if trace_function is None else trace_function
    if not callable(tracer):
        raise TypeError("trace_function must be callable")

    outputs: list[WallStrikeSamples] = []
    for bundle in bundles:
        if not isinstance(bundle, StrikeSeedBundle):
            raise TypeError("bundles must contain StrikeSeedBundle values")
        groups: list[list[int]] = []
        group_values: list[float] = []
        for seed_index, seed_phi in enumerate(bundle.phi):
            group_index = next(
                (
                    index
                    for index, value in enumerate(group_values)
                    if np.isclose(seed_phi, value, rtol=0.0, atol=1.0e-13)
                ),
                None,
            )
            if group_index is None:
                group_values.append(float(seed_phi))
                groups.append([seed_index])
            else:
                groups[group_index].append(seed_index)

        strike_R: list[np.ndarray] = []
        strike_Z: list[np.ndarray] = []
        strike_phi: list[np.ndarray] = []
        strike_L: list[np.ndarray] = []
        strike_indices: list[np.ndarray] = []
        resolved = np.zeros(bundle.R.size, dtype=bool)
        for group_phi, group_index_values in zip(group_values, groups):
            group_indices = np.asarray(group_index_values, dtype=int)
            wall_hits = tracer(
                field,
                bundle.R[group_indices],
                bundle.Z[group_indices],
                float(group_phi),
                turns,
                step,
                wall_phi,
                wall_R,
                wall_Z,
                extend_phi=bool(extend_phi),
                direction=bundle.direction,
            )
            strike = strike_line_from_wall_hits(wall_hits, direction=bundle.direction)
            local_indices = np.asarray(strike["seed_index"], dtype=int).ravel()
            if np.any(local_indices < 0) or np.any(local_indices >= group_indices.size):
                raise ValueError("wall tracer returned an invalid seed_index")
            global_indices = group_indices[local_indices]
            resolved[global_indices] = True
            strike_R.append(np.asarray(strike["R"], dtype=float).ravel())
            strike_Z.append(np.asarray(strike["Z"], dtype=float).ravel())
            strike_phi.append(np.asarray(strike["phi"], dtype=float).ravel())
            strike_L.append(np.asarray(strike["connection_length"], dtype=float).ravel())
            strike_indices.append(global_indices)

        if strike_indices:
            seed_indices = np.concatenate(strike_indices)
            order = np.argsort(seed_indices, kind="stable")
            seed_indices = seed_indices[order]
            hit_R = np.concatenate(strike_R)[order]
            hit_Z = np.concatenate(strike_Z)[order]
            hit_phi = np.concatenate(strike_phi)[order]
            connection_length = np.concatenate(strike_L)[order]
        else:  # pragma: no cover - every validated bundle creates a group
            seed_indices = np.empty(0, dtype=int)
            hit_R = np.empty(0, dtype=float)
            hit_Z = np.empty(0, dtype=float)
            hit_phi = np.empty(0, dtype=float)
            connection_length = np.empty(0, dtype=float)
        unresolved_weight = float(np.sum(bundle.weights[~resolved]))
        metadata = dict(bundle.metadata)
        metadata.update(
            {
                "seed_bundle_label": bundle.label,
                "launched_seed_count": int(bundle.R.size),
                "resolved_seed_count": int(seed_indices.size),
                "unresolved_seed_count": int(bundle.R.size - seed_indices.size),
                "tracer": (
                    "pyna.toroidal.flt.trace_wall_hits_twall_field"
                    if trace_function is None
                    else (
                        f"{getattr(tracer, '__module__', '')}."
                        f"{getattr(tracer, '__qualname__', type(tracer).__name__)}"
                    ).strip(".")
                ),
                "wall_term": 1,
                "first_wall_hit": True,
                "seed_order_preserved": True,
            }
        )
        outputs.append(
            WallStrikeSamples(
                label=bundle.label,
                mode=bundle.mode,
                R=hit_R,
                Z=hit_Z,
                phi=hit_phi,
                weights=bundle.weights[seed_indices],
                connection_length=connection_length,
                seed_index=seed_indices,
                direction=bundle.direction,
                unresolved_weight=unresolved_weight,
                weight_kind=bundle.weight_kind,
                metadata=metadata,
            )
        )
    return tuple(outputs)


def _strict_edges(values: Sequence[float], name: str) -> np.ndarray:
    edges = np.asarray(values, dtype=float).ravel()
    if edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{name} must contain at least two finite increasing values")
    return edges


def _wall_cell_areas(
    wall: object,
    phi_edges: np.ndarray,
    s_edges: np.ndarray,
    *,
    period: float,
) -> np.ndarray:
    if hasattr(wall, "cell_areas") and callable(getattr(wall, "cell_areas")):
        areas = np.asarray(wall.cell_areas(phi_edges, s_edges), dtype=float)
    else:
        # Reuse the existing wall-area implementation without invoking FusionSC.
        from pyna.toroidal.control.fusionsc_heat import FusionSCWallSurfaceSpec

        wall_phi, wall_R, wall_Z, _ = _wall_arrays_and_period(wall, period)
        tolerance = max(1.0e-12, 1.0e-10 * period)
        if abs(float(wall_phi[-1] - wall_phi[0]) - period) <= tolerance:
            wall_phi = wall_phi[:-1]
            wall_R = wall_R[:-1]
            wall_Z = wall_Z[:-1]
        area_wall = FusionSCWallSurfaceSpec(
            phi_values=wall_phi,
            R=wall_R,
            Z=wall_Z,
            toroidal_period=period,
        )
        areas = np.asarray(area_wall.cell_areas(phi_edges, s_edges), dtype=float)
    expected = (phi_edges.size - 1, s_edges.size - 1)
    if areas.shape != expected or not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("wall cell areas must be positive, finite, and match heat bins")
    return areas


def wall_heat_state_from_strikes(
    strikes: Sequence[WallStrikeSamples],
    wall: object,
    *,
    phi_edges: Sequence[float],
    s_edges: Sequence[float],
    total_power: float | None = None,
    field_period: float | None = None,
) -> BoundaryTopologyHeatState:
    """Convert resolved strikes to a power-conserving standard heat state."""

    if not strikes:
        raise ValueError("strikes must not be empty")
    if not all(isinstance(value, WallStrikeSamples) for value in strikes):
        raise TypeError("strikes must contain WallStrikeSamples values")
    kinds = {value.weight_kind for value in strikes}
    if len(kinds) != 1:
        raise ValueError("relative and power strike weights cannot be mixed in one heat state")
    weight_kind = next(iter(kinds))
    launched_weight = float(sum(value.launched_weight for value in strikes))
    if launched_weight <= 0.0:
        raise ValueError("strike bundles contain no launched weight")
    if weight_kind == "relative":
        if total_power is None:
            raise ValueError("relative strike weights require total_power")
        launched_power = float(total_power)
        if not np.isfinite(launched_power) or launched_power <= 0.0:
            raise ValueError("total_power must be positive and finite")
        power_scale = launched_power / launched_weight
    else:
        launched_power = launched_weight
        power_scale = 1.0
        if total_power is not None and not np.isclose(
            float(total_power), launched_power, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError("total_power cannot renormalize absolute power weights")

    phi_bins = _strict_edges(phi_edges, "phi_edges")
    s_bins = _strict_edges(s_edges, "s_edges")
    wall_phi, wall_R, wall_Z, period = _wall_arrays_and_period(wall, field_period)
    tolerance = max(1.0e-12, 1.0e-10 * period)
    if not np.isclose(phi_bins[-1] - phi_bins[0], period, rtol=0.0, atol=tolerance):
        raise ValueError("phi_edges must cover exactly one wall field period")
    if abs(float(s_bins[0])) > tolerance or abs(float(s_bins[-1]) - 1.0) > tolerance:
        raise ValueError("s_edges must cover the complete normalized wall interval [0, 1]")

    hit_R = np.concatenate([value.R for value in strikes])
    hit_Z = np.concatenate([value.Z for value in strikes])
    hit_phi = np.concatenate([value.phi for value in strikes])
    resolved_weights = np.concatenate([value.weights for value in strikes])
    resolved_power = resolved_weights * power_scale
    unresolved_power = float(
        sum(value.unresolved_weight for value in strikes) * power_scale
    )
    expected_deposited_power = float(np.sum(resolved_power))
    if hit_R.size:
        projection = project_points_to_toroidal_surface(
            hit_R,
            hit_Z,
            hit_phi,
            wall_phi,
            wall_R,
            wall_Z,
            field_period=period,
        )
        binned_phi = float(phi_bins[0]) + np.mod(projection.phi - float(phi_bins[0]), period)
        cell_power, _, _ = np.histogram2d(
            binned_phi,
            projection.s,
            bins=(phi_bins, s_bins),
            weights=resolved_power,
        )
        projection_distance_max = float(np.max(projection.distance))
        projection_distance_rms = float(np.sqrt(np.mean(projection.distance**2)))
    else:
        cell_power = np.zeros((phi_bins.size - 1, s_bins.size - 1), dtype=float)
        projection_distance_max = float("nan")
        projection_distance_rms = float("nan")
    deposited_power = float(np.sum(cell_power))
    if not np.isclose(
        deposited_power,
        expected_deposited_power,
        rtol=1.0e-12,
        atol=max(1.0e-14, 1.0e-12 * max(launched_power, 1.0)),
    ):
        raise RuntimeError("wall binning lost resolved strike power")
    if not np.isclose(
        deposited_power + unresolved_power,
        launched_power,
        rtol=1.0e-12,
        atol=max(1.0e-14, 1.0e-12 * max(launched_power, 1.0)),
    ):
        raise RuntimeError("resolved and unresolved strike powers do not conserve launched power")

    areas = _wall_cell_areas(wall, phi_bins, s_bins, period=period)
    heat = cell_power / areas
    integrated = float(np.sum(heat * areas))
    if not np.isclose(integrated, deposited_power, rtol=1.0e-12, atol=1.0e-14):
        raise RuntimeError("wall heat normalization failed power conservation")
    quantitative = all(bool(value.metadata.get("quantitative", False)) for value in strikes)
    metadata = {
        "model": "topology_strike_heat_bridge",
        "topology_modes": tuple(dict.fromkeys(value.mode for value in strikes)),
        "source_labels": tuple(value.label for value in strikes),
        "weight_kind": weight_kind,
        "quantitative": bool(quantitative),
        "proxy": not bool(quantitative),
        "transport_model": "first_wall_hit_no_cross_field_transport",
        "projection": "continuous_toroidal_section_poloidal_segment",
        "wall_geometry": "sampled_toroidal_surface",
        "launched_power": launched_power,
        "deposited_power": deposited_power,
        "unresolved_power": unresolved_power,
        "resolved_power_fraction": deposited_power / launched_power,
        "projection_distance_max": projection_distance_max,
        "projection_distance_rms": projection_distance_rms,
        "normalization": "sum(heat * cell_areas) == deposited_power",
        "field_period": period,
    }
    return BoundaryTopologyHeatState(
        heat=heat,
        phi_values=0.5 * (phi_bins[:-1] + phi_bins[1:]),
        s_values=0.5 * (s_bins[:-1] + s_bins[1:]),
        cell_areas=areas,
        metadata=metadata,
    )


def sum_boundary_heat_states(
    states: Sequence[BoundaryTopologyHeatState],
) -> BoundaryTopologyHeatState:
    """Conservatively superpose compatible standard heat states."""

    if not states:
        raise ValueError("states must not be empty")
    if not all(isinstance(value, BoundaryTopologyHeatState) for value in states):
        raise TypeError("states must contain BoundaryTopologyHeatState values")
    reference = states[0]
    reference_area = None if reference.cell_areas is None else np.asarray(reference.cell_areas, dtype=float)
    for state in states[1:]:
        if not np.allclose(state.phi_values, reference.phi_values, rtol=0.0, atol=1.0e-12):
            raise ValueError("heat states must share phi_values")
        if not np.allclose(state.s_values, reference.s_values, rtol=0.0, atol=1.0e-12):
            raise ValueError("heat states must share s_values")
        area = None if state.cell_areas is None else np.asarray(state.cell_areas, dtype=float)
        if (reference_area is None) != (area is None):
            raise ValueError("heat states must use the same cell-area convention")
        if reference_area is not None and not np.allclose(area, reference_area, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError("heat states must share cell_areas")
    component_heat = [np.asarray(state.heat, dtype=float) for state in states]
    if not all(np.all(np.isfinite(value)) and np.all(value >= 0.0) for value in component_heat):
        raise ValueError("heat states must contain finite non-negative heat flux")
    heat = np.sum(component_heat, axis=0)
    if reference_area is None:
        deposited_power = float(np.sum(heat))
    else:
        deposited_power = float(np.sum(heat * reference_area))
    component_deposited = [
        float(np.sum(state.heat))
        if state.cell_areas is None
        else float(np.sum(state.heat * state.cell_areas))
        for state in states
    ]
    unresolved_power = float(sum(float(state.metadata.get("unresolved_power", 0.0)) for state in states))
    launched_power = float(
        sum(
            float(state.metadata.get("launched_power", deposited + float(state.metadata.get("unresolved_power", 0.0))))
            for state, deposited in zip(states, component_deposited)
        )
    )
    quantitative = all(bool(state.metadata.get("quantitative", False)) for state in states)
    metadata = {
        "model": "summed_boundary_heat_states",
        "component_count": len(states),
        "component_models": tuple(state.metadata.get("model") for state in states),
        "quantitative": quantitative,
        "proxy": not quantitative,
        "launched_power": launched_power,
        "deposited_power": deposited_power,
        "unresolved_power": unresolved_power,
        "resolved_power_fraction": (
            deposited_power / launched_power if launched_power > 0.0 else float("nan")
        ),
        "normalization": (
            "sum(heat * cell_areas) == deposited_power"
            if reference_area is not None
            else "sum(heat) == deposited_power"
        ),
    }
    return BoundaryTopologyHeatState(
        heat=heat,
        phi_values=reference.phi_values,
        s_values=reference.s_values,
        cell_areas=reference_area,
        metadata=metadata,
    )


__all__ = [
    "StrikeSeedBundle",
    "WallStrikeSamples",
    "island_strike_seed_bundles",
    "manifold_strike_seed_bundles",
    "sum_boundary_heat_states",
    "trace_wall_strikes_field",
    "wall_heat_state_from_strikes",
]
