"""Dimension-agnostic tracing workflow helpers.

This module keeps orchestration optional.  The public tracing functions use
``pyna.cache`` for local disk caching, while the ``*_flow`` helpers build
Prefect flows only when Prefect is installed.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

from pyna.cache import CacheStore, memoize


_CACHE_SCHEMA_VERSION = 1
_CACHE_MARKER = "pyna.workflow.tracing"
_CACHE_SCHEMA_NAME = "pyna.workflow.tracing.result"
_CACHE_COMPATIBILITY = "v1 append-only; readers ignore unknown fields"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=_json_default)


def _json_loads(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _source_signature(source: Any, explicit: Any = None) -> Any:
    """Resolve a stable source identity for cache records when available."""

    if explicit is not None:
        return explicit
    value = getattr(source, "cache_signature", None)
    if callable(value):
        return value()
    if value is not None:
        return value
    return None


def _signature_hash(signature: Any) -> dict[str, Any]:
    signature_json = _json_dumps(signature)
    return {
        "algorithm": "sha256",
        "source_signature_json_sha256": hashlib.sha256(signature_json.encode("utf-8")).hexdigest(),
    }


def _pyna_version() -> str | None:
    for package_name in ("pyna-chaos", "pyna"):
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    if arr.dtype == object:
        h.update(repr(arr.tolist()).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(arr).tobytes())
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": h.hexdigest(),
    }


def _result_summary(record: dict[str, Any]) -> dict[str, Any]:
    kind = str(record.get("kind", ""))
    summary: dict[str, Any] = {"kind": kind}
    if "states" in record:
        states = np.asarray(record["states"])
        summary["states"] = _array_summary(states)
        summary["n_samples"] = int(states.shape[0]) if states.ndim >= 1 else 0
        summary["ambient_dim"] = int(states.shape[1]) if states.ndim >= 2 else 1
    if "times" in record:
        summary["times"] = _array_summary(record["times"])
    if bool(record.get("has_steps", False)) and "steps" in record:
        summary["steps"] = _array_summary(record["steps"])
    if "data" in record:
        summary["data"] = _array_summary(record["data"])
    return summary


def _add_cache_audit_fields(
    record: dict[str, Any],
    *,
    trace_id: str | None,
    trace_function: str,
    source_role: str,
    source: Any,
    request: dict[str, Any],
    source_signature: Any = None,
) -> dict[str, Any]:
    request_json = _json_dumps(request)
    record["trace_id"] = "" if trace_id is None else str(trace_id)
    record["request_json"] = request_json
    record["request_hash_json"] = _json_dumps(
        {
            "algorithm": "sha256",
            "request_json_sha256": hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
        }
    )
    record["producer_json"] = _json_dumps(
        {
            "pyna_version": _pyna_version(),
            "cache_schema_name": _CACHE_SCHEMA_NAME,
            "cache_schema_version": int(_CACHE_SCHEMA_VERSION),
            "trace_function": trace_function,
            source_role: _source_type(source),
        }
    )
    record["source_signature_json"] = _json_dumps(source_signature)
    record["source_signature_hash_json"] = _json_dumps(_signature_hash(source_signature))
    record["result_summary_json"] = _json_dumps(_result_summary(record))
    record["complete"] = True
    return record


def _validate_cache_record_source_signature(record: Any, expected: Any) -> None:
    if expected is None:
        return
    if not isinstance(record, dict) or record.get("__pyna_cache__") != _CACHE_MARKER:
        raise ValueError("workflow cache entry has no source signature to validate")
    if "source_signature_json" not in record:
        raise ValueError("workflow cache entry has no source signature")
    actual = _json_loads(record["source_signature_json"])
    if actual != expected:
        raise ValueError(
            "workflow cache entry source signature does not match the requested source"
        )


def _resolve_use_cache(use_cache: bool, cache: bool | None) -> bool:
    if cache is None:
        return bool(use_cache)
    if bool(use_cache) != bool(cache) and use_cache is not True:
        raise ValueError("cache and use_cache specify conflicting values")
    return bool(cache)


def _require_trace_id(trace_id: str | None, *, cache_kind: str) -> str:
    if trace_id is None or str(trace_id).strip() == "":
        raise ValueError(
            f"{cache_kind} cache requires a stable trace_id; pass use_cache=False "
            "for one-off runs or provide a reproducible identity for the system "
            "and inputs."
        )
    return str(trace_id)


def _method_kwargs(
    dt: Any = None,
    t_eval: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = dict(extra or {})
    if dt is not None:
        kwargs["dt"] = dt
    if t_eval is not None:
        kwargs["t_eval"] = t_eval
    return kwargs


def _trace_trajectory_uncached(
    *,
    system: Any,
    trace_id: str | None,
    x0: Any,
    t_span: Any,
    dt: Any = None,
    t_eval: Any = None,
    method_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Call a system's native trajectory/integrate method."""

    del trace_id  # cache identity only; the numerical system owns semantics
    kwargs = _method_kwargs(dt=dt, t_eval=t_eval, extra=method_kwargs)
    for method_name in ("trajectory", "integrate"):
        method = getattr(system, method_name, None)
        if callable(method):
            return method(x0, t_span, **kwargs)

    raise TypeError(
        f"{type(system).__name__} does not expose callable trajectory(...) "
        "or integrate(...); implement one of those methods or wrap the RHS "
        "with pyna.dynamics.CallableFlow."
    )


def _trace_orbit_uncached(
    *,
    map_obj: Any,
    trace_id: str | None,
    x0: Any,
    n_iter: int,
) -> Any:
    """Iterate a map-like object and return a generic Orbit."""

    del trace_id
    from pyna.topo.workflow import orbit_from_map

    return orbit_from_map(map_obj, x0, int(n_iter))


def _source_type(value: Any) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


def _encode_cache_record(value: Any, *, expected_kind: str) -> dict[str, Any]:
    """Encode supported tracing results as a versioned, array-based record."""

    from pyna.topo.core import Orbit, Trajectory

    if isinstance(value, Trajectory):
        return {
            "__pyna_cache__": _CACHE_MARKER,
            "schema_name": _CACHE_SCHEMA_NAME,
            "schema_version": int(_CACHE_SCHEMA_VERSION),
            "compatibility": _CACHE_COMPATIBILITY,
            "kind": "trajectory",
            "source_type": _source_type(value),
            "states": np.asarray(value.states, dtype=float),
            "times": np.asarray(value.times, dtype=float),
            "time_name": str(value.time_name),
            "coordinate_names_json": _json_dumps(
                list(value.coordinate_names) if value.coordinate_names is not None else None
            ),
            "metadata_json": _json_dumps(dict(value.metadata)),
        }

    if isinstance(value, Orbit):
        steps = None if value.steps is None else np.asarray(value.steps)
        return {
            "__pyna_cache__": _CACHE_MARKER,
            "schema_name": _CACHE_SCHEMA_NAME,
            "schema_version": int(_CACHE_SCHEMA_VERSION),
            "compatibility": _CACHE_COMPATIBILITY,
            "kind": "orbit",
            "source_type": _source_type(value),
            "states": np.asarray(value.states, dtype=float),
            "has_steps": bool(steps is not None),
            "steps": np.asarray([], dtype=int) if steps is None else steps,
            "coordinate_names_json": _json_dumps(
                list(value.coordinate_names) if value.coordinate_names is not None else None
            ),
            "metadata_json": _json_dumps(dict(value.metadata)),
        }

    if isinstance(value, np.ndarray):
        return {
            "__pyna_cache__": _CACHE_MARKER,
            "schema_name": _CACHE_SCHEMA_NAME,
            "schema_version": int(_CACHE_SCHEMA_VERSION),
            "compatibility": _CACHE_COMPATIBILITY,
            "kind": "ndarray",
            "expected_kind": str(expected_kind),
            "source_type": "numpy.ndarray",
            "data": np.asarray(value),
            "metadata_json": _json_dumps({}),
        }

    raise TypeError(
        f"workflow cache cannot store {type(value).__name__} as a stable schema; "
        "return a pyna.topo Trajectory/Orbit or numpy array, or pass use_cache=False."
    )


def _decode_cache_record(record: Any, *, expected_source_signature: Any = None) -> Any:
    """Decode a versioned workflow cache record.

    Non-record values are returned unchanged so older pickle-backed caches remain
    readable during the transition to schema records.
    """

    _validate_cache_record_source_signature(record, expected_source_signature)

    if not isinstance(record, dict) or record.get("__pyna_cache__") != _CACHE_MARKER:
        return record

    schema_name = record.get("schema_name", _CACHE_SCHEMA_NAME)
    if schema_name != _CACHE_SCHEMA_NAME:
        raise ValueError(f"workflow cache entry has unknown schema {schema_name!r}")

    schema_version = int(record.get("schema_version", 0))
    if schema_version > _CACHE_SCHEMA_VERSION:
        raise ValueError(
            "workflow cache entry was written by a newer pyna cache schema "
            f"({schema_version}); this version supports {_CACHE_SCHEMA_VERSION}."
        )
    if schema_version < 1:
        raise ValueError("workflow cache entry has an unsupported schema version")
    if "complete" in record and not bool(record["complete"]):
        raise ValueError("workflow cache entry is not complete")

    kind = str(record.get("kind", ""))
    metadata = _json_loads(record.get("metadata_json", "{}"))
    coordinate_names = _json_loads(record.get("coordinate_names_json", "null"))
    coordinate_names_tuple = None if coordinate_names is None else tuple(coordinate_names)

    if kind == "trajectory":
        from pyna.topo.core import Trajectory

        return Trajectory(
            states=np.asarray(record["states"], dtype=float),
            times=np.asarray(record["times"], dtype=float),
            time_name=str(record.get("time_name", "t")),
            coordinate_names=coordinate_names_tuple,
            metadata=dict(metadata),
        )

    if kind == "orbit":
        from pyna.topo.core import Orbit

        has_steps = bool(record.get("has_steps", False))
        return Orbit(
            states=np.asarray(record["states"], dtype=float),
            steps=np.asarray(record["steps"]) if has_steps else None,
            coordinate_names=coordinate_names_tuple,
            metadata=dict(metadata),
        )

    if kind == "ndarray":
        return np.asarray(record["data"])

    raise ValueError(f"workflow cache entry has unknown kind {kind!r}")


def _trace_trajectory_cache_record(
    *,
    system: Any,
    trace_id: str | None,
    x0: Any,
    t_span: Any,
    dt: Any = None,
    t_eval: Any = None,
    method_kwargs: dict[str, Any] | None = None,
    source_signature: Any = None,
) -> dict[str, Any]:
    result = _trace_trajectory_uncached(
        system=system,
        trace_id=trace_id,
        x0=x0,
        t_span=t_span,
        dt=dt,
        t_eval=t_eval,
        method_kwargs=method_kwargs,
    )
    record = _encode_cache_record(result, expected_kind="trajectory")
    return _add_cache_audit_fields(
        record,
        trace_id=trace_id,
        trace_function="trace_trajectory",
        source_role="system_type",
        source=system,
        source_signature=source_signature,
        request={
            "x0": x0,
            "t_span": t_span,
            "dt": dt,
            "t_eval": t_eval,
            "method_kwargs": dict(method_kwargs or {}),
        },
    )


def _trace_orbit_cache_record(
    *,
    map_obj: Any,
    trace_id: str | None,
    x0: Any,
    n_iter: int,
    source_signature: Any = None,
) -> dict[str, Any]:
    result = _trace_orbit_uncached(
        map_obj=map_obj,
        trace_id=trace_id,
        x0=x0,
        n_iter=int(n_iter),
    )
    record = _encode_cache_record(result, expected_kind="orbit")
    return _add_cache_audit_fields(
        record,
        trace_id=trace_id,
        trace_function="trace_orbit",
        source_role="map_type",
        source=map_obj,
        source_signature=source_signature,
        request={
            "x0": x0,
            "n_iter": int(n_iter),
        },
    )


def _memoized(
    fn: Any,
    *,
    namespace: str,
    ignore: list[str],
    cache_store: CacheStore | None,
    cache_backend: str,
) -> Any:
    return memoize(
        fn,
        namespace=namespace,
        ignore=ignore,
        cache=cache_store,
        backend=None if cache_store is not None else cache_backend,
    )


def _flush_memoized_cache(fn: Any) -> None:
    store = getattr(fn, "_pyna_cache_store", None)
    if store is not None and hasattr(store, "flush"):
        store.flush()


def trace_trajectory(
    system: Any,
    x0: Any,
    t_span: Any,
    *,
    trace_id: str | None = None,
    source_signature: Any = None,
    dt: Any = None,
    t_eval: Any = None,
    cache_namespace: str = "workflow.trajectory",
    use_cache: bool = True,
    cache: bool | None = None,
    cache_store: CacheStore | None = None,
    cache_backend: str = "npz",
    cache_sync: bool = False,
    **kwargs: Any,
) -> Any:
    """Trace a finite-dimensional continuous trajectory.

    ``system`` must expose ``trajectory(x0, t_span, ...)`` or
    ``integrate(x0, t_span, ...)``.  Local caching is independent of Prefect.
    When caching is enabled, callers must provide a stable ``trace_id`` because
    arbitrary Python system objects often have process-local ``repr`` values.
    Pass ``source_signature`` or implement ``system.cache_signature()`` to bind
    the cache entry to the system parameters behind that id.  Set
    ``cache_sync=True`` when the result must be durable before returning.
    """

    should_cache = _resolve_use_cache(use_cache, cache)
    method_kwargs = dict(kwargs)
    if not should_cache:
        return _trace_trajectory_uncached(
            system=system,
            trace_id=trace_id,
            x0=x0,
            t_span=t_span,
            dt=dt,
            t_eval=t_eval,
            method_kwargs=method_kwargs,
        )

    resolved_source_signature = _source_signature(system, source_signature)
    stable_id = _require_trace_id(trace_id, cache_kind="trajectory")
    cached_trace = _memoized(
        _trace_trajectory_cache_record,
        namespace=cache_namespace,
        ignore=["system"],
        cache_store=cache_store,
        cache_backend=cache_backend,
    )
    record = cached_trace(
        system=system,
        trace_id=stable_id,
        x0=x0,
        t_span=t_span,
        dt=dt,
        t_eval=t_eval,
        method_kwargs=method_kwargs,
        source_signature=resolved_source_signature,
    )
    if cache_sync:
        _flush_memoized_cache(cached_trace)
    return _decode_cache_record(record, expected_source_signature=resolved_source_signature)


def trace_orbit(
    map_obj: Any,
    x0: Any,
    n_iter: int,
    *,
    trace_id: str | None = None,
    source_signature: Any = None,
    cache_namespace: str = "workflow.orbit",
    use_cache: bool = True,
    cache: bool | None = None,
    cache_store: CacheStore | None = None,
    cache_backend: str = "npz",
    cache_sync: bool = False,
) -> Any:
    """Trace a finite-dimensional discrete orbit.

    ``map_obj`` may expose ``orbit_geometry``/``orbit`` for sampled output or
    ``step`` for one-step advancement; the result is the generic
    :class:`pyna.topo.core.Orbit` shape used by ``TopologyWorkflow``.  Pass
    ``source_signature`` or implement ``map_obj.cache_signature()`` to bind the
    cache entry to the map parameters behind the trace id.  Set
    ``cache_sync=True`` when the result must be durable before returning.
    """

    should_cache = _resolve_use_cache(use_cache, cache)
    if not should_cache:
        return _trace_orbit_uncached(
            map_obj=map_obj,
            trace_id=trace_id,
            x0=x0,
            n_iter=int(n_iter),
        )

    resolved_source_signature = _source_signature(map_obj, source_signature)
    stable_id = _require_trace_id(trace_id, cache_kind="orbit")
    cached_trace = _memoized(
        _trace_orbit_cache_record,
        namespace=cache_namespace,
        ignore=["map_obj"],
        cache_store=cache_store,
        cache_backend=cache_backend,
    )
    record = cached_trace(
        map_obj=map_obj,
        trace_id=stable_id,
        x0=x0,
        n_iter=int(n_iter),
        source_signature=resolved_source_signature,
    )
    if cache_sync:
        _flush_memoized_cache(cached_trace)
    return _decode_cache_record(record, expected_source_signature=resolved_source_signature)


def _trace_trajectory_task(
    system: Any,
    x0: Any,
    t_span: Any,
    trace_id: str | None,
    source_signature: Any,
    dt: Any,
    t_eval: Any,
    cache_namespace: str,
    use_cache: bool,
    cache_store: CacheStore | None,
    cache_backend: str,
    cache_sync: bool,
    method_kwargs: dict[str, Any],
) -> Any:
    return trace_trajectory(
        system,
        x0,
        t_span,
        trace_id=trace_id,
        source_signature=source_signature,
        dt=dt,
        t_eval=t_eval,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        cache_store=cache_store,
        cache_backend=cache_backend,
        cache_sync=cache_sync,
        **method_kwargs,
    )


def _trace_orbit_task(
    map_obj: Any,
    x0: Any,
    n_iter: int,
    trace_id: str | None,
    source_signature: Any,
    cache_namespace: str,
    use_cache: bool,
    cache_store: CacheStore | None,
    cache_backend: str,
    cache_sync: bool,
) -> Any:
    return trace_orbit(
        map_obj,
        x0,
        n_iter,
        trace_id=trace_id,
        source_signature=source_signature,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        cache_store=cache_store,
        cache_backend=cache_backend,
        cache_sync=cache_sync,
    )


def _optional_prefect_flow_task() -> tuple[Any, Any]:
    from pyna.workflow.prefect import optional_prefect

    return optional_prefect()


def _build_prefect_trace_trajectory_flow(flow: Any, task: Any) -> Any:
    trajectory_task = task(_trace_trajectory_task)

    @flow(name="pyna-trace-trajectory")
    def prefect_trace_trajectory_flow(
        system: Any,
        x0: Any,
        t_span: Any,
        *,
        trace_id: str | None = None,
        source_signature: Any = None,
        dt: Any = None,
        t_eval: Any = None,
        cache_namespace: str = "workflow.trajectory",
        use_cache: bool = True,
        cache_store: CacheStore | None = None,
        cache_backend: str = "npz",
        cache_sync: bool = False,
        **kwargs: Any,
    ) -> Any:
        return trajectory_task(
            system,
            x0,
            t_span,
            trace_id,
            source_signature,
            dt,
            t_eval,
            cache_namespace,
            use_cache,
            cache_store,
            cache_backend,
            cache_sync,
            kwargs,
        )

    return prefect_trace_trajectory_flow


def _build_prefect_trace_orbit_flow(flow: Any, task: Any) -> Any:
    orbit_task = task(_trace_orbit_task)

    @flow(name="pyna-trace-orbit")
    def prefect_trace_orbit_flow(
        map_obj: Any,
        x0: Any,
        n_iter: int,
        *,
        trace_id: str | None = None,
        source_signature: Any = None,
        cache_namespace: str = "workflow.orbit",
        use_cache: bool = True,
        cache_store: CacheStore | None = None,
        cache_backend: str = "npz",
        cache_sync: bool = False,
    ) -> Any:
        return orbit_task(
            map_obj,
            x0,
            n_iter,
            trace_id,
            source_signature,
            cache_namespace,
            use_cache,
            cache_store,
            cache_backend,
            cache_sync,
        )

    return prefect_trace_orbit_flow


def trace_trajectory_flow(
    system: Any,
    x0: Any,
    t_span: Any,
    *,
    trace_id: str | None = None,
    source_signature: Any = None,
    dt: Any = None,
    t_eval: Any = None,
    cache_namespace: str = "workflow.trajectory",
    use_cache: bool = True,
    cache_store: CacheStore | None = None,
    cache_backend: str = "npz",
    cache_sync: bool = False,
    **kwargs: Any,
) -> Any:
    """Run the Prefect flow for generic continuous trajectory tracing."""

    flow, task = _optional_prefect_flow_task()
    prefect_flow = _build_prefect_trace_trajectory_flow(flow, task)
    return prefect_flow(
        system,
        x0,
        t_span,
        trace_id=trace_id,
        source_signature=source_signature,
        dt=dt,
        t_eval=t_eval,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        cache_store=cache_store,
        cache_backend=cache_backend,
        cache_sync=cache_sync,
        **kwargs,
    )


def trace_orbit_flow(
    map_obj: Any,
    x0: Any,
    n_iter: int,
    *,
    trace_id: str | None = None,
    source_signature: Any = None,
    cache_namespace: str = "workflow.orbit",
    use_cache: bool = True,
    cache_store: CacheStore | None = None,
    cache_backend: str = "npz",
    cache_sync: bool = False,
) -> Any:
    """Run the Prefect flow for generic discrete orbit tracing."""

    flow, task = _optional_prefect_flow_task()
    prefect_flow = _build_prefect_trace_orbit_flow(flow, task)
    return prefect_flow(
        map_obj,
        x0,
        n_iter,
        trace_id=trace_id,
        source_signature=source_signature,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        cache_store=cache_store,
        cache_backend=cache_backend,
        cache_sync=cache_sync,
    )


def build_prefect_trace_trajectory_flow() -> Any:
    """Return the canonical Prefect flow for generic trajectory tracing."""

    flow, task = _optional_prefect_flow_task()
    return _build_prefect_trace_trajectory_flow(flow, task)


def build_prefect_trace_orbit_flow() -> Any:
    """Return the canonical Prefect flow for generic orbit tracing."""

    flow, task = _optional_prefect_flow_task()
    return _build_prefect_trace_orbit_flow(flow, task)


__all__ = [
    "build_prefect_trace_orbit_flow",
    "build_prefect_trace_trajectory_flow",
    "trace_orbit",
    "trace_orbit_flow",
    "trace_trajectory",
    "trace_trajectory_flow",
]
