"""Dimension-agnostic tracing workflow helpers.

This module keeps orchestration optional.  The public tracing functions use
``pyna.cache`` for local disk caching, while the ``*_flow`` helpers build
Prefect flows only when Prefect is installed.
"""

from __future__ import annotations

from typing import Any

from pyna.cache import CacheStore, memoize


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


def trace_trajectory(
    system: Any,
    x0: Any,
    t_span: Any,
    *,
    trace_id: str | None = None,
    dt: Any = None,
    t_eval: Any = None,
    cache_namespace: str = "workflow.trajectory",
    use_cache: bool = True,
    cache: bool | None = None,
    cache_store: CacheStore | None = None,
    cache_backend: str = "pickle",
    **kwargs: Any,
) -> Any:
    """Trace a finite-dimensional continuous trajectory.

    ``system`` must expose ``trajectory(x0, t_span, ...)`` or
    ``integrate(x0, t_span, ...)``.  Local caching is independent of Prefect.
    When caching is enabled, callers must provide a stable ``trace_id`` because
    arbitrary Python system objects often have process-local ``repr`` values.
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

    stable_id = _require_trace_id(trace_id, cache_kind="trajectory")
    cached_trace = _memoized(
        _trace_trajectory_uncached,
        namespace=cache_namespace,
        ignore=["system"],
        cache_store=cache_store,
        cache_backend=cache_backend,
    )
    return cached_trace(
        system=system,
        trace_id=stable_id,
        x0=x0,
        t_span=t_span,
        dt=dt,
        t_eval=t_eval,
        method_kwargs=method_kwargs,
    )


def trace_orbit(
    map_obj: Any,
    x0: Any,
    n_iter: int,
    *,
    trace_id: str | None = None,
    cache_namespace: str = "workflow.orbit",
    use_cache: bool = True,
    cache: bool | None = None,
    cache_store: CacheStore | None = None,
    cache_backend: str = "pickle",
) -> Any:
    """Trace a finite-dimensional discrete orbit.

    ``map_obj`` may expose ``orbit_geometry`` or ``orbit``; the result is the
    generic :class:`pyna.topo.core.Orbit` shape used by ``TopologyWorkflow``.
    """

    should_cache = _resolve_use_cache(use_cache, cache)
    if not should_cache:
        return _trace_orbit_uncached(
            map_obj=map_obj,
            trace_id=trace_id,
            x0=x0,
            n_iter=int(n_iter),
        )

    stable_id = _require_trace_id(trace_id, cache_kind="orbit")
    cached_trace = _memoized(
        _trace_orbit_uncached,
        namespace=cache_namespace,
        ignore=["map_obj"],
        cache_store=cache_store,
        cache_backend=cache_backend,
    )
    return cached_trace(
        map_obj=map_obj,
        trace_id=stable_id,
        x0=x0,
        n_iter=int(n_iter),
    )


def _trace_trajectory_task(
    system: Any,
    x0: Any,
    t_span: Any,
    trace_id: str | None,
    dt: Any,
    t_eval: Any,
    cache_namespace: str,
    use_cache: bool,
    cache_store: CacheStore | None,
    cache_backend: str,
    method_kwargs: dict[str, Any],
) -> Any:
    return trace_trajectory(
        system,
        x0,
        t_span,
        trace_id=trace_id,
        dt=dt,
        t_eval=t_eval,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        cache_store=cache_store,
        cache_backend=cache_backend,
        **method_kwargs,
    )


def _trace_orbit_task(
    map_obj: Any,
    x0: Any,
    n_iter: int,
    trace_id: str | None,
    cache_namespace: str,
    use_cache: bool,
    cache_store: CacheStore | None,
    cache_backend: str,
) -> Any:
    return trace_orbit(
        map_obj,
        x0,
        n_iter,
        trace_id=trace_id,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        cache_store=cache_store,
        cache_backend=cache_backend,
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
        dt: Any = None,
        t_eval: Any = None,
        cache_namespace: str = "workflow.trajectory",
        use_cache: bool = True,
        cache_store: CacheStore | None = None,
        cache_backend: str = "pickle",
        **kwargs: Any,
    ) -> Any:
        return trajectory_task(
            system,
            x0,
            t_span,
            trace_id,
            dt,
            t_eval,
            cache_namespace,
            use_cache,
            cache_store,
            cache_backend,
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
        cache_namespace: str = "workflow.orbit",
        use_cache: bool = True,
        cache_store: CacheStore | None = None,
        cache_backend: str = "pickle",
    ) -> Any:
        return orbit_task(
            map_obj,
            x0,
            n_iter,
            trace_id,
            cache_namespace,
            use_cache,
            cache_store,
            cache_backend,
        )

    return prefect_trace_orbit_flow


def trace_trajectory_flow(
    system: Any,
    x0: Any,
    t_span: Any,
    *,
    trace_id: str | None = None,
    dt: Any = None,
    t_eval: Any = None,
    cache_namespace: str = "workflow.trajectory",
    use_cache: bool = True,
    cache_store: CacheStore | None = None,
    cache_backend: str = "pickle",
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
        dt=dt,
        t_eval=t_eval,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        cache_store=cache_store,
        cache_backend=cache_backend,
        **kwargs,
    )


def trace_orbit_flow(
    map_obj: Any,
    x0: Any,
    n_iter: int,
    *,
    trace_id: str | None = None,
    cache_namespace: str = "workflow.orbit",
    use_cache: bool = True,
    cache_store: CacheStore | None = None,
    cache_backend: str = "pickle",
) -> Any:
    """Run the Prefect flow for generic discrete orbit tracing."""

    flow, task = _optional_prefect_flow_task()
    prefect_flow = _build_prefect_trace_orbit_flow(flow, task)
    return prefect_flow(
        map_obj,
        x0,
        n_iter,
        trace_id=trace_id,
        cache_namespace=cache_namespace,
        use_cache=use_cache,
        cache_store=cache_store,
        cache_backend=cache_backend,
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
