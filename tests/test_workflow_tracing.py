"""Tests for generic workflow trajectory tracing."""

from __future__ import annotations

import builtins
import importlib.util
import inspect
import sys

import numpy as np
import pytest

from pyna.cache import CacheStore
from pyna.topo import Orbit, Trajectory
from pyna.workflow.tracing import (
    build_prefect_trace_trajectory_flow,
    trace_orbit,
    trace_trajectory,
    trace_trajectory_flow,
)


class CountingTrajectoryFlow:
    """Small dimension-agnostic flow that counts RHS evaluations."""

    def __init__(self) -> None:
        self.rhs_calls = 0
        self.trajectory_calls = 0

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        self.rhs_calls += 1
        offsets = 0.01 * np.arange(x.size, dtype=float)
        return 0.2 + 0.05 * np.asarray(x, dtype=float) + offsets

    def trajectory(
        self,
        x0,
        t_span,
        *,
        dt=None,
        t_eval=None,
        **kwargs,
    ) -> Trajectory:
        del dt, kwargs
        self.trajectory_calls += 1
        times = _times_from(t_span, t_eval)
        states = _euler_trace(self.rhs, x0, times)
        return Trajectory(
            states=states,
            times=times,
            coordinate_names=tuple(f"x{i}" for i in range(states.shape[1])),
            metadata={"trajectory_calls": self.trajectory_calls},
        )


class IntegrateOnlyFlow:
    """Flow exposing integrate(...) but not trajectory(...)."""

    def __init__(self) -> None:
        self.rhs_calls = 0
        self.integrate_calls = 0

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        self.rhs_calls += 1
        offsets = 0.02 * np.arange(x.size, dtype=float)
        return -0.1 * np.asarray(x, dtype=float) + offsets + 0.3

    def integrate(
        self,
        x0,
        t_span,
        *,
        dt=None,
        t_eval=None,
        **kwargs,
    ) -> Trajectory:
        del dt, kwargs
        self.integrate_calls += 1
        times = _times_from(t_span, t_eval)
        states = _euler_trace(self.rhs, x0, times)
        return Trajectory(
            states=states,
            times=times,
            coordinate_names=tuple(f"q{i}" for i in range(states.shape[1])),
            metadata={"integrate_calls": self.integrate_calls},
        )


class CountingMap:
    """Small finite-dimensional map with an orbit(...) method."""

    def __init__(self) -> None:
        self.orbit_calls = 0

    def orbit(self, x0, n_iter):
        self.orbit_calls += 1
        x = np.asarray(x0, dtype=float).copy()
        states = [x.copy()]
        for _ in range(int(n_iter)):
            x = x + np.array([1.0, -0.5])
            states.append(x.copy())
        return np.vstack(states)


def _times_from(t_span, t_eval) -> np.ndarray:
    if t_eval is not None:
        return np.asarray(t_eval, dtype=float)
    return np.linspace(float(t_span[0]), float(t_span[1]), 6)


def _euler_trace(rhs, x0, times: np.ndarray) -> np.ndarray:
    x = np.asarray(x0, dtype=float)
    states = [x.copy()]
    for i in range(len(times) - 1):
        h = float(times[i + 1] - times[i])
        x = x + h * rhs(float(times[i]), x)
        states.append(x.copy())
    return np.vstack(states)


def _block_prefect_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "prefect", raising=False)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "prefect" or name.startswith("prefect."):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prefect" or name.startswith("prefect."):
            raise ModuleNotFoundError("No module named 'prefect'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def test_trace_trajectory_returns_trajectory_and_uses_trace_id_cache(tmp_path):
    store = CacheStore(tmp_path, backend="pickle", async_write=False)
    flow = CountingTrajectoryFlow()
    x0 = np.array([1.0, -0.5, 0.25])
    t_span = (0.0, 0.7)
    t_eval = np.linspace(t_span[0], t_span[1], 8)

    first = trace_trajectory(
        flow,
        x0,
        t_span,
        t_eval=t_eval,
        trace_id="generic-3d",
        use_cache=True,
        cache_store=store,
    )

    assert isinstance(first, Trajectory)
    assert first.states.shape == (len(t_eval), x0.size)
    np.testing.assert_allclose(first.times, t_eval)
    first_rhs_calls = flow.rhs_calls
    entries_after_first = len(store)
    assert first_rhs_calls == len(t_eval) - 1
    assert entries_after_first >= 1

    second = trace_trajectory(
        flow,
        x0,
        t_span,
        t_eval=t_eval,
        trace_id="generic-3d",
        use_cache=True,
        cache_store=store,
    )

    assert isinstance(second, Trajectory)
    np.testing.assert_allclose(second.states, first.states)
    np.testing.assert_allclose(second.times, first.times)
    second_extra_rhs_calls = flow.rhs_calls - first_rhs_calls
    assert second_extra_rhs_calls < first_rhs_calls / 2
    assert len(store) == entries_after_first

    rhs_calls_before_new_trace = flow.rhs_calls
    trace_trajectory(
        flow,
        x0,
        t_span,
        t_eval=t_eval,
        trace_id="generic-3d-new",
        use_cache=True,
        cache_store=store,
    )

    assert flow.rhs_calls - rhs_calls_before_new_trace >= first_rhs_calls
    assert len(store) > entries_after_first


def test_trace_trajectory_cache_payload_is_versioned_schema(tmp_path):
    store = CacheStore(tmp_path, backend="npz", async_write=False)
    flow = CountingTrajectoryFlow()

    trajectory = trace_trajectory(
        flow,
        np.array([0.0, 1.0]),
        (0.0, 0.2),
        t_eval=np.linspace(0.0, 0.2, 3),
        trace_id="schema-check",
        cache_store=store,
    )

    assert isinstance(trajectory, Trajectory)
    assert len(store) == 1
    key = next(iter(store._index))
    record = store.get(key)
    assert record["__pyna_cache__"] == "pyna.workflow.tracing"
    assert int(record["schema_version"]) == 1
    assert record["kind"] == "trajectory"
    assert "states" in record
    assert "metadata_json" in record


def test_trace_trajectory_cache_requires_trace_id(tmp_path):
    store = CacheStore(tmp_path, backend="pickle", async_write=False)
    flow = CountingTrajectoryFlow()

    with pytest.raises(ValueError, match="trace_id"):
        trace_trajectory(
            flow,
            np.array([0.1, 0.2]),
            (0.0, 1.0),
            t_eval=np.linspace(0.0, 1.0, 5),
            use_cache=True,
            cache_store=store,
        )

    assert flow.rhs_calls == 0
    assert len(store) == 0


def test_trace_trajectory_use_cache_false_does_not_require_trace_id(tmp_path):
    store = CacheStore(tmp_path, backend="pickle", async_write=False)
    flow = CountingTrajectoryFlow()
    t_eval = np.linspace(0.0, 0.4, 5)

    first = trace_trajectory(
        flow,
        np.array([0.2, -0.3, 0.4]),
        (0.0, 0.4),
        t_eval=t_eval,
        use_cache=False,
        cache_store=store,
    )
    first_rhs_calls = flow.rhs_calls

    second = trace_trajectory(
        flow,
        np.array([0.2, -0.3, 0.4]),
        (0.0, 0.4),
        t_eval=t_eval,
        use_cache=False,
        cache_store=store,
    )

    assert isinstance(first, Trajectory)
    assert isinstance(second, Trajectory)
    assert first_rhs_calls == len(t_eval) - 1
    assert flow.rhs_calls == 2 * first_rhs_calls
    assert len(store) == 0


def test_trace_trajectory_cache_false_alias_does_not_require_trace_id(tmp_path):
    if "cache" not in inspect.signature(trace_trajectory).parameters:
        pytest.skip("trace_trajectory does not expose a cache= alias")

    store = CacheStore(tmp_path, backend="pickle", async_write=False)
    flow = CountingTrajectoryFlow()
    t_eval = np.linspace(0.0, 0.4, 5)

    first = trace_trajectory(
        flow,
        np.array([0.2, -0.3, 0.4]),
        (0.0, 0.4),
        t_eval=t_eval,
        cache=False,
        cache_store=store,
    )
    first_rhs_calls = flow.rhs_calls

    second = trace_trajectory(
        flow,
        np.array([0.2, -0.3, 0.4]),
        (0.0, 0.4),
        t_eval=t_eval,
        cache=False,
        cache_store=store,
    )

    assert isinstance(first, Trajectory)
    assert isinstance(second, Trajectory)
    assert first_rhs_calls == len(t_eval) - 1
    assert flow.rhs_calls == 2 * first_rhs_calls
    assert len(store) == 0


def test_trace_trajectory_uses_integrate_fallback_without_prefect():
    flow = IntegrateOnlyFlow()
    t_eval = np.linspace(0.0, 0.3, 4)

    trajectory = trace_trajectory(
        flow,
        np.array([1.0, 0.5, -0.25, 0.125]),
        (0.0, 0.3),
        t_eval=t_eval,
        use_cache=False,
    )

    assert isinstance(trajectory, Trajectory)
    assert trajectory.states.shape == (len(t_eval), 4)
    assert flow.integrate_calls == 1
    assert flow.rhs_calls == len(t_eval) - 1


def test_trace_orbit_returns_orbit_and_uses_trace_id_cache(tmp_path):
    store = CacheStore(tmp_path, backend="pickle", async_write=False)
    map_obj = CountingMap()

    first = trace_orbit(
        map_obj,
        np.array([0.0, 1.0]),
        4,
        trace_id="shift-map",
        cache_store=store,
    )

    assert isinstance(first, Orbit)
    assert first.n_samples == 5
    np.testing.assert_allclose(first.states[-1], np.array([4.0, -1.0]))
    assert map_obj.orbit_calls == 1

    second = trace_orbit(
        map_obj,
        np.array([0.0, 1.0]),
        4,
        trace_id="shift-map",
        cache_store=store,
    )

    assert isinstance(second, Orbit)
    np.testing.assert_allclose(second.states, first.states)
    assert map_obj.orbit_calls == 1


def test_trace_orbit_cache_payload_is_versioned_schema(tmp_path):
    store = CacheStore(tmp_path, backend="npz", async_write=False)
    map_obj = CountingMap()

    orbit = trace_orbit(
        map_obj,
        np.array([0.0, 1.0]),
        2,
        trace_id="schema-map",
        cache_store=store,
    )

    assert isinstance(orbit, Orbit)
    assert len(store) == 1
    key = next(iter(store._index))
    record = store.get(key)
    assert record["__pyna_cache__"] == "pyna.workflow.tracing"
    assert int(record["schema_version"]) == 1
    assert record["kind"] == "orbit"
    assert "states" in record
    assert "steps" in record


def test_trace_orbit_cache_requires_trace_id(tmp_path):
    with pytest.raises(ValueError, match="trace_id"):
        trace_orbit(
            CountingMap(),
            np.array([0.0, 1.0]),
            2,
            cache_store=CacheStore(tmp_path, backend="pickle", async_write=False),
        )


def test_build_prefect_trace_trajectory_flow_requires_workflow_extra(monkeypatch):
    _block_prefect_imports(monkeypatch)

    with pytest.raises(RuntimeError, match=r"pyna-chaos\[workflow\]"):
        build_prefect_trace_trajectory_flow()


def test_trace_trajectory_flow_requires_workflow_extra(monkeypatch):
    _block_prefect_imports(monkeypatch)

    with pytest.raises(RuntimeError, match=r"pyna-chaos\[workflow\]"):
        trace_trajectory_flow(
            CountingTrajectoryFlow(),
            np.array([0.0, 1.0]),
            (0.0, 1.0),
            use_cache=False,
        )
