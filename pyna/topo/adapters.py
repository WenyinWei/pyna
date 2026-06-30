"""Adapters between user data and pyna topology objects.

Adapters are deliberately conservative: they normalize representation, but they
do not claim a sampled curve is invariant.  Promotion from ``Trajectory`` to
``Cycle`` or from ``Orbit`` to ``PeriodicOrbit`` is exposed as an explicit
operation with optional closure/iteration checks.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from pyna.topo.core import Cycle, LinearStabilityData, Orbit, PeriodicOrbit, SectionPoint, Trajectory


def _as_state(x: Any, *, name: str = "state") -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _as_states(x: Any, *, name: str = "states") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, dim).")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _stability_data(value: Any) -> Optional[LinearStabilityData]:
    if value is None:
        return None
    if isinstance(value, LinearStabilityData):
        return value
    return LinearStabilityData(jacobian=np.asarray(value, dtype=float))


def _metadata(base: Optional[dict] = None, **extra: Any) -> dict:
    out = dict(base or {})
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def _section_point_sequence(obj: Any) -> Optional[list]:
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, (str, bytes)):
        return None
    try:
        seq = list(obj)
    except TypeError:
        return None
    return seq if seq and all(isinstance(pt, SectionPoint) for pt in seq) else None


def as_section_point(
    obj: Any,
    *,
    section_value: Optional[float] = None,
    section_label: Optional[str] = None,
    stability_data: Any = None,
    coordinate_names: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
) -> Any:
    """Return ``obj`` as a generic section point."""

    stability = _stability_data(stability_data)
    if isinstance(obj, SectionPoint) and stability is None:
        return obj

    if isinstance(obj, SectionPoint):
        state = obj.state.copy() if obj.state is not None else None
        section_value = obj.section_value if section_value is None else section_value
        section_label = obj.section_label if section_label is None else section_label
        stability = obj.stability_data if stability is None else stability
        base_metadata = dict(obj.metadata)
    elif hasattr(obj, "state") and getattr(obj, "state") is not None:
        state = _as_state(getattr(obj, "state"), name="state")
        base_metadata = {}
    elif hasattr(obj, "R") and hasattr(obj, "Z"):
        state = np.array([float(obj.R), float(obj.Z)], dtype=float)
        section_value = float(getattr(obj, "phi", section_value)) if getattr(obj, "phi", None) is not None else section_value
        base_metadata = {}
    else:
        state = _as_state(obj, name="section point")
        base_metadata = {}

    return SectionPoint(
        state=state,
        section_value=section_value,
        section_label=section_label,
        stability_data=stability,
        metadata=_metadata(
            base_metadata,
            **dict(metadata or {}),
            coordinate_names=tuple(coordinate_names) if coordinate_names is not None else None,
        ),
    )


def as_trajectory(
    obj: Any,
    *,
    times: Optional[Sequence[float]] = None,
    time_name: str = "t",
    coordinate_names: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
) -> Trajectory:
    """Return ``obj`` as a generic sampled continuous-time trajectory."""

    if isinstance(obj, Trajectory) and times is None and coordinate_names is None and metadata is None:
        return obj

    if isinstance(obj, Trajectory):
        states = obj.states
        t_arr = obj.times if times is None else np.asarray(times, dtype=float)
        names = obj.coordinate_names if coordinate_names is None else tuple(coordinate_names)
        meta = _metadata(obj.metadata, **dict(metadata or {}))
    elif hasattr(obj, "states") and hasattr(obj, "times"):
        states = _as_states(getattr(obj, "states"), name="states")
        t_arr = np.asarray(getattr(obj, "times") if times is None else times, dtype=float)
        names = tuple(coordinate_names) if coordinate_names is not None else None
        meta = dict(metadata or {})
    elif hasattr(obj, "y") and hasattr(obj, "t"):
        y = np.asarray(getattr(obj, "y"), dtype=float)
        t_arr = np.asarray(getattr(obj, "t") if times is None else times, dtype=float)
        states = y.T if y.ndim == 2 and y.shape[1] == t_arr.size and y.shape[0] != t_arr.size else y
        states = _as_states(states, name="y")
        names = tuple(coordinate_names) if coordinate_names is not None else None
        meta = dict(metadata or {})
    else:
        states = _as_states(obj, name="trajectory")
        t_arr = np.arange(states.shape[0], dtype=float) if times is None else np.asarray(times, dtype=float)
        names = tuple(coordinate_names) if coordinate_names is not None else None
        meta = dict(metadata or {})

    return Trajectory(states=states, times=t_arr, time_name=time_name, coordinate_names=names, metadata=meta)


def as_orbit(
    obj: Any,
    *,
    steps: Optional[Sequence[int]] = None,
    coordinate_names: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
) -> Orbit:
    """Return ``obj`` as a generic sampled discrete-time orbit."""

    if isinstance(obj, Orbit) and steps is None and coordinate_names is None and metadata is None:
        return obj
    if isinstance(obj, Orbit):
        states = obj.states
        step_arr = obj.steps if steps is None else np.asarray(steps)
        names = obj.coordinate_names if coordinate_names is None else tuple(coordinate_names)
        meta = _metadata(obj.metadata, **dict(metadata or {}))
    elif hasattr(obj, "states"):
        states = _as_states(getattr(obj, "states"), name="states")
        step_arr = np.asarray(steps) if steps is not None else None
        names = tuple(coordinate_names) if coordinate_names is not None else None
        meta = dict(metadata or {})
    else:
        states = _as_states(obj, name="orbit")
        step_arr = np.asarray(steps) if steps is not None else None
        names = tuple(coordinate_names) if coordinate_names is not None else None
        meta = dict(metadata or {})
    return Orbit(states=states, steps=step_arr, coordinate_names=names, metadata=meta)


def as_periodic_orbit(
    obj: Any,
    *,
    map_obj: Optional[Any] = None,
    period: Optional[int] = None,
    section_value: Optional[float] = None,
    section_label: Optional[str] = None,
    stability_data: Any = None,
    coordinate_names: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
    verify: bool = False,
    closure_tol: float = 1e-8,
    drop_closing_duplicate: bool = True,
) -> PeriodicOrbit:
    """Return ``obj`` as a periodic orbit with optional map/closure checks."""

    stability = _stability_data(stability_data)
    if isinstance(obj, PeriodicOrbit) and not verify and stability is None and metadata is None:
        return obj

    points = None
    section_points = None
    if isinstance(obj, PeriodicOrbit):
        points = list(obj.points)
        states = np.asarray([pt.state for pt in points if pt.state is not None], dtype=float)
        period = obj.period if period is None else period
        stability = obj.stability_data if stability is None else stability
        orbit_trace = obj.orbit_trace
        meta = _metadata(obj.metadata, **dict(metadata or {}))
    else:
        if isinstance(obj, Orbit):
            states = obj.states
            orbit_trace = obj
        else:
            section_points = _section_point_sequence(obj)
            if section_points is None:
                states = _as_states(obj, name="periodic orbit")
                orbit_trace = Orbit(states=states, coordinate_names=tuple(coordinate_names) if coordinate_names else None)
            else:
                points = section_points
                states = np.asarray([pt.state for pt in points if pt.state is not None], dtype=float)
                orbit_trace = Orbit(states=states, coordinate_names=tuple(coordinate_names) if coordinate_names else None)
                period = len(points) if period is None else period
                meta = dict(metadata or {})
        if points is None:
            meta = dict(metadata or {})
            if drop_closing_duplicate and states.shape[0] > 1 and np.allclose(states[0], states[-1], atol=closure_tol, rtol=0.0):
                states_for_points = states[:-1]
            else:
                states_for_points = states
            points = [
                as_section_point(
                    state,
                    section_value=section_value,
                    section_label=section_label,
                    stability_data=stability,
                    coordinate_names=coordinate_names,
                )
                for state in states_for_points
            ]
            period = len(points) if period is None else period

    if verify:
        if states.size == 0:
            raise ValueError("cannot verify an empty periodic orbit.")
        if map_obj is not None:
            check_states = states[:-1] if states.shape[0] > 1 and np.allclose(states[0], states[-1], atol=closure_tol, rtol=0.0) else states
            for i, state in enumerate(check_states):
                expected = check_states[(i + 1) % check_states.shape[0]]
                actual = np.asarray(map_obj.step(state), dtype=float).reshape(-1)
                if not np.allclose(actual, expected, atol=closure_tol, rtol=0.0):
                    raise ValueError("map iteration does not close the requested periodic orbit.")
        elif not np.allclose(states[0], states[-1], atol=closure_tol, rtol=0.0):
            raise ValueError("periodic orbit samples are not closed; pass map_obj or disable verify.")

    return PeriodicOrbit(
        points=points,
        period=period,
        stability_data=stability,
        representative_state=points[0].state.copy() if points and points[0].state is not None else None,
        orbit_trace=orbit_trace,
        metadata=meta,
    )


def as_cycle(
    obj: Any,
    *,
    times: Optional[Sequence[float]] = None,
    period_value: Optional[float] = None,
    return_map_orbit: Optional[PeriodicOrbit] = None,
    metadata: Optional[dict] = None,
    require_closed: bool = False,
    closure_tol: float = 1e-8,
) -> Cycle:
    """Return ``obj`` as a continuous-time cycle representation."""

    if isinstance(obj, Cycle) and not require_closed and metadata is None:
        return obj

    trajectory = as_trajectory(obj, times=times) if not isinstance(obj, Cycle) else obj.trajectory
    if require_closed and trajectory is not None:
        if not np.allclose(trajectory.states[0], trajectory.states[-1], atol=closure_tol, rtol=0.0):
            raise ValueError("trajectory is not closed; refusing to promote it to Cycle.")
    if period_value is None and trajectory is not None and trajectory.n_samples > 1:
        period_value = float(trajectory.times[-1] - trajectory.times[0])
    return Cycle(
        trajectory=trajectory,
        period_value=period_value,
        return_map_orbit=return_map_orbit,
        metadata=_metadata(getattr(obj, "metadata", None), **dict(metadata or {})),
    )


__all__ = [
    "as_section_point",
    "as_trajectory",
    "as_orbit",
    "as_periodic_orbit",
    "as_cycle",
]
