"""Workflow facade for teaching and day-to-day topology construction.

``TopologyWorkflow`` composes factories, builders and bridges into the sequence
users actually perform in notebooks:

1. create or receive a flow/map,
2. integrate trajectories or iterate orbits,
3. explicitly promote closed samples to invariant objects,
4. cut continuous geometry into section-level discrete geometry.

It is a facade, not a new mathematical layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from pyna.topo.adapters import as_cycle
from pyna.topo.bridges import CoreSectionCutBridge, SectionCutBridge
from pyna.topo.builders import GeometryBuilder, IslandChainBuilder, TubeChainBuilder
from pyna.topo.core import Cycle, IslandChain, Orbit, PeriodicOrbit, Trajectory, Tube, TubeChain
from pyna.topo.factories import DynamicalSystemFactory, GeometryFactory, PoincareMapFactory


def _map_coordinate_names(map_obj: Any) -> tuple[str, ...] | None:
    names = getattr(getattr(map_obj, "phase_space", None), "coordinate_names", None)
    return tuple(names) if names else None


def _state_matrix_from_result(value: Any) -> np.ndarray | None:
    """Return sampled states from common map-orbit result conventions."""

    if isinstance(value, Orbit):
        return np.asarray(value.states, dtype=float)

    if isinstance(value, tuple) and value and all(np.ndim(item) == 1 for item in value):
        arrays = [np.asarray(item, dtype=float) for item in value]
        sizes = {arr.shape[0] for arr in arrays}
        if len(sizes) == 1:
            return np.column_stack(arrays)

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        return arr
    return None


def _state_from_result(value: Any) -> np.ndarray:
    states = _state_matrix_from_result(value)
    if states is not None:
        if states.shape[0] == 0:
            raise ValueError("map result produced no states")
        return np.asarray(states[-1], dtype=float).reshape(-1)

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.copy()
    raise TypeError("map step result cannot be interpreted as a state")


def _call_with_state_and_count(method: Callable[..., Any], state: np.ndarray, count: int) -> Any:
    try:
        return method(state, int(count))
    except TypeError as state_error:
        try:
            return method(*state.tolist(), int(count))
        except TypeError:
            raise state_error


def _call_with_state(method: Callable[..., Any], state: np.ndarray) -> Any:
    try:
        return method(state)
    except TypeError as state_error:
        try:
            return method(*state.tolist())
        except TypeError:
            raise state_error


def _orbit_from_states(states: Any, *, coordinate_names: tuple[str, ...] | None) -> Orbit:
    if isinstance(states, Orbit):
        return states
    state_arr = _state_matrix_from_result(states)
    if state_arr is None:
        raise TypeError("map orbit result must be a sampled state array or Orbit")
    return Orbit(
        states=np.asarray(state_arr, dtype=float),
        steps=np.arange(state_arr.shape[0]),
        coordinate_names=coordinate_names,
    )


def _orbit_from_step(
    step: Callable[..., Any],
    x0: np.ndarray,
    n_iter: int,
    *,
    coordinate_names: tuple[str, ...] | None,
) -> Orbit:
    x = x0.copy()
    sampled = [x.copy()]
    for _ in range(n_iter):
        x = _state_from_result(_call_with_state(step, x))
        sampled.append(x.copy())
    return _orbit_from_states(np.vstack(sampled), coordinate_names=coordinate_names)


def orbit_from_map(map_obj: Any, x0: Sequence[float], n_iter: int) -> Orbit:
    """Iterate a map-like object and return an ``Orbit`` geometry object."""

    n_iter = int(n_iter)
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    x0_arr = np.asarray(x0, dtype=float).reshape(-1)
    names = _map_coordinate_names(map_obj)

    orbit_geometry = getattr(map_obj, "orbit_geometry", None)
    if callable(orbit_geometry):
        return _orbit_from_states(orbit_geometry(x0_arr, n_iter), coordinate_names=names)

    orbit = getattr(map_obj, "orbit", None)
    if callable(orbit):
        return _orbit_from_states(_call_with_state_and_count(orbit, x0_arr, n_iter), coordinate_names=names)

    step = getattr(map_obj, "step", None)
    if callable(step):
        return _orbit_from_step(step, x0_arr, n_iter, coordinate_names=names)

    raise TypeError(
        f"{type(map_obj).__name__} cannot produce a discrete orbit; "
        "implement orbit_geometry(...), orbit(...), or step(...)."
    )


def make_poincare_map(flow: Any, section: Any, **kwargs: Any) -> Any:
    """Create an executable Poincare map for ``flow`` and ``section``."""

    return PoincareMapFactory.create(flow=flow, section=section, **kwargs)


def section_cut(obj: Any, section: Any, *, bridge: Optional[SectionCutBridge] = None, **kwargs: Any) -> Any:
    """Cut a supported continuous-time geometry object by ``section``."""

    section_bridge = CoreSectionCutBridge() if bridge is None else bridge
    if isinstance(obj, TubeChain):
        return section_bridge.cut_tube_chain(obj, section, **kwargs)
    if isinstance(obj, Tube):
        return section_bridge.cut_tube(obj, section, **kwargs)
    if hasattr(obj, "section_cut"):
        return obj.section_cut(section, **kwargs)
    raise TypeError(f"{type(obj).__name__} does not implement section_cut().")


@dataclass
class TopologyWorkflow:
    """High-level workflow facade for examples and user scripts."""

    closure_tol: float = 1e-8
    geometry: GeometryBuilder = field(init=False)
    section_bridge: SectionCutBridge = field(default_factory=CoreSectionCutBridge)

    def __post_init__(self) -> None:
        self.geometry = GeometryBuilder(closure_tol=float(self.closure_tol))

    # ── System construction ────────────────────────────────────────────────

    def system(self, kind: str, **params: Any) -> Any:
        """Build a ready-to-use dynamical system by stable factory key."""

        return DynamicalSystemFactory.create(kind, **params)

    def geometry_object(self, kind: str, **params: Any) -> Any:
        """Build a topology object through the geometry factory."""

        return GeometryFactory.create(kind, **params)

    # ── Continuous-time workflow ───────────────────────────────────────────

    def trajectory(self, flow: Any, x0: Sequence[float], t_span: tuple, **kwargs: Any) -> Trajectory:
        """Integrate ``flow`` and return a sampled trajectory."""

        if not hasattr(flow, "trajectory"):
            raise TypeError(
                f"{type(flow).__name__} does not expose trajectory(); "
                "wrap it with pyna.dynamics.CallableFlow or implement trajectory()."
            )
        return flow.trajectory(x0, t_span, **kwargs)

    def closed_cycle(
        self,
        trajectory: Any,
        *,
        period_value: Optional[float] = None,
        closure_tol: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Cycle:
        """Promote a closed sampled trajectory to a ``Cycle``."""

        return self.geometry.cycle(
            trajectory,
            require_closed=True,
            closure_tol=self.closure_tol if closure_tol is None else float(closure_tol),
            period_value=period_value,
            metadata=metadata,
        )

    def cycle(self, obj: Any, *, require_closed: bool = False, **kwargs: Any) -> Cycle:
        """Build a ``Cycle`` with an explicit closure policy."""

        return self.geometry.cycle(obj, require_closed=require_closed, **kwargs)

    def poincare_map(self, flow: Any, section: Any, **kwargs: Any) -> Any:
        """Create an executable Poincare map for ``flow`` and ``section``."""

        return make_poincare_map(flow, section, **kwargs)

    # ── Discrete-time workflow ─────────────────────────────────────────────

    def orbit(self, map_obj: Any, x0: Sequence[float], n_iter: int) -> Orbit:
        """Iterate ``map_obj`` and return a sampled orbit."""

        return orbit_from_map(map_obj, x0, n_iter)

    def periodic_orbit(
        self,
        samples: Any,
        *,
        map_obj: Optional[Any] = None,
        verify: bool = True,
        **kwargs: Any,
    ) -> PeriodicOrbit:
        """Promote map samples to a ``PeriodicOrbit``."""

        return self.geometry.periodic_orbit(samples, map_obj=map_obj, verify=verify, **kwargs)

    # ── Resonance objects ──────────────────────────────────────────────────

    def tube(
        self,
        O_cycle: Any,
        X_cycles: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
        require_closed: bool = False,
        closure_tol: Optional[float] = None,
    ) -> Tube:
        """Build a generic ``Tube`` from one O-cycle and optional X-cycles."""

        tol = self.closure_tol if closure_tol is None else float(closure_tol)
        return Tube(
            O_cycle=as_cycle(O_cycle, require_closed=require_closed, closure_tol=tol),
            X_cycles=[
                as_cycle(cycle, require_closed=require_closed, closure_tol=tol)
                for cycle in (X_cycles or [])
            ],
            label=label,
        )

    def tube_chain(
        self,
        O_cycles: Sequence[Any],
        X_cycles: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
        require_closed: bool = False,
    ) -> TubeChain:
        """Build a generic ``TubeChain`` through ``TubeChainBuilder``."""

        return TubeChainBuilder.from_cycles(
            O_cycles,
            X_cycles,
            label=label,
            require_closed=require_closed,
            closure_tol=self.closure_tol,
        )

    def island_chain(
        self,
        O_orbits: Sequence[Any],
        X_orbits: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
        proximity_tol: Optional[float] = None,
    ) -> IslandChain:
        """Build a generic ``IslandChain`` from O/X periodic orbits."""

        return IslandChainBuilder.from_periodic_orbits(
            O_orbits,
            X_orbits,
            label=label,
            proximity_tol=proximity_tol,
        )

    def section_cut(self, obj: Any, section: Any, **kwargs: Any) -> Any:
        """Cut any supported object by a section.

        Core ``Tube`` and ``TubeChain`` objects use the configured bridge.
        Other objects delegate to their own ``section_cut`` implementation.
        """

        return section_cut(obj, section, bridge=self.section_bridge, **kwargs)

    # ── Small teaching helpers ─────────────────────────────────────────────

    def closing_error(self, samples: Any) -> float:
        """Return ``||x_final - x_initial||`` for trajectory/orbit-like samples."""

        states = getattr(samples, "states", samples)
        arr = np.asarray(states, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            raise ValueError("closing_error requires samples with shape (n_samples, dim).")
        return float(np.linalg.norm(arr[-1] - arr[0]))


__all__ = [
    "TopologyWorkflow",
    "make_poincare_map",
    "orbit_from_map",
    "section_cut",
]
