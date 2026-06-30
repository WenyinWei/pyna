from __future__ import annotations

import numpy as np
import pytest

from pyna.dynamics import CallableFlow, CallableMap
from pyna.topo.builders import GeometryBuilder, IslandChainBuilder
from pyna.topo.core import Cycle, PeriodicOrbit, Trajectory
from pyna.topo.dynamics import GeneralPoincareMap
from pyna.topo.factories import (
    DynamicalSystemFactory,
    GeometryFactory,
    PoincareMapFactory,
    Registry,
)
from pyna.topo.adapters import as_periodic_orbit
from pyna.topo.core import SectionPoint
from pyna.topo.section import HyperplaneSection, ToroidalSection


def test_registry_register_duplicate_unknown_and_copy_are_explicit():
    registry = Registry()
    registry.register("toy", lambda value: value + 1)

    assert registry.create("toy", value=4) == 5
    assert tuple(registry.keys()) == ("toy",)

    with pytest.raises(KeyError, match="already registered"):
        registry.register("toy", lambda value: value)

    with pytest.raises(KeyError, match="unknown registry key"):
        registry.create("missing")

    local = registry.copy()
    local.register("other", lambda value: value * 2)
    assert local.create("other", value=3) == 6
    assert "other" not in registry.keys()


def test_dynamical_system_factory_builds_flow_and_map_from_spec():
    flow = DynamicalSystemFactory.create(
        "callable-flow",
        rhs=lambda x, t: np.array([x[0]]),
        dim=1,
        coordinate_names=("x",),
        label="linear flow",
    )
    assert isinstance(flow, CallableFlow)
    assert flow.phase_space.coordinate_names == ("x",)

    cmap = DynamicalSystemFactory.from_spec(
        {
            "kind": "callable-map",
            "params": {
                "step_func": lambda x: np.array([x[0] + 1.0]),
                "dim": 1,
                "label": "translation",
            },
        }
    )
    assert isinstance(cmap, CallableMap)
    np.testing.assert_allclose(cmap.step(np.array([2.0])), np.array([3.0]))


def test_poincare_map_factory_returns_executable_general_map_by_default():
    flow = CallableFlow(lambda x, t: np.array([1.0, 0.0]), dim=2)
    section = HyperplaneSection(np.array([1.0, 0.0]), 0.5, phase_dim=2)

    pm = PoincareMapFactory.create(flow=flow, section=section, dt=0.1)
    assert isinstance(pm, GeneralPoincareMap)

    assert isinstance(flow.make_poincare_map(section), GeneralPoincareMap)


def test_poincare_map_factory_auto_detects_flow_field_cache(monkeypatch):
    import pyna.topo.dynamics as dyn

    calls = {}

    class FakeMCFMap:
        def __init__(self, field_cache, **kwargs):
            calls["field_cache"] = field_cache
            calls["kwargs"] = kwargs

    class FlowWithFieldCache:
        field_cache = {"prepared": True}

    monkeypatch.setattr(dyn, "MCFPoincareMap", FakeMCFMap)
    pm = PoincareMapFactory.create(flow=FlowWithFieldCache(), section=ToroidalSection(0.25))
    assert isinstance(pm, FakeMCFMap)
    assert calls["field_cache"] == {"prepared": True}
    assert calls["kwargs"]["phi_section"] == pytest.approx(0.25)


def test_geometry_builder_requires_explicit_closed_trajectory_promotion():
    builder = GeometryBuilder(closure_tol=1e-10)
    closed = Trajectory(
        states=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        times=np.array([0.0, 0.5, 1.0]),
    )
    cycle = builder.cycle(closed, require_closed=True)
    assert isinstance(cycle, Cycle)
    assert cycle.period_value == pytest.approx(1.0)

    open_traj = Trajectory(
        states=np.array([[0.0, 0.0], [1.0, 0.0]]),
        times=np.array([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="not closed"):
        builder.cycle(open_traj, require_closed=True)


def test_periodic_orbit_builder_verifies_map_iteration():
    builder = GeometryBuilder()
    identity = CallableMap(lambda x: x.copy(), dim=1)
    po = builder.periodic_orbit([[0.0]], map_obj=identity, verify=True)
    assert isinstance(po, PeriodicOrbit)
    assert po.period == 1

    shift = CallableMap(lambda x: x + 1.0, dim=1)
    with pytest.raises(ValueError, match="does not close"):
        builder.periodic_orbit([[0.0]], map_obj=shift, verify=True)


def test_periodic_orbit_adapter_verifies_section_point_sequences():
    shift = CallableMap(lambda x: x + 1.0, dim=1)
    points = [SectionPoint(state=np.array([0.0]))]

    with pytest.raises(ValueError, match="does not close"):
        as_periodic_orbit(points, map_obj=shift, verify=True)


def test_geometry_factory_and_island_chain_builder_promote_discrete_objects():
    po_a = GeometryFactory.create("periodic-orbit", obj=[[0.0, 0.0]], verify=False)
    po_b = GeometryFactory.create("periodic-orbit", obj=[[0.1, 0.0]], verify=False)

    chain = IslandChainBuilder.from_periodic_orbits(
        [po_a],
        [po_b],
        label="test-chain",
        proximity_tol=0.5,
    )
    assert chain.n_islands == 1
    assert chain.period == 1
    assert chain.islands[0].step() is chain.islands[0]
    assert len(chain.X_points) == 1


def test_island_chain_builder_assigns_x_orbits_to_nearest_o_once():
    po_left = GeometryFactory.create("periodic-orbit", obj=[[0.0, 0.0]], verify=False)
    po_right = GeometryFactory.create("periodic-orbit", obj=[[10.0, 0.0]], verify=False)
    x_right = GeometryFactory.create("periodic-orbit", obj=[[9.5, 0.0]], verify=False)

    chain = IslandChainBuilder.from_periodic_orbits(
        [po_left, po_right],
        [x_right],
        proximity_tol=100.0,
    )

    assert len(chain.islands[0].X_orbits) == 0
    assert len(chain.islands[1].X_orbits) == 1
    assert chain.X_points[0].state[0] == pytest.approx(9.5)
