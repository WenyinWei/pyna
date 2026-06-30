from __future__ import annotations

import numpy as np
import pytest

from pyna.topo import CoreTube, TopologyWorkflow
from pyna.topo.core import Cycle, IslandChain, PeriodicOrbit, Trajectory
from pyna.topo.dynamics import GeneralPoincareMap
from pyna.topo.section import HyperplaneSection


def _crossing_cycle(y: float) -> Cycle:
    return Cycle(
        trajectory=Trajectory(
            states=np.array([[-1.0, y], [1.0, y]]),
            times=np.array([0.0, 1.0]),
        )
    )


def test_workflow_teaches_flow_to_cycle_to_section_cut():
    wf = TopologyWorkflow(closure_tol=1e-3)
    flow = wf.system(
        "callable-flow",
        rhs=lambda x, t: np.array([x[1], -x[0]]),
        dim=2,
        coordinate_names=("q", "p"),
    )
    traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2.0 * np.pi), dt=0.01)

    assert wf.closing_error(traj) < 1e-3
    cycle = wf.closed_cycle(traj)
    assert isinstance(cycle, Cycle)

    section = HyperplaneSection(np.array([1.0, 0.0]), 1.0, phase_dim=2)
    orbit = wf.section_cut(cycle, section)
    assert isinstance(orbit, PeriodicOrbit)
    assert orbit.period >= 1

    pmap = wf.poincare_map(flow, section, dt=0.05)
    assert isinstance(pmap, GeneralPoincareMap)


def test_workflow_refuses_open_trajectory_cycle_promotion():
    wf = TopologyWorkflow()
    open_traj = Trajectory(
        states=np.array([[0.0, 0.0], [1.0, 0.0]]),
        times=np.array([0.0, 1.0]),
    )

    with pytest.raises(ValueError, match="not closed"):
        wf.closed_cycle(open_traj)


def test_workflow_builds_tube_and_island_chain_from_cycles():
    wf = TopologyWorkflow()
    section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)

    tube = wf.tube(_crossing_cycle(0.0), [_crossing_cycle(0.25)], label="teaching-tube")
    assert isinstance(tube, CoreTube)

    chain = wf.section_cut(tube, section)
    assert isinstance(chain, IslandChain)
    assert chain.n_islands == 1
    assert len(chain.X_points) == 1

    tube_chain = wf.tube_chain([_crossing_cycle(0.0), _crossing_cycle(0.5)], label="chain")
    merged = wf.section_cut(tube_chain, section)
    assert merged.n_islands == 2


def test_workflow_builds_map_orbits_and_island_chain():
    wf = TopologyWorkflow()
    identity = wf.system("callable-map", step_func=lambda x: x.copy(), dim=2)

    orbit = wf.orbit(identity, [0.0, 0.0], 2)
    assert orbit.n_samples == 3
    assert orbit.period_guess == 1

    po_o = wf.periodic_orbit([[0.0, 0.0]], map_obj=identity)
    po_x = wf.periodic_orbit([[0.2, 0.0]], map_obj=identity)
    chain = wf.island_chain([po_o], [po_x], proximity_tol=0.5)
    assert chain.n_islands == 1
    assert chain.X_points[0].state[0] == pytest.approx(0.2)


def test_thin_object_and_toroidal_bridge_classes_are_not_public_exports():
    import pyna.topo as topo

    assert "ObjectSectionCutBridge" not in topo.__all__
    assert "ToroidalSectionCutBridge" not in topo.__all__
    assert "ContinuousDiscreteBridge" not in topo.__all__
