from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.builders import TubeChainBuilder
from pyna.topo.core import Cycle, IslandChain, PeriodicOrbit, Trajectory, Tube, TubeChain
from pyna.topo.section import HyperplaneSection
from pyna.topo.workflow import section_cut


def _crossing_cycle(y: float) -> Cycle:
    return Cycle(
        trajectory=Trajectory(
            states=np.array([[-1.0, y], [1.0, y]]),
            times=np.array([0.0, 1.0]),
        ),
        period_value=1.0,
    )


def test_cycle_tube_and_tube_chain_cut_to_discrete_geometry():
    section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
    o_cycle = _crossing_cycle(0.0)
    x_cycle = _crossing_cycle(0.2)

    po = o_cycle.section_cut(section)
    assert isinstance(po, PeriodicOrbit)
    assert po.period == 1

    tube = Tube(O_cycle=o_cycle, X_cycles=[x_cycle], label="core-tube")
    chain = tube.section_cut(section)
    assert isinstance(chain, IslandChain)
    assert chain.n_islands == 1
    assert chain.period == 1
    np.testing.assert_allclose(chain.O_points[0].state, np.array([0.0, 0.0]))
    np.testing.assert_allclose(chain.X_points[0].state, np.array([0.0, 0.2]))

    tube_chain = TubeChain(tubes=[tube, Tube(O_cycle=_crossing_cycle(-0.3), label="second")])
    merged = tube_chain.section_cut(section)
    assert merged.n_islands == 2
    assert merged.period == 2
    assert merged.metadata["n_tubes_included"] == 2


def test_workflow_section_cut_function_matches_object_methods():
    section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
    tube = Tube(O_cycle=_crossing_cycle(0.0), X_cycles=[_crossing_cycle(0.4)])

    bridge_chain = section_cut(tube, section)
    method_chain = tube.section_cut(section)
    assert bridge_chain.n_islands == method_chain.n_islands == 1
    np.testing.assert_allclose(
        bridge_chain.islands[0].O_point.state,
        method_chain.islands[0].O_point.state,
    )


def test_section_cut_function_handles_tube_chain():
    section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
    tube_chain = TubeChainBuilder.from_cycles(
        [_crossing_cycle(0.0), _crossing_cycle(0.5)],
        label="built-chain",
    )

    chain = section_cut(tube_chain, section)
    assert chain.n_islands == 2
    assert chain.label == "built-chain"


def test_discrete_objects_reject_second_section_cut():
    section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
    chain = Tube(O_cycle=_crossing_cycle(0.0)).section_cut(section)

    with pytest.raises(ValueError, match="already a reduced discrete object"):
        chain.section_cut(section)

    with pytest.raises(ValueError, match="already a reduced discrete object"):
        chain.islands[0].section_cut(section)
