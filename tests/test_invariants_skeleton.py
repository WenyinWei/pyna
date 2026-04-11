"""Tests for the new invariant-object skeleton (invariants.py + context.py)."""

import numpy as np
import pytest

from pyna.topo.toroidal_invariants import Cycle, FixedPoint, MonodromyData, PeriodicOrbit
from pyna.topo.invariants import Stability
from pyna.topo.invariant import InvariantTorus
from pyna.topo.toroidal_island import Island, IslandChain
from pyna.topo.toroidal_tube import Tube, TubeChain


# ---------------------------------------------------------------------------
# Scenario 1: Cycle stability + unstable_seeds
# ---------------------------------------------------------------------------

def test_cycle_hyperbolic():
    DPm_X = np.array([[np.exp(0.2), 0.0], [0.0, np.exp(-0.2)]])
    mono_X = MonodromyData(DPm=DPm_X, eigenvalues=np.linalg.eigvals(DPm_X))
    fp = FixedPoint(phi=0.0, R=1.3, Z=0.0, DPm=DPm_X)
    cycle_X = Cycle(winding=(3, 1), sections={0.0: [fp]}, monodromy=mono_X)

    assert cycle_X.stability == Stability.HYPERBOLIC
    expected_residue = (2.0 - (np.exp(0.2) + np.exp(-0.2))) / 4.0
    assert abs(cycle_X.monodromy.greene_residue - expected_residue) < 1e-10
    seeds_R, seeds_Z = cycle_X.unstable_seeds(phi=0.0, n_seeds=8, init_length=1e-4)
    assert len(seeds_R) == 8


# ---------------------------------------------------------------------------
# Scenario 2: Island hierarchy + rotation profile
# ---------------------------------------------------------------------------

def test_island_hierarchy():
    DPm_X = np.array([[np.exp(0.2), 0.0], [0.0, np.exp(-0.2)]])
    # O_point: rotation matrix with theta = 2*pi*0.35 → rotation_vector = (0.35,)
    theta = 2 * np.pi * 0.35
    DPm_O = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    O_fp = FixedPoint(phi=0.0, R=1.05, Z=0.0, DPm=DPm_O)
    X_fp = FixedPoint(phi=0.0, R=1.10, Z=0.0, DPm=DPm_X)

    island = Island(O_orbit=PeriodicOrbit(points=[O_fp]), X_orbits=[PeriodicOrbit(points=[X_fp])])
    t1 = InvariantTorus(rotation_vector=(0.31,), ambient_dim=2)
    t2 = InvariantTorus(rotation_vector=(0.305,), ambient_dim=2)
    island.add_torus(t1, r=0.5)
    island.add_torus(t2, r=0.8)

    prof = island.rotation_profile
    assert abs(prof(0.5)[0] - 0.31) < 1e-10
    assert abs(prof(0.8)[0] - 0.305) < 1e-10
    # central rotation_vector = 0.35 > 0.305
    assert prof(0.0)[0] > prof(0.8)[0]

    sub_chain = IslandChain(islands=[], winding=(1, 1))
    island.add_child_chain(sub_chain)
    assert sub_chain.parent_island is island
    assert sub_chain.depth == 1


# ---------------------------------------------------------------------------
# Scenario 3: TubeChain.section_cut
# ---------------------------------------------------------------------------

def test_tubechain_section_cut():
    DPm_X = np.array([[np.exp(0.2), 0.0], [0.0, np.exp(-0.2)]])
    theta = 2 * np.pi * 0.35
    DPm_O = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    O_fp = FixedPoint(phi=0.0, R=1.05, Z=0.0, DPm=DPm_O)
    X_fp = FixedPoint(phi=0.0, R=1.10, Z=0.0, DPm=DPm_X)
    O_fp2 = FixedPoint(phi=np.pi / 4, R=1.05, Z=0.01, DPm=DPm_O)
    X_fp2 = FixedPoint(phi=np.pi / 4, R=1.10, Z=0.01, DPm=DPm_X)

    O_cycle = Cycle(winding=(10, 3), sections={0.0: [O_fp], np.pi / 4: [O_fp2]})
    X_cycle = Cycle(winding=(10, 3), sections={0.0: [X_fp], np.pi / 4: [X_fp2]})

    tube = Tube(O_cycle=O_cycle, X_cycles=[X_cycle])
    tc = TubeChain(O_cycles=[O_cycle], X_cycles=[X_cycle], tubes=[tube])

    island_chain = tc.section_cut(0.0)
    assert isinstance(island_chain, IslandChain)
    assert len(island_chain.O_points) == 1
    assert len(island_chain.X_points) == 1
    assert island_chain.section_xpoints(0.0)[0] is X_fp
    assert island_chain.section_opoints(0.0)[0] is O_fp

    s = tc.summary()
    assert "TubeChain" in s


# ---------------------------------------------------------------------------
# Scenario 4: DynamicsContext factory
# ---------------------------------------------------------------------------

def test_dynamics_context():
    from pyna.context import DynamicsContext

    class MockFlow:
        phase_dim = 3

    ctx = DynamicsContext(MockFlow())
    c = ctx.cycle(winding=(10, 3))
    assert c.ambient_dim == 3

    t = ctx.tube(O_cycle=c)
    assert t.ambient_dim == 3

    island_chain = ctx.island_chain(O_points=[], X_points=[])
    assert island_chain.ambient_dim == 2


# ---------------------------------------------------------------------------
# Scenario 5: Tube.section_cut period-3 and period-1
# ---------------------------------------------------------------------------

def test_tube_section_cut_period3():
    """period-3 tube: section_cut returns 3 Islands with step() ring."""
    DPm_X = np.array([[np.exp(0.2), 0.0], [0.0, np.exp(-0.2)]])
    DPm_O = np.array([[np.cos(2*np.pi*0.35), -np.sin(2*np.pi*0.35)],
                      [np.sin(2*np.pi*0.35),  np.cos(2*np.pi*0.35)]])
    mono_X = MonodromyData(DPm=DPm_X, eigenvalues=np.linalg.eigvals(DPm_X))

    o_fps = [FixedPoint(phi=0.0, R=1.05+0.01*k, Z=0.01*k, DPm=DPm_O) for k in range(3)]
    x_fps = [FixedPoint(phi=0.0, R=1.10+0.01*k, Z=0.01*k, DPm=DPm_X) for k in range(3)]

    o_cycle = Cycle(winding=(3, 1), sections={0.0: o_fps})
    x_cycle = Cycle(winding=(3, 1), sections={0.0: x_fps}, monodromy=mono_X)

    tube = Tube(O_cycle=o_cycle, X_cycles=[x_cycle])
    islands = tube.section_cut(0.0)

    assert len(islands) == 3
    assert islands[0].step() is islands[1]
    assert islands[1].step() is islands[2]
    assert islands[2].step() is islands[0]   # 循环
    assert islands[0].step_back() is islands[2]

    # TubeChain
    tc = TubeChain(O_cycles=[o_cycle], X_cycles=[x_cycle], tubes=[tube])
    chain = tc.section_cut(0.0)
    assert isinstance(chain, IslandChain)
    assert len(chain.islands) == 3
    assert len(chain.O_points) == 3


def test_tube_section_cut_period1():
    """period-1 tube: section_cut returns 1 Island, step() self-loop."""
    DPm_O = np.array([[np.cos(2*np.pi*0.35), -np.sin(2*np.pi*0.35)],
                      [np.sin(2*np.pi*0.35),  np.cos(2*np.pi*0.35)]])
    o_fp = FixedPoint(phi=0.0, R=1.05, Z=0.0, DPm=DPm_O)
    o_cycle = Cycle(winding=(1, 0), sections={0.0: [o_fp]})
    tube = Tube(O_cycle=o_cycle, X_cycles=[])
    islands = tube.section_cut(0.0)
    assert len(islands) == 1
    assert islands[0].step() is islands[0]   # m=1 自指


def test_root_tube_axis_ordering():
    """Tube.section_cut uses root_tube() to sort Islands by polar angle."""
    import numpy as np

    DPm_O_axis = np.array([[np.cos(2*np.pi*0.35), -np.sin(2*np.pi*0.35)],
                            [np.sin(2*np.pi*0.35),  np.cos(2*np.pi*0.35)]])
    DPm_O = np.array([[np.cos(2*np.pi*0.3), -np.sin(2*np.pi*0.3)],
                      [np.sin(2*np.pi*0.3),  np.cos(2*np.pi*0.3)]])

    # Root Tube: magnetic axis at R=1.0, Z=0.0
    axis_fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, DPm=DPm_O_axis)
    axis_cycle = Cycle(winding=(1, 0), sections={0.0: [axis_fp]})
    axis_tube = Tube(O_cycle=axis_cycle, X_cycles=[])

    # Island tube: m=4 island chain, O-points at different angles (shuffled)
    # angles = 90, 0, 270, 180 -> sorted should be -180, 0, 90, 180 (arctan2 order)
    angles_deg = [90, 0, 270, 180]
    r_island = 0.2
    o_fps = [
        FixedPoint(phi=0.0,
                   R=1.0 + r_island * np.cos(np.radians(a)),
                   Z=r_island * np.sin(np.radians(a)),
                   DPm=DPm_O)
        for a in angles_deg
    ]
    o_cycle = Cycle(winding=(4, 1), sections={0.0: o_fps})
    island_tube = Tube(O_cycle=o_cycle, X_cycles=[])

    # Build hierarchy: axis_tube -> TubeChain -> island_tube
    tc = TubeChain(O_cycles=[o_cycle], X_cycles=[], tubes=[island_tube])
    axis_tube.add_child_chain(tc)   # tc.parent_tube = axis_tube
    island_tube.parent_chain = tc   # island_tube knows its parent chain

    # section_cut should sort by polar angle
    islands = island_tube.section_cut(0.0)
    assert len(islands) == 4

    # Verify polar angles are monotonically increasing
    angles_out = [np.degrees(np.arctan2(isl.O_point.Z - 0.0,
                                         isl.O_point.R - 1.0)) for isl in islands]
    for i in range(len(angles_out) - 1):
        assert angles_out[i] < angles_out[i+1], f"Not sorted: {angles_out}"

    # step() ring links correct
    assert islands[0].step() is islands[1]
    assert islands[3].step() is islands[0]

    # Island.root_island() traversal should not crash
    ic = tc.section_cut(0.0)
    for isl in ic.islands:
        assert isl.root_island() is not None
