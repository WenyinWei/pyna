import numpy as np
import pytest

from pyna.topo.toroidal_island import Island
from pyna.topo.toroidal_invariants import Cycle, FixedPoint, PeriodicOrbit
from pyna.topo import (
    island_section_points,
    island_chain_section_points,
)


def test_island_section_points_basic():
    fp_O = FixedPoint(phi=0.0, R=1.0, Z=0.0, DPm=np.eye(2), kind='O')
    fp_X1 = FixedPoint(phi=0.0, R=1.1, Z=0.1, DPm=np.array([[2.,0],[0,.5]]), kind='X')
    fp_X2 = FixedPoint(phi=0.0, R=0.9, Z=-0.1, DPm=np.array([[2.,0],[0,.5]]), kind='X')
    isl = Island(
        O_orbit=PeriodicOrbit(points=[fp_O]),
        X_orbits=[PeriodicOrbit(points=[fp_X1]), PeriodicOrbit(points=[fp_X2])],
        label='10/3',
    )
    sec = island_section_points(isl)
    assert len(sec['O_points']) == 1
    assert len(sec['X_points']) == 2
    assert np.allclose(sec['O_points'][0], [1.0, 0.0])


def test_island_chain_section_points_from_cycle():
    fp1 = FixedPoint(phi=0.0, R=1.1, Z=0.2, DPm=np.diag([2.0, 0.5]), kind='X')
    fp2 = FixedPoint(phi=0.0, R=1.0, Z=0.1, DPm=np.diag([0.8, 0.8]), kind='O')
    cycle = Cycle(winding=(10, 3), sections={0.0: [fp1, fp2]},
                  monodromy=fp1.monodromy, ambient_dim=2)
    sec = island_chain_section_points(cycle, phi=0.0)
    assert len(sec['X_points']) == 1
    assert len(sec['O_points']) == 1
