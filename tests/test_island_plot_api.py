import numpy as np
import pytest

from pyna.topo.island import Island
from pyna.topo.invariants import Cycle, FixedPoint
from pyna.topo import (
    island_section_points,
    island_chain_section_points,
)


def test_island_section_points_basic():
    isl = Island(
        period_n=10,
        O_point=np.array([1.0, 0.0]),
        X_points=[np.array([1.1, 0.1]), np.array([0.9, -0.1])],
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
