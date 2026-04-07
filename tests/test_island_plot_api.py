import numpy as np

from pyna.topo.island import Island
from pyna.topo.island_chain import ChainFixedPoint, IslandChainOrbit
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


def test_island_chain_section_points_from_orbit():
    fp1 = ChainFixedPoint(phi=0.0, R=1.1, Z=0.2, DPm=np.diag([2.0, 0.5]), DX_pol_accum=np.eye(2))
    fp2 = ChainFixedPoint(phi=0.0, R=1.0, Z=0.1, DPm=np.diag([0.8, 0.8]), DX_pol_accum=np.eye(2))
    chain = IslandChainOrbit(m=10, n=3, Np=2, fixed_points=[fp1, fp2], seed_phi=0.0, seed_RZ=(1.1, 0.2))
    sec = island_chain_section_points(chain, phi=0.0)
    assert len(sec['X_points']) == 1
    assert len(sec['O_points']) == 1
