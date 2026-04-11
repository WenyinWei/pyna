"""Tests for island chain connectivity.

Verifies n_independent_orbits, is_connected, n_points_per_orbit, and
orbit_groups for IslandChain (island.py).
"""
import sys
sys.path.insert(0, r'C:\Users\Legion\Nutstore\1\Repo\pyna')

import numpy as np
import pytest

from pyna.topo.toroidal_invariants import FixedPoint, PeriodicOrbit
from pyna.topo.toroidal_island import Island, IslandChain


def make_chain(m: int, n: int, n_islands: int = None) -> IslandChain:
    """Create an IslandChain with dummy Island objects."""
    from math import gcd
    if n_islands is None:
        n_islands = m // gcd(m, n)
    islands = [
        Island(O_orbit=PeriodicOrbit(points=[
            FixedPoint(phi=0.0, R=3.0 + i * 0.01, Z=0.0, DPm=np.eye(2), kind='O')
        ]))
        for i in range(n_islands)
    ]
    return IslandChain(m=m, n=n, islands=islands)


class TestIslandChainConnectivity:

    def test_hao_10_3_n_independent_orbits(self):
        chain = make_chain(m=10, n=3)
        assert chain.n_independent_orbits == 1

    def test_hao_10_3_is_connected(self):
        chain = make_chain(m=10, n=3)
        assert chain.is_connected is True

    def test_hao_10_3_orbit_groups(self):
        chain = make_chain(m=10, n=3)
        groups = chain.orbit_groups
        assert len(groups) == 1
        assert len(groups[0]) == 10

    def test_w7x_5_5_n_independent_orbits(self):
        chain = make_chain(m=5, n=5)
        assert chain.n_independent_orbits == 5

    def test_w7x_5_5_is_connected(self):
        chain = make_chain(m=5, n=5)
        assert chain.is_connected is False

    def test_w7x_5_5_has_single_island_after_section_cut(self):
        chain = make_chain(m=5, n=5)
        assert chain.n_islands == 1
        groups = chain.orbit_groups
        assert len(groups) == 5
        assert sum(len(g) for g in groups) == 1

    def test_generic_6_4_orbit_groups(self):
        chain = make_chain(m=6, n=4)
        groups = chain.orbit_groups
        assert len(groups) == 2
        assert len(groups[0]) == 3
        assert len(groups[1]) == 3

    def test_orbit_groups_all_islands_covered(self):
        """Every island should appear in exactly one orbit group."""
        chain = make_chain(m=6, n=4)
        groups = chain.orbit_groups
        flat = [isl for grp in groups for isl in grp]
        assert len(flat) == len(chain.islands)
        assert set(id(x) for x in flat) == set(id(x) for x in chain.islands)
