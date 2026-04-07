"""Tests for island chain connectivity properties.

Verifies n_independent_orbits, is_connected, n_points_per_orbit, and
visit_sequence for both IslandChainOrbit (island_chain.py) and
IslandChain (island.py).
"""
import sys
sys.path.insert(0, r'C:\Users\Legion\Nutstore\1\Repo\pyna')

import numpy as np
import pytest

from pyna.topo.island_chain import IslandChainOrbit, ChainFixedPoint
from pyna.topo.island import Island, IslandChain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_orbit(m: int, n: int) -> IslandChainOrbit:
    """Create a minimal IslandChainOrbit with dummy fixed points."""
    fp = ChainFixedPoint(
        phi=0.0, R=3.0, Z=0.0,
        DPm=np.eye(2),
        DX_pol_accum=np.eye(2),
    )
    return IslandChainOrbit(m=m, n=n, Np=1, fixed_points=[fp], seed_phi=0.0, seed_RZ=(3.0, 0.0))


def make_chain(m: int, n: int, n_islands: int = None) -> IslandChain:
    """Create an IslandChain with dummy Island objects."""
    if n_islands is None:
        n_islands = m
    islands = [
        Island(period_n=m, O_point=np.array([3.0 + i * 0.01, 0.0]))
        for i in range(n_islands)
    ]
    return IslandChain(m=m, n=n, islands=islands)


# ---------------------------------------------------------------------------
# IslandChainOrbit tests
# ---------------------------------------------------------------------------

class TestIslandChainOrbitConnectivity:

    def test_hao_10_3_n_independent_orbits(self):
        orbit = make_orbit(m=10, n=3)
        assert orbit.n_independent_orbits == 1

    def test_hao_10_3_is_connected(self):
        orbit = make_orbit(m=10, n=3)
        assert orbit.is_connected is True

    def test_hao_10_3_n_points_per_orbit(self):
        orbit = make_orbit(m=10, n=3)
        assert orbit.n_points_per_orbit == 10

    def test_hao_10_3_visit_sequence(self):
        orbit = make_orbit(m=10, n=3)
        seq = orbit.visit_sequence()
        assert seq == [[0, 3, 6, 9, 2, 5, 8, 1, 4, 7]]

    def test_w7x_5_5_n_independent_orbits(self):
        orbit = make_orbit(m=5, n=5)
        assert orbit.n_independent_orbits == 5

    def test_w7x_5_5_is_connected(self):
        orbit = make_orbit(m=5, n=5)
        assert orbit.is_connected is False

    def test_w7x_5_5_n_points_per_orbit(self):
        orbit = make_orbit(m=5, n=5)
        assert orbit.n_points_per_orbit == 1

    def test_w7x_5_5_visit_sequence(self):
        orbit = make_orbit(m=5, n=5)
        seq = orbit.visit_sequence()
        assert seq == [[0], [1], [2], [3], [4]]

    def test_generic_6_4_n_independent_orbits(self):
        orbit = make_orbit(m=6, n=4)
        assert orbit.n_independent_orbits == 2

    def test_generic_6_4_n_points_per_orbit(self):
        orbit = make_orbit(m=6, n=4)
        assert orbit.n_points_per_orbit == 3

    def test_generic_6_4_visit_sequence(self):
        orbit = make_orbit(m=6, n=4)
        seq = orbit.visit_sequence()
        assert seq == [[0, 2, 4], [1, 3, 5]]

    def test_generic_6_4_is_connected(self):
        orbit = make_orbit(m=6, n=4)
        assert orbit.is_connected is False

    def test_visit_sequence_covers_all_indices(self):
        """All indices 0..m-1 should appear exactly once across all orbits."""
        for m, n in [(10, 3), (5, 5), (6, 4)]:
            orbit = make_orbit(m=m, n=n)
            seq = orbit.visit_sequence()
            all_indices = [idx for grp in seq for idx in grp]
            assert sorted(all_indices) == list(range(m)), f"Failed for m={m}, n={n}"

    def test_n_per_orbit_zero_n(self):
        """When n=0 (degenerate), n_independent_orbits should be 1."""
        orbit = make_orbit(m=5, n=0)
        assert orbit.n_independent_orbits == 1


# ---------------------------------------------------------------------------
# IslandChain tests
# ---------------------------------------------------------------------------

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

    def test_w7x_5_5_orbit_groups(self):
        chain = make_chain(m=5, n=5)
        groups = chain.orbit_groups
        assert len(groups) == 5
        for g in groups:
            assert len(g) == 1

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
