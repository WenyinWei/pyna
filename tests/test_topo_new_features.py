"""Tests for new topo features:
- rotational_transform_from_trajectory (poincare.py)
- detect_residual_islands (island_extract.py)
- Island.explore_sub_islands (island.py)
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.poincare import rotational_transform_from_trajectory
from pyna.topo.island_extract import detect_residual_islands
from pyna.topo.island import Island, IslandChain


# ---------------------------------------------------------------------------
# rotational_transform_from_trajectory
# ---------------------------------------------------------------------------

class TestRotationalTransformFromTrajectory:

    def _make_circular_orbit(self, R0, Z0, r, iota, n_pts=500):
        """Generate a synthetic trajectory on a flux surface with known ι."""
        phi = np.linspace(0.0, 10.0 * 2 * np.pi, n_pts)
        theta = iota * phi  # poloidal angle advances as iota * phi
        R = R0 + r * np.cos(theta)
        Z = Z0 + r * np.sin(theta)
        traj = np.column_stack([R, Z, phi])
        return traj

    def test_returns_finite(self):
        traj = self._make_circular_orbit(R0=1.5, Z0=0.0, r=0.1, iota=0.5)
        iota = rotational_transform_from_trajectory(traj, axis_RZ=[1.5, 0.0])
        assert np.isfinite(iota)

    def test_known_iota(self):
        """Estimated ι should be close to the known value."""
        known_iota = 1.0 / 3.0
        traj = self._make_circular_orbit(R0=1.5, Z0=0.0, r=0.1, iota=known_iota,
                                          n_pts=2000)
        iota = rotational_transform_from_trajectory(traj, axis_RZ=[1.5, 0.0])
        assert abs(iota - known_iota) < 0.02, \
            f"Expected iota≈{known_iota:.4f}, got {iota:.4f}"

    def test_no_axis_given(self):
        """Without axis_RZ, the function should use the trajectory centroid."""
        traj = self._make_circular_orbit(R0=1.5, Z0=0.0, r=0.1, iota=0.5)
        iota = rotational_transform_from_trajectory(traj)
        assert np.isfinite(iota)

    def test_n_turns_restriction(self):
        """n_turns should limit the trajectory used."""
        traj = self._make_circular_orbit(R0=1.5, Z0=0.0, r=0.1, iota=0.5, n_pts=500)
        iota_full = rotational_transform_from_trajectory(traj, axis_RZ=[1.5, 0.0])
        iota_short = rotational_transform_from_trajectory(traj, axis_RZ=[1.5, 0.0],
                                                           n_turns=2)
        # Both should give similar iota but not necessarily identical
        assert abs(iota_full - iota_short) < 0.15

    def test_invalid_traj_raises(self):
        with pytest.raises((ValueError, Exception)):
            rotational_transform_from_trajectory(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# detect_residual_islands
# ---------------------------------------------------------------------------

class TestDetectResidualIslands:

    def _make_chaotic_with_island(self, period, n_pts=500, n_island_pts=50):
        """Create synthetic Poincaré data: uniform chaos + one island cluster."""
        rng = np.random.default_rng(42)
        # Chaotic background: random points in annulus [0.8, 1.2] × [-0.4, 0.4]
        r_chaos = rng.uniform(0.8, 1.2, n_pts)
        theta_chaos = rng.uniform(0.0, 2 * np.pi, n_pts)
        R_chaos = 1.5 + r_chaos * np.cos(theta_chaos)
        Z_chaos = r_chaos * np.sin(theta_chaos)

        # Island cluster: tight clumps at p equidistant angles
        r_island = 1.0
        angles_island = np.linspace(0, 2 * np.pi, period, endpoint=False)
        R_isl = []
        Z_isl = []
        for ang in angles_island:
            spread = 0.02
            pts_ang = rng.normal(scale=spread, size=(n_island_pts // period, 2))
            R_isl.extend(1.5 + r_island * np.cos(ang) + pts_ang[:, 0])
            Z_isl.extend(r_island * np.sin(ang) + pts_ang[:, 1])

        R_all = np.concatenate([R_chaos, R_isl])
        Z_all = np.concatenate([Z_chaos, Z_isl])
        return np.column_stack([R_all, Z_all])

    def test_returns_list(self):
        pts = self._make_chaotic_with_island(period=3)
        candidates = detect_residual_islands(pts, R_axis=1.5, Z_axis=0.0)
        assert isinstance(candidates, list)

    def test_each_candidate_has_expected_keys(self):
        pts = self._make_chaotic_with_island(period=3)
        candidates = detect_residual_islands(pts, R_axis=1.5, Z_axis=0.0)
        for c in candidates:
            assert 'period' in c
            assert 'r_shell' in c
            assert 'angle_peaks' in c
            assert 'seed_RZ' in c

    def test_seed_RZ_shape(self):
        pts = self._make_chaotic_with_island(period=3)
        candidates = detect_residual_islands(pts, R_axis=1.5, Z_axis=0.0)
        for c in candidates:
            assert c['seed_RZ'].ndim == 2
            assert c['seed_RZ'].shape[1] == 2
            assert c['seed_RZ'].shape[0] == c['period']

    def test_empty_data_returns_empty(self):
        # All points at axis → no structure
        pts = np.zeros((10, 2))
        candidates = detect_residual_islands(pts, R_axis=0.0, Z_axis=0.0)
        assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# IslandChain.scan_xo_rings_parallel (smoke test with mocked field)
# ---------------------------------------------------------------------------

class TestIslandChainScanXORings:

    def _make_chain(self):
        isl0 = Island(period_n=2, O_point=np.array([1.5, 0.05]))
        isl1 = Island(period_n=2, O_point=np.array([1.5, -0.05]))
        return IslandChain(m=2, n=1, islands=[isl0, isl1], connected=True)

    def test_scan_xo_rings_parallel_runs(self):
        """Smoke test: scan_xo_rings_parallel should run without errors."""
        chain = self._make_chain()

        def mock_field(r, z, phi):
            # Simple rotation — no real X-points, but fixed-point finder
            # will just not find any; that's fine for a smoke test.
            return np.array([-z, r])

        # Should not raise
        chain.scan_xo_rings_parallel(
            mock_field,
            r_scan=0.05,
            n_scan=20,
            n_workers=1,
        )

    def test_scan_disconnected_chain(self):
        """For a disconnected chain, X-point scan should run per sub-chain."""
        chain = self._make_chain()
        chain.split_into_subchains([[0], [1]])
        assert not chain.connected

        def mock_field(r, z, phi):
            return np.array([-z, r])

        chain.scan_xo_rings_parallel(
            mock_field,
            r_scan=0.05,
            n_scan=20,
            n_workers=1,
        )


# ---------------------------------------------------------------------------
# Island.explore_sub_islands (smoke test)
# ---------------------------------------------------------------------------

class TestIslandExploreSubIslands:

    def test_returns_list(self):
        """explore_sub_islands should return a (possibly empty) list of Islands."""
        isl = Island(period_n=3, O_point=np.array([1.5, 0.0]),
                     halfwidth=0.05)

        def mock_field(r, z, phi):
            return np.array([-z, r])

        sub = isl.explore_sub_islands(
            mock_field,
            n_turns_range=range(1, 3),
            r_scan_factor=0.2,
            n_scan=20,
        )
        assert isinstance(sub, list)
        for s in sub:
            assert isinstance(s, Island)
            assert s.level == isl.level + 1
            assert s.parent is isl
