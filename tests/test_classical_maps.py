"""Tests for pyna.topo.classical_maps — Hénon map and standard map."""

import numpy as np
import pytest
from pyna.topo.classical_maps import HenonMap, StandardMap, poincare_map_henon


class TestHenonMap:

    def test_step_area_preserving(self):
        """Jacobian determinant should be 1 at any point."""
        m = HenonMap(K=0.3)
        for x, y in [(0.3, 0.1), (-0.2, 0.4), (0.0, 0.0)]:
            J = m.jacobian_at(x, y)
            assert np.isclose(np.linalg.det(J), 1.0, atol=1e-12)

    def test_fixed_points_location(self):
        """Fixed points should satisfy the map equation."""
        m = HenonMap(K=0.4)
        o, x = m.fixed_points()
        for pt in [o, x]:
            x0, y0 = pt
            x1, y1 = m.step(x0, y0)
            assert np.isclose(x1, x0, atol=1e-12)
            assert np.isclose(y1, y0, atol=1e-12)

    def test_o_point_eigenvalues_unit_circle(self):
        """O-point monodromy eigenvalues should lie on the unit circle (|λ|=1).
        
        For x* = (-1+sqrt(1+4K))/2 to be elliptic we need |tr(M)| < 2.
        tr = -4*x* - 1 (since J = [[-4x,-1],[1,0]]).
        Small positive K gives x* ≈ K (small), so |tr| < 2.
        """
        m = HenonMap(K=0.05)  # small K → clearly elliptic O-point
        o, _ = m.fixed_points()
        M = m.monodromy_at_fixed_point(o)
        eigs = np.linalg.eigvals(M)
        assert all(np.isclose(abs(e), 1.0, atol=1e-10) for e in eigs)

    def test_x_point_eigenvalues_outside_unit_circle(self):
        """X-point monodromy eigenvalues: |λ1|>1 and |λ2|<1 (saddle)."""
        m = HenonMap(K=0.5)
        _, x = m.fixed_points()
        M = m.monodromy_at_fixed_point(x)
        eigs = np.abs(np.linalg.eigvals(M))
        assert max(eigs) > 1.0
        assert min(eigs) < 1.0

    def test_iterate_bounded_for_small_K(self):
        """For small K near O-point, orbit stays bounded."""
        m = HenonMap(K=0.05)
        o, _ = m.fixed_points()
        xs, ys = m.iterate(o[0] + 0.01, o[1], n_steps=500)
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(ys))
        assert np.all(np.abs(xs) < 5)

    def test_no_fixed_points_for_negative_K(self):
        """No real fixed points when discriminant < 0 (K < -0.25)."""
        m = HenonMap(K=-0.5)
        o, x = m.fixed_points()
        assert o is None
        assert x is None

    def test_poincare_map_henon_returns_dict(self):
        results = poincare_map_henon([0.1, 0.3], n_steps=100, n_seeds=5)
        assert 0.1 in results
        xs, ys = results[0.1]
        assert len(xs) > 0
        assert xs.shape == ys.shape


class TestStandardMap:

    def test_step_area_preserving(self):
        """Jacobian determinant should be 1 for any (p, x)."""
        m = StandardMap(K=0.5)
        for p, x in [(1.0, 0.5), (2.0, 3.1), (0.0, np.pi)]:
            J = m.jacobian_at(p, x)
            assert np.isclose(np.linalg.det(J), 1.0, atol=1e-12)

    def test_iterate_bounded_small_K(self):
        """For small K, iterates should remain in [0, 2π]."""
        m = StandardMap(K=0.1)
        ps, xs = m.iterate(np.pi, np.pi / 2, n_steps=200)
        assert np.all(ps >= 0) and np.all(ps <= 2 * np.pi)
        assert np.all(xs >= 0) and np.all(xs <= 2 * np.pi)

    def test_step_modular(self):
        """Output should be in [0, 2π)."""
        m = StandardMap(K=0.5)
        for _ in range(50):
            p = np.random.uniform(0, 2 * np.pi)
            x = np.random.uniform(0, 2 * np.pi)
            pn, xn = m.step(p, x)
            assert 0 <= pn < 2 * np.pi
            assert 0 <= xn < 2 * np.pi
