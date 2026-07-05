"""Tests for pyna.topo.island_extract."""
import numpy as np
import pytest
from pyna.topo.island_extract import (
    extract_island_width,
    extract_island_width_newton,
    IslandChain,
)


def make_synthetic_4island_data(n_pts=200, R_axis=2.0, Z_axis=0.0,
                                 r_res=0.4, island_w=0.03, noise=0.005):
    """Generate synthetic Poincaré scatter data for a 4-island chain."""
    rng = np.random.default_rng(42)
    mode_m = 4
    angles_O = np.linspace(0, 2 * np.pi, mode_m, endpoint=False) + 0.1
    pts = []
    per_island = n_pts // mode_m
    for angle in angles_O:
        R_O = R_axis + r_res * np.cos(angle)
        Z_O = Z_axis + r_res * np.sin(angle)
        # Scatter around each O-point
        r_rand = r_res + rng.uniform(-island_w, island_w, per_island)
        th_rand = angle + rng.uniform(-0.3, 0.3, per_island)
        R_pts = R_axis + r_rand * np.cos(th_rand)
        Z_pts = Z_axis + r_rand * np.sin(th_rand)
        pts.extend(zip(R_pts, Z_pts))
    return np.array(pts)


def make_synthetic_2island_data(n_pts=200, R_axis=2.0, Z_axis=0.0,
                                 r_res=0.4, island_w=0.025):
    """Generate two island point-cloud clusters centred on known O-points."""
    rng = np.random.default_rng(123)
    angles_O = np.array([0.0, np.pi])
    pts = []
    per_island = n_pts // 2
    for angle in angles_O:
        r_rand = r_res + rng.uniform(-island_w, island_w, per_island)
        th_rand = angle + rng.uniform(-0.18, 0.18, per_island)
        R_pts = R_axis + r_rand * np.cos(th_rand)
        Z_pts = Z_axis + r_rand * np.sin(th_rand)
        pts.extend(zip(R_pts, Z_pts))
    return np.array(pts)


def _piecewise_fixed_point_map(R, Z, period):
    """Synthetic Poincare map with two O-points and two X-points."""
    del period
    x = np.asarray([R, Z], dtype=float)
    centers = np.asarray([
        [2.4, 0.0],
        [1.6, 0.0],
        [2.0, 0.4],
        [2.0, -0.4],
    ], dtype=float)
    kinds = np.asarray(["O", "O", "X", "X"])
    idx = int(np.argmin(np.linalg.norm(centers - x[None, :], axis=1)))
    center = centers[idx]
    if kinds[idx] == "O":
        angle = 0.7
        mat = np.asarray([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
    else:
        mat = np.asarray([[1.8, 0.0], [0.0, 0.55]])
    return center + mat @ (x - center)


def _failing_map(R, Z, period):
    del R, Z, period
    return np.array([np.nan, np.nan])


def _psi(R, Z):
    return ((R - 2.0) ** 2 + Z ** 2) / 0.5 ** 2


def _assert_point_set_close(actual, expected, tol=1.0e-7):
    actual = np.asarray(actual, dtype=float).reshape((-1, 2))
    expected = np.asarray(expected, dtype=float).reshape((-1, 2))
    assert len(actual) == len(expected)
    for point in expected:
        assert np.min(np.linalg.norm(actual - point[None, :], axis=1)) < tol


def test_4island_detects_correct_count():
    pts = make_synthetic_4island_data()
    chain = extract_island_width(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=4,
        psi_func=lambda R, Z: ((R - 2.0)**2 + Z**2) / 0.5**2,
    )
    assert isinstance(chain, IslandChain)
    assert len(chain.O_points) == 4, f"Expected 4 O-points, got {len(chain.O_points)}"


def test_half_width_positive():
    pts = make_synthetic_4island_data()
    chain = extract_island_width(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=4,
        psi_func=lambda R, Z: ((R - 2.0)**2 + Z**2) / 0.5**2,
    )
    assert chain.half_width_r > 0


def test_x_points_count():
    pts = make_synthetic_4island_data()
    chain = extract_island_width(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=4,
        psi_func=lambda R, Z: ((R - 2.0)**2 + Z**2) / 0.5**2,
    )
    assert len(chain.X_points) == 4


def test_legacy_max_newton_iter_is_compatibility_only():
    pts = make_synthetic_4island_data()
    chain_a = extract_island_width(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=4,
        psi_func=_psi,
        max_newton_iter=1,
    )
    chain_b = extract_island_width(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=4,
        psi_func=_psi,
        max_newton_iter=999,
    )
    np.testing.assert_allclose(chain_a.O_points, chain_b.O_points)
    np.testing.assert_allclose(chain_a.X_points, chain_b.X_points)
    assert chain_a.half_width_r == pytest.approx(chain_b.half_width_r)


def test_newton_requires_exactly_one_map_source():
    pts = make_synthetic_2island_data()
    with pytest.raises(ValueError, match="exactly one"):
        extract_island_width_newton(
            pts, R_axis=2.0, Z_axis=0.0,
            mode_m=2,
            psi_func=_psi,
        )
    with pytest.raises(ValueError, match="exactly one"):
        extract_island_width_newton(
            pts, R_axis=2.0, Z_axis=0.0,
            mode_m=2,
            psi_func=_psi,
            map_func=_piecewise_fixed_point_map,
            tracer=object(),
        )


def test_newton_refines_and_classifies_o_and_x_points():
    pts = make_synthetic_2island_data()
    chain = extract_island_width_newton(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=2,
        psi_func=_psi,
        map_func=_piecewise_fixed_point_map,
        fallback_to_point_cloud=False,
        tol=1.0e-10,
    )

    _assert_point_set_close(chain.O_points, [[2.4, 0.0], [1.6, 0.0]])
    _assert_point_set_close(chain.X_points, [[2.0, 0.4], [2.0, -0.4]])
    assert np.count_nonzero(chain.fixed_point_kinds == "O") == 2
    assert np.count_nonzero(chain.fixed_point_kinds == "X") == 2
    assert np.all(chain.fixed_point_residuals < 1.0e-10)
    assert chain.half_width_r > 0.0


def test_newton_fallback_returns_point_cloud_seeds_when_map_fails():
    pts = make_synthetic_4island_data()
    chain = extract_island_width_newton(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=4,
        psi_func=_psi,
        map_func=_failing_map,
        fallback_to_point_cloud=True,
    )

    assert len(chain.O_points) == 4
    assert len(chain.X_points) == 4
    assert chain.fixed_point_residuals.size == 0
    assert chain.metadata["used_point_cloud_fallback_O"] is True
    assert chain.metadata["used_point_cloud_fallback_X"] is True


def test_newton_without_fallback_returns_only_converged_points():
    pts = make_synthetic_4island_data()
    chain = extract_island_width_newton(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=4,
        psi_func=_psi,
        map_func=_failing_map,
        fallback_to_point_cloud=False,
    )

    assert chain.O_points.shape == (0, 2)
    assert chain.X_points.shape == (0, 2)
    assert np.isnan(chain.half_width_r)
    assert np.isnan(chain.half_width_psi)
