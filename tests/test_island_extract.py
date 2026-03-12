"""Tests for pyna.topo.island_extract."""
import numpy as np
import pytest
from pyna.topo.island_extract import extract_island_width, IslandChain


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
    assert chain.half_width_R > 0


def test_x_points_count():
    pts = make_synthetic_4island_data()
    chain = extract_island_width(
        pts, R_axis=2.0, Z_axis=0.0,
        mode_m=4,
        psi_func=lambda R, Z: ((R - 2.0)**2 + Z**2) / 0.5**2,
    )
    assert len(chain.X_points) == 4
