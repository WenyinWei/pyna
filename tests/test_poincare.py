"""Tests for pyna.topo.poincare."""
import numpy as np
import pytest
from pyna.topo.poincare import ToroidalSection, PoincareMap


def test_toroidal_section_detects_crossing():
    sec = ToroidalSection(phi0=0.0)
    # Step from phi slightly below 0 to slightly above 0
    pt_prev = np.array([1.0, 0.0, -0.05])
    pt_curr = np.array([1.0, 0.0,  0.05])
    crossing = sec.detect_crossing(pt_prev, pt_curr)
    assert crossing is not None
    # phi of crossing should be ~0 (mod 2pi)
    assert abs(crossing[2] % (2 * np.pi)) < 0.1


def test_toroidal_section_no_crossing():
    sec = ToroidalSection(phi0=np.pi)
    # Step from phi=0.1 to 0.2 — doesn't cross pi
    pt_prev = np.array([1.0, 0.0, 0.1])
    pt_curr = np.array([1.0, 0.0, 0.2])
    crossing = sec.detect_crossing(pt_prev, pt_curr)
    assert crossing is None


def test_poincare_map_accumulates():
    sec = ToroidalSection(phi0=0.0)
    pmap = PoincareMap([sec])

    # Simulate a trajectory that crosses phi=0 twice
    traj = np.array([
        [1.5, 0.0, 6.0],   # phi ~ 2pi-0.28
        [1.5, 0.0, 6.4],   # crosses phi=0 (mod 2pi)
        [1.5, 0.0, 0.3],
        [1.5, 0.0, 6.1],   # crosses phi=0 again
        [1.5, 0.0, 6.5],
    ])
    pmap.record_trajectory(traj)
    arr = pmap.crossing_array(0)
    assert arr.shape[1] == 3
    assert arr.shape[0] >= 1


def test_crossing_array_empty():
    sec = ToroidalSection(phi0=0.0)
    pmap = PoincareMap([sec])
    arr = pmap.crossing_array(0)
    assert arr.shape == (0, 3)
