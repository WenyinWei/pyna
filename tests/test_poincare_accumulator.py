"""Tests for PoincareAccumulator (renamed from PoincareMap) and related helpers."""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.poincare import (
    PoincareAccumulator,
    PoincareMap,          # backward compat alias
    ToroidalSection,      # backward compat alias
    poincare_from_fieldlines,
    rotational_transform_from_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_trajectory(n_turns: int = 3) -> np.ndarray:
    """Return a helical trajectory with n_turns toroidal turns."""
    phi = np.linspace(0, 2 * np.pi * n_turns, 300 * n_turns)
    R = 1.0 + 0.1 * np.cos(phi * 1.5)   # some winding
    Z = 0.1 * np.sin(phi * 1.5)
    return np.column_stack([R, Z, phi])


# ---------------------------------------------------------------------------
# PoincareAccumulator method API
# ---------------------------------------------------------------------------

def test_poincare_accumulator_has_record_step():
    acc = PoincareAccumulator([ToroidalSection(0.0)])
    assert hasattr(acc, "record_step")


def test_poincare_accumulator_has_record_trajectory():
    acc = PoincareAccumulator([ToroidalSection(0.0)])
    assert hasattr(acc, "record_trajectory")


def test_poincare_accumulator_has_crossing_array():
    acc = PoincareAccumulator([ToroidalSection(0.0)])
    assert hasattr(acc, "crossing_array")


def test_poincare_accumulator_record_trajectory_returns_crossings():
    traj = _make_simple_trajectory(n_turns=3)
    sec = ToroidalSection(0.0)
    acc = PoincareAccumulator([sec])
    acc.record_trajectory(traj)
    crossings = acc.crossing_array(0)
    # We should get roughly n_turns crossings
    assert crossings.ndim == 2
    assert crossings.shape[1] == 3
    assert len(crossings) >= 1


def test_poincare_accumulator_empty_crossings_shape():
    sec = ToroidalSection(0.0)
    acc = PoincareAccumulator([sec])
    out = acc.crossing_array(0)
    assert out.shape == (0, 3)


# ---------------------------------------------------------------------------
# Backward compat: PoincareMap is still PoincareAccumulator
# ---------------------------------------------------------------------------

def test_poincare_map_alias_is_accumulator():
    assert PoincareMap is PoincareAccumulator


def test_poincare_map_alias_works():
    pm = PoincareMap([ToroidalSection(0.0)])
    assert isinstance(pm, PoincareAccumulator)


# ---------------------------------------------------------------------------
# poincare_from_fieldlines returns PoincareAccumulator
# ---------------------------------------------------------------------------

def _simple_field(rzphi):
    """Trivial helical field: dR=0, dZ=0, dphi=1, with some poloidal winding."""
    R, Z, phi = rzphi
    return np.array([0.05 * np.cos(phi), 0.05 * np.sin(phi), 1.0])


class _FakeBackend:
    """Minimal backend that returns a pre-built trajectory."""
    def __init__(self, traj):
        self._traj = traj

    def trace(self, start_pt, t_max):
        return self._traj


def test_poincare_from_fieldlines_returns_accumulator():
    traj = _make_simple_trajectory(n_turns=2)
    backend = _FakeBackend(traj)
    start_pts = np.array([[1.0, 0.0, 0.0]])
    sections = [ToroidalSection(0.0)]
    result = poincare_from_fieldlines(
        field_func=_simple_field,
        start_pts=start_pts,
        sections=sections,
        t_max=10.0,
        backend=backend,
    )
    assert isinstance(result, PoincareAccumulator)


# ---------------------------------------------------------------------------
# rotational_transform_from_trajectory still works
# ---------------------------------------------------------------------------

def test_rotational_transform_from_trajectory_available():
    assert callable(rotational_transform_from_trajectory)


def test_rotational_transform_from_trajectory_runs():
    traj = _make_simple_trajectory(n_turns=5)
    iota = rotational_transform_from_trajectory(traj, axis_RZ=[1.0, 0.0])
    assert isinstance(iota, float)
    assert not np.isnan(iota)
