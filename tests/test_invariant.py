"""Tests for pyna.topo.invariant -- InvariantObject class hierarchy.

All tests use synthetic/analytic data only (no real HAO data files).
Updated to use the new class hierarchy:
  - FixedPoint (replaces ChainFixedPoint)
  - Cycle (replaces IslandChainOrbit)
  - Island/IslandChain from island.py
  - InvariantTorus from invariant.py
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.invariants import (
    FixedPoint,
    Cycle,
    MonodromyData,
    Stability,
)
from pyna.topo.invariant import (
    InvariantObject,
    InvariantTorus,
)
from pyna.topo.island import Island, IslandChain
from pyna.topo.resonance import ResonanceNumber
from pyna.topo.dynamics import MCFPoincareMap, MCF_2D


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fp(phi=0.0, R=1.5, Z=0.0, kind='X'):
    """Make a FixedPoint with synthetic DPm."""
    if kind == 'X':
        # Hyperbolic: eigenvalues 3 and 1/3, Tr=3.333 > 2
        DPm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])
    else:
        # Elliptic: DPm = rotation by small angle, Tr < 2
        th = 0.4
        DPm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return FixedPoint(phi=phi, R=R, Z=Z, DPm=DPm)


def _make_cycle(m=10, n=3, kind='X'):
    """Make a minimal Cycle with one section."""
    fp = _make_fp(phi=0.0, kind=kind)
    mono = MonodromyData(DPm=fp.DPm, eigenvalues=np.linalg.eigvals(fp.DPm))
    return Cycle(winding=(m, n), sections={0.0: [fp]}, monodromy=mono)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_invariant_object_is_abstract():
    """InvariantObject cannot be instantiated directly."""
    with pytest.raises(TypeError):
        InvariantObject()


def test_fixed_point_auto_kind():
    """FixedPoint auto-derives kind from DPm."""
    fp_x = _make_fp(kind='X')
    assert fp_x.kind == 'X'
    fp_o = _make_fp(kind='O')
    assert fp_o.kind == 'O'


def test_fixed_point_greene_residue():
    """FixedPoint computes Greene's residue correctly."""
    fp_x = _make_fp(kind='X')
    assert fp_x.greene_residue < 0  # hyperbolic
    fp_o = _make_fp(kind='O')
    assert 0 < fp_o.greene_residue < 1  # elliptic


def test_fixed_point_array_interface():
    """FixedPoint supports array-like indexing."""
    fp = _make_fp(R=1.5, Z=0.3)
    assert fp[0] == pytest.approx(1.5)
    assert fp[1] == pytest.approx(0.3)
    assert len(fp) == 2
    arr = np.asarray(fp)
    assert arr.shape == (2,)


def test_fixed_point_diagnostics():
    """FixedPoint.diagnostics() returns expected keys."""
    fp = _make_fp(kind='X')
    d = fp.diagnostics()
    assert d['invariant_type'] == 'FixedPoint'
    assert d['kind'] == 'X'
    assert 'greene_residue' in d


def test_cycle_stability():
    """Cycle stability classification works."""
    cycle_x = _make_cycle(m=3, n=1, kind='X')
    assert cycle_x.stability == Stability.HYPERBOLIC

    cycle_o = _make_cycle(m=3, n=1, kind='O')
    assert cycle_o.stability == Stability.ELLIPTIC


def test_cycle_section_cut():
    """Cycle.section_cut returns FixedPoints at section."""
    cycle = _make_cycle(m=5, n=2, kind='O')
    fps = cycle.section_cut(0.0)
    assert len(fps) == 1
    assert isinstance(fps[0], FixedPoint)


def test_cycle_diagnostics():
    """Cycle.diagnostics() returns expected keys."""
    cycle = _make_cycle(m=10, n=3, kind='X')
    d = cycle.diagnostics()
    assert d['invariant_type'] == 'Cycle'
    assert d['winding'] == (10, 3)


def test_island_chain_from_fixed_points():
    """IslandChain.from_fixed_points creates islands from O/X points."""
    o_pts = [np.array([1.5, 0.0]), np.array([1.6, 0.1])]
    x_pts = [np.array([1.55, 0.05])]
    chain = IslandChain.from_fixed_points(o_pts, x_pts, m=2, n=1, proximity_tol=0.2)
    assert chain.n_islands == 2
    assert chain.m == 2
    assert chain.n == 1


def test_invariant_torus_construction():
    """InvariantTorus can be constructed with a crossing dict."""
    phi0 = 0.0
    R_rand = np.random.uniform(1.4, 1.6, 50)
    Z_rand = np.random.uniform(-0.1, 0.1, 50)
    crossings = np.column_stack([R_rand, Z_rand])
    torus = InvariantTorus(
        crossings={phi0: crossings},
        rotational_transform=0.3,
        label='test KAM',
    )
    assert torus.rotational_transform == pytest.approx(0.3)
    assert torus.is_resonant is False
    assert torus.safety_factor == pytest.approx(1.0 / 0.3)
    pts = torus.section_cut(phi0)
    assert len(pts) == 1
    assert pts[0].shape == (50, 2)


def test_invariant_torus_diagnostics():
    """InvariantTorus.diagnostics has expected structure."""
    torus = InvariantTorus(
        crossings={0.0: np.zeros((10, 2))},
        rotational_transform=0.25,
    )
    d = torus.diagnostics()
    assert d['invariant_type'] == 'InvariantTorus'
    assert d['is_resonant'] is False
    assert 0.0 in d['crossing_counts']


def test_mcf_poincare_map_phase_space():
    """MCFPoincareMap.phase_space == MCF_2D."""
    NR, NZ, NPhi = 5, 5, 4
    fc = {
        'R_grid': np.linspace(0.5, 2.0, NR),
        'Z_grid': np.linspace(-1.0, 1.0, NZ),
        'Phi_grid': np.linspace(0, 2 * np.pi, NPhi, endpoint=False),
        'BR':   np.zeros((NR, NZ, NPhi)),
        'BPhi': np.ones((NR, NZ, NPhi)) * 1.0,
        'BZ':   np.zeros((NR, NZ, NPhi)),
    }
    pm = MCFPoincareMap(fc, Np=2, phi_section=0.0)
    assert pm.phase_space is MCF_2D
    assert pm.Np == 2
    assert pm.phi_section == pytest.approx(0.0)
    assert pm.n_turns == 1


def test_invariant_torus_repr():
    """InvariantTorus repr contains iota value."""
    torus = InvariantTorus(
        crossings={0.0: np.zeros((5, 2))},
        rotational_transform=0.333,
    )
    r = repr(torus)
    assert '0.333' in r
