"""Tests for pyna.topo.invariant -- InvariantObject class hierarchy.

All tests use synthetic/analytic data only (no real HAO data files).
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.island_chain import IslandChainOrbit, ChainFixedPoint
from pyna.topo.invariant import (
    InvariantObject,
    PeriodicOrbit,
    InvariantTorus,
    InvariantManifold,
    StableManifold,
    UnstableManifold,
)
from pyna.topo.resonance import ResonanceNumber
from pyna.topo.dynamics import MCFPoincareMap, MCF_2D


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chain_fp(phi=0.0, R=1.5, Z=0.0, kind='X'):
    """Make a ChainFixedPoint with synthetic DPm."""
    if kind == 'X':
        # Hyperbolic: eigenvalues 3 and 1/3, Tr=3.333 > 2
        DPm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])
    else:
        # Elliptic: DPm = rotation by small angle, Tr < 2
        th = 0.4
        DPm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return ChainFixedPoint(phi=phi, R=R, Z=Z, DPm=DPm, DX_pol_accum=np.eye(2))


def _make_orbit(m=10, n=3, Np=2, kind='X'):
    """Make a minimal IslandChainOrbit with one section."""
    fp = _make_chain_fp(phi=0.0, kind=kind)
    return IslandChainOrbit(
        m=m, n=n, Np=Np,
        fixed_points=[fp],
        seed_phi=0.0,
        seed_RZ=(1.5, 0.0),
        section_phis=[0.0],
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_invariant_object_is_abstract():
    """InvariantObject cannot be instantiated directly."""
    with pytest.raises(TypeError):
        InvariantObject()


def test_resonance_number_in_periodic_orbit():
    """PeriodicOrbit.resonance returns correct ResonanceNumber."""
    orbit = _make_orbit(m=10, n=3)
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    r = po.resonance
    assert isinstance(r, ResonanceNumber)
    assert r.m == 10
    assert r.n_pol == 3
    assert str(r) == '10/3'


def test_periodic_orbit_from_island_chain_orbit():
    """Wrapping IslandChainOrbit preserves fixed_points and backward compat."""
    orbit = _make_orbit(m=10, n=3, kind='X')
    po = PeriodicOrbit.from_island_chain_orbit(orbit, label='test X orbit')
    assert po.orbit is orbit
    assert po.m == 10
    assert po.n == 3
    assert po.Np == 2
    assert len(po.fixed_points) == 1
    assert po.label == 'test X orbit'


def test_periodic_orbit_stability_x():
    """X-orbit has stability == 'X' and negative Greene residue."""
    orbit = _make_orbit(m=3, n=1, kind='X')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    assert po.stability == 'X'
    assert po.greene_residue < 0


def test_periodic_orbit_stability_o():
    """O-orbit has stability == 'O' and Greene residue in (0, 1)."""
    orbit = _make_orbit(m=3, n=1, kind='O')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    assert po.stability == 'O'
    assert 0 < po.greene_residue < 1


def test_periodic_orbit_section_cut():
    """section_cut returns Island objects with period_n == m."""
    from pyna.topo.island import Island
    orbit = _make_orbit(m=5, n=2, kind='O')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    islands = po.section_cut(0.0)
    assert len(islands) == 1
    assert isinstance(islands[0], Island)
    assert islands[0].period_n == 5
    # Back-reference to PeriodicOrbit
    assert islands[0].periodic_orbit is po


def test_periodic_orbit_diagnostics():
    """diagnostics() returns expected keys."""
    orbit = _make_orbit(m=10, n=3, kind='X')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    d = po.diagnostics()
    assert d['invariant_type'] == 'PeriodicOrbit'
    assert d['stability'] == 'X'
    assert 'greene_residue' in d
    assert d['resonance'] == '10/3'


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


def test_stable_manifold_init():
    """StableManifold can be constructed from a PeriodicOrbit."""
    orbit = _make_orbit(m=3, n=1, kind='X')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    mf = StableManifold(po)
    assert mf.branch == 'stable'
    assert mf.periodic_orbit is po
    assert mf.points is None  # not grown yet


def test_unstable_manifold_init():
    """UnstableManifold can be constructed from a PeriodicOrbit."""
    orbit = _make_orbit(m=3, n=1, kind='X')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    mf = UnstableManifold(po)
    assert mf.branch == 'unstable'
    assert mf.points is None


def test_manifold_section_cut_raises_before_grow():
    """InvariantManifold.section_cut raises before grow() is called."""
    orbit = _make_orbit(m=3, n=1, kind='X')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    mf = StableManifold(po)
    with pytest.raises(RuntimeError, match='grow'):
        mf.section_cut(0.0)


def test_manifold_invalid_branch():
    """InvariantManifold raises ValueError for invalid branch."""
    orbit = _make_orbit(m=3, n=1, kind='X')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    with pytest.raises(ValueError, match='branch'):
        # Directly call ABC init via StableManifold with wrong branch
        StableManifold.__bases__[0].__init__(
            StableManifold.__new__(StableManifold), po, 'diagonal'
        )


def test_mcf_poincare_map_phase_space():
    """MCFPoincareMap.phase_space == MCF_2D."""
    # Build a minimal dummy field cache
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


def test_periodic_orbit_repr():
    """PeriodicOrbit repr contains m, n, stability."""
    orbit = _make_orbit(m=10, n=3, kind='X')
    po = PeriodicOrbit.from_island_chain_orbit(orbit)
    r = repr(po)
    assert '10' in r and '3' in r and 'X' in r


def test_invariant_torus_repr():
    """InvariantTorus repr contains iota value."""
    torus = InvariantTorus(
        crossings={0.0: np.zeros((5, 2))},
        rotational_transform=0.333,
    )
    r = repr(torus)
    assert '0.333' in r
