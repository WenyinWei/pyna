"""Tests for GeneralPoincareMap and general Section integration.

Uses purely synthetic systems (StandardMap, HyperplaneSection) -- no real
field data or external files required.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.dynamics import StandardMap, GeneralPoincareMap, PoincareMap
from pyna.topo.section import HyperplaneSection, ParametricSection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_standard_map(K: float = 0.5) -> StandardMap:
    return StandardMap(K=K)


def _theta_zero_section():
    """Hyperplane theta=0 in (theta, I) phase space."""
    return HyperplaneSection(
        normal_vec=np.array([1.0, 0.0]),
        offset=0.0,
        phase_dim=2,
    )


# ── GeneralPoincareMap basics ─────────────────────────────────────────────────

def test_general_poincare_map_phase_space():
    """GeneralPoincareMap.phase_space has dim=1 (section of 2D system)."""
    sm = _make_standard_map()
    sec = _theta_zero_section()
    pm = GeneralPoincareMap(sm, sec, dt=0.1, t_max=50.0)
    assert pm.phase_space.dim == 1


def test_general_poincare_map_repr():
    """repr includes flow and section class names."""
    sm = _make_standard_map()
    sec = _theta_zero_section()
    pm = GeneralPoincareMap(sm, sec)
    r = repr(pm)
    assert 'StandardMap' in r
    assert 'HyperplaneSection' in r


def test_general_poincare_map_has_flow_and_section():
    """flow and section properties return the objects."""
    sm = _make_standard_map()
    sec = _theta_zero_section()
    pm = GeneralPoincareMap(sm, sec)
    assert pm.flow is sm
    assert pm.section is sec


def test_mcf_poincare_map_is_discrete_map():
    """MCFPoincareMap is a DiscreteMap."""
    from pyna.topo.dynamics import MCFPoincareMap, DiscreteMap
    NR, NZ, NPhi = 5, 5, 4
    fc = {
        'R_grid': np.linspace(0.5, 2.0, NR),
        'Z_grid': np.linspace(-1.0, 1.0, NZ),
        'Phi_grid': np.linspace(0, 2 * np.pi, NPhi, endpoint=False),
        'BR':   np.zeros((NR, NZ, NPhi)),
        'BPhi': np.ones((NR, NZ, NPhi)),
        'BZ':   np.zeros((NR, NZ, NPhi)),
    }
    pm = MCFPoincareMap(fc, Np=2, phi_section=0.0)
    assert isinstance(pm, DiscreteMap)


# ── Section.f / contains / normal ─────────────────────────────────────────────

def test_hyperplane_section_f():
    """HyperplaneSection.f is a·x - c."""
    sec = HyperplaneSection(
        normal_vec=np.array([1.0, 0.0]),
        offset=1.0,
        phase_dim=2,
    )
    assert sec.f(np.array([1.0, 0.5])) == pytest.approx(0.0)
    assert sec.f(np.array([2.0, 0.5])) == pytest.approx(1.0)
    assert sec.f(np.array([0.0, 0.5])) == pytest.approx(-1.0)


def test_hyperplane_section_contains():
    sec = HyperplaneSection(
        normal_vec=np.array([0.0, 1.0]),
        offset=0.0,
        phase_dim=2,
    )
    assert sec.contains(np.array([5.0, 0.0]))
    assert not sec.contains(np.array([5.0, 1.0]))


def test_hyperplane_section_normal():
    n = np.array([3.0, 4.0])
    sec = HyperplaneSection(normal_vec=n, offset=0.0, phase_dim=2)
    assert np.allclose(sec.normal(np.zeros(2)), n)


def test_parametric_section_f_and_contains():
    """ParametricSection with circular section."""
    # Section: x^2 + y^2 = 1
    def f(x): return x[0]**2 + x[1]**2 - 1.0
    def grad(x): return np.array([2*x[0], 2*x[1]])

    sec = ParametricSection(f_func=f, grad_func=grad, phase_dim=2)
    assert sec.contains(np.array([1.0, 0.0]), tol=1e-10)
    assert not sec.contains(np.array([0.5, 0.0]))
    assert sec.f(np.array([0.5, 0.5])) == pytest.approx(0.5**2 + 0.5**2 - 1.0)


# ── GeneralPoincareMap._detect crossing ───────────────────────────────────────

def test_general_poincare_map_detects_sign_change():
    """_detect fires when f(y) changes sign."""
    sm = _make_standard_map()
    # Section: x[0] = 1.0 (theta=1)
    sec = HyperplaneSection(np.array([1.0, 0.0]), 1.0, phase_dim=2)
    pm = GeneralPoincareMap(sm, sec, dt=0.1, t_max=20.0)

    # Directly test _detect
    y_before = np.array([0.8, 0.5])   # f = 0.8 - 1.0 = -0.2 < 0
    y_after  = np.array([1.2, 0.5])   # f = 1.2 - 1.0 = +0.2 > 0
    hit = pm._detect(y_before, y_after, 0.0, 0.1)
    assert hit is not None
    assert len(hit) == 2
    # Interpolated crossing should be near (1.0, 0.5)
    assert abs(hit[0] - 1.0) < 0.1


def test_general_poincare_map_no_crossing_same_sign():
    """_detect returns None when both points are on the same side."""
    sm = _make_standard_map()
    sec = HyperplaneSection(np.array([1.0, 0.0]), 1.0, phase_dim=2)
    pm = GeneralPoincareMap(sm, sec, dt=0.1, t_max=20.0)

    y_prev = np.array([0.5, 0.3])
    y_curr = np.array([0.7, 0.3])
    assert pm._detect(y_prev, y_curr, 0.0, 0.1) is None


# ── Tube.section_cut with general section ─────────────────────────────────────

def test_tube_section_cut_general_section_with_orbit():
    """Tube.section_cut(HyperplaneSection) returns Islands via orbit scan."""
    from pyna.topo.invariants import Cycle, FixedPoint
    from pyna.topo.tube import Tube

    th = 0.4
    DPm_O = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    fp = FixedPoint(phi=0.0, R=1.5, Z=0.0, DPm=DPm_O, kind='O')
    cycle = Cycle(winding=(3, 1), sections={0.0: [fp]},
                  monodromy=fp.monodromy, ambient_dim=2)

    n_pts = 100
    angle = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    R_orbit = 1.5 + 0.05 * np.cos(angle)
    Z_orbit = 0.05 * np.sin(angle)
    phi_orbit = angle

    tube = Tube(
        o_cycle=cycle, x_cycles=[], label='test-tube',
        _orbit_R=R_orbit, _orbit_Z=Z_orbit,
        _orbit_phi=phi_orbit, _orbit_alive=np.ones(n_pts, dtype=bool),
    )

    sec = HyperplaneSection(
        normal_vec=np.array([0.0, 1.0]),
        offset=0.0,
        phase_dim=2,
    )
    islands = tube.section_cut(sec)
    assert isinstance(islands, list)
    for isl in islands:
        # O_point is a FixedPoint; verify it has R, Z coordinates
        assert hasattr(isl.O_point, 'R') and hasattr(isl.O_point, 'Z')
        assert isl.tube is tube


def test_tube_section_cut_toroidal_still_works():
    """Tube.section_cut with ToroidalSection still uses the fast path."""
    from pyna.topo.invariants import Cycle, FixedPoint
    from pyna.topo.tube import Tube
    from pyna.topo.section import ToroidalSection
    from pyna.topo.island import Island

    th = 0.4
    DPm_O = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    fp = FixedPoint(phi=0.0, R=1.5, Z=0.0, DPm=DPm_O, kind='O')
    cycle = Cycle(winding=(3, 1), sections={0.0: [fp]},
                  monodromy=fp.monodromy, ambient_dim=2)
    tube = Tube(o_cycle=cycle, x_cycles=[])
    sec = ToroidalSection(0.0)
    islands = tube.section_cut(sec)
    assert len(islands) == 1
    assert isinstance(islands[0], Island)


# ── GeneralPoincareMap.trajectory ─────────────────────────────────────────────

def test_general_poincare_map_trajectory_returns_array():
    """trajectory() returns a 2D array; use a simple harmonic oscillator flow."""
    from pyna.topo.dynamics import ContinuousFlow

    # Simple harmonic oscillator: dx/dt = y, dy/dt = -x  (period 2pi)
    class HarmonicOscillator(ContinuousFlow):
        @property
        def phase_space(self):
            from pyna.topo.dynamics import PhaseSpace
            return PhaseSpace(dim=2, coordinate_names=('x', 'y'), symplectic=True)
        def vector_field(self, x, t=0.0):
            return np.array([x[1], -x[0]])

    flow = HarmonicOscillator()
    # Section: x=0 plane
    sec = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
    pm = GeneralPoincareMap(flow, sec, dt=0.05, t_max=20.0, direction=1)
    # Start at (0.5, 0) — on the section; integrate and find next crossing
    result = pm.trajectory(np.array([0.5, 0.0]), n_crossings=3)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 2  # (R, Z) or (x, y) coords


def test_general_poincare_map_exports_from_topo():
    """GeneralPoincareMap is exported from pyna.topo."""
    from pyna.topo import GeneralPoincareMap as GPM
    assert GPM is not None
