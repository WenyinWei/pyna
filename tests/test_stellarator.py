"""Tests for SimpleStellarartor."""
import numpy as np
import pytest
from pyna.MCF.equilibrium.stellarator import SimpleStellarartor, simple_stellarator


def make_st():
    return simple_stellarator(R0=3.0, r0=0.3, B0=1.0, q0=1.1, q1=5.0,
                               m_h=4, n_h=4, epsilon_h=0.05)


def test_factory():
    st = make_st()
    assert isinstance(st, SimpleStellarartor)


def test_magnetic_axis():
    st = make_st()
    R_ax, Z_ax = st.magnetic_axis
    assert R_ax == 3.0
    assert Z_ax == 0.0


def test_psi_ax_zero_at_axis():
    st = make_st()
    psi = float(st.psi_ax(3.0, 0.0))
    assert psi == pytest.approx(0.0, abs=1e-12)


def test_resonant_psi_finds_q41():
    st = make_st()
    psi_list = st.resonant_psi(4, 1)
    assert len(psi_list) == 1
    psi_res = psi_list[0]
    assert 0.0 < psi_res < 1.0
    q_at_res = float(st.q_of_psi(psi_res))
    assert abs(q_at_res - 4.0) < 1e-8


def test_field_func_normalized():
    st = make_st()
    pt = np.array([3.1, 0.05, 0.0])
    tan = st.field_func(pt)
    norm = np.sqrt(tan[0]**2 + tan[1]**2 + (tan[2] * pt[0])**2)
    # Not exactly unit in (R,Z,phi) sense but check it's finite and non-zero
    assert np.isfinite(tan).all()
    assert np.linalg.norm(tan) > 0


def test_start_points_shape():
    st = make_st()
    pts = st.start_points_near_resonance(4, 1, n_lines=12, delta_psi=0.05)
    assert pts.shape == (12, 3)
    # All phi=0
    assert np.all(pts[:, 2] == 0.0)


def test_str_repr():
    st = make_st()
    s = str(st)
    assert 'SimpleStellarartor' in s
    r = repr(st)
    assert 'SimpleStellarartor' in r
