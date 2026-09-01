"""Tests for CoilFieldAnalyticRectangularSection."""
from importlib.util import find_spec

import numpy as np
import pytest
from pyna.toroidal.coils import (
    BRBZ_induced_by_rectangular_winding_pack_gauss_legendre,
    BRBZ_induced_by_thick_finitelen_solenoid,
    CoilFieldAnalyticCircular,
    CoilFieldAnalyticRectangularSection,
)


# A representative PF coil: Rc=1.5m, Zc=0m, dR=0.05m, dZ=0.05m, 100 turns, 1A
RC, ZC, DR, DZ, TURNS = 1.5, 0.0, 0.05, 0.05, 100


def test_instantiation():
    coil = CoilFieldAnalyticRectangularSection(RC, ZC, DR, DZ, TURNS, current=1.0)
    assert coil.Rc == RC
    assert coil.Zc == ZC
    assert coil.dR == DR
    assert coil.dZ == DZ
    assert coil.turns == TURNS
    assert coil.current == 1.0


def test_B_at_finite():
    coil = CoilFieldAnalyticRectangularSection(RC, ZC, DR, DZ, TURNS, current=1.0)
    BR, BZ, Bphi = coil.B_at(1.5, 1.0, 0.0)
    assert np.all(np.isfinite(BR))
    assert np.all(np.isfinite(BZ))
    assert np.all(Bphi == 0.0)


def test_B_at_grid_shape():
    coil = CoilFieldAnalyticRectangularSection(RC, ZC, DR, DZ, TURNS, current=1.0)
    R = np.linspace(0.5, 2.5, 10)
    Z = np.linspace(-1.0, 1.0, 10)
    BR_grid, BZ_grid = coil.B_at_grid(R, Z)
    assert BR_grid.shape == (10, 10)
    assert BZ_grid.shape == (10, 10)


def test_far_field_vs_circular():
    """Thin rectangular section should approach single circular loop in far field."""
    # Very thin coil ~ single loop
    Rc = 1.0
    turns = 1
    I = 1.0
    eps = 1e-4  # very thin cross-section
    rect_coil = CoilFieldAnalyticRectangularSection(Rc, 0.0, eps, eps, turns, current=I)
    circ_coil = CoilFieldAnalyticCircular(Rc, center_xyz=(0.0, 0.0, 0.0), current=I)

    R_test, Z_test = 0.5, 2.5
    BR_rect, BZ_rect, _ = rect_coil.B_at(R_test, Z_test, 0.0)
    BR_circ, BZ_circ, _ = circ_coil.B_at(R_test, Z_test, 0.0)

    # Check relative agreement within 5%
    assert abs(BR_rect - BR_circ) / (abs(BR_circ) + 1e-20) < 0.05, (
        f"BR mismatch: rect={float(BR_rect):.4e}, circ={float(BR_circ):.4e}"
    )
    assert abs(BZ_rect - BZ_circ) / (abs(BZ_circ) + 1e-20) < 0.05, (
        f"BZ mismatch: rect={float(BZ_rect):.4e}, circ={float(BZ_circ):.4e}"
    )


def test_current_setter():
    coil = CoilFieldAnalyticRectangularSection(RC, ZC, DR, DZ, TURNS, current=1.0)
    coil.current = 2.0
    assert coil.current == 2.0


def test_divergence_free():
    coil = CoilFieldAnalyticRectangularSection(RC, ZC, DR, DZ, TURNS)
    assert coil.divergence_free() is True


def test_pf1_gl32_reproduces_fresh_winding_pack_witnesses():
    R = np.array([1.9, 1.9, 1.2, 2.5015625])
    Z = np.array([0.3, 0.0, 0.3, 0.3])
    BR, BZ = BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
        0.62866,
        0.25132,
        0.16078,
        0.45177,
        140,
        1000.0,
        R,
        Z,
        quadrature_order=32,
        backend="cpu",
    )
    np.testing.assert_allclose(
        BR,
        [
            2.3357270234772816e-4,
            -1.1449010827491548e-3,
            1.9092564263199366e-3,
            7.211637226194412e-5,
        ],
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        BZ,
        [
            -2.827509900151661e-3,
            -2.5811866873508447e-3,
            -1.316794069499984e-2,
            -1.1850660734820846e-3,
        ],
        rtol=2.0e-12,
    )


def test_historical_public_name_dispatches_to_gl_solver():
    geometry = (0.54827, 0.70905, 0.025435, 0.45177, 1000.0, 140)
    BR_compat, BZ_compat = BRBZ_induced_by_thick_finitelen_solenoid(
        *geometry,
        1.9,
        0.3,
        quadrature_order=16,
        backend="cpu",
    )
    BR_gl, BZ_gl = BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
        0.62866,
        0.25132,
        0.16078,
        0.45177,
        140,
        1000.0,
        1.9,
        0.3,
        quadrature_order=16,
        backend="cpu",
    )
    np.testing.assert_array_equal(BR_compat, BR_gl)
    np.testing.assert_array_equal(BZ_compat, BZ_gl)


@pytest.mark.skipif(find_spec("cupy") is None, reason="CuPy is unavailable")
def test_rectangular_winding_pack_cuda_matches_cpu_float64():
    R = np.array([1.2, 1.9, 2.5])
    Z = np.array([0.3, 0.0, 0.3])
    args = (0.62866, 0.25132, 0.16078, 0.45177, 140, 1000.0, R, Z)
    BR_cpu, BZ_cpu = BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
        *args, quadrature_order=32, backend="cpu"
    )
    BR_cuda, BZ_cuda = BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
        *args, quadrature_order=32, backend="cuda"
    )
    np.testing.assert_allclose(BR_cuda, BR_cpu, rtol=1.0e-12, atol=1.0e-15)
    np.testing.assert_allclose(BZ_cuda, BZ_cpu, rtol=1.0e-12, atol=1.0e-15)


@pytest.mark.skipif(find_spec("cupy") is None, reason="CuPy is unavailable")
def test_cuda_request_uses_finite_cpu_axis_limit():
    BR, BZ = BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
        0.62866,
        0.25132,
        0.16078,
        0.45177,
        140,
        1000.0,
        np.array([0.0, 1.9]),
        np.array([0.25132, 0.3]),
        backend="cuda",
    )
    assert BR[0] == 0.0
    assert np.all(np.isfinite(BR))
    assert np.all(np.isfinite(BZ))

