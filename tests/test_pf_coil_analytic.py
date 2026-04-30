"""Tests for CoilFieldAnalyticRectangularSection."""
import numpy as np
import pytest
from pyna.toroidal.coils import CoilFieldAnalyticRectangularSection, CoilFieldAnalyticCircular


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

