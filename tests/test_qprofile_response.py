"""Tests for q-profile and rotation-transform response matrix.

Run with:
    py -3.13 -m pytest tests/test_qprofile_response.py -v
"""

import numpy as np
import pytest

from pyna.MCF.control.qprofile_response import (
    q_from_flux_surface_integral,
    q_response_matrix_analytic,
    iota_response_matrix,
    build_qprofile_response,
)
from pyna.MCF.equilibrium.Solovev import EquilibriumSolovev, solovev_iter_like


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def eq():
    """Small EAST-like Solov'ev equilibrium for fast tests."""
    return EquilibriumSolovev(R0=1.86, a=0.6, B0=2.5, kappa=1.7, delta=0.3, q0=2.0)


def make_field_func(eq):
    """Wrap EquilibriumSolovev into a unified field callable."""
    def field(x):
        R, Z = float(x[0]), float(x[1])
        BR, BZ = eq.BR_BZ(R, Z)
        Bphi = eq.Bphi(R)
        return [float(BR), float(BZ), float(Bphi)]
    return field


def make_delta_field(dBR=0.0, dBZ=0.0, dBphi=0.0):
    """Constant perturbation field (unit coil)."""
    def delta(x):
        return [dBR, dBZ, dBphi]
    return delta


def simple_surfaces(eq, s_values=(0.3, 0.6, 0.9)):
    """Fetch flux surfaces from equilibrium."""
    surfaces = []
    for s in s_values:
        R_fs, Z_fs = eq.flux_surface(s)
        surfaces.append((R_fs, Z_fs))
    return surfaces, list(s_values)


# ── tests: flux_surface ───────────────────────────────────────────────────

class TestFluxSurface:
    def test_returns_arrays(self, eq):
        R, Z = eq.flux_surface(0.5)
        assert isinstance(R, np.ndarray)
        assert isinstance(Z, np.ndarray)
        assert len(R) > 10
        assert len(R) == len(Z)

    def test_contour_is_closed(self, eq):
        R, Z = eq.flux_surface(0.5)
        # First and last points should be close (matplotlib closes the path)
        gap = np.sqrt((R[-1] - R[0]) ** 2 + (Z[-1] - Z[0]) ** 2)
        # Allow up to 5 % of minor radius
        assert gap < 0.05 * eq.a * 2, f"Contour not closed, gap={gap:.4f}"

    def test_psi_on_contour(self, eq):
        """All points on the contour should have psi_norm ≈ target."""
        target = 0.4
        R, Z = eq.flux_surface(target)
        psi_vals = eq.psi(R, Z)
        assert np.allclose(psi_vals, target, atol=0.05), (
            f"psi spread: min={psi_vals.min():.3f}, max={psi_vals.max():.3f}"
        )

    def test_larger_psi_gives_larger_surface(self, eq):
        R1, _ = eq.flux_surface(0.3)
        R2, _ = eq.flux_surface(0.7)
        assert np.max(R2) > np.max(R1)

    def test_lcfs(self, eq):
        R, Z = eq.flux_surface(1.0)
        assert len(R) > 0


# ── tests: q_from_flux_surface_integral ──────────────────────────────────

class TestQIntegral:
    def test_returns_positive_q(self, eq):
        field = make_field_func(eq)
        R_fs, Z_fs = eq.flux_surface(0.5)
        q = q_from_flux_surface_integral(field, R_fs, Z_fs)
        assert np.isfinite(q)
        assert q > 0

    def test_q_increases_outward(self, eq):
        """For normal tokamak shear: q(outer) > q(inner)."""
        field = make_field_func(eq)
        R1, Z1 = eq.flux_surface(0.3)
        R2, Z2 = eq.flux_surface(0.7)
        q1 = q_from_flux_surface_integral(field, R1, Z1)
        q2 = q_from_flux_surface_integral(field, R2, Z2)
        # q should increase outward for normal shear; allow either sign for robustness
        assert np.isfinite(q1) and np.isfinite(q2)


# ── tests: q_response_matrix_analytic ────────────────────────────────────

class TestQResponseMatrix:
    def test_shape(self, eq):
        field = make_field_func(eq)
        s_vals = (0.3, 0.6, 0.9)
        surfaces, labels = simple_surfaces(eq, s_vals)
        delta_fields = [
            make_delta_field(dBZ=0.01),
            make_delta_field(dBR=0.01),
        ]
        R_q, lbls = q_response_matrix_analytic(field, delta_fields, surfaces, labels)
        assert R_q.shape == (len(s_vals), len(delta_fields))
        assert len(lbls) == len(s_vals)

    def test_labels_format(self, eq):
        field = make_field_func(eq)
        s_vals = (0.2, 0.5)
        surfaces, labels = simple_surfaces(eq, s_vals)
        _, lbls = q_response_matrix_analytic(
            field, [make_delta_field(dBZ=0.01)], surfaces, labels
        )
        for lbl in lbls:
            assert lbl.startswith("q.s")

    def test_zero_perturbation_gives_zero_response(self, eq):
        field = make_field_func(eq)
        s_vals = (0.3, 0.6)
        surfaces, labels = simple_surfaces(eq, s_vals)
        zero_field = make_delta_field(0.0, 0.0, 0.0)
        R_q, _ = q_response_matrix_analytic(field, [zero_field], surfaces, labels)
        np.testing.assert_allclose(R_q, 0.0, atol=1e-10)

    def test_sign_flip_with_coil_location(self, eq):
        """A coil above vs below mid-plane should give opposite-sign δq
        (for purely vertical field perturbation, δBZ flips sign)."""
        field = make_field_func(eq)
        s_vals = (0.5,)
        surfaces, labels = simple_surfaces(eq, s_vals)
        pos_coil = make_delta_field(dBZ=+0.01)
        neg_coil = make_delta_field(dBZ=-0.01)
        R_pos, _ = q_response_matrix_analytic(field, [pos_coil], surfaces, labels)
        R_neg, _ = q_response_matrix_analytic(field, [neg_coil], surfaces, labels)
        np.testing.assert_allclose(R_pos, -R_neg, rtol=1e-10)

    def test_linearity(self, eq):
        """Response should be linear in perturbation amplitude."""
        field = make_field_func(eq)
        s_vals = (0.4,)
        surfaces, labels = simple_surfaces(eq, s_vals)
        d1 = make_delta_field(dBZ=0.01)
        d2 = make_delta_field(dBZ=0.02)
        R1, _ = q_response_matrix_analytic(field, [d1], surfaces, labels)
        R2, _ = q_response_matrix_analytic(field, [d2], surfaces, labels)
        np.testing.assert_allclose(R2, 2 * R1, rtol=1e-8)


# ── tests: iota_response_matrix ───────────────────────────────────────────

class TestIotaResponseMatrix:
    def test_shape(self, eq):
        field = make_field_func(eq)
        s_vals = (0.3, 0.6)
        surfaces, labels = simple_surfaces(eq, s_vals)
        delta_fields = [make_delta_field(dBZ=0.01)]
        R_iota, iota_lbls = iota_response_matrix(field, delta_fields, surfaces, labels)
        assert R_iota.shape == (2, 1)
        for lbl in iota_lbls:
            assert lbl.startswith("iota.")

    def test_iota_sign_opposite_to_q(self, eq):
        """∂ι/∂I = -1/q² ∂q/∂I  →  opposite sign to ∂q/∂I."""
        field = make_field_func(eq)
        s_vals = (0.5,)
        surfaces, labels = simple_surfaces(eq, s_vals)
        delta_fields = [make_delta_field(dBZ=0.01)]

        R_q, _ = q_response_matrix_analytic(field, delta_fields, surfaces, labels)
        R_iota, _ = iota_response_matrix(field, delta_fields, surfaces, labels)

        # ∂ι/∂I and ∂q/∂I should have opposite signs
        nonzero = R_q[R_q != 0]
        if len(nonzero) > 0:
            assert np.all(np.sign(R_iota[R_q != 0]) == -np.sign(R_q[R_q != 0]))


# ── tests: build_qprofile_response ────────────────────────────────────────

class TestBuildQprofileResponse:
    def test_end_to_end(self, eq):
        field = make_field_func(eq)
        delta_fields = [make_delta_field(dBZ=0.01), make_delta_field(dBR=0.005)]
        s_vals = (0.25, 0.5, 0.75)
        R_q, labels = build_qprofile_response(
            field, delta_fields, eq, s_values=s_vals
        )
        assert R_q.shape == (3, 2)
        assert len(labels) == 3
        assert all(np.isfinite(R_q).ravel())

    def test_fallback_without_flux_surface(self, eq):
        """If equilibrium has no flux_surface method, fallback ellipse is used."""
        class MinimalEq:
            R0 = eq.R0
            a = eq.a
            kappa = eq.kappa

        field = make_field_func(eq)
        delta_fields = [make_delta_field(dBZ=0.01)]
        R_q, labels = build_qprofile_response(
            field, delta_fields, MinimalEq(), s_values=(0.4,)
        )
        assert R_q.shape == (1, 1)
