"""Tests for pyna.topo.perturbation â€?DPm change under field perturbation.

Test strategy
-------------
We use analytically tractable model fields where the monodromy matrix and its
perturbation can be computed or bounded analytically.

Model: rotation field
    dR/dÏ† = âˆ?Z âˆ?Z0),   dZ/dÏ† = (R âˆ?R0)
This gives circular orbits around (R0, Z0).  The monodromy matrix for a
rotation through angle Î¸ = 2Ï€Â·m is M = R(Î¸) (the rotation matrix), so
    DPm = I  (after an integer number of full turns).
Because DPm âˆ?I = 0 this is an O-point (elliptic), which we use only for
the Î´X_cyc test (hyperbolic eigenvalue test requires a different setup).

Model: hyperbolic field (saddle)
    dR/dÏ† = Î»Â·R,   dZ/dÏ† = âˆ’Î»Â·Z   (Î» > 0)
The flow map is:  R(Ï†) = R0Â·e^{Î»Ï†},  Z(Ï†) = Z0Â·e^{âˆ’Î»Ï†}.
The monodromy for one turn (Ï† = 0 â†?2Ï€):
    M = diag(e^{2Ï€Î»}, e^{âˆ?Ï€Î»})
which has eigenvalues e^{Â±2Ï€Î»} (hyperbolic when Î» â‰?0).

The variational equation for M is dM/dÏ† = AÂ·M with A = diag(Î», âˆ’Î?.
For a perturbation ÎµÂ·A_pert:
    Î´DPm = integral_0^{2Ï€} DX_pol(2Ï€, Ï†)Â·(ÎµÂ·A_pert)Â·DX_pol(Ï†, 0) dÏ†
which we can compute analytically for constant A_pert.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.perturbation import (
    DPm_finite_difference,
    DPm_shift_under_field_perturbation,
    eigenvalue_perturbation,
)
from pyna.topo.variational import PoincareMapVariationalEquations


# ---------------------------------------------------------------------------
# Helpers / model fields
# ---------------------------------------------------------------------------

_R0, _Z0 = 1.8, 0.0
_LAMBDA = 0.05   # saddle strength (small so numerics are easy)


def _rotation_field(R, Z, phi):
    """Circular orbit around (_R0, _Z0). One-turn DPm = I (O-point)."""
    return np.array([-(Z - _Z0), R - _R0])


def _saddle_field(R, Z, phi):
    """Saddle orbit through the origin.

    dR/dÏ† = Î»Â·R,  dZ/dÏ† = âˆ’Î»Â·Z  (X-point at origin, DPm = diag(e^{Â±2Ï€Î»})).
    """
    return np.array([_LAMBDA * R, -_LAMBDA * Z])


def _saddle_DPm_exact(lam: float = _LAMBDA, m: int = 1) -> np.ndarray:
    """Exact monodromy matrix for the saddle field after m turns."""
    return np.diag([np.exp(2 * np.pi * m * lam), np.exp(-2 * np.pi * m * lam)])


def _perturb_field_factory(base_func, eps: float, direction: np.ndarray):
    """Return a field that is base + eps * constant-direction perturbation."""
    d = np.asarray(direction, dtype=float)

    def pert(R, Z, phi):
        return np.asarray(base_func(R, Z, phi), dtype=float) + eps * d

    return pert


# ---------------------------------------------------------------------------
# CycleVariationalData mock (minimal, matching the real dataclass interface)
# ---------------------------------------------------------------------------

class _MockCycleData:
    """Minimal CycleVariationalData for tests (no monodromy.py dependency)."""

    def __init__(self, x0: np.ndarray, DPm: np.ndarray):
        # trajectory: first row = x0; we only need row 0
        self.trajectory = np.array([x0])
        self.DPm = DPm


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestEigenvaluePerturbation:
    """eigenvalue_perturbation: first-order formula."""

    def test_identity_perturbation_zero_shift(self):
        """Î´DPm = 0 â†?Î´Î» = 0."""
        DPm = _saddle_DPm_exact()
        result = eigenvalue_perturbation(DPm, np.zeros((2, 2)))
        assert np.allclose(result["delta_eigenvalues"], 0, atol=1e-12)

    def test_diagonal_perturbation_known_shift(self):
        """For diagonal DPm and diagonal Î´DPm the result is exact."""
        lam = _LAMBDA
        DPm = _saddle_DPm_exact(lam)
        # Î´DPm = diag(a, b) â†?Î´Î»_u = a, Î´Î»_s = b
        a, b = 0.01, -0.01
        delta_DPm = np.diag([a, b])
        result = eigenvalue_perturbation(DPm, delta_DPm)
        # Hyperbolic eigenvalue is e^{2Ï€Î»} (> 1), its perturbation is a
        assert abs(result["delta_lambda_u"] - a) < 1e-10

    def test_returns_new_eigenvalue(self):
        DPm = _saddle_DPm_exact()
        delta_DPm = np.diag([0.001, -0.001])
        result = eigenvalue_perturbation(DPm, delta_DPm)
        expected = result["lambda_u"] + result["delta_lambda_u"]
        assert abs(result["new_lambda_u"] - expected) < 1e-14

    def test_area_preserving_base(self):
        """DPm from saddle should have det â‰?1."""
        DPm = _saddle_DPm_exact()
        assert abs(np.linalg.det(DPm) - 1.0) < 1e-12


class TestDPmFiniteDifference:
    """DPm_finite_difference: Python fallback path (no field cache)."""

    def test_no_perturbation_zero_delta(self):
        """Perturbing by 0 should give Î´DPm â‰?0."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        DPm_base, DPm_pert, delta_DPm, _ = DPm_finite_difference(
            x0,
            field_func_base=_saddle_field,
            field_func_pert=_saddle_field,   # same field
            phi_span=phi_span,
            fd_eps_state=1e-5,
        )
        assert np.allclose(delta_DPm, 0, atol=1e-8)

    def test_saddle_base_matches_exact(self):
        """DPm_base for saddle field should match analytic diag(e^{Â±2Ï€Î»})."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        DPm_base, _, _, _ = DPm_finite_difference(
            x0,
            field_func_base=_saddle_field,
            field_func_pert=_saddle_field,
            phi_span=phi_span,
            fd_eps_state=1e-5,
        )
        expected = _saddle_DPm_exact()
        assert np.allclose(DPm_base, expected, atol=1e-5), (
            f"DPm_base={DPm_base} vs expected={expected}"
        )

    def test_small_perturbation_linear_scaling(self):
        """Î´DPm should scale linearly with perturbation amplitude Îµ."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        direction = np.array([1.0, 0.0])

        results = []
        for eps in [1e-3, 2e-3]:
            pert_f = _perturb_field_factory(_saddle_field, eps, direction)
            _, _, dDPm, _ = DPm_finite_difference(
                x0, _saddle_field, pert_f, phi_span, fd_eps_state=1e-5,
            )
            results.append(dDPm / eps)

        # dDPm/eps should be the same to within numerical error
        assert np.allclose(results[0], results[1], rtol=0.05), (
            f"Î´DPm/Îµ not constant: {results[0]} vs {results[1]}"
        )

    def test_det_pert_near_one(self):
        """DPm_pert should still be area-preserving (det â‰?1)."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        eps = 1e-3
        pert_f = _perturb_field_factory(_saddle_field, eps, np.array([0.0, 1.0]))
        _, DPm_pert, _, _ = DPm_finite_difference(
            x0, _saddle_field, pert_f, phi_span, fd_eps_state=1e-5,
        )
        # Area-preserving: det â‰?1 + O(ÎµÂ²) â‰?1 for small Îµ
        assert abs(np.linalg.det(DPm_pert) - 1.0) < 0.01


class TestDPmShiftUnderFieldPerturbation:
    """dpm_shift_under_field_perturbation: full pipeline test."""

    def _make_cycle_data(self, x0, field_func, phi_span):
        veq = PoincareMapVariationalEquations(field_func, fd_eps=1e-5)
        _, DPm = veq.tangent_map(x0, phi_span, order=1)
        return _MockCycleData(x0, DPm)

    def test_zero_perturbation(self):
        """Îµ = 0: all shifts should be zero."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        cycle_data = self._make_cycle_data(x0, _saddle_field, phi_span)

        result = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=lambda R, Z, phi: np.zeros(2),
            field_func=_saddle_field,
            phi_start=0.0,
            island_period=1,
            fd_eps=1e-5,
        )
        assert np.allclose(result["delta_DPm"], 0, atol=1e-8)
        assert np.allclose(result["delta_X_cyc"], 0, atol=1e-6)
        assert abs(result["delta_lambda_u"]) < 1e-8

    def test_output_keys_present(self):
        """Check all expected keys are in the output dict."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        cycle_data = self._make_cycle_data(x0, _saddle_field, phi_span)
        result = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=lambda R, Z, phi: np.zeros(2),
            field_func=_saddle_field,
        )
        for key in ["delta_X_cyc", "delta_DPm", "DPm_base", "DPm_pert",
                    "delta_lambda_u", "new_lambda_u", "lambda_u",
                    "eigenvalue_analysis", "used_cyna"]:
            assert key in result, f"Missing key: {key}"

    def test_DPm_base_matches_cycle_data(self):
        """DPm_base returned should match the cycle_data.DPm."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        cycle_data = self._make_cycle_data(x0, _saddle_field, phi_span)
        result = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=lambda R, Z, phi: np.zeros(2),
            field_func=_saddle_field,
            fd_eps=1e-5,
        )
        assert np.allclose(result["DPm_base"], cycle_data.DPm, atol=1e-5)

    def test_eigenvalue_consistency(self):
        """new_lambda_u should equal lambda_u + delta_lambda_u."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        cycle_data = self._make_cycle_data(x0, _saddle_field, phi_span)
        eps = 2e-3
        pert_f = _perturb_field_factory(_saddle_field, eps, np.array([1.0, 0.0]))
        result = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=None,
            field_func=_saddle_field,
            field_func_pert=pert_f,
            fd_eps=1e-5,
        )
        expected = result["lambda_u"] + result["delta_lambda_u"]
        assert abs(result["new_lambda_u"] - expected) < 1e-14

    def test_fd_vs_explicit_pert_consistency(self):
        """delta_B_func and field_func_pert paths should give same Î´DPm."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        cycle_data = self._make_cycle_data(x0, _saddle_field, phi_span)
        eps = 1e-3
        direction = np.array([0.5, -0.3])

        def delta_B(R, Z, phi):
            return eps * direction

        def pert_f(R, Z, phi):
            return np.asarray(_saddle_field(R, Z, phi), float) + eps * direction

        result_db = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=delta_B,
            field_func=_saddle_field,
            fd_eps=1e-5,
        )
        result_pert = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=None,
            field_func=_saddle_field,
            field_func_pert=pert_f,
            fd_eps=1e-5,
        )
        assert np.allclose(result_db["delta_DPm"], result_pert["delta_DPm"],
                           atol=1e-8), (
            f"delta_B_func path: {result_db['delta_DPm']}\n"
            f"field_func_pert path: {result_pert['delta_DPm']}"
        )

    def test_lambda_u_is_hyperbolic(self):
        """lambda_u should be the larger-magnitude eigenvalue (|Î»| > 1)."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        cycle_data = self._make_cycle_data(x0, _saddle_field, phi_span)
        result = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=lambda R, Z, phi: np.zeros(2),
            field_func=_saddle_field,
        )
        assert abs(result["lambda_u"]) > 1.0, (
            f"|lambda_u| = {abs(result['lambda_u']):.4f} should be > 1"
        )

    def test_no_perturbation_no_cyna_flag(self):
        """used_cyna should be False when no field cache is supplied."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        cycle_data = self._make_cycle_data(x0, _saddle_field, phi_span)
        result = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=lambda R, Z, phi: np.zeros(2),
            field_func=_saddle_field,
        )
        assert result["used_cyna"] is False


class TestDeltaXCycPeriodicity:
    """Î´X_cyc self-consistency: check the periodicity condition."""

    def test_delta_X_cyc_satisfies_periodicity(self):
        """(DPm - I)Â·Î´X_cyc â‰?-âˆ®Î´f dÏ† for a simple perturbation."""
        x0 = np.array([1.0, 0.0])
        phi_span = (0.0, 2 * np.pi)
        eps = 5e-3
        direction = np.array([1.0, 0.0])

        veq = PoincareMapVariationalEquations(_saddle_field, fd_eps=1e-5)
        _, DPm = veq.tangent_map(x0, phi_span)

        def delta_f(R, Z, phi):
            return eps * direction

        # âˆ?Î´f dÏ† â‰?2Ï€ * eps * direction  (constant perturbation)
        integral_df_exact = 2 * np.pi * eps * direction

        cycle_data = _MockCycleData(x0, DPm)

        result = DPm_shift_under_field_perturbation(
            cycle_data=cycle_data,
            delta_B_func=delta_f,
            field_func=_saddle_field,
            fd_eps=1e-5,
        )
        dX = result["delta_X_cyc"]
        lhs = (DPm - np.eye(2)) @ dX
        # Should equal -âˆ®Î´f dÏ†
        assert np.allclose(lhs, -integral_df_exact, atol=1e-4), (
            f"Periodicity mismatch: LHS={lhs}, -âˆ®Î´f dÏ†={-integral_df_exact}"
        )
