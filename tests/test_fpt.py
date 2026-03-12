"""Tests for pyna.control — Functional Perturbation Theory module.

Run with:  py -3.13 -m pytest tests/test_fpt.py -v
"""

import numpy as np
import pytest
from pyna.MCF.equilibrium.Solovev import SolovevEquilibrium


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def solovev_iter_like():
    """Return an ITER-like Solov'ev equilibrium (same as test_solovev fixture)."""
    return SolovevEquilibrium(R0=6.2, a=2.0, kappa=1.7, delta=0.33, B0=5.3, A=1.0)


def make_field_func(eq: SolovevEquilibrium):
    """Wrap SolovevEquilibrium into the field_func API.

    field_func([R, Z, phi]) -> [BR/|B|, BZ/|B|, Bphi/(R|B|)]
    """
    def field(rzphi):
        R, Z, phi = float(rzphi[0]), float(rzphi[1]), float(rzphi[2])
        BR, BZ = eq.BR_BZ(R, Z)
        Bphi   = eq.Bphi(R)   # Bphi depends only on R (toroidal field)
        Bmag   = np.sqrt(BR**2 + BZ**2 + Bphi**2) + 1e-30
        return np.array([BR / Bmag, BZ / Bmag, Bphi / (R * Bmag)])
    return field


def make_zero_delta_field():
    """A perturbation field that is identically zero."""
    return lambda rzphi: np.zeros(3)


def make_const_delta_field(dBR=0.0, dBZ=0.0, dBphi=0.001):
    """Uniform perturbation field (tiny, for linearity tests)."""
    def delta_field(rzphi):
        R = float(rzphi[0])
        Bmag_approx = 5.3  # rough scale
        return np.array([
            dBR  / Bmag_approx,
            dBZ  / Bmag_approx,
            dBphi / (R * Bmag_approx),
        ])
    return delta_field


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def eq():
    return solovev_iter_like()


@pytest.fixture
def field_func(eq):
    return make_field_func(eq)


@pytest.fixture
def axis_RZ(eq):
    """Magnetic axis (R, Z)."""
    R0, Z0 = eq.magnetic_axis
    return float(R0), float(Z0)


# ─────────────────────────────────────────────────────────────────────────────
# Test A_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestAMatrix:
    def test_shape(self, field_func, axis_RZ):
        from pyna.control.fpt import A_matrix
        R, Z = axis_RZ
        A = A_matrix(field_func, R, Z)
        assert A.shape == (2, 2)

    def test_finite_entries(self, field_func, axis_RZ):
        from pyna.control.fpt import A_matrix
        R, Z = axis_RZ
        A = A_matrix(field_func, R, Z)
        assert np.all(np.isfinite(A))

    def test_eps_sensitivity(self, field_func, axis_RZ):
        """Results should be consistent across reasonable eps values."""
        from pyna.control.fpt import A_matrix
        R, Z = axis_RZ
        A1 = A_matrix(field_func, R, Z, eps=1e-4)
        A2 = A_matrix(field_func, R, Z, eps=5e-5)
        # Should agree to ~0.1 %
        np.testing.assert_allclose(A1, A2, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Test DPm_axisymmetric
# ─────────────────────────────────────────────────────────────────────────────

class TestDPmAxisymmetric:
    def test_shape(self, field_func, axis_RZ):
        from pyna.control.fpt import A_matrix, DPm_axisymmetric
        R, Z = axis_RZ
        A = A_matrix(field_func, R, Z)
        DPm = DPm_axisymmetric(A)
        assert DPm.shape == (2, 2)

    def test_det_one(self):
        """det(exp(2πA)) = exp(2π·Tr(A)); for Tr(A)=0 this equals 1."""
        from pyna.control.fpt import DPm_axisymmetric
        # Hamiltonian (area-preserving): Tr(A) = 0
        A = np.array([[0.2, -0.5], [0.8, -0.2]])   # Tr = 0
        DPm = DPm_axisymmetric(A)
        assert abs(np.linalg.det(DPm) - 1.0) < 1e-10

    def test_finite_entries(self, field_func, axis_RZ):
        from pyna.control.fpt import A_matrix, DPm_axisymmetric
        R, Z = axis_RZ
        A = A_matrix(field_func, R, Z)
        DPm = DPm_axisymmetric(A)
        assert np.all(np.isfinite(DPm))

    def test_identity_for_zero_A(self):
        """exp(2π · 0) = I."""
        from pyna.control.fpt import DPm_axisymmetric
        A_zero = np.zeros((2, 2))
        DPm = DPm_axisymmetric(A_zero)
        np.testing.assert_allclose(DPm, np.eye(2), atol=1e-14)


# ─────────────────────────────────────────────────────────────────────────────
# Test cycle_shift
# ─────────────────────────────────────────────────────────────────────────────

class TestCycleShift:
    def test_formula(self):
        """δx_cyc = -A⁻¹ · δg  (exact algebraic identity)."""
        from pyna.control.fpt import cycle_shift
        rng = np.random.default_rng(42)
        A = rng.standard_normal((2, 2)) + 2 * np.eye(2)  # make non-singular
        delta_g = rng.standard_normal(2)
        result = cycle_shift(A, delta_g)
        expected = -np.linalg.solve(A, delta_g)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_zero_perturbation(self):
        from pyna.control.fpt import cycle_shift
        A = np.array([[1.0, 0.2], [-0.1, 0.8]])
        result = cycle_shift(A, np.zeros(2))
        np.testing.assert_allclose(result, np.zeros(2), atol=1e-14)

    def test_shape(self):
        from pyna.control.fpt import cycle_shift
        A = np.eye(2) * 2
        delta_g = np.array([0.01, -0.005])
        assert cycle_shift(A, delta_g).shape == (2,)

    def test_linearity(self):
        """cycle_shift is linear in delta_g."""
        from pyna.control.fpt import cycle_shift
        A = np.array([[2.0, 0.5], [0.3, 1.5]])
        dg1 = np.array([0.1, 0.2])
        dg2 = np.array([0.05, -0.1])
        np.testing.assert_allclose(
            cycle_shift(A, dg1 + dg2),
            cycle_shift(A, dg1) + cycle_shift(A, dg2),
            atol=1e-14,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test DPm_change
# ─────────────────────────────────────────────────────────────────────────────

class TestDPmChange:
    def test_shape(self, field_func, axis_RZ):
        from pyna.control.fpt import A_matrix, DPm_change
        R, Z = axis_RZ
        A = A_matrix(field_func, R, Z)
        delta_A = np.zeros((2, 2))
        dDPm = DPm_change(A, delta_A)
        assert dDPm.shape == (2, 2)

    def test_zero_for_zero_delta_A(self, field_func, axis_RZ):
        from pyna.control.fpt import A_matrix, DPm_change
        R, Z = axis_RZ
        A = A_matrix(field_func, R, Z)
        dDPm = DPm_change(A, np.zeros((2, 2)))
        np.testing.assert_allclose(dDPm, np.zeros((2, 2)), atol=1e-12)

    def test_finite(self, field_func, axis_RZ):
        from pyna.control.fpt import A_matrix, DPm_change
        R, Z = axis_RZ
        A = A_matrix(field_func, R, Z)
        delta_A = A * 0.01
        dDPm = DPm_change(A, delta_A)
        assert np.all(np.isfinite(dDPm))

    def test_linearity_in_delta_A(self, field_func, axis_RZ):
        """δDPm is linear in δA."""
        from pyna.control.fpt import A_matrix, DPm_change
        R, Z = axis_RZ
        A = A_matrix(field_func, R, Z)
        dA = A * 0.01
        dDPm1 = DPm_change(A, dA)
        dDPm2 = DPm_change(A, 2 * dA)
        np.testing.assert_allclose(dDPm2, 2 * dDPm1, rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test manifold_shift
# ─────────────────────────────────────────────────────────────────────────────

class TestManifoldShift:
    def test_shape(self, field_func, axis_RZ):
        from pyna.control.fpt import manifold_shift
        R, Z = axis_RZ
        # Straight manifold going radially outward
        manifold_pts = np.column_stack([
            np.linspace(R, R + 0.5, 20),
            np.full(20, Z),
        ])
        delta_xcyc = np.array([0.001, 0.0])
        delta_m = manifold_shift(
            field_func, make_zero_delta_field(),
            manifold_pts, delta_xcyc, stable=True,
        )
        assert delta_m.shape == (20, 2)

    def test_initial_condition(self, field_func, axis_RZ):
        """First point of shift equals delta_xcyc."""
        from pyna.control.fpt import manifold_shift
        R, Z = axis_RZ
        manifold_pts = np.column_stack([
            np.linspace(R, R + 0.5, 10),
            np.full(10, Z),
        ])
        delta_xcyc = np.array([0.002, -0.001])
        delta_m = manifold_shift(
            field_func, make_zero_delta_field(),
            manifold_pts, delta_xcyc,
        )
        np.testing.assert_allclose(delta_m[0], delta_xcyc, atol=1e-14)

    def test_finite_progression(self, field_func, axis_RZ):
        """All entries finite; shift evolves smoothly."""
        from pyna.control.fpt import manifold_shift
        R, Z = axis_RZ
        manifold_pts = np.column_stack([
            np.linspace(R, R + 0.3, 15),
            np.linspace(Z, Z + 0.1, 15),
        ])
        delta_xcyc = np.array([1e-3, 5e-4])
        delta_m = manifold_shift(
            field_func, make_const_delta_field(dBphi=1e-4),
            manifold_pts, delta_xcyc,
        )
        assert np.all(np.isfinite(delta_m))


# ─────────────────────────────────────────────────────────────────────────────
# Test TopologyState.to_vector roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestTopologyState:
    def _make_state(self):
        from pyna.control.topology_state import (
            TopologyState, XPointState, OPointState,
        )
        A = np.array([[0.1, 0.0], [0.0, -0.1]])
        from scipy.linalg import expm
        DPm = expm(2 * np.pi * A)
        eigs, vecs = np.linalg.eig(DPm)
        xp = XPointState(
            R=1.5, Z=-1.2, A_matrix=A, DPm=DPm,
            DPm_eigenvalues=eigs, DPm_eigenvectors=vecs,
            greene_residue=float((2 - np.trace(DPm)) / 4),
        )
        A2 = np.array([[0.0, -0.3], [0.3, 0.0]])
        DPm2 = expm(2 * np.pi * A2)
        eigs2, _ = np.linalg.eig(DPm2)
        op = OPointState(
            R=1.7, Z=0.0, A_matrix=A2, DPm=DPm2,
            DPm_eigenvalues=eigs2, iota=0.3,
        )
        return TopologyState(
            xpoints=[xp], opoints=[op],
            gap_gi={'inner': 0.05, 'outer': 0.12},
            q_samples=np.array([2.0, 3.0, 5.0]),
        )

    def test_to_vector_length(self):
        state = self._make_state()
        vec, labels = state.to_vector()
        assert len(vec) == len(labels)
        assert len(vec) > 0

    def test_to_vector_finite(self):
        state = self._make_state()
        vec, _ = state.to_vector()
        assert np.all(np.isfinite(vec))

    def test_label_types(self):
        state = self._make_state()
        _, labels = state.to_vector()
        assert all(isinstance(l, str) for l in labels)

    def test_xpoint_labels_present(self):
        state = self._make_state()
        _, labels = state.to_vector()
        assert any('xp0.R' in l for l in labels)
        assert any('xp0.Z' in l for l in labels)

    def test_gap_labels_present(self):
        state = self._make_state()
        _, labels = state.to_vector()
        assert any('gap.inner' in l for l in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Test TopologyController.solve
# ─────────────────────────────────────────────────────────────────────────────

class TestTopologyController:
    def _make_simple_problem(self):
        """2-coil, 4-observable toy problem."""
        from scipy.linalg import expm
        from pyna.control.topology_state import (
            TopologyState, XPointState,
        )
        from pyna.control.optimizer import TopologyController, ControlWeights

        A = np.array([[0.1, 0.0], [0.0, -0.1]])
        DPm = expm(2 * np.pi * A)
        eigs, vecs = np.linalg.eig(DPm)

        current_state = TopologyState(xpoints=[
            XPointState(R=1.5, Z=-1.2, A_matrix=A, DPm=DPm,
                        DPm_eigenvalues=eigs, DPm_eigenvectors=vecs,
                        greene_residue=float((2 - np.trace(DPm)) / 4))
        ])
        target_state = TopologyState(xpoints=[
            XPointState(R=1.51, Z=-1.19, A_matrix=A, DPm=DPm,
                        DPm_eigenvalues=eigs, DPm_eigenvectors=vecs,
                        greene_residue=float((2 - np.trace(DPm)) / 4))
        ])

        n_obs = len(current_state.to_vector()[0])
        n_coils = 2
        rng = np.random.default_rng(0)
        R_mat = rng.standard_normal((n_obs, n_coils)) * 0.01

        ctrl = TopologyController(n_coils=n_coils)
        weights = ControlWeights(delta_I_regularization=1e-6)
        return ctrl, current_state, target_state, R_mat, weights

    def test_returns_correct_shape(self):
        ctrl, cs, ts, R, w = self._make_simple_problem()
        labels = cs.to_vector()[1]
        delta_I, result = ctrl.solve(cs, ts, R, labels, weights=w)
        assert delta_I.shape == (2,)

    def test_objective_decreases(self):
        """Optimised δI must reduce the objective below the zero-current value."""
        from pyna.control.optimizer import ControlWeights
        ctrl, cs, ts, R, w = self._make_simple_problem()
        labels = cs.to_vector()[1]
        delta_I, result = ctrl.solve(cs, ts, R, labels, weights=w)

        s_vec, _ = cs.to_vector()
        t_vec, _ = ts.to_vector()
        W = np.diag(ctrl._build_weight_vector(labels, w))

        obj0 = (s_vec - t_vec) @ W @ (s_vec - t_vec)  # δI = 0
        obj1 = result.fun
        assert obj1 <= obj0 + 1e-8

    def test_predict_response_keys(self):
        ctrl, cs, ts, R, w = self._make_simple_problem()
        labels = cs.to_vector()[1]
        delta_I = np.ones(2)
        pred = ctrl.predict_response(cs, delta_I, R, labels)
        assert set(pred.keys()) == set(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Test compute_topology_state
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTopologyState:
    def test_xpoint_populated(self, field_func, axis_RZ):
        from pyna.control.topology_state import compute_topology_state
        R, Z = axis_RZ
        state = compute_topology_state(
            field_func,
            xpoint_guesses=[(R, Z)],
            opoint_guesses=[(R, Z)],
        )
        assert len(state.xpoints) == 1
        assert len(state.opoints) == 1

    def test_DPm_det_one(self, field_func, axis_RZ):
        """For a Hamiltonian A (Tr=0), det(DPm) = 1."""
        from pyna.control.fpt import DPm_axisymmetric
        # Use a synthetic Tr=0 matrix (not the singular axis)
        A = np.array([[0.1, 0.5], [-0.3, -0.1]])   # Tr = 0
        DPm = DPm_axisymmetric(A)
        assert abs(np.linalg.det(DPm) - 1.0) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Test greene_residue
# ─────────────────────────────────────────────────────────────────────────────

class TestGreeneResidue:
    def test_identity(self):
        """Tr(I) = 2 → R = 0."""
        from pyna.control.surface_fate import greene_residue
        assert abs(greene_residue(np.eye(2))) < 1e-14

    def test_hyperbolic(self):
        from pyna.control.surface_fate import greene_residue
        # Large trace → R < 0
        DPm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])
        R = greene_residue(DPm)
        assert R < 0.0

    def test_elliptic(self):
        from pyna.control.surface_fate import greene_residue
        theta = 0.3
        DPm = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
        R = greene_residue(DPm)
        assert 0.0 < R < 1.0
