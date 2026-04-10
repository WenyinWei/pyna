"""Tests for pyna.MCF.coords.island_healing.

Tests use a synthetic Solov'ev-like axisymmetric equilibrium with a known
m/n = 3/1 island chain planted at the boundary.  The field is analytic so
exact O/X-point positions are known.

Run with::

    pytest tests/test_island_healing.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.island import Island, IslandChain
from pyna.topo.invariants import FixedPoint, PeriodicOrbit
from pyna.MCF.coords.island_healing import (
    assign_island_chain_pest_angles,
    build_r1_boundary,
    heal_pest_mesh_at_island_chain,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic PEST mesh + island chain
# ---------------------------------------------------------------------------

def _make_synthetic_pest(
    ns: int = 30,
    ntheta: int = 129,
    R0: float = 3.0,
    a: float = 0.5,
) -> tuple:
    """Concentric elliptic surfaces as a synthetic PEST-like mesh.

    Returns (S, TET, R_mesh, Z_mesh, q_iS, Rmaxis, Zmaxis).
    q profile is linear: q(S) = 1.5 + S.
    """
    S = np.linspace(0.0, 0.9, ns)
    TET = np.linspace(0.0, 2.0 * np.pi, ntheta)
    R_mesh = np.empty((ns, ntheta))
    Z_mesh = np.empty((ns, ntheta))
    Rmaxis, Zmaxis = R0, 0.0

    for i, s in enumerate(S):
        r = s * a
        R_mesh[i, :] = R0 + r * np.cos(TET)
        Z_mesh[i, :] = r * np.sin(TET)

    q_iS = 1.5 + S
    return S, TET, R_mesh, Z_mesh, q_iS, Rmaxis, Zmaxis


def _make_island_chain_3_1(
    R0: float = 3.0,
    a_bdy: float = 0.5,   # boundary radius
) -> IslandChain:
    """Synthetic 3/1 island chain at the boundary.

    O-points at θ = 0, 2π/3, 4π/3 (geometric angle).
    X-points at θ = π/3, π, 5π/3.
    All points lie on a circle of radius a_bdy.
    """
    m, n = 3, 1
    theta_O_geo = [0.0, 2*np.pi/3, 4*np.pi/3]
    theta_X_geo = [np.pi/3, np.pi, 5*np.pi/3]

    O_pts = [np.array([R0 + a_bdy * np.cos(t), a_bdy * np.sin(t)]) for t in theta_O_geo]
    X_pts = [np.array([R0 + a_bdy * np.cos(t), a_bdy * np.sin(t)]) for t in theta_X_geo]

    def _fp(arr, kind):
        return FixedPoint(phi=0.0, R=float(arr[0]), Z=float(arr[1]),
                          DPm=np.eye(2) if kind == 'O' else np.array([[2.,0],[0.,.5]]),
                          kind=kind)

    islands = [
        Island(O_orbit=PeriodicOrbit(points=[_fp(op, 'O')]),
               X_orbits=[PeriodicOrbit(points=[_fp(X_pts[k], 'X')])])
        for k, op in enumerate(O_pts)
    ]
    return IslandChain(m=m, n=n, islands=islands)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssignPestAngles:

    def setup_method(self):
        (self.S, self.TET, self.R_mesh, self.Z_mesh,
         self.q_iS, self.Rmaxis, self.Zmaxis) = _make_synthetic_pest()
        self.chain = _make_island_chain_3_1(R0=self.Rmaxis)

    def test_returns_correct_count(self):
        theta_O, theta_X = assign_island_chain_pest_angles(
            self.chain, self.R_mesh, self.Z_mesh, self.TET,
            self.Rmaxis, self.Zmaxis,
        )
        assert len(theta_O) == self.chain.m   # 3 O-point angles
        assert len(theta_X) == self.chain.m   # 3 X-point angles

    def test_spacing_is_pi_over_m(self):
        """Consecutive O (and X) angles must be separated by 2π/m."""
        theta_O, theta_X = assign_island_chain_pest_angles(
            self.chain, self.R_mesh, self.Z_mesh, self.TET,
            self.Rmaxis, self.Zmaxis,
        )
        m = self.chain.m
        expected = 2.0 * np.pi / m
        diffs_O = np.diff(np.sort(theta_O))
        diffs_X = np.diff(np.sort(theta_X))
        np.testing.assert_allclose(diffs_O, expected, atol=0.05,
            err_msg="O-points not equally spaced by 2π/m in θ*")
        np.testing.assert_allclose(diffs_X, expected, atol=0.05,
            err_msg="X-points not equally spaced by 2π/m in θ*")

    def test_O_X_interleaved_by_half_spacing(self):
        """X-point angles should be offset from O-point angles by π/m."""
        theta_O, theta_X = assign_island_chain_pest_angles(
            self.chain, self.R_mesh, self.Z_mesh, self.TET,
            self.Rmaxis, self.Zmaxis,
        )
        m = self.chain.m
        half = np.pi / m
        for tx in theta_X:
            # Distance from tx to nearest theta_O should be ~π/m
            diffs = np.abs(np.angle(np.exp(1j * (theta_O - tx))))
            nearest = np.min(diffs)
            assert abs(nearest - half) < 0.1, (
                f"X-point at θ*={tx:.3f} not offset by π/m={half:.3f} "
                f"from nearest O-point (got {nearest:.3f})"
            )

    def test_island_pest_theta_set(self):
        assign_island_chain_pest_angles(
            self.chain, self.R_mesh, self.Z_mesh, self.TET,
            self.Rmaxis, self.Zmaxis,
        )
        for isl in self.chain.islands:
            assert hasattr(isl, 'pest_theta'), "Island.pest_theta not set"
            assert isl.pest_theta is not None

    def test_chain_attributes_set(self):
        theta_O, theta_X = assign_island_chain_pest_angles(
            self.chain, self.R_mesh, self.Z_mesh, self.TET,
            self.Rmaxis, self.Zmaxis,
        )
        assert hasattr(self.chain, 'pest_theta_O')
        assert hasattr(self.chain, 'pest_theta_X')
        np.testing.assert_array_equal(self.chain.pest_theta_O, theta_O)
        np.testing.assert_array_equal(self.chain.pest_theta_X, theta_X)


class TestBuildR1Boundary:

    def setup_method(self):
        (self.S, self.TET, self.R_mesh, self.Z_mesh,
         self.q_iS, self.Rmaxis, self.Zmaxis) = _make_synthetic_pest()
        self.chain = _make_island_chain_3_1(R0=self.Rmaxis)
        assign_island_chain_pest_angles(
            self.chain, self.R_mesh, self.Z_mesh, self.TET,
            self.Rmaxis, self.Zmaxis,
        )

    def test_output_shape(self):
        R_bdy, Z_bdy = build_r1_boundary(
            self.chain,
            self.chain.pest_theta_O,
            self.chain.pest_theta_X,
            self.TET,
        )
        assert R_bdy.shape == (len(self.TET),)
        assert Z_bdy.shape == (len(self.TET),)

    def test_boundary_passes_near_opoints(self):
        """r=1 curve should pass close to the O-points."""
        R_bdy, Z_bdy = build_r1_boundary(
            self.chain,
            self.chain.pest_theta_O,
            self.chain.pest_theta_X,
            self.TET,
        )
        for isl in self.chain.islands:
            op = isl.O_point
            dists = np.sqrt((R_bdy - op[0])**2 + (Z_bdy - op[1])**2)
            assert np.min(dists) < 0.15, (
                f"Boundary does not pass near O-point {op}; "
                f"min distance = {np.min(dists):.4f}"
            )

    def test_boundary_passes_near_xpoints(self):
        """r=1 curve should pass close to the X-points."""
        R_bdy, Z_bdy = build_r1_boundary(
            self.chain,
            self.chain.pest_theta_O,
            self.chain.pest_theta_X,
            self.TET,
        )
        for isl in self.chain.islands:
            for xp in isl.X_points:
                dists = np.sqrt((R_bdy - xp[0])**2 + (Z_bdy - xp[1])**2)
                assert np.min(dists) < 0.15, (
                    f"Boundary does not pass near X-point {xp}; "
                    f"min distance = {np.min(dists):.4f}"
                )

    def test_boundary_is_closed(self):
        """First and last point should coincide (periodicity)."""
        R_bdy, Z_bdy = build_r1_boundary(
            self.chain,
            self.chain.pest_theta_O,
            self.chain.pest_theta_X,
            self.TET,
        )
        assert abs(R_bdy[0] - R_bdy[-1]) < 0.01
        assert abs(Z_bdy[0] - Z_bdy[-1]) < 0.01


class TestHealPestMesh:

    def setup_method(self):
        (self.S, self.TET, self.R_mesh, self.Z_mesh,
         self.q_iS, self.Rmaxis, self.Zmaxis) = _make_synthetic_pest()
        self.chain = _make_island_chain_3_1(R0=self.Rmaxis)

    def test_output_shapes(self):
        n_heal = 15
        S_out, TET, R_out, Z_out, q_out = heal_pest_mesh_at_island_chain(
            self.S, self.TET, self.R_mesh, self.Z_mesh, self.q_iS,
            self.chain, self.Rmaxis, self.Zmaxis,
            n_heal=n_heal,
        )
        ns_orig = len(self.S)
        # n_heal interior surfaces + 1 surface at r=1
        assert len(S_out) == ns_orig + n_heal + 1, (
            f"Expected {ns_orig + n_heal + 1} surfaces, got {len(S_out)}"
        )
        assert R_out.shape == (len(S_out), len(self.TET))
        assert Z_out.shape == (len(S_out), len(self.TET))
        assert q_out.shape == (len(S_out),)

    def test_S_ends_at_1(self):
        S_out, _, _, _, _ = heal_pest_mesh_at_island_chain(
            self.S, self.TET, self.R_mesh, self.Z_mesh, self.q_iS,
            self.chain, self.Rmaxis, self.Zmaxis,
        )
        assert abs(S_out[-1] - 1.0) < 1e-12, f"Last S should be 1.0, got {S_out[-1]}"

    def test_S_monotone(self):
        S_out, _, _, _, _ = heal_pest_mesh_at_island_chain(
            self.S, self.TET, self.R_mesh, self.Z_mesh, self.q_iS,
            self.chain, self.Rmaxis, self.Zmaxis,
        )
        assert np.all(np.diff(S_out) >= 0), "S_out is not monotonically non-decreasing"

    def test_q_at_r1_is_rational(self):
        _, _, _, _, q_out = heal_pest_mesh_at_island_chain(
            self.S, self.TET, self.R_mesh, self.Z_mesh, self.q_iS,
            self.chain, self.Rmaxis, self.Zmaxis,
        )
        q_mn = self.chain.m / self.chain.n
        assert abs(q_out[-1] - q_mn) < 1e-10, (
            f"q at r=1 should be {q_mn}, got {q_out[-1]}"
        )

    def test_existing_surfaces_unchanged(self):
        """Mesh surfaces below S_last_good must be preserved exactly."""
        S_out, _, R_out, Z_out, _ = heal_pest_mesh_at_island_chain(
            self.S, self.TET, self.R_mesh, self.Z_mesh, self.q_iS,
            self.chain, self.Rmaxis, self.Zmaxis,
        )
        ns_orig = len(self.S)
        np.testing.assert_array_equal(
            S_out[:ns_orig], self.S,
            err_msg="Original S values were modified"
        )
        np.testing.assert_array_equal(
            R_out[:ns_orig, :], self.R_mesh,
            err_msg="Original R_mesh was modified"
        )

    def test_last_surface_near_island_ring(self):
        """The r=1 surface should pass close to O-points and X-points."""
        S_out, TET, R_out, Z_out, _ = heal_pest_mesh_at_island_chain(
            self.S, self.TET, self.R_mesh, self.Z_mesh, self.q_iS,
            self.chain, self.Rmaxis, self.Zmaxis,
            n_heal=30,
        )
        R_r1 = R_out[-1, :]
        Z_r1 = Z_out[-1, :]
        for isl in self.chain.islands:
            op = isl.O_point
            dists = np.sqrt((R_r1 - op[0])**2 + (Z_r1 - op[1])**2)
            assert np.min(dists) < 0.15, (
                f"r=1 surface not near O-point {op}; min dist={np.min(dists):.4f}"
            )

    def test_disconnected_chain(self):
        """Disconnected (split) island chain should not raise."""
        chain = _make_island_chain_3_1(R0=self.Rmaxis)
        chain.split_into_subchains([[0], [1], [2]])
        # Should complete without error
        S_out, _, R_out, Z_out, q_out = heal_pest_mesh_at_island_chain(
            self.S, self.TET, self.R_mesh, self.Z_mesh, self.q_iS,
            chain, self.Rmaxis, self.Zmaxis,
        )
        assert S_out[-1] == pytest.approx(1.0)

    def test_linear_interp(self):
        """interp_radial='linear' should also produce valid output."""
        S_out, _, R_out, Z_out, _ = heal_pest_mesh_at_island_chain(
            self.S, self.TET, self.R_mesh, self.Z_mesh, self.q_iS,
            self.chain, self.Rmaxis, self.Zmaxis,
            interp_radial='linear',
        )
        assert S_out[-1] == pytest.approx(1.0)
        assert not np.any(np.isnan(R_out)), "NaN in healed R_mesh (linear interp)"
        assert not np.any(np.isnan(Z_out)), "NaN in healed Z_mesh (linear interp)"
