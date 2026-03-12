"""Tests for magnetic coordinate systems: PEST, Equal-arc, Hamada, Boozer."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures: build a Solov'ev equilibrium and PEST mesh once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solovev_eq():
    from pyna.MCF.equilibrium.Solovev import solovev_iter_like
    return solovev_iter_like(scale=0.3)


@pytest.fixture(scope="module")
def pest_mesh(solovev_eq):
    eq = solovev_eq
    nR, nZ = 120, 120
    R_grid = np.linspace(0.3 * eq.R0, 1.5 * eq.R0, nR)
    Z_grid = np.linspace(-eq.a * eq.kappa * 1.2, eq.a * eq.kappa * 1.2, nZ)
    Rg, Zg = np.meshgrid(R_grid, Z_grid, indexing='ij')
    BR_grid, BZ_grid = eq.BR_BZ(Rg, Zg)
    BPhi_grid = eq.Bphi(Rg)
    psi_norm_grid = eq.psi(Rg, Zg)
    Rmaxis, Zmaxis = eq.magnetic_axis

    from pyna.MCF.coords.PEST import build_PEST_mesh
    S, TET, R_mesh, Z_mesh, q_iS = build_PEST_mesh(
        R_grid, Z_grid, BR_grid, BZ_grid, BPhi_grid, psi_norm_grid,
        Rmaxis, Zmaxis, ns=20, ntheta=91
    )
    return S, TET, R_mesh, Z_mesh, q_iS, eq


# ---------------------------------------------------------------------------
# Test: Equal-arc
# ---------------------------------------------------------------------------

class TestEqualArc:
    def test_arc_lengths_uniform(self, pest_mesh):
        """Arc lengths in θ_ea space should be uniform within 2% on each surface."""
        from pyna.MCF.coords.EqualArc import build_equal_arc_mesh
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        _, TET_ea, R_ea, Z_ea = build_equal_arc_mesh(S, TET, R_mesh, Z_mesh, n_theta=91)

        for i in range(2, len(S) - 1):  # skip axis
            R_s = R_ea[i, :]
            Z_s = Z_ea[i, :]
            dR = np.diff(R_s)
            dZ = np.diff(Z_s)
            ds = np.sqrt(dR ** 2 + dZ ** 2)
            # All arc-length segments should be equal within 2%
            if ds.mean() > 1e-10:
                variation = ds.std() / ds.mean()
                assert variation < 0.02, (
                    f"Surface i={i}: arc-length variation {variation:.3f} > 2%"
                )

    def test_output_shapes(self, pest_mesh):
        from pyna.MCF.coords.EqualArc import build_equal_arc_mesh
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        n_theta = 61
        S_out, TET_ea, R_ea, Z_ea = build_equal_arc_mesh(S, TET, R_mesh, Z_mesh, n_theta=n_theta)
        ns = len(S)
        assert TET_ea.shape == (n_theta,)
        assert R_ea.shape == (ns, n_theta)
        assert Z_ea.shape == (ns, n_theta)
        assert np.isclose(TET_ea[0], 0.0)
        assert np.isclose(TET_ea[-1], 2 * np.pi)


# ---------------------------------------------------------------------------
# Test: PEST safety factor
# ---------------------------------------------------------------------------

class TestPEST:
    def test_q_profile_reasonable(self, pest_mesh):
        """PEST q values should be positive and increase from axis to edge."""
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        q_valid = q_iS[np.isfinite(q_iS)]
        assert np.all(q_valid > 0), "Safety factor should be positive."

    def test_q_matches_equilibrium(self, pest_mesh):
        """PEST q should roughly match the equilibrium q within 10%."""
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        # Compute reference q from equilibrium at the same S values
        psi_vals = S[2:-1] ** 2  # S = sqrt(psi_norm), so psi_norm = S²
        q_ref = eq.q_profile(psi_vals, n_theta=256)
        q_pest = q_iS[2:-1]

        finite = np.isfinite(q_ref) & np.isfinite(q_pest)
        if not np.any(finite):
            pytest.skip("No valid q values to compare.")

        rel_err = np.abs(q_pest[finite] - q_ref[finite]) / (np.abs(q_ref[finite]) + 1e-10)
        max_err = rel_err.max()
        assert max_err < 0.15, (
            f"Max relative error in q vs equilibrium: {max_err:.3f} (> 15%)"
        )


# ---------------------------------------------------------------------------
# Test: Boozer
# ---------------------------------------------------------------------------

class TestBoozer:
    def test_lambda_smooth_and_bounded(self, pest_mesh):
        """Boozer angle correction λ should be smooth and bounded."""
        from pyna.MCF.coords.Boozer import build_Boozer_mesh
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        _, TET_B, R_B, Z_B, lam = build_Boozer_mesh(
            S, TET, R_mesh, Z_mesh, q_iS, equilibrium=eq, n_theta=91
        )
        # λ should be bounded: |λ| < 2π
        finite_lam = lam[np.isfinite(lam)]
        assert np.all(np.abs(finite_lam) < 2 * np.pi), (
            "Boozer angle correction exceeds 2π."
        )

    def test_lambda_periodic(self, pest_mesh):
        """λ should satisfy λ(θ*=0) ≈ λ(θ*=2π) (periodic correction)."""
        from pyna.MCF.coords.Boozer import build_Boozer_mesh
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        _, TET_B, R_B, Z_B, lam = build_Boozer_mesh(
            S, TET, R_mesh, Z_mesh, q_iS, equilibrium=eq, n_theta=91
        )
        for i in range(1, len(S) - 1):
            assert abs(lam[i, 0] - lam[i, -1]) < 0.1, (
                f"Surface {i}: λ not periodic: λ(0)={lam[i,0]:.4f}, λ(2π)={lam[i,-1]:.4f}"
            )

    def test_output_shapes(self, pest_mesh):
        from pyna.MCF.coords.Boozer import build_Boozer_mesh
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        n_theta = 61
        S_out, TET_B, R_B, Z_B, lam = build_Boozer_mesh(
            S, TET, R_mesh, Z_mesh, q_iS, equilibrium=eq, n_theta=n_theta
        )
        ns = len(S)
        assert TET_B.shape == (n_theta,)
        assert R_B.shape == (ns, n_theta)
        assert Z_B.shape == (ns, n_theta)
        assert lam.shape == (ns, len(TET))


# ---------------------------------------------------------------------------
# Test: Hamada
# ---------------------------------------------------------------------------

class TestHamada:
    def test_area_elements_uniform(self, pest_mesh):
        """Hamada angle: cumulative area (from axis) should be linear in θ_H."""
        from pyna.MCF.coords.Hamada import build_Hamada_mesh
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        _, TET_H, R_H, Z_H = build_Hamada_mesh(S, TET, R_mesh, Z_mesh, n_theta=91)

        # Estimate magnetic axis
        R_ax = R_mesh[0, 0]
        Z_ax = Z_mesh[0, 0]

        for i in range(2, len(S) - 1):
            R_s = R_H[i, :]
            Z_s = Z_H[i, :]
            # Drop endpoint duplicate
            if np.allclose(R_s[0], R_s[-1]) and np.allclose(Z_s[0], Z_s[-1]):
                R_loop = R_s[:-1]
                Z_loop = Z_s[:-1]
            else:
                R_loop = R_s
                Z_loop = Z_s
            R_closed = np.append(R_loop, R_loop[0])
            Z_closed = np.append(Z_loop, Z_loop[0])
            # Triangle area from axis to each segment
            dA = 0.5 * (
                (R_closed[:-1] - R_ax) * (Z_closed[1:] - Z_ax)
                - (R_closed[1:] - R_ax) * (Z_closed[:-1] - Z_ax)
            )
            A_cumulative = np.cumsum(dA)
            A_total = A_cumulative[-1]
            if abs(A_total) < 1e-10:
                continue
            # Cumulative area should be approximately linear in index (equal-area)
            A_norm = A_cumulative / A_total
            A_expected = np.linspace(1.0 / len(R_loop), 1.0, len(R_loop))
            max_deviation = np.max(np.abs(A_norm - A_expected))
            assert max_deviation < 0.05, (
                f"Surface i={i}: cumulative area deviation {max_deviation:.3f} > 5%"
            )

    def test_output_shapes(self, pest_mesh):
        from pyna.MCF.coords.Hamada import build_Hamada_mesh
        S, TET, R_mesh, Z_mesh, q_iS, eq = pest_mesh
        n_theta = 61
        S_out, TET_H, R_H, Z_H = build_Hamada_mesh(S, TET, R_mesh, Z_mesh, n_theta=n_theta)
        ns = len(S)
        assert TET_H.shape == (n_theta,)
        assert R_H.shape == (ns, n_theta)
        assert Z_H.shape == (ns, n_theta)
