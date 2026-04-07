"""tests/test_island_healed_coords.py
=====================================
Unit tests for pyna.topo.island_healed_coords.

Uses synthetic analytic data so no real field cache is needed.

Synthetic model
---------------
Inner surfaces: ellipses centred at (R0, Z0) = (1.0, 0.0), with
    R(r, θ) = R0 + r * a * cos(θ)
    Z(r, θ) = Z0 + r * b * sin(θ)
where a=0.15, b=0.12.

Xcyc at r_island=1.0: placed at θ_X_inner = π (left of axis, R=R0-a, Z=Z0).
Ocyc at r_island=1.0: placed at θ_O_inner = 0 (right of axis, R=R0+a, Z=Z0).
"""
from __future__ import annotations

import sys
import numpy as np
import pytest

sys.path.insert(0, r'C:\Users\Legion\Nutstore\1\Repo\pyna')

from pyna.topo.island_healed_coords import (
    InnerFourierSection,
    XOCycAnchor,
    IslandHealedCoordMap,
)

# ── Synthetic geometry ────────────────────────────────────────────────────────
R0, Z0 = 1.0, 0.0
A_ELL, B_ELL = 0.15, 0.12   # semi-axes of elliptic surfaces

N_THETA = 128
N_R     = 16
N_FOURIER = 6
R_INNER_FIT = 0.80
R_ISLAND = 1.0
BLEND_WIDTH = 0.10

theta_arr = np.linspace(-np.pi, np.pi, N_THETA, endpoint=False)
r_norms   = np.linspace(0.05, R_INNER_FIT, N_R)


def make_surface(r_norm: float) -> tuple:
    R = R0 + r_norm * A_ELL * np.cos(theta_arr)
    Z = Z0 + r_norm * B_ELL * np.sin(theta_arr)
    return R, Z


# Build R_surf, Z_surf  (n_r, n_theta)
R_surf = np.array([make_surface(r)[0] for r in r_norms])
Z_surf = np.array([make_surface(r)[1] for r in r_norms])

# One Poincaré section at phi=0
sec = InnerFourierSection.from_poincare_surfaces(
    phi_ref=0.0,
    R_ax=R0, Z_ax=Z0,
    r_norms=r_norms,
    R_surf=R_surf,
    Z_surf=Z_surf,
    theta_arr=theta_arr,
    n_fourier=N_FOURIER,
)

# Xcyc and Ocyc synthetic positions
R_xring = R0 - A_ELL   # θ_inner = π
Z_xring = Z0
R_oring = R0 + A_ELL   # θ_inner = 0
Z_oring = Z0

_, theta_x_inn = sec.project(R_xring, Z_xring, r_init=0.95)
_, theta_o_inn = sec.project(R_oring, Z_oring, r_init=0.95)

anchor = XOCycAnchor(
    phi=0.0,
    R_x=R_xring, Z_x=Z_xring,
    R_o=R_oring, Z_o=Z_oring,
    r_island=R_ISLAND,
    theta_x_inner=float(theta_x_inn),
    theta_o_inner=float(theta_o_inn),
)

coord = IslandHealedCoordMap(
    sections=[sec],
    anchors=[anchor],
    r_inner_fit=R_INNER_FIT,
    r_island=R_ISLAND,
    blend_width=BLEND_WIDTH,
    theta_X=np.pi,
    theta_O=0.0,
)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInnerFourierSection:
    """Tests for InnerFourierSection."""

    def test_axis_roundtrip(self):
        """Axis (r=0) should map to (R0, Z0)."""
        R, Z = sec.eval_RZ(0.0, 0.0)
        assert abs(R - R0) < 1e-4, f"Axis R off: {R}"
        assert abs(Z - Z0) < 1e-4, f"Axis Z off: {Z}"

    def test_surface_roundtrip(self):
        """Forward-then-project should recover (r, θ) to < 1 mm."""
        for r_test in [0.2, 0.5, 0.75]:
            for theta_test in [0.0, np.pi / 3, -np.pi / 2, np.pi]:
                R_t, Z_t = sec.eval_RZ(r_test, theta_test)
                r_back, theta_back = sec.project(R_t, Z_t, r_init=r_test)
                assert abs(r_back - r_test) < 1e-3, \
                    f"r roundtrip fail at (r={r_test}, θ={theta_test:.2f}): Δr={abs(r_back-r_test):.2e}"
                dth = abs((theta_back - theta_test + np.pi) % (2 * np.pi) - np.pi)
                assert dth < 1e-2, \
                    f"θ roundtrip fail at (r={r_test}, θ={theta_test:.2f}): Δθ={dth:.2e}"

    def test_fourier_fit_quality(self):
        """Fourier fit residual < 0.1 mm for r in fit range."""
        for ir, r_test in enumerate(r_norms[::3]):
            R_true, Z_true = make_surface(r_test)
            R_fit = np.array([sec.eval_RZ(r_test, th)[0] for th in theta_arr])
            Z_fit = np.array([sec.eval_RZ(r_test, th)[1] for th in theta_arr])
            rms_R = np.sqrt(np.mean((R_fit - R_true) ** 2))
            rms_Z = np.sqrt(np.mean((Z_fit - Z_true) ** 2))
            assert rms_R < 1e-4, f"Fourier fit R residual too large: {rms_R:.3e} m"
            assert rms_Z < 1e-4, f"Fourier fit Z residual too large: {rms_Z:.3e} m"

    def test_grad_r_direction_ellipse(self):
        """ê_r at (r=0.5, θ=0) should point roughly in +R direction (along semi-axis)."""
        eR, eZ = sec.grad_r_direction(0.5, 0.0)
        # On the ellipse, ê_r at θ=0 is (cos0*a, sin0*b) / norm ≈ (1, 0)
        assert eR > 0.9, f"ê_r at θ=0 should point mostly +R, got eR={eR:.3f}"
        assert abs(eZ) < 0.5, f"ê_r at θ=0 should have small eZ, got {eZ:.3f}"

    def test_grad_r_direction_top(self):
        """ê_r at (r=0.5, θ=π/2) should point mostly in +Z direction."""
        eR, eZ = sec.grad_r_direction(0.5, np.pi / 2)
        assert eZ > 0.9, f"ê_r at θ=π/2 should point mostly +Z, got eZ={eZ:.3f}"

    def test_project_outside_fit_range(self):
        """Projection outside fit range (r>1) should still converge."""
        # Point well outside: R=R0+0.25, Z=0
        r_out, theta_out = sec.project(R0 + 0.25, Z0, r_init=1.2)
        assert r_out > R_INNER_FIT, f"r outside fit should be > r_inner: {r_out}"
        assert abs(theta_out) < 0.2, f"θ at θ=0 should be near 0: {theta_out}"


class TestXOAnchorProjection:
    """Tests that X/O ring positions project correctly."""

    def test_theta_xring(self):
        """Xcyc inner-Fourier θ should be near π (left side of ellipse)."""
        theta_x = anchor.theta_x_inner
        # π or -π (both sides of ±π wrap)
        assert abs(abs(theta_x) - np.pi) < 0.15, \
            f"Xcyc θ should be near ±π, got {np.degrees(theta_x):.1f}°"

    def test_theta_oring(self):
        """Ocyc inner-Fourier θ should be near 0 (right side)."""
        theta_o = anchor.theta_o_inner
        assert abs(theta_o) < 0.15, \
            f"Ocyc θ should be near 0, got {np.degrees(theta_o):.1f}°"


class TestIslandHealedCoordMap:
    """Integration tests for the full coordinate map."""

    def test_inner_zone_identity(self):
        """For r ≪ r_inner_fit, to_rtheta should give the inner-Fourier result."""
        r_test = 0.3
        theta_test = np.pi / 4
        R_t, Z_t = sec.eval_RZ(r_test, theta_test)
        r_out, theta_out = coord.to_rtheta(R_t, Z_t, phi=0.0, r_init=r_test)
        assert abs(r_out - r_test) < 2e-3, f"Inner zone r: {r_out:.4f} vs {r_test}"
        dth = abs((theta_out - theta_test + np.pi) % (2 * np.pi) - np.pi)
        assert dth < 0.05, f"Inner zone θ: {np.degrees(theta_out):.2f}° vs {np.degrees(theta_test):.2f}°"

    def test_o_ring_maps_to_theta_O(self):
        """Ocyc position at r=r_island should project to θ ≈ theta_O = 0."""
        r_out, theta_out = coord.to_rtheta(R_oring, Z_oring, phi=0.0, r_init=0.95)
        # In the blend zone the Ocyc should be mapped toward 0
        # (exact only if fully in island zone)
        assert abs(theta_out) < 0.5, \
            f"Ocyc θ should be near 0, got {np.degrees(theta_out):.1f}°"

    def test_x_ring_maps_toward_theta_X(self):
        """Xcyc position at r=r_island should project toward θ ≈ theta_X = π."""
        r_out, theta_out = coord.to_rtheta(R_xring, Z_xring, phi=0.0, r_init=0.95)
        dist_to_pi = abs(abs(theta_out) - np.pi)
        assert dist_to_pi < 0.5, \
            f"Xcyc θ should be near ±π, got {np.degrees(theta_out):.1f}°"

    def test_grad_r_inner_zone(self):
        """ê_r in inner zone should match InnerFourierSection result."""
        r_test, theta_test = 0.4, 0.0
        eR_coord, eZ_coord = coord.grad_r_direction(r_test, theta_test, phi=0.0)
        eR_sec,   eZ_sec   = sec.grad_r_direction(r_test, theta_test)
        assert abs(eR_coord - eR_sec) < 0.05, f"ê_r eR mismatch: {eR_coord:.3f} vs {eR_sec:.3f}"
        assert abs(eZ_coord - eZ_sec) < 0.05, f"ê_r eZ mismatch: {eZ_coord:.3f} vs {eZ_sec:.3f}"

    def test_grad_r_is_unit_vector(self):
        """ê_r should always be a unit vector."""
        for r_test in [0.2, 0.6, 0.9, 1.1]:
            for theta_test in [0.0, 1.0, -1.5]:
                eR, eZ = coord.grad_r_direction(r_test, theta_test, phi=0.0)
                norm = np.sqrt(eR ** 2 + eZ ** 2)
                assert abs(norm - 1.0) < 1e-6, f"ê_r not unit at r={r_test}: |ê_r|={norm:.6f}"

    def test_eval_RZ_inner_zone(self):
        """eval_RZ in inner zone should match inner section."""
        r_test, theta_test = 0.5, np.pi / 3
        R_coord, Z_coord = coord.eval_RZ(r_test, theta_test, phi=0.0)
        R_sec,   Z_sec   = sec.eval_RZ(r_test, theta_test)
        assert abs(R_coord - R_sec) < 1e-4, f"eval_RZ R: {R_coord:.4f} vs {R_sec:.4f}"
        assert abs(Z_coord - Z_sec) < 1e-4, f"eval_RZ Z: {Z_coord:.4f} vs {Z_sec:.4f}"

    def test_consistency_on_grid(self):
        """to_rtheta followed by eval_RZ should recover (R, Z) to < 2 mm in inner zone."""
        for r_test in [0.2, 0.5, 0.7]:
            for theta_test in [0.0, 1.0, 2.5, -1.5]:
                R_orig, Z_orig = sec.eval_RZ(r_test, theta_test)
                r_back, theta_back = coord.to_rtheta(R_orig, Z_orig, phi=0.0, r_init=r_test)
                R_rec,  Z_rec  = coord.eval_RZ(r_back, theta_back, phi=0.0)
                err_R = abs(R_rec - R_orig)
                err_Z = abs(Z_rec - Z_orig)
                assert err_R < 2e-3, \
                    f"Roundtrip R error at (r={r_test},θ={theta_test:.1f}): {err_R:.2e} m"
                assert err_Z < 2e-3, \
                    f"Roundtrip Z error at (r={r_test},θ={theta_test:.1f}): {err_Z:.2e} m"


class TestWithIslandConstraint:
    """Tests for InnerFourierSection.with_island_constraint()."""

    # Use synthetic elliptic section; X-cycle at theta_inner=pi (left side),
    # O-cycle at theta_inner=0 (right side), placed at r=r_island=1.0

    R_x = R0 - A_ELL   # X-cycle: exactly on the ellipse at r=1, theta=pi
    Z_x = Z0
    R_o = R0 + A_ELL   # O-cycle: exactly on the ellipse at r=1, theta=0
    Z_o = Z0
    r_island = 1.0

    @pytest.fixture(scope='class')
    def sec_constrained(self):
        return sec.with_island_constraint(
            self.R_x, self.Z_x, self.R_o, self.Z_o, r_island=self.r_island
        )

    def test_xcycle_position_within_1mm(self, sec_constrained):
        """After constraint, eval_RZ(1.0, theta_X) should be within 1 mm of X-cycle."""
        theta_x = float(np.arctan2(self.Z_x - Z0, self.R_x - R0))
        R_c, Z_c = sec_constrained.eval_RZ(self.r_island, theta_x)
        err_R = abs(R_c - self.R_x)
        err_Z = abs(Z_c - self.Z_x)
        assert err_R < 1e-3, f"X-cycle R error: {err_R*1000:.2f} mm"
        assert err_Z < 1e-3, f"X-cycle Z error: {err_Z*1000:.2f} mm"

    def test_ocycle_position_within_1mm(self, sec_constrained):
        """After constraint, eval_RZ(1.0, theta_O) should be within 1 mm of O-cycle."""
        theta_o = float(np.arctan2(self.Z_o - Z0, self.R_o - R0))
        R_c, Z_c = sec_constrained.eval_RZ(self.r_island, theta_o)
        err_R = abs(R_c - self.R_o)
        err_Z = abs(Z_c - self.Z_o)
        assert err_R < 1e-3, f"O-cycle R error: {err_R*1000:.2f} mm"
        assert err_Z < 1e-3, f"O-cycle Z error: {err_Z*1000:.2f} mm"

    def test_project_xcycle_gives_r_near_1(self, sec_constrained):
        """After constraint, project(R_x, Z_x)[0] should be close to 1.0."""
        r_proj, _ = sec_constrained.project(self.R_x, self.Z_x, r_init=0.95)
        assert abs(r_proj - self.r_island) < 0.05, \
            f"X-cycle projected r: {r_proj:.4f}, expected ~{self.r_island}"

    def test_inner_surfaces_unchanged(self, sec_constrained):
        """Inner surfaces (r <= r_inner_fit=0.80) should be unchanged to < 0.1 mm."""
        for r_test in [0.2, 0.5, 0.75]:
            for theta_test in [0.0, 1.0, -1.5, np.pi]:
                R_orig, Z_orig = sec.eval_RZ(r_test, theta_test)
                R_new, Z_new = sec_constrained.eval_RZ(r_test, theta_test)
                err_R = abs(R_new - R_orig)
                err_Z = abs(Z_new - Z_orig)
                assert err_R < 1e-4, \
                    f"Inner surface R changed at (r={r_test}, θ={theta_test:.2f}): Δ={err_R:.2e}"
                assert err_Z < 1e-4, \
                    f"Inner surface Z changed at (r={r_test}, θ={theta_test:.2f}): Δ={err_Z:.2e}"

    def test_returns_new_section(self, sec_constrained):
        """with_island_constraint should return a new object, not mutate the original."""
        assert sec_constrained is not sec
        # Original section should still extrapolate without the constraint node
        assert len(sec.r_nodes) == len(sec_constrained.r_nodes) - 1


# ---------------------------------------------------------------------------
# Tests for build_from_island_chain helpers
# ---------------------------------------------------------------------------

class TestExtractOrbitFromChain:
    """Unit tests for _extract_orbit_from_chain and _split_orbit_by_kind."""

    def _make_fake_fp(self, kind, phi, R, Z):
        """Create a minimal fake ChainFixedPoint-like namespace."""
        import types
        fp = types.SimpleNamespace(kind=kind, phi=float(phi), R=float(R), Z=float(Z))
        return fp

    def _make_fake_orbit(self, x_count=2, o_count=2):
        """Orbit with x_count X-type and o_count O-type fake fixed points."""
        import types
        fps = []
        for i in range(x_count):
            fps.append(self._make_fake_fp('X', phi=i * 0.5, R=1.1, Z=0.0))
        for i in range(o_count):
            fps.append(self._make_fake_fp('O', phi=i * 0.5 + 0.25, R=0.9, Z=0.0))
        return types.SimpleNamespace(fixed_points=fps)

    def test_accepts_orbit_directly(self):
        from pyna.topo.island_healed_coords import _extract_orbit_from_chain
        orbit = self._make_fake_orbit()
        result = _extract_orbit_from_chain(orbit, label='test')
        assert result is orbit

    def test_accepts_chain_with_orbit(self):
        from pyna.topo.island_healed_coords import _extract_orbit_from_chain
        import types
        orbit = self._make_fake_orbit()
        chain = types.SimpleNamespace(orbit=orbit)  # no fixed_points attr
        result = _extract_orbit_from_chain(chain, label='test')
        assert result is orbit

    def test_raises_on_none(self):
        from pyna.topo.island_healed_coords import _extract_orbit_from_chain
        with pytest.raises(TypeError, match='must not be None'):
            _extract_orbit_from_chain(None, label='secondary_chain')

    def test_raises_on_chain_without_orbit_attr(self):
        from pyna.topo.island_healed_coords import _extract_orbit_from_chain
        import types
        bad = types.SimpleNamespace(foo='bar')
        with pytest.raises(AttributeError, match="no 'orbit' attribute"):
            _extract_orbit_from_chain(bad, label='secondary_chain')

    def test_raises_on_chain_with_none_orbit(self):
        from pyna.topo.island_healed_coords import _extract_orbit_from_chain
        import types
        chain = types.SimpleNamespace(orbit=None)
        with pytest.raises(ValueError, match='orbit is None'):
            _extract_orbit_from_chain(chain, label='secondary_chain')

    def test_split_orbit_by_kind_basic(self):
        from pyna.topo.island_healed_coords import _split_orbit_by_kind
        orbit = self._make_fake_orbit(x_count=3, o_count=4)
        xring, oring = _split_orbit_by_kind(orbit)
        assert len(xring.fixed_points) == 3
        assert all(fp.kind == 'X' for fp in xring.fixed_points)
        assert len(oring.fixed_points) == 4
        assert all(fp.kind == 'O' for fp in oring.fixed_points)

    def test_split_raises_no_x_points(self):
        from pyna.topo.island_healed_coords import _split_orbit_by_kind
        import types
        orbit = types.SimpleNamespace(
            fixed_points=[self._make_fake_fp('O', 0.0, 0.9, 0.0)]
        )
        with pytest.raises(ValueError, match='no X-type fixed points'):
            _split_orbit_by_kind(orbit)

    def test_split_raises_no_o_points(self):
        from pyna.topo.island_healed_coords import _split_orbit_by_kind
        import types
        orbit = types.SimpleNamespace(
            fixed_points=[self._make_fake_fp('X', 0.0, 1.1, 0.0)]
        )
        with pytest.raises(ValueError, match='no O-type fixed points'):
            _split_orbit_by_kind(orbit)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

