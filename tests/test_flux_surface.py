"""tests/test_flux_surface.py
==============================
Tests for FluxSurface, FluxSurfaceMap, XPointOrbit, and Island extensions.

These tests are designed to:
1. Build a FluxSurfaceMap from synthetic Poincaré data (no field cache required).
2. Verify round-trip accuracy: to_RZ(to_rtheta(R, Z, phi)) ≈ (R, Z) < 1 mm.
3. (Optional, HAO field_cache) Trace X-point orbit and verify closure < 1 mm.
4. Project external "coil" positions and verify r > 1 (outside LCFS).

Run with: pytest tests/test_flux_surface.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.flux_surface import FluxSurface, FluxSurfaceMap, XPointOrbit
from pyna.topo.island import Island


# ---------------------------------------------------------------------------
# Helpers — generate synthetic elliptical flux surfaces
# ---------------------------------------------------------------------------

def make_ellipse_surface(r_norm, R_ax=0.85, Z_ax=0.0, a=0.3, phi=0.0,
                          n_pts=128, n_fourier=8, noise=0.0):
    """Synthetic elliptical surface."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    R_pts = R_ax + a * r_norm * np.cos(theta)
    Z_pts = Z_ax + a * r_norm * 0.7 * np.sin(theta)
    if noise > 0:
        rng = np.random.default_rng(42)
        R_pts += rng.normal(0, noise, n_pts)
        Z_pts += rng.normal(0, noise, n_pts)
    return FluxSurface.from_poincare(R_pts, Z_pts, R_ax, Z_ax, r_norm, phi,
                                      n_fourier=n_fourier)


def make_map(n_surf=12, r_max=0.80, R_ax=0.85, Z_ax=0.0, a=0.3):
    """Build a FluxSurfaceMap from n_surf synthetic elliptical surfaces."""
    r_vals = np.linspace(0.05, r_max, n_surf)
    surfaces = [make_ellipse_surface(r, R_ax=R_ax, Z_ax=Z_ax, a=a) for r in r_vals]
    return FluxSurfaceMap.from_surfaces(surfaces, r_max_fit=r_max, r_extrapolate_max=2.0)


# ---------------------------------------------------------------------------
# Test FluxSurface
# ---------------------------------------------------------------------------

class TestFluxSurface:
    def test_from_poincare_roundtrip(self):
        """Fourier fit should reproduce circle to < 0.1 mm."""
        R_ax, Z_ax, a = 0.85, 0.0, 0.3
        r_norm = 0.5
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        R_pts = R_ax + a * r_norm * np.cos(theta)
        Z_pts = Z_ax + a * r_norm * np.sin(theta)

        surf = FluxSurface.from_poincare(R_pts, Z_pts, R_ax, Z_ax, r_norm, phi=0.0, n_fourier=6)
        R_fit, Z_fit = surf.RZ(theta)
        err = np.sqrt((R_fit - R_pts) ** 2 + (Z_fit - Z_pts) ** 2)
        assert err.max() < 1e-4, f"Max fit error {err.max()*1e3:.3f} mm > 0.1 mm"

    def test_contains(self):
        """O-point inside surface, exterior point outside."""
        surf = make_ellipse_surface(0.5)
        assert surf.contains(0.85, 0.0), "Magnetic axis should be inside r=0.5 surface"
        assert not surf.contains(0.85 + 0.3, 0.0), "Far-right point should be outside"

    def test_area(self):
        """Area of ellipse r*a × r*a*0.7 should match π·a·b."""
        R_ax, Z_ax, a, r = 0.85, 0.0, 0.3, 0.5
        surf = make_ellipse_surface(r, R_ax=R_ax, Z_ax=Z_ax, a=a)
        expected = np.pi * (a * r) * (a * r * 0.7)
        got = surf.area()
        assert abs(got - expected) / expected < 0.01, \
            f"Area {got:.4f} vs expected {expected:.4f}"

    def test_fit_residual_small(self):
        """Residual on clean data should be < 0.1 mm."""
        surf = make_ellipse_surface(0.5, noise=0.0)
        assert surf.fit_residual < 1e-4, f"Residual {surf.fit_residual*1e3:.3f} mm"


# ---------------------------------------------------------------------------
# Test FluxSurfaceMap
# ---------------------------------------------------------------------------

class TestFluxSurfaceMap:
    @pytest.fixture(scope="class")
    def fmap(self):
        return make_map()

    def test_to_RZ_to_rtheta_roundtrip(self, fmap):
        """Round-trip (R,Z) → (r,θ) → (R,Z) should be < 1 mm."""
        test_pts = [
            (0.85 + 0.3 * 0.4, 0.0),      # along R-axis, r~0.4
            (0.85 + 0.3 * 0.6, 0.0),      # r~0.6
            (0.85, 0.3 * 0.5 * 0.7),      # along Z-axis top
            (0.85 - 0.3 * 0.3, -0.3 * 0.3 * 0.7),  # lower-left
        ]
        for R, Z in test_pts:
            r, theta = fmap.to_rtheta(R, Z, phi=0.0)
            R2, Z2 = fmap.to_RZ(r, theta, phi=0.0)
            err = np.sqrt((R2 - R) ** 2 + (Z2 - Z) ** 2)
            assert err < 1e-3, \
                f"Round-trip error at ({R:.3f},{Z:.3f}): {err*1e3:.2f} mm > 1 mm"

    def test_extrapolation_coil(self, fmap):
        """Point outside LCFS (r>1) should return r > 1."""
        R_outside = 0.85 + 0.3 * 1.3   # well outside minor radius
        Z_outside = 0.0
        r, _ = fmap.to_rtheta(R_outside, Z_outside, phi=0.0)
        assert r > 1.0, f"Expected r > 1 for outside-LCFS point, got {r:.3f}"

    def test_project_points_batch(self, fmap):
        """Batch projection should produce same results as individual calls."""
        R_arr = np.array([0.85 + 0.3 * 0.3, 0.85 + 0.3 * 0.5])
        Z_arr = np.array([0.0, 0.0])
        phi_arr = np.array([0.0, 0.0])
        r_arr, theta_arr = fmap.project_points(R_arr, Z_arr, phi_arr)
        assert len(r_arr) == 2
        for i in range(2):
            r_i, th_i = fmap.to_rtheta(R_arr[i], Z_arr[i], phi_arr[i])
            assert abs(r_arr[i] - r_i) < 1e-6

    def test_from_surfaces_too_few(self):
        """from_surfaces should raise with < 3 surfaces."""
        surf = make_ellipse_surface(0.3)
        with pytest.raises(ValueError):
            FluxSurfaceMap.from_surfaces([surf, surf], r_max_fit=0.5)


# ---------------------------------------------------------------------------
# Test Island extensions
# ---------------------------------------------------------------------------

class TestIslandExtensions:
    def _make_island(self):
        return Island(
            period_n=10,
            O_point=np.array([0.85, 0.0]),
            halfwidth=0.05,
        )

    def test_build_flux_surface_map(self):
        """build_flux_surface_map should create a FluxSurfaceMap on self."""
        island = self._make_island()
        R_ax, Z_ax = 0.85, 0.0
        a = 0.3

        def mock_tracer(R0, Z0, phi0, n_turns):
            """Return points on an elliptical orbit."""
            # Infer r from distance to axis
            r_n = np.sqrt((R0 - R_ax) ** 2 + (Z0 - Z_ax) ** 2) / (a * island.halfwidth / 0.05)
            theta = np.linspace(0, 2 * np.pi, n_turns, endpoint=False)
            R_pts = R_ax + a * r_n * np.cos(theta)
            Z_pts = Z_ax + a * r_n * 0.7 * np.sin(theta)
            return R_pts, Z_pts

        fmap = island.build_flux_surface_map(
            tracer=mock_tracer,
            phi_sections=[0.0],
            n_r=8,
            n_turns=200,
            n_fourier=6,
            r_max_fit=0.80,
        )
        assert fmap is island.flux_surface_map
        assert len(fmap.surfaces) >= 3

    def test_coil_theta_projection(self):
        """Coil positions outside LCFS should have r > 1."""
        fmap = make_map()
        island = self._make_island()
        island.flux_surface_map = fmap

        # Coils far outside: R = 0.85 ± 0.6 m
        coil_R = np.array([0.85 + 0.6, 0.85 - 0.6, 0.85])
        coil_Z = np.array([0.0, 0.0, 0.6])
        coil_phi = np.zeros(3)

        result = island.coil_theta_projection(coil_R, coil_Z, coil_phi)
        assert result.shape == (3, 2)
        r_vals = result[:, 0]
        assert np.all(r_vals > 1.0), \
            f"Expected all coil r > 1, got {r_vals}"

    def test_coil_theta_projection_no_map(self):
        """Should raise RuntimeError when no flux_surface_map."""
        island = self._make_island()
        with pytest.raises(RuntimeError):
            island.coil_theta_projection([1.0], [0.0], [0.0])


# ---------------------------------------------------------------------------
# Test XPointOrbit (synthetic, no field_cache required)
# ---------------------------------------------------------------------------

class TestXPointOrbit:
    def _make_synthetic_orbit(self, period=10):
        """Synthetic closed orbit: circle in (R-R_ax, Z-Z_ax) plane."""
        R_ax, Z_ax = 0.85, 0.0
        phi_arr = np.linspace(0, 2 * np.pi * period, 500 * period)
        # X-point orbits the axis many times; use a simple helical path
        theta_pol = phi_arr * 3 / period  # poloidal winding
        R_arr = R_ax + 0.28 * np.cos(theta_pol)
        Z_arr = Z_ax + 0.28 * 0.7 * np.sin(theta_pol)
        return XPointOrbit(phi_arr=phi_arr, R_arr=R_arr, Z_arr=Z_arr, period=period)

    def test_project_to_map(self):
        """project_to_map should populate r_arr and theta_arr."""
        fmap = make_map()
        orbit = self._make_synthetic_orbit(period=3)
        orbit2 = orbit.project_to_map(fmap)
        assert orbit2.r_arr is not None
        assert orbit2.theta_arr is not None
        assert len(orbit2.r_arr) == len(orbit.phi_arr)
        # X-point should be near LCFS (r ≈ 0.85–0.95 for our synthetic geometry)
        r_median = np.median(orbit2.r_arr)
        assert 0.5 < r_median < 1.5, f"Unexpected median r = {r_median:.3f}"

    def test_theta_at_phi(self):
        """theta_at_phi should return value in [0, 2π)."""
        fmap = make_map()
        orbit = self._make_synthetic_orbit(period=3)
        orbit2 = orbit.project_to_map(fmap)
        th = orbit2.theta_at_phi(np.pi)
        assert 0.0 <= th < 2 * np.pi

    def test_theta_at_phi_no_map(self):
        """Should raise RuntimeError before projection."""
        orbit = self._make_synthetic_orbit()
        with pytest.raises(RuntimeError):
            orbit.theta_at_phi(0.0)


# ---------------------------------------------------------------------------
# Optional HAO integration test (skipped if field_cache unavailable)
# ---------------------------------------------------------------------------

def _try_load_hao_field_cache():
    try:
        import pickle
        with open(r"D:\haodata\field_cache.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@pytest.mark.skipif(
    _try_load_hao_field_cache() is None,
    reason="HAO field_cache.pkl not found",
)
class TestHAOIntegration:
    @pytest.fixture(scope="class")
    def field_cache(self):
        import pickle
        with open(r"D:\haodata\field_cache.pkl", "rb") as f:
            return pickle.load(f)

    def test_x_orbit_closure(self, field_cache):
        """X-point orbit should close to < 1 mm after period turns."""
        # Use known HAO X-point coordinates (m/n=10/3 island)
        R_xpt, Z_xpt = 1.08, 0.0  # approximate — replace with actual value
        orbit = XPointOrbit.trace(R_xpt, Z_xpt, phi0=0.0, period=10,
                                   field_cache=field_cache, dphi_out=0.05)
        if len(orbit.R_arr) < 10:
            pytest.skip("Orbit too short — X-point may be outside domain")
        err = np.sqrt((orbit.R_arr[-1] - orbit.R_arr[0]) ** 2 +
                      (orbit.Z_arr[-1] - orbit.Z_arr[0]) ** 2)
        assert err < 1e-3, f"X-point orbit closure error {err*1e3:.2f} mm > 1 mm"
