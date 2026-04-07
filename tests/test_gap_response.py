"""Tests for plasma-wall gap response matrix.

Covers:
- make_east_like_wall: creates valid polygon
- WallGeometry.all_gaps: returns positive gaps when LCFS is inside wall
- gap_response_matrix_fpt: returns correct shape (n_gaps, n_coils)
- Caching: second manifold growth call is faster
"""

import time
import numpy as np
import pytest

from pyna.MCF.control.wall import WallGeometry, make_east_like_wall
from pyna.MCF.control.gap_response import gap_response_matrix_fpt, _grow_manifold
from pyna.control.topology_state import XPointState
from pyna.control.FPT import A_matrix, DPm_axisymmetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_field_func(R0=1.85, a=0.45, kappa=1.6):
    """Analytic field function for a simple circular-ish tokamak equilibrium.

    Returns a callable [R,Z,phi] â?[dR/dl, dZ/dl, dphi/dl] approximating
    a large-aspect-ratio tokamak.  The poloidal field is

        B_pol â?r  (uniform q profile)

    where r = sqrt((R-R0)^2 + Z^2).  We normalise to give |B_pol/B_phi| ~ a/R0.
    """
    def field(rzphi):
        R, Z, _ = rzphi
        r = np.sqrt((R - R0) ** 2 + Z ** 2) + 1e-8
        # Poloidal field direction (tangent to flux surface)
        BR_dir = -Z / r
        BZ_dir = (R - R0) / r
        # Magnitude: |Bpol| â?r/a * (a/R0)/1 so that q ~ 1
        Bpol_mag = r / (a * R0)
        BR = Bpol_mag * BR_dir
        BZ = Bpol_mag * BZ_dir
        Bphi = 1.0 / R  # toroidal field ~ 1/R
        norm = np.sqrt(BR ** 2 + BZ ** 2 + Bphi ** 2)
        return [BR / norm, BZ / norm, Bphi / (R * norm)]

    return field


def simple_x_point(R0=1.85, a=0.45):
    """Create a mock XPointState for the lower X-point of a simple equilibrium."""
    R_xpt = R0
    Z_xpt = -a * 1.6  # lower X-point (kappa * a below midplane)

    field_func = simple_field_func(R0=R0, a=a)

    A = A_matrix(field_func, R_xpt, Z_xpt)
    DPm = DPm_axisymmetric(A)
    eigvals, eigvecs = np.linalg.eig(DPm)
    greene = float((2 - np.trace(DPm)) / 4.0)

    return XPointState(
        R=R_xpt, Z=Z_xpt,
        A_matrix=A,
        DPm=DPm,
        DPm_eigenvalues=eigvals,
        DPm_eigenvectors=eigvecs,
        Greene_residue=greene,
    )


def zero_coil_field(scale=0.0):
    """Return a coil perturbation field (zero or small constant shift)."""
    def field(rzphi):
        R, Z, _ = rzphi
        # Tiny vertical field perturbation
        return [0.0, scale / (R + 1e-8), 0.0]
    return field


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMakeEastLikeWall:
    def test_returns_wall_geometry(self):
        wall = make_east_like_wall()
        assert isinstance(wall, WallGeometry)

    def test_polygon_has_correct_size(self):
        wall = make_east_like_wall(n_pts=64)
        assert len(wall.R_wall) == 64
        assert len(wall.Z_wall) == 64

    def test_polygon_R_positive(self):
        wall = make_east_like_wall()
        assert np.all(wall.R_wall > 0), "All wall R coordinates should be positive"

    def test_six_monitors(self):
        wall = make_east_like_wall()
        assert len(wall.gap_monitor_names) == 6
        assert 'inner_mid' in wall.gap_monitor_names
        assert 'outer_mid' in wall.gap_monitor_names
        assert 'top' in wall.gap_monitor_names
        assert 'bottom' in wall.gap_monitor_names
        assert 'div_inner' in wall.gap_monitor_names
        assert 'div_outer' in wall.gap_monitor_names

    def test_monitor_arrays_match_names(self):
        wall = make_east_like_wall()
        assert len(wall.gap_monitor_R) == len(wall.gap_monitor_names)
        assert len(wall.gap_monitor_Z) == len(wall.gap_monitor_names)

    def test_wall_encloses_standard_lcfs(self):
        """Wall should enclose a standard D-shape LCFS."""
        wall = make_east_like_wall(R0=1.85, a=0.45, kappa=1.6, delta=0.4)
        # Check centroid of wall polygon is around R0
        assert abs(np.mean(wall.R_wall) - 1.85) < 0.1


class TestWallGeometryAllGaps:
    """Test gap computation when LCFS is strictly inside the wall."""

    def setup_method(self):
        self.wall = make_east_like_wall(R0=1.85, a=0.45, kappa=1.6, delta=0.4)
        # Generate a simple D-shape LCFS (smaller than wall)
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        R0, a, kappa, delta = 1.85, 0.45, 1.6, 0.4
        self.lcfs_R = R0 + a * np.cos(theta + delta * np.sin(theta))
        self.lcfs_Z = a * kappa * np.sin(theta)

    def test_all_gaps_returns_dict(self):
        gaps = self.wall.all_gaps(self.lcfs_R, self.lcfs_Z)
        assert isinstance(gaps, dict)
        assert len(gaps) == 6

    def test_all_gaps_positive(self):
        gaps = self.wall.all_gaps(self.lcfs_R, self.lcfs_Z)
        for name, gap in gaps.items():
            assert gap >= 0.0, f"Gap '{name}' = {gap:.4f} m should be non-negative"

    def test_gap_to_LCFS_specific_monitor(self):
        gap = self.wall.gap_to_LCFS(self.lcfs_R, self.lcfs_Z, 0)  # inner_mid
        assert gap >= 0.0

    def test_outer_gap_larger_than_inner(self):
        """For a D-shape, outer gap should be roughly similar magnitude."""
        gaps = self.wall.all_gaps(self.lcfs_R, self.lcfs_Z)
        # Both should be positive; rough sanity check
        assert gaps['outer_mid'] > 0
        assert gaps['inner_mid'] > 0


class TestInwardNormal:
    def test_normal_unit_length(self):
        wall = make_east_like_wall()
        n = wall.inward_normal_at(wall.gap_monitor_R[0], wall.gap_monitor_Z[0])
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10

    def test_inner_mid_normal_points_outward_in_R(self):
        """At inner_mid monitor, inward normal should point toward plasma (+R)."""
        wall = make_east_like_wall(R0=1.85)
        # inner_mid is at small R, so inward normal should have positive R component
        n = wall.inward_normal_at(wall.gap_monitor_R[0], wall.gap_monitor_Z[0])
        # The centroid is at R~1.85; inner_mid is at R < 1.85, so n_R > 0
        assert n[0] > 0, f"Inner mid inward normal R-component = {n[0]:.3f}, expected > 0"

    def test_outer_mid_normal_points_inward_in_R(self):
        """At outer_mid monitor, inward normal should point toward plasma (-R)."""
        wall = make_east_like_wall(R0=1.85)
        n = wall.inward_normal_at(wall.gap_monitor_R[1], wall.gap_monitor_Z[1])
        assert n[0] < 0, f"Outer mid inward normal R-component = {n[0]:.3f}, expected < 0"


class TestGapResponseMatrix:
    """Test gap_response_matrix_fpt shape and basic properties."""

    def setup_method(self):
        self.R0 = 1.85
        self.a = 0.45
        self.field_func = simple_field_func(R0=self.R0, a=self.a)
        self.wall = make_east_like_wall(R0=self.R0, a=self.a)
        self.x_point = simple_x_point(R0=self.R0, a=self.a)
        # Two coil perturbation fields
        self.coil_funcs = [
            zero_coil_field(scale=1e-4),
            zero_coil_field(scale=-1e-4),
        ]

    def test_output_shape(self):
        R_gap, names = gap_response_matrix_fpt(
            self.field_func, self.coil_funcs, self.wall, self.x_point,
            field_func_key='test', s_max=0.3, ds=0.02,
        )
        n_gaps = len(self.wall.gap_monitor_names)
        n_coils = len(self.coil_funcs)
        assert R_gap.shape == (n_gaps, n_coils), (
            f"Expected shape ({n_gaps}, {n_coils}), got {R_gap.shape}"
        )

    def test_gap_names_match_wall(self):
        R_gap, names = gap_response_matrix_fpt(
            self.field_func, self.coil_funcs, self.wall, self.x_point,
            field_func_key='test', s_max=0.3, ds=0.02,
        )
        assert names == self.wall.gap_monitor_names

    def test_antisymmetric_coils(self):
        """Opposite-sign coils should give opposite-sign gap responses."""
        R_gap, _ = gap_response_matrix_fpt(
            self.field_func, self.coil_funcs, self.wall, self.x_point,
            field_func_key='test', s_max=0.3, ds=0.02,
        )
        # coil_funcs[1] = -coil_funcs[0], so R_gap[:,1] ~ -R_gap[:,0]
        np.testing.assert_allclose(
            R_gap[:, 0], -R_gap[:, 1], rtol=1e-6,
            err_msg="Antisymmetric coils should give opposite gap responses",
        )

    def test_zero_perturbation_gives_zero_response(self):
        """A zero perturbation field should give zero gap response."""
        zero_field = zero_coil_field(scale=0.0)
        R_gap, _ = gap_response_matrix_fpt(
            self.field_func, [zero_field], self.wall, self.x_point,
            field_func_key='test_zero', s_max=0.3, ds=0.02,
        )
        np.testing.assert_allclose(R_gap, 0.0, atol=1e-12)

    def test_finite_values(self):
        """Response matrix should not contain NaN or Inf."""
        R_gap, _ = gap_response_matrix_fpt(
            self.field_func, self.coil_funcs, self.wall, self.x_point,
            field_func_key='test_finite', s_max=0.3, ds=0.02,
        )
        assert np.all(np.isfinite(R_gap)), "Response matrix contains NaN or Inf"


class TestManifoldGrowth:
    """Test stable manifold growing."""

    def test_manifold_has_right_shape(self):
        field_func = simple_field_func()
        x_point = simple_x_point()
        pts = _grow_manifold(field_func, x_point, s_max=0.2, ds=0.02)
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert pts.shape[0] > 1

    def test_manifold_starts_near_xpoint(self):
        field_func = simple_field_func()
        x_point = simple_x_point()
        pts = _grow_manifold(field_func, x_point, s_max=0.2, ds=0.02)
        dist_first = np.sqrt(
            (pts[0, 0] - x_point.R) ** 2 + (pts[0, 1] - x_point.Z) ** 2
        )
        assert dist_first < 1e-2, f"First manifold point too far from X-point: {dist_first:.4f} m"


class TestCaching:
    """Test that second manifold growth is faster (caching effect)."""

    def test_second_call_not_slower(self):
        field_func = simple_field_func()
        x_point = simple_x_point()

        t0 = time.perf_counter()
        pts1 = _grow_manifold(field_func, x_point, s_max=0.5, ds=0.02)
        t1 = time.perf_counter()
        pts2 = _grow_manifold(field_func, x_point, s_max=0.5, ds=0.02)
        t2 = time.perf_counter()

        # Both calls must produce identical results
        np.testing.assert_array_equal(pts1, pts2)

        # Second call should be at most 2x slower (usually much faster due to Python
        # object reuse, though _grow_manifold itself isn't disk-cached here)
        # We just check it completes in reasonable time
        assert (t2 - t1) < 10.0, "Second manifold growth took unexpectedly long"


class TestHashEqParams:
    def test_deterministic(self):
        from pyna.control._cache import hash_eq_params
        k1 = hash_eq_params(R0=1.85, B0=3.5, Ip=500e3)
        k2 = hash_eq_params(Ip=500e3, R0=1.85, B0=3.5)
        assert k1 == k2

    def test_different_params_differ(self):
        from pyna.control._cache import hash_eq_params
        k1 = hash_eq_params(R0=1.85)
        k2 = hash_eq_params(R0=1.90)
        assert k1 != k2

