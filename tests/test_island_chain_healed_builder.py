"""tests/test_island_chain_healed_builder.py
=============================================
Unit tests for the chain-driven healed coordinate builder:
  - fp_by_section_from_orbits
  - build_from_orbits (input validation and high-level smoke test)

Uses synthetic stub objects (no real field cache required).
"""
from __future__ import annotations

import sys
import numpy as np
import pytest

sys.path.insert(0, r'C:\Users\Legion\Nutstore\1\Repo\pyna')

from pyna.topo.island_healed_coords import (
    fp_by_section_from_orbits,
    build_from_orbits,
)

# ── Minimal stubs ─────────────────────────────────────────────────────────────

class _FP:
    """Minimal ChainFixedPoint stub exposing .phi, .R, .Z."""
    def __init__(self, phi, R, Z, kind='X'):
        self.phi = phi
        self.R = R
        self.Z = Z
        self.kind = kind


class _Orbit:
    """Minimal IslandChainOrbit stub."""
    def __init__(self, fps):
        self.fixed_points = fps


# Synthetic geometry (same ellipse as test_island_healed_coords)
R0, Z0 = 1.0, 0.0
A_ELL  = 0.15

# Two X-fixed-points and two O-fixed-points at phi=0 and phi=pi
X_FPS = [
    _FP(phi=0.0,    R=R0 - A_ELL, Z=Z0, kind='X'),
    _FP(phi=np.pi,  R=R0 - A_ELL, Z=Z0, kind='X'),
]
O_FPS = [
    _FP(phi=0.0,    R=R0 + A_ELL, Z=Z0, kind='O'),
    _FP(phi=np.pi,  R=R0 + A_ELL, Z=Z0, kind='O'),
]

XORBIT = _Orbit(X_FPS)
OORBIT = _Orbit(O_FPS)


# ── Tests: fp_by_section_from_orbits ─────────────────────────────────────────

class TestFpBySectionFromOrbits:
    """Unit tests for fp_by_section_from_orbits."""

    def test_returns_dict(self):
        result = fp_by_section_from_orbits(XORBIT, OORBIT)
        assert isinstance(result, dict)

    def test_all_phi_sections_covered(self):
        result = fp_by_section_from_orbits(XORBIT, OORBIT)
        # Should have at least one entry per unique phi in the fixed points
        assert len(result) >= 1

    def test_each_section_has_xpts_and_opts_keys(self):
        result = fp_by_section_from_orbits(XORBIT, OORBIT)
        for phi_key, sec_data in result.items():
            assert 'xpts' in sec_data, f"Missing 'xpts' at phi={phi_key}"
            assert 'opts' in sec_data, f"Missing 'opts' at phi={phi_key}"

    def test_xpts_are_x_type(self):
        result = fp_by_section_from_orbits(XORBIT, OORBIT)
        for phi_key, sec_data in result.items():
            for xpt in sec_data['xpts']:
                assert xpt.kind == 'X', f"Expected kind='X', got {xpt.kind}"

    def test_opts_are_o_type(self):
        result = fp_by_section_from_orbits(XORBIT, OORBIT)
        for phi_key, sec_data in result.items():
            for opt in sec_data['opts']:
                assert opt.kind == 'O', f"Expected kind='O', got {opt.kind}"

    def test_explicit_phi_sections(self):
        """When phi_sections is given, the result should use those keys."""
        phis = [0.0, np.pi]
        result = fp_by_section_from_orbits(XORBIT, OORBIT, phi_sections=phis)
        assert len(result) == 2

    def test_empty_orbits(self):
        """Empty orbits should return an empty dict."""
        result = fp_by_section_from_orbits(_Orbit([]), _Orbit([]))
        assert result == {}

    def test_phi_tol_controls_assignment(self):
        """With very tight phi_tol, fixed points far from section should not appear."""
        # Sections at pi/2, pi*3/2; fixed points at 0 and pi → should be excluded
        phis = [np.pi / 2, 3 * np.pi / 2]
        result = fp_by_section_from_orbits(XORBIT, OORBIT,
                                           phi_sections=phis, phi_tol=0.01)
        for sec_data in result.values():
            assert len(sec_data['xpts']) == 0
            assert len(sec_data['opts']) == 0

    def test_r_and_z_accessible(self):
        """Fixed points stored in the dict must expose .R and .Z."""
        result = fp_by_section_from_orbits(XORBIT, OORBIT)
        for sec_data in result.values():
            for pt in sec_data['xpts'] + sec_data['opts']:
                assert hasattr(pt, 'R'), "Fixed point missing .R"
                assert hasattr(pt, 'Z'), "Fixed point missing .Z"


# ── Tests: build_from_orbits input validation ─────────────────────────────────

class TestBuildFromOrbitsValidation:
    """Input-validation tests for build_from_orbits.

    These tests check that the function raises informative errors for bad
    inputs, without requiring a real coords_npz file.
    """

    def test_missing_coords_npz_raises(self):
        """Passing a non-existent coords_npz should raise FileNotFoundError."""
        with pytest.raises((FileNotFoundError, OSError)):
            build_from_orbits(
                coords_npz='/nonexistent/path/coords.npz',
                xring_orbit=XORBIT,
                oring_orbit=OORBIT,
            )

    def test_single_fixedpoint_orbit_raises(self):
        """An orbit with only 1 fixed point should raise ValueError (need >=2)."""
        x_single = _Orbit([X_FPS[0]])
        o_single = _Orbit([O_FPS[0]])

        # Build a minimal valid npz in memory and save to a temp file
        import tempfile, os
        n_r, n_theta, n_phi = 4, 16, 2
        r_vals   = np.linspace(0.1, 1.0, n_r)
        th_vals  = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        phi_vals = np.array([0.0, np.pi])
        R_AX = np.full(n_phi, R0)
        Z_AX = np.full(n_phi, Z0)
        R_surf = np.zeros((n_r, n_theta, n_phi))
        Z_surf = np.zeros((n_r, n_theta, n_phi))
        for ip in range(n_phi):
            for ir, r in enumerate(r_vals):
                R_surf[ir, :, ip] = R0 + r * A_ELL * np.cos(th_vals)
                Z_surf[ir, :, ip] = Z0 + r * 0.12  * np.sin(th_vals)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            tmp_path = f.name
        try:
            np.savez(tmp_path,
                     R_surf=R_surf, Z_surf=Z_surf,
                     r_vals=r_vals, theta_vals=th_vals,
                     phi_vals=phi_vals, R_AX=R_AX, Z_AX=Z_AX)
            with pytest.raises(ValueError, match='>=2'):
                build_from_orbits(
                    coords_npz=tmp_path,
                    xring_orbit=x_single,
                    oring_orbit=o_single,
                )
        finally:
            os.unlink(tmp_path)

    def test_smoke_with_minimal_npz(self):
        """build_from_orbits with a minimal synthetic npz should return an IslandHealedCoordMap."""
        from pyna.topo.island_healed_coords import IslandHealedCoordMap
        import tempfile, os

        n_r, n_theta, n_phi = 8, 32, 2
        r_vals   = np.linspace(0.1, 1.0, n_r)
        th_vals  = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        phi_vals = np.array([0.0, np.pi])
        R_AX = np.full(n_phi, R0)
        Z_AX = np.full(n_phi, Z0)
        R_surf = np.zeros((n_r, n_theta, n_phi))
        Z_surf = np.zeros((n_r, n_theta, n_phi))
        for ip in range(n_phi):
            for ir, r in enumerate(r_vals):
                R_surf[ir, :, ip] = R0 + r * A_ELL * np.cos(th_vals)
                Z_surf[ir, :, ip] = Z0 + r * 0.12  * np.sin(th_vals)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            tmp_path = f.name
        try:
            np.savez(tmp_path,
                     R_surf=R_surf, Z_surf=Z_surf,
                     r_vals=r_vals, theta_vals=th_vals,
                     phi_vals=phi_vals, R_AX=R_AX, Z_AX=Z_AX)
            coord = build_from_orbits(
                coords_npz=tmp_path,
                xring_orbit=XORBIT,
                oring_orbit=OORBIT,
                r_inner_fit=0.80,
                r_island=1.0,
                blend_width=0.10,
                n_fourier=4,
                n_anchors=8,
            )
            assert isinstance(coord, IslandHealedCoordMap)
            # Basic sanity: project a point in the inner zone
            R_t, Z_t = R0 + 0.3 * A_ELL, Z0
            r_out, theta_out = coord.to_rtheta(R_t, Z_t, phi=0.0, r_init=0.3)
            assert 0.0 < r_out < 1.5, f"Unexpected r: {r_out}"
        finally:
            os.unlink(tmp_path)


# ── Integration: pyna.topo exports ────────────────────────────────────────────

class TestTopoExports:
    """Verify that chain-driven builder symbols are exported from pyna.topo."""

    def test_fp_by_section_from_orbits_exported(self):
        from pyna.topo import fp_by_section_from_orbits as f
        assert callable(f)

    def test_build_from_orbits_exported(self):
        from pyna.topo import build_from_orbits as f
        assert callable(f)

    def test_build_from_trajectory_npz_exported(self):
        from pyna.topo import build_from_trajectory_npz as f
        assert callable(f)

    def test_all_list_contains_chain_builders(self):
        import pyna.topo as topo
        assert 'fp_by_section_from_orbits' in topo.__all__
        assert 'build_from_orbits' in topo.__all__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
