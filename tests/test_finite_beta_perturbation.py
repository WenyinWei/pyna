"""Tests for finite-beta perturbation framework."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add pyna to path
PYNA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PYNA_ROOT))

from pyna.toroidal.equilibrium.finite_beta_perturbation import (
    compute_diamagnetic_current,
    compute_pfirsch_schlueter_current,
    compute_bootstrap_current,
    compute_pressure_gradient,
    CoilVacuumField,
    FiniteBetaPerturbation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VACUUM_DIR = "/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields"


def make_test_field(nR=16, nZ=16, nPhi=8):
    """Create a simple test magnetic field."""
    R = np.linspace(0.8, 1.2, nR)
    Z = np.linspace(-0.2, 0.2, nZ)
    Phi = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)

    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")

    # Simple stellarator-like field
    B0 = 1.0  # T
    BR = 0.1 * np.sin(PP) * np.cos(2 * np.pi * ZZ / 0.4)
    BPhi = B0 / RR + 0.05 * np.cos(3 * PP)
    BZ = 0.1 * np.cos(PP) * np.sin(2 * np.pi * RR / 0.4)

    return np.stack([BR, BPhi, BZ], axis=0), R, Z, Phi


def make_test_pressure(R, Z, Phi, p0=1e4):
    """Create a simple pressure profile."""
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    psi_n = (RR - R.min()) / (R.max() - R.min())
    return p0 * np.maximum(0, 1 - psi_n) ** 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPressureGradient:
    def test_uniform_pressure(self):
        """Uniform pressure should have zero gradient."""
        nR, nZ, nPhi = 16, 16, 8
        R = np.linspace(0.8, 1.2, nR)
        Z = np.linspace(-0.2, 0.2, nZ)
        Phi = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)

        p = np.ones((nR, nZ, nPhi)) * 1e4
        grad_p = compute_pressure_gradient(p, R, Z, Phi)

        assert np.allclose(grad_p, 0, atol=1e-10)

    def test_radial_gradient(self):
        """Pressure varying only in R should have gradient only in dR."""
        nR, nZ, nPhi = 20, 16, 8
        R = np.linspace(0.8, 1.2, nR)
        Z = np.linspace(-0.2, 0.2, nZ)
        Phi = np.linspace(0, 2 * np.pi, nPhi, endpoint=False)

        RR, _, _ = np.meshgrid(R, Z, Phi, indexing="ij")
        p = (RR - R.min()) / (R.max() - R.min())

        grad_p = compute_pressure_gradient(p, R, Z, Phi)

        # dR should be non-zero
        assert np.abs(grad_p[0]).max() > 1e-10
        # dPhi and dZ should be ~zero
        assert np.allclose(grad_p[1], 0, atol=1e-10)
        assert np.allclose(grad_p[2], 0, atol=1e-10)


class TestCurrentComponents:
    def test_diamagnetic_current_shape(self):
        """Diamagnetic current should have correct shape."""
        B_field, R, Z, Phi = make_test_field()
        p = make_test_pressure(R, Z, Phi)
        grad_p = compute_pressure_gradient(p, R, Z, Phi)

        J_dia = compute_diamagnetic_current(B_field, grad_p)

        assert J_dia.shape == B_field.shape
        assert J_dia.shape[0] == 3  # R, Phi, Z components

    def test_pfirsch_schlueter_current_shape(self):
        """PS current should have correct shape."""
        B_field, R, Z, Phi = make_test_field()
        p = make_test_pressure(R, Z, Phi)
        grad_p = compute_pressure_gradient(p, R, Z, Phi)

        J_PS = compute_pfirsch_schlueter_current(
            B_field, grad_p, R, Z, Phi,
        )

        assert J_PS.shape == B_field.shape

    def test_bootstrap_current_shape(self):
        """Bootstrap current should have correct shape."""
        B_field, R, Z, Phi = make_test_field()
        p = make_test_pressure(R, Z, Phi)

        psi_n = np.zeros_like(p)
        RR, _, _ = np.meshgrid(R, Z, Phi, indexing="ij")
        psi_n[:] = (RR - R.min()) / (R.max() - R.min())

        J_BS = compute_bootstrap_current(B_field, p, psi_n, R, Z, Phi)

        assert J_BS.shape == B_field.shape

    def test_diamagnetic_current_perpendicular_to_B(self):
        """J_dia should be perpendicular to B (J × B = ∇p)."""
        B_field, R, Z, Phi = make_test_field()
        p = make_test_pressure(R, Z, Phi)
        grad_p = compute_pressure_gradient(p, R, Z, Phi)

        J_dia = compute_diamagnetic_current(B_field, grad_p)

        # J · B should be small (not exactly zero due to numerical effects)
        J_dot_B = J_dia[0] * B_field[0] + J_dia[1] * B_field[1] + J_dia[2] * B_field[2]
        B_mag = np.sqrt(B_field[0]**2 + B_field[1]**2 + B_field[2]**2)

        # Normalise by |B|²
        ratio = np.abs(J_dot_B).max() / (np.abs(B_mag).max() + 1e-20)
        # Should be small compared to J_dia magnitude
        J_mag = np.sqrt(J_dia[0]**2 + J_dia[1]**2 + J_dia[2]**2)
        assert ratio < 0.1 * J_mag.max()


class TestCoilVacuumField:
    @pytest.mark.skipif(
        not os.path.exists(VACUUM_DIR),
        reason=f"Vacuum field directory {VACUUM_DIR} not found",
    )
    def test_load_coil_from_npz(self):
        """Should load a coil vacuum field from .npz file."""
        import glob
        files = sorted(glob.glob(os.path.join(VACUUM_DIR, "dipole_coil_*.npz")))
        if not files:
            pytest.skip("No coil files found")

        coil = CoilVacuumField.from_npz(files[0])

        assert coil.BR.shape == coil.BPhi.shape == coil.BZ.shape
        assert len(coil.R_grid) == coil.BR.shape[0]
        assert len(coil.Z_grid) == coil.BR.shape[1]
        assert len(coil.Phi_grid) == coil.BR.shape[2]
        assert coil.coil_index >= 0


class TestFiniteBetaPerturbation:
    @pytest.mark.skipif(
        not os.path.exists(VACUUM_DIR),
        reason=f"Vacuum field directory {VACUUM_DIR} not found",
    )
    def test_small_beta_climb(self):
        """Run a small beta climb with few coils."""
        import glob
        files = sorted(glob.glob(os.path.join(VACUUM_DIR, "dipole_coil_*.npz")))[:10]
        if len(files) < 2:
            pytest.skip("Need at least 2 coil files")

        def p_func(psi_n):
            return max(0, 1 - psi_n) ** 2

        solver = FiniteBetaPerturbation(
            coil_files=files,
            p_profile_func=p_func,
            beta_values=[0.0, 0.01],
            verbose=False,
        )

        history = solver.run()

        assert len(history) == 2
        assert history[0].beta == 0.0
        assert history[1].beta == 0.01
        assert history[0].B_total.shape == solver.B_vacuum.shape
