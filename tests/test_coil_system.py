"""Tests for CoilSet, biot_savart_field, StellaratorControlCoils, and IMAS compat."""
from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def on_axis_Bz_circular_loop(a, I, Z):
    """Analytic on-axis Bz of a circular loop of radius a at height Z above loop."""
    mu0_over_4pi = 1e-7
    return mu0_over_4pi * 2 * np.pi * a**2 * I / (a**2 + Z**2)**1.5


# ---------------------------------------------------------------------------
# 1. biot_savart_field: circular loop Bz on axis
# ---------------------------------------------------------------------------

def make_circular_loop_pts(a, Z0=0.0, N=200):
    """Build XYZ points for a circular loop of radius a in Z=Z0 plane."""
    phi = np.linspace(0, 2 * np.pi, N + 1)
    X = a * np.cos(phi)
    Y = a * np.sin(phi)
    Z = np.full_like(X, Z0)
    return np.column_stack([X, Y, Z])


class TestBiotSavart:
    def test_on_axis_Bz(self):
        """Bz on the axis of a circular loop must match analytic formula."""
        from pyna.MCF.coils.coil_system import biot_savart_field

        a = 0.5    # loop radius (m)
        I = 1000.0 # current (A)
        Z0 = 0.0   # loop at Z=0
        pts = make_circular_loop_pts(a, Z0, N=500)

        # Evaluate at R≈0 (small R to avoid R=0 singularity), Z = 0.3 m
        Z_eval = 0.3
        R_eval = 1e-4  # essentially on axis
        R_grid = np.array([[R_eval]])
        Z_grid = np.array([[Z_eval]])

        # phi=0 → X=R_eval, Y=0
        BR, BZ, BPhi = biot_savart_field(pts, I, R_grid, Z_grid)

        analytic = on_axis_Bz_circular_loop(a, I, Z_eval - Z0)
        # Allow 2% tolerance (numerical integration with 500 segments)
        assert abs(float(BZ[0, 0]) - analytic) / abs(analytic) < 0.02, (
            f"BZ={float(BZ[0,0]):.6e}, analytic={analytic:.6e}"
        )

    def test_zero_current(self):
        """Zero current → zero field."""
        from pyna.MCF.coils.coil_system import biot_savart_field

        pts = make_circular_loop_pts(0.5, N=100)
        R = np.array([[0.5]])
        Z = np.array([[0.0]])
        BR, BZ, BP = biot_savart_field(pts, 0.0, R, Z)
        assert np.allclose(BR, 0) and np.allclose(BZ, 0) and np.allclose(BP, 0)

    def test_sign_convention(self):
        """Positive current in +phi direction → positive Bz at centre (small R)."""
        from pyna.MCF.coils.coil_system import biot_savart_field

        a = 0.3
        I = 500.0
        pts = make_circular_loop_pts(a, Z0=0.0, N=300)
        R_grid = np.array([[1e-4]])
        Z_grid = np.array([[0.0]])
        _, BZ, _ = biot_savart_field(pts, I, R_grid, Z_grid)
        assert float(BZ[0, 0]) > 0, "Positive current loop should give positive Bz at centre"


# ---------------------------------------------------------------------------
# 2. CoilSet
# ---------------------------------------------------------------------------

class TestCoilSet:
    def test_add_and_len(self):
        from pyna.MCF.coils.coil_system import CoilSet

        cs = CoilSet()
        pts = make_circular_loop_pts(0.5, N=50)
        cs.add_coil(pts, 100.0)
        cs.add_coil(pts, -100.0)
        assert len(cs) == 2

    def test_scale_currents(self):
        from pyna.MCF.coils.coil_system import CoilSet

        cs = CoilSet()
        pts = make_circular_loop_pts(0.5, N=50)
        cs.add_coil(pts, 100.0)
        cs.scale_currents(2.0)
        assert abs(cs.coils[0][1] - 200.0) < 1e-10

    def test_set_current(self):
        from pyna.MCF.coils.coil_system import CoilSet

        cs = CoilSet()
        pts = make_circular_loop_pts(0.5, N=50)
        cs.add_coil(pts, 100.0)
        cs.set_current(0, 42.0)
        assert abs(cs.coils[0][1] - 42.0) < 1e-10

    def test_field_on_grid_shape(self):
        """field_on_grid returns arrays of correct shape."""
        from pyna.MCF.coils.coil_system import CoilSet

        cs = CoilSet()
        pts = make_circular_loop_pts(0.5, N=50)
        cs.add_coil(pts, 100.0)

        R_1d = np.linspace(0.1, 0.9, 5)
        Z_1d = np.linspace(-0.3, 0.3, 4)
        BR, BZ, BP = cs.field_on_grid(R_1d, Z_1d)
        assert BR.shape == (5, 4)
        assert BZ.shape == (5, 4)
        assert BP.shape == (5, 4)

    def test_get_set_currents(self):
        from pyna.MCF.coils.coil_system import CoilSet

        cs = CoilSet()
        for I in [10.0, 20.0, 30.0]:
            cs.add_coil(make_circular_loop_pts(0.5, N=20), I)
        arr = cs.get_currents()
        assert list(arr) == pytest.approx([10.0, 20.0, 30.0])
        cs.set_currents([1.0, 2.0, 3.0])
        arr2 = cs.get_currents()
        assert list(arr2) == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 3. StellaratorControlCoils
# ---------------------------------------------------------------------------

class TestStellaratorControlCoils:
    def test_correct_number_of_coils(self):
        from pyna.MCF.coils.coil_system import StellaratorControlCoils

        N = 12
        scc = StellaratorControlCoils(R0=3.0, r_coil=0.35, N_coils=N,
                                       m_target=4, n_target=4, I0=1000.0)
        assert len(scc) == N

    def test_coil_pts_shape(self):
        """Each coil should have 5 points (closed rectangle)."""
        from pyna.MCF.coils.coil_system import StellaratorControlCoils

        scc = StellaratorControlCoils(R0=3.0, r_coil=0.35, N_coils=8,
                                       m_target=4, n_target=4, I0=1.0)
        for pts, _ in scc.coils:
            assert pts.shape == (5, 3), f"Expected (5,3) got {pts.shape}"

    def test_current_phasing(self):
        """Currents should follow cos(n_target * phi_k) phasing."""
        from pyna.MCF.coils.coil_system import StellaratorControlCoils

        N = 16
        n_t = 4
        I0 = 500.0
        scc = StellaratorControlCoils(R0=3.0, r_coil=0.35, N_coils=N,
                                       m_target=4, n_target=n_t, I0=I0)
        for k, (_, I) in enumerate(scc.coils):
            phi_k = 2 * np.pi * k / N
            expected = I0 * np.cos(n_t * phi_k)
            assert abs(I - expected) < 1e-10, f"Coil {k}: got {I}, expected {expected}"

    def test_coil_pts_closed(self):
        """First and last point of each saddle coil should match (closed loop)."""
        from pyna.MCF.coils.coil_system import StellaratorControlCoils

        scc = StellaratorControlCoils(R0=3.0, r_coil=0.35, N_coils=8,
                                       m_target=2, n_target=3, I0=1.0)
        for pts, _ in scc.coils:
            assert np.allclose(pts[0], pts[-1]), "Coil not closed (first != last point)"

    def test_repr(self):
        from pyna.MCF.coils.coil_system import StellaratorControlCoils

        scc = StellaratorControlCoils(3.0, 0.35, 8, 4, 4)
        assert 'StellaratorControlCoils' in repr(scc)


# ---------------------------------------------------------------------------
# 4. IMAS compatibility
# ---------------------------------------------------------------------------

class TestIMASEquilibriumIDS:
    def test_round_trip_json(self):
        """to_json / from_json round-trip preserves all fields."""
        from pyna.imas_compat import IMASEquilibriumIDS

        ids = IMASEquilibriumIDS(
            ip=1e6,
            b0=2.5,
            r0=3.0,
            psi=np.linspace(0, 1, 20),
            q=np.linspace(1.1, 5.0, 20),
            r_boundary=np.array([3.3, 3.0, 2.7]),
            z_boundary=np.array([0.0, 0.2, 0.0]),
        )
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            ids.to_json(path)
            ids2 = IMASEquilibriumIDS.from_json(path)
            assert ids2.ip == pytest.approx(ids.ip)
            assert ids2.b0 == pytest.approx(ids.b0)
            assert ids2.r0 == pytest.approx(ids.r0)
            assert np.allclose(ids2.psi, ids.psi)
            assert np.allclose(ids2.q, ids.q)
        finally:
            os.unlink(path)

    def test_from_stellarator(self):
        from pyna.imas_compat import IMASEquilibriumIDS
        from pyna.MCF.equilibrium.stellarator import SimpleStellarartor

        st = SimpleStellarartor()
        ids = IMASEquilibriumIDS.from_stellarator(st, n_psi=32, n_theta=64)
        assert ids.b0 == pytest.approx(st.B0)
        assert ids.r0 == pytest.approx(st.R0)
        assert len(ids.psi) == 32
        assert len(ids.r_boundary) == 64


class TestIMASCoilsNonAxisymmetric:
    def test_from_coil_set_round_trip(self):
        from pyna.imas_compat import IMASCoilsNonAxisymmetric
        from pyna.MCF.coils.coil_system import CoilSet

        cs = CoilSet()
        for I in [100.0, -100.0]:
            cs.add_coil(make_circular_loop_pts(0.5, N=10), I)

        ids = IMASCoilsNonAxisymmetric.from_coil_set(cs)
        assert len(ids.coil_names) == 2
        assert ids.coil_current[0] == pytest.approx(100.0)
        assert ids.coil_current[1] == pytest.approx(-100.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            ids.to_json(path)
            ids2 = IMASCoilsNonAxisymmetric.from_json(path)
            assert len(ids2.coil_names) == 2
            assert np.allclose(ids2.coil_conductor[0], ids.coil_conductor[0])
        finally:
            os.unlink(path)


class TestIMASPoincareMapping:
    def test_from_poincare_data(self):
        from pyna.imas_compat import IMASPoincareMapping

        R = np.array([3.1, 3.2, 3.3])
        Z = np.array([0.0, 0.1, 0.0])
        pm = IMASPoincareMapping.from_poincare_data(R, Z, phi_section=0.0)
        assert len(pm.r_crossings[0]) == 3
        assert pm.phi_sections[0] == pytest.approx(0.0)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            pm.to_json(path)
            pm2 = IMASPoincareMapping.from_json(path)
            assert np.allclose(pm2.r_crossings[0], R.tolist())
        finally:
            os.unlink(path)
