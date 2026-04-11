"""Tests for pyna.topo.fixed_points -- find_periodic_orbit and classify_fixed_point."""
import numpy as np
import pytest
from pyna.toroidal.equilibrium.stellarator import simple_stellarator
from pyna.topo.fixed_points import find_periodic_orbit, classify_fixed_point


def make_field_func_2d(st):
    """Wrap StellaratorSimple.field_func into (R, Z, phi) -> [dR/dphi, dZ/dphi]."""
    def ff2d(R: float, Z: float, phi: float) -> np.ndarray:
        tang = np.asarray(st.field_func(np.array([R, Z, phi])), dtype=float)
        dphi_ds = tang[2]
        if abs(dphi_ds) < 1e-15:
            return np.array([0.0, 0.0])
        return np.array([tang[0] / dphi_ds, tang[1] / dphi_ds])
    return ff2d


@pytest.fixture(scope="module")
def stellarator_setup():
    """Set up a stellarator with a q=5/4 island (period-4 island chain).

    m_h=4, n_h=4 perturbation drives islands near the q=5/4 surface.
    q0=1.1, q1=5.0 -> q=5/4 at psi=(1.25-1.1)/(5-1.1)=0.0385
    r_res = 0.3 * sqrt(0.0385) ~ 0.059 m
    """
    st = simple_stellarator(R0=3.0, r0=0.3, B0=1.0, q0=1.1, q1=5.0,
                            m_h=4, n_h=4, epsilon_h=0.05)
    ff2d = make_field_func_2d(st)
    psi_res = (5 / 4 - 1.1) / (5.0 - 1.1)
    r_res = 0.3 * np.sqrt(psi_res)   # ~0.059 m
    seed = np.array([3.0 + r_res, 0.0])
    return st, ff2d, seed, r_res


def test_find_periodic_orbit_finds_fixed_points(stellarator_setup):
    """find_periodic_orbit should find >= 2 period-4 fixed points."""
    st, ff2d, seed, r_res = stellarator_setup
    fps = find_periodic_orbit(
        ff2d, seed=seed, n_turns=4, r_scan=r_res * 0.6,
        n_scan=300, verbose=True
    )
    print(f"\nFound {len(fps)} fixed points")
    for fp in fps:
        print(f"  R={fp[0]:.6f}  Z={fp[1]:.6f}")
    assert len(fps) >= 2, f"Expected >= 2 fixed points, got {len(fps)}"


def test_classify_fixed_points(stellarator_setup):
    """All fixed points should have det(J) close to 1; at least one must be an X-point."""
    st, ff2d, seed, r_res = stellarator_setup
    fps = find_periodic_orbit(
        ff2d, seed=seed, n_turns=4, r_scan=r_res * 0.6,
        n_scan=300, verbose=False
    )
    assert len(fps) >= 2

    types = []
    for fp in fps:
        fp_type, J, det_J = classify_fixed_point(fp, ff2d, n_turns=4)
        print(f"  R={fp[0]:.6f}  Z={fp[1]:.6f}  type={fp_type}  det(J)={det_J:.6f}")
        # det should be close to 1 (area-preserving map)
        assert abs(det_J - 1.0) < 0.02, (
            f"det(J)={det_J:.6f} deviates from 1 by more than 0.02 at {fp}")
        types.append(fp_type)

    assert "X" in types, f"Expected at least one X-point, got types: {types}"
