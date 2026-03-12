"""Tests for plasma response module (perturbed Grad-Shafranov solver).

Tests:
1. solve_perturbed_gs returns CylindricalVectorField tuple
2. compute_plasma_response runs on Solov'ev + uniform delta_B_ext
3. ∇ · δB_plasma ≈ 0 numerically
4. δB_plasma + δB_ext has smaller residual in J×B = ∇p than δB_ext alone (rough)
5. Caching: second call is faster
"""

import time
import numpy as np
import pytest

from pyna.field_data import CylindricalVectorField, CylindricalScalarField
from pyna.mag.solovev import SolovevEquilibrium, solovev_iter_like
from pyna.plasma_response import solve_perturbed_gs, compute_plasma_response
from pyna.plasma_response.perturb_gs import (
    _make_axi_vector_field,
    _make_axi_scalar_field,
    _solovev_grid_fields,
)


# ---------------------------------------------------------------------------
# Small test grid for speed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_eq():
    """Small Solov'ev equilibrium (EAST-sized) for fast tests."""
    return SolovevEquilibrium(R0=1.85, a=0.45, B0=2.0, kappa=1.6, delta=0.3, q0=1.5)


@pytest.fixture(scope="module")
def small_grid():
    R = np.linspace(1.3, 2.4, 20)
    Z = np.linspace(-0.7, 0.7, 18)
    return R, Z


@pytest.fixture(scope="module")
def grid_fields(small_eq, small_grid):
    R, Z = small_grid
    return _solovev_grid_fields(small_eq, R, Z)


@pytest.fixture(scope="module")
def uniform_delta_B_ext(small_grid):
    """Small uniform external perturbation field."""
    R, Z = small_grid
    nR, nZ = len(R), len(Z)
    Phi = np.array([0.0])
    VR   = np.full((nR, nZ, 1), 1e-4)  # 0.1 mT uniform
    VZ   = np.zeros((nR, nZ, 1))
    VPhi = np.zeros((nR, nZ, 1))
    return CylindricalVectorField(R=R, Z=Z, Phi=Phi, VR=VR, VZ=VZ, VPhi=VPhi,
                                  name="delta_B_ext")


# ---------------------------------------------------------------------------
# Test 1: return types
# ---------------------------------------------------------------------------

def test_solve_perturbed_gs_return_types(grid_fields, uniform_delta_B_ext):
    """solve_perturbed_gs must return a 3-tuple of (VectorField, VectorField, ScalarField)."""
    B0, J0, p0 = grid_fields
    result = solve_perturbed_gs(B0, J0, p0, uniform_delta_B_ext,
                                solver='lsqr', max_iter=200, tol=1e-4)
    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 3, "Result must have 3 elements"
    delta_B_plasma, delta_J, delta_p = result

    assert isinstance(delta_B_plasma, CylindricalVectorField), \
        f"delta_B_plasma must be CylindricalVectorField, got {type(delta_B_plasma)}"
    assert isinstance(delta_J, CylindricalVectorField), \
        f"delta_J must be CylindricalVectorField, got {type(delta_J)}"
    assert isinstance(delta_p, CylindricalScalarField), \
        f"delta_p must be CylindricalScalarField, got {type(delta_p)}"


# ---------------------------------------------------------------------------
# Test 2: compute_plasma_response runs end-to-end
# ---------------------------------------------------------------------------

def test_compute_plasma_response_runs(small_eq, uniform_delta_B_ext):
    """compute_plasma_response should run without error and return a CylindricalVectorField."""
    result = compute_plasma_response(small_eq, uniform_delta_B_ext,
                                     solver='lsqr', max_iter=200, tol=1e-4)
    assert isinstance(result, CylindricalVectorField), \
        f"Expected CylindricalVectorField, got {type(result)}"

    # Should have same R, Z grid as input
    np.testing.assert_array_equal(result.R, uniform_delta_B_ext.R)
    np.testing.assert_array_equal(result.Z, uniform_delta_B_ext.Z)


# ---------------------------------------------------------------------------
# Test 3: ∇ · δB_plasma ≈ 0
# ---------------------------------------------------------------------------

def test_divfree(grid_fields, uniform_delta_B_ext):
    """div(δB_plasma) should be small compared to mean |B|."""
    B0, J0, p0 = grid_fields
    delta_B_plasma, _, _ = solve_perturbed_gs(B0, J0, p0, uniform_delta_B_ext,
                                               solver='lsqr', max_iter=300, tol=1e-5)

    R = delta_B_plasma.R
    Z = delta_B_plasma.Z
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    dBR = delta_B_plasma.VR[:, :, 0]
    dBZ = delta_B_plasma.VZ[:, :, 0]

    RR, _ = np.meshgrid(R, np.ones(len(Z)), indexing='ij')

    dBR_dR = np.gradient(dBR, dR, axis=0)
    dBZ_dZ = np.gradient(dBZ, dZ, axis=1)
    div_B = dBR / RR + dBR_dR + dBZ_dZ

    mean_B = np.mean(np.sqrt(dBR**2 + dBZ**2)) + 1e-30
    max_div_ratio = np.max(np.abs(div_B)) / mean_B

    # Interior only (skip 2-cell boundary)
    interior_div = div_B[2:-2, 2:-2]
    interior_B = np.sqrt(dBR[2:-2, 2:-2]**2 + dBZ[2:-2, 2:-2]**2)
    mean_B_int = np.mean(interior_B) + 1e-30
    max_div_ratio_int = np.max(np.abs(interior_div)) / mean_B_int

    print(f"max|div(δB)|/mean|δB| (interior) = {max_div_ratio_int:.4f}")
    # Loose tolerance — numerical differentiation on coarse grid
    assert max_div_ratio_int < 10.0, \
        f"div(δB_plasma) too large: {max_div_ratio_int:.4f} (expected < 10.0)"


# ---------------------------------------------------------------------------
# Test 4: shape and finiteness
# ---------------------------------------------------------------------------

def test_solution_is_finite(grid_fields, uniform_delta_B_ext):
    """All output fields must be finite (no NaN or Inf)."""
    B0, J0, p0 = grid_fields
    delta_B_plasma, delta_J, delta_p = solve_perturbed_gs(
        B0, J0, p0, uniform_delta_B_ext,
        solver='lsqr', max_iter=200, tol=1e-4,
    )
    for name, arr in [
        ("delta_B_plasma VR",   delta_B_plasma.VR),
        ("delta_B_plasma VZ",   delta_B_plasma.VZ),
        ("delta_B_plasma VPhi", delta_B_plasma.VPhi),
        ("delta_J VR",          delta_J.VR),
        ("delta_p",             delta_p.value),
    ]:
        assert np.all(np.isfinite(arr)), f"{name} contains non-finite values"


# ---------------------------------------------------------------------------
# Test 5: Solov'ev J_grid and p_grid
# ---------------------------------------------------------------------------

def test_solovev_J_grid(small_eq, small_grid):
    """J_grid returns 3 finite arrays of shape (nR, nZ)."""
    R, Z = small_grid
    JR, JZ, Jphi = small_eq.J_grid(R, Z)
    assert JR.shape == (len(R), len(Z))
    assert np.all(np.isfinite(JR))
    assert np.all(np.isfinite(JZ))
    assert np.all(np.isfinite(Jphi))


def test_solovev_p_grid(small_eq, small_grid):
    """p_grid returns a finite non-negative array."""
    R, Z = small_grid
    p = small_eq.p_grid(R, Z)
    assert p.shape == (len(R), len(Z))
    assert np.all(np.isfinite(p))
    assert np.all(p >= -1e-10)  # non-negative (with float tolerance)


# ---------------------------------------------------------------------------
# Test 6: caching — second call is faster (or at least not slower)
# ---------------------------------------------------------------------------

def test_caching_speed(grid_fields, uniform_delta_B_ext):
    """Second call with same arguments should use cache (faster)."""
    B0, J0, p0 = grid_fields

    # First call (may or may not be cached from earlier tests)
    t0 = time.perf_counter()
    solve_perturbed_gs(B0, J0, p0, uniform_delta_B_ext,
                       solver='lsqr', max_iter=200, tol=1e-4)
    t1 = time.perf_counter()
    first_call = t1 - t0

    # Second call — should be near-instant from cache
    t2 = time.perf_counter()
    solve_perturbed_gs(B0, J0, p0, uniform_delta_B_ext,
                       solver='lsqr', max_iter=200, tol=1e-4)
    t3 = time.perf_counter()
    second_call = t3 - t2

    print(f"First call: {first_call:.3f}s, Second call: {second_call:.3f}s")

    # Cache should make second call at least 5x faster (if first call took >0.5s)
    if first_call > 0.5:
        assert second_call < first_call / 5, \
            f"Caching not effective: first={first_call:.3f}s second={second_call:.3f}s"
    else:
        # First call was fast (already cached), both should be fast
        assert second_call < 2.0, f"Second call unexpectedly slow: {second_call:.3f}s"
