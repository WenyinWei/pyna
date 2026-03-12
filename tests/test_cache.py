"""Tests for pyna centralized cache infrastructure.

Run with:  py -3.13 -m pytest tests/test_cache.py -v
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from pyna.mag.solovev import SolovevEquilibrium
from pyna.control._cached_fpt import CachedFPTAnalyzer
from pyna.cache import eq_hash, array_hash, cache_info, clear_cache


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_eq():
    return SolovevEquilibrium(R0=6.2, a=2.0, B0=5.3, kappa=1.7, delta=0.33, q0=1.5)


def make_field_func(eq):
    def field(rzphi):
        R, Z, phi = float(rzphi[0]), float(rzphi[1]), float(rzphi[2])
        BR, BZ = eq.BR_BZ(R, Z)
        Bphi = eq.Bphi(R)
        Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2) + 1e-30
        return np.array([BR / Bmag, BZ / Bmag, Bphi / (R * Bmag)])
    return field


def make_coil_func(dBR=1e-4, dBZ=0.0):
    def coil(rzphi):
        R = float(rzphi[0])
        Bscale = 5.3
        Bmag = Bscale
        return np.array([dBR / Bmag, dBZ / Bmag, 0.0 / (R * Bmag)])
    return coil


@pytest.fixture
def eq():
    return make_eq()


@pytest.fixture
def field_func(eq):
    return make_field_func(eq)


@pytest.fixture
def analyzer(field_func):
    return CachedFPTAnalyzer(field_func, eq_hash_str='test_eq', eps=1e-4)


# ─── Tests: eq_hash / array_hash ──────────────────────────────────────────────

def test_eq_hash_deterministic():
    h1 = eq_hash(R0=1.86, B0=2.0, kappa=1.7)
    h2 = eq_hash(R0=1.86, B0=2.0, kappa=1.7)
    assert h1 == h2


def test_eq_hash_different_for_different_params():
    h1 = eq_hash(R0=1.86, B0=2.0)
    h2 = eq_hash(R0=1.86, B0=2.1)
    assert h1 != h2


def test_eq_hash_order_independent():
    h1 = eq_hash(R0=1.86, B0=2.0, kappa=1.7)
    h2 = eq_hash(kappa=1.7, B0=2.0, R0=1.86)
    assert h1 == h2


def test_array_hash_deterministic():
    arr = np.array([1.0, 2.0, 3.0])
    h1 = array_hash(arr)
    h2 = array_hash(arr)
    assert h1 == h2


def test_array_hash_different_arrays():
    h1 = array_hash(np.array([1.0, 2.0]))
    h2 = array_hash(np.array([1.0, 3.0]))
    assert h1 != h2


# ─── Tests: CachedFPTAnalyzer.A_matrix ────────────────────────────────────────

def test_A_matrix_returns_2x2(analyzer):
    A = analyzer.A_matrix(6.2, 0.0)
    assert A.shape == (2, 2)


def test_A_matrix_same_as_uncached(field_func, analyzer):
    from pyna.control.fpt import A_matrix as raw_A
    R, Z = 6.2, 0.0
    A_cached = analyzer.A_matrix(R, Z)
    A_raw = raw_A(field_func, R, Z, eps=1e-4)
    np.testing.assert_allclose(A_cached, A_raw, rtol=1e-10)


def test_A_matrix_is_cached_in_memory(analyzer):
    """Second call must return same object (no recomputation)."""
    A1 = analyzer.A_matrix(6.2, 0.0)
    A2 = analyzer.A_matrix(6.2, 0.0)
    assert A1 is A2   # exact same object — no copy


def test_A_matrix_call_count(field_func):
    """Verify A_matrix inner function called only once for repeated queries."""
    call_count = [0]
    original_A = None

    from pyna.control import fpt as fpt_module
    original_A = fpt_module.A_matrix

    def counting_A(ff, R, Z, phi=0.0, eps=1e-4):
        call_count[0] += 1
        return original_A(ff, R, Z, phi, eps)

    analyzer = CachedFPTAnalyzer(field_func, eq_hash_str='count_test2')

    # Pre-populate cache using the real function
    from pyna.control.fpt import A_matrix as _A
    key = (round(6.2, 8), round(0.0, 8), round(0.0, 8))
    analyzer._A_cache[key] = _A(field_func, 6.2, 0.0)

    # Now patch — further calls should hit cache and NOT call counting_A
    fpt_module.A_matrix = counting_A
    try:
        for _ in range(5):
            analyzer.A_matrix(6.2, 0.0)
        assert call_count[0] == 0, f"Expected 0 inner calls, got {call_count[0]}"
    finally:
        fpt_module.A_matrix = original_A


# ─── Tests: DPm ───────────────────────────────────────────────────────────────

def test_DPm_returns_2x2(analyzer):
    DPm = analyzer.DPm(6.2, 0.0)
    assert DPm.shape == (2, 2)


def test_DPm_same_as_uncached(field_func, analyzer):
    from pyna.control.fpt import A_matrix as raw_A, DPm_axisymmetric
    R, Z = 6.2, 0.0
    DPm_cached = analyzer.DPm(R, Z)
    A = raw_A(field_func, R, Z, eps=1e-4)
    DPm_raw = DPm_axisymmetric(A)
    np.testing.assert_allclose(DPm_cached, DPm_raw, rtol=1e-10)


def test_DPm_cached_in_memory(analyzer):
    DPm1 = analyzer.DPm(6.2, 0.0)
    DPm2 = analyzer.DPm(6.2, 0.0)
    assert DPm1 is DPm2


# ─── Tests: coil_field_at ─────────────────────────────────────────────────────

def test_coil_field_cached(analyzer):
    coil = make_coil_func()
    f1 = analyzer.coil_field_at(coil, 6.2, 0.0)
    f2 = analyzer.coil_field_at(coil, 6.2, 0.0)
    assert f1 is f2


def test_coil_field_correct(analyzer):
    coil = make_coil_func(dBR=1e-3)
    f = analyzer.coil_field_at(coil, 6.2, 0.0)
    expected = np.asarray(coil([6.2, 0.0, 0.0]))
    np.testing.assert_array_equal(f, expected)


# ─── Tests: cycle_shift ───────────────────────────────────────────────────────

def test_cycle_shift_zero_for_zero_perturbation(analyzer):
    """Zero perturbation → zero shift."""
    zero_coil = lambda rzphi: np.zeros(3)
    shift = analyzer.cycle_shift(6.2, 0.0, zero_coil)
    np.testing.assert_allclose(shift, 0.0, atol=1e-10)


def test_cycle_shift_linearity(analyzer):
    """Shift should scale linearly with perturbation amplitude."""
    coil1 = make_coil_func(dBR=1e-5)
    coil2 = make_coil_func(dBR=2e-5)
    s1 = analyzer.cycle_shift(6.2, 0.0, coil1)
    s2 = analyzer.cycle_shift(6.2, 0.0, coil2)
    np.testing.assert_allclose(s2, 2 * s1, rtol=1e-3)


# ─── Tests: cache_stats ───────────────────────────────────────────────────────

def test_cache_stats_initial(analyzer):
    stats = analyzer.cache_stats()
    assert stats['A_matrix_entries'] == 0
    assert stats['DPm_entries'] == 0
    assert stats['coil_field_entries'] == 0
    assert stats['manifold_entries'] == 0


def test_cache_stats_count_correctly(analyzer):
    coil = make_coil_func()
    analyzer.A_matrix(6.2, 0.0)
    analyzer.A_matrix(6.2, 0.5)   # different Z
    analyzer.DPm(6.2, 0.0)
    analyzer.coil_field_at(coil, 6.2, 0.0)

    stats = analyzer.cache_stats()
    assert stats['A_matrix_entries'] == 2
    assert stats['DPm_entries'] == 1
    assert stats['coil_field_entries'] == 1


def test_cache_stats_after_multiple_coils(analyzer):
    coils = [make_coil_func(dBR=(i + 1) * 1e-5) for i in range(4)]
    for cf in coils:
        analyzer.coil_field_at(cf, 6.2, 0.0)
    stats = analyzer.cache_stats()
    assert stats['coil_field_entries'] == 4


# ─── Tests: clear_all ────────────────────────────────────────────────────────

def test_clear_all_resets_caches(analyzer):
    coil = make_coil_func()
    analyzer.A_matrix(6.2, 0.0)
    analyzer.DPm(6.2, 0.0)
    analyzer.coil_field_at(coil, 6.2, 0.0)

    analyzer.clear_all()
    stats = analyzer.cache_stats()
    assert stats['A_matrix_entries'] == 0
    assert stats['DPm_entries'] == 0
    assert stats['coil_field_entries'] == 0


def test_clear_coil_cache_only(analyzer):
    coil = make_coil_func()
    analyzer.A_matrix(6.2, 0.0)
    analyzer.coil_field_at(coil, 6.2, 0.0)

    analyzer.clear_coil_cache()
    stats = analyzer.cache_stats()
    # A_matrix preserved, coil cleared
    assert stats['A_matrix_entries'] == 1
    assert stats['coil_field_entries'] == 0


# ─── Tests: SolovevEquilibrium additions ─────────────────────────────────────

def test_find_opoint_near_magnetic_axis(eq):
    R_ax, Z_ax = eq.magnetic_axis
    R_op, Z_op = eq.find_opoint()
    assert abs(R_op - R_ax) < 1e-6
    assert abs(Z_op - Z_ax) < 1e-6


def test_flux_surface_returns_arrays(eq):
    R_fs, Z_fs = eq.flux_surface(0.5)
    assert len(R_fs) > 10
    assert len(R_fs) == len(Z_fs)


def test_flux_surface_psi_close_to_target(eq):
    psi_norm_target = 0.5
    R_fs, Z_fs = eq.flux_surface(psi_norm_target)
    psi_vals = eq.psi(R_fs, Z_fs)
    np.testing.assert_allclose(psi_vals, psi_norm_target, atol=0.02)


def test_cache_info_returns_dict():
    info = cache_info()
    assert 'cache_dir' in info
    assert 'total_size_mb' in info
    assert 'n_entries' in info
    assert info['total_size_mb'] >= 0
