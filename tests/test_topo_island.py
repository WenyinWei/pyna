"""Tests for pyna.topo.island — rational surfaces and island widths.

All tests use a synthetic parabolic q profile:
    q(S) = q0 + (q1 - q0) * S^2

with q0 = 1.5, q1 = 4.0.  Rational surfaces are analytically at:
    S_res(m, n) = sqrt((m/n - q0) / (q1 - q0))
"""
import numpy as np
import pytest
from pyna.topo.island import (
    locate_rational_surface,
    locate_all_rational_surfaces,
    island_halfwidth,
    all_rational_q,
    Island,
    IslandChain,
    ChainRole,
)

# ---------------------------------------------------------------------------
# Synthetic q profile
# ---------------------------------------------------------------------------
Q0, Q1 = 1.5, 4.0
S = np.linspace(0.01, 1.0, 500)
q_profile = Q0 + (Q1 - Q0) * S**2


def _analytic_S_res(m: int, n: int) -> float:
    """Analytic S location where q(S) = m/n for the synthetic profile."""
    q_target = m / n
    if not (Q0 <= q_target <= Q1):
        return float("nan")
    return float(np.sqrt((q_target - Q0) / (Q1 - Q0)))


# ---------------------------------------------------------------------------
# locate_rational_surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,n", [(2, 1), (3, 1), (5, 2), (7, 2)])
def test_locate_rational_surface_single(m, n):
    roots = locate_rational_surface(S, q_profile, m, n)
    analytic = _analytic_S_res(m, n)
    assert len(roots) == 1, f"Expected exactly one root for q={m}/{n}"
    assert abs(roots[0] - analytic) < 1e-2, (
        f"q={m}/{n}: root {roots[0]:.4f} vs analytic {analytic:.4f}"
    )


def test_locate_rational_surface_no_root():
    # q < q0 everywhere => no root
    roots = locate_rational_surface(S, q_profile, 1, 1)  # q=1.0 < q0=1.5
    assert len(roots) == 0


def test_locate_rational_surface_with_nans():
    q_with_nans = q_profile.copy()
    q_with_nans[::10] = np.nan  # inject NaNs
    roots = locate_rational_surface(S, q_with_nans, 2, 1)
    analytic = _analytic_S_res(2, 1)
    assert len(roots) == 1
    assert abs(roots[0] - analytic) < 0.05


# ---------------------------------------------------------------------------
# locate_all_rational_surfaces
# ---------------------------------------------------------------------------

def test_locate_all_rational_surfaces_structure():
    result = locate_all_rational_surfaces(S, q_profile, m_max=4, n_max=2)
    # Keys should span [-4, ..., 4]
    assert set(result.keys()) == set(range(-4, 5))
    for m in result:
        assert set(result[m].keys()) == {1, 2}


def test_locate_all_rational_surfaces_known():
    result = locate_all_rational_surfaces(S, q_profile, m_max=6, n_max=2)
    # q = 2/1 = 2.0 should be within [1.5, 4.0]
    assert len(result[2][1]) == 1
    analytic = _analytic_S_res(2, 1)
    assert abs(result[2][1][0] - analytic) < 1e-2


# ---------------------------------------------------------------------------
# island_halfwidth
# ---------------------------------------------------------------------------

def test_island_halfwidth_positive():
    S_res = _analytic_S_res(2, 1)
    # Synthetic tilde_b: constant 1e-3 over S
    tilde_b = np.full_like(S, 1e-3)
    w = island_halfwidth(2, 1, S_res, S, q_profile, tilde_b)
    assert w > 0.0
    assert np.isfinite(w)


def test_island_halfwidth_scales_with_amplitude():
    S_res = _analytic_S_res(2, 1)
    tilde_b_small = np.full_like(S, 1e-4)
    tilde_b_large = np.full_like(S, 4e-4)
    w_small = island_halfwidth(2, 1, S_res, S, q_profile, tilde_b_small)
    w_large = island_halfwidth(2, 1, S_res, S, q_profile, tilde_b_large)
    # w ∝ sqrt(|b|), so 4x amplitude → 2x width
    assert abs(w_large / w_small - 2.0) < 0.1


# ---------------------------------------------------------------------------
# all_rational_q
# ---------------------------------------------------------------------------

def test_all_rational_q_basic():
    result = all_rational_q(4, 2)
    # All returned q values should be unique
    q_vals = [group[0][0] / group[0][1] for group in result]
    assert len(q_vals) == len(set(q_vals))


def test_all_rational_q_filter():
    result = all_rational_q(6, 3, q_min=1.5, q_max=4.0)
    for group in result:
        q = group[0][0] / group[0][1]
        assert 1.5 <= q <= 4.0


def test_all_rational_q_grouping():
    # q = 2/2 = 1/1, they should be in different groups or
    # 2/2 grouped with 1/1... depends on algorithm.
    # m=2,n=2 equals m=1,n=1; verify they share the same group.
    result = all_rational_q(4, 4, q_min=0.9, q_max=1.1)
    q1_groups = [g for g in result if g[0][0] / g[0][1] == 1.0]
    assert len(q1_groups) == 1


# ---------------------------------------------------------------------------
# ChainRole / Island.chain / Island.connected_to / IslandChain new fields
# ---------------------------------------------------------------------------

def test_chain_role_enum_values():
    assert ChainRole.PRIMARY.value == "primary"
    assert ChainRole.SECONDARY.value == "secondary"
    assert ChainRole.NESTED.value == "nested"


def test_island_has_chain_and_connected_to_fields():
    isl = Island(period_n=2, O_point=np.array([3.1, 0.0]))
    assert isl.chain is None
    assert isl.connected_to == []


def test_island_chain_default_role():
    chain = IslandChain(m=2, n=1)
    assert chain.role is ChainRole.PRIMARY
    assert chain.orbit is None
    assert chain.parent_chain is None
    assert chain.primary_chain_ref is None


def test_island_chain_back_links_islands():
    isl0 = Island(period_n=2, O_point=np.array([3.1, 0.05]))
    isl1 = Island(period_n=2, O_point=np.array([3.1, -0.05]))
    chain = IslandChain(m=2, n=1, islands=[isl0, isl1])
    assert isl0.chain is chain
    assert isl1.chain is chain


def test_from_fixed_points_back_link():
    O_pts = [np.array([3.1, 0.05]), np.array([3.1, -0.05])]
    X_pts = [np.array([3.15, 0.0])]
    chain = IslandChain.from_fixed_points(O_pts, X_pts, m=2, n=1)
    for isl in chain.islands:
        assert isl.chain is chain


def test_chain_role_can_be_set():
    chain = IslandChain(m=3, n=1, role=ChainRole.SECONDARY)
    assert chain.role is ChainRole.SECONDARY


def test_chain_export_from_topo():
    from pyna.topo import ChainRole as CR
    assert CR.PRIMARY is ChainRole.PRIMARY
