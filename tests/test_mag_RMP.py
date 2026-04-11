"""Tests for pyna.mag.RMP RMP spectrum and island-width pipeline."""
import numpy as np
import pytest
from pyna.toroidal.equilibrium.axisymmetric import EquilibriumTokamakCircularSynthetic
from pyna.toroidal.coils.RMP import normalize_b, RMP_spectrum_2d, island_width_at_rational_surfaces


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def eq():
    return EquilibriumTokamakCircularSynthetic(nS=60, nTET=64)


# ---------------------------------------------------------------------------
# normalize_b
# ---------------------------------------------------------------------------

def test_normalize_b_scalar():
    result = normalize_b(0.01, 2.0)
    assert abs(result - 0.005) < 1e-14


def test_normalize_b_array():
    B = np.array([0.01, 0.02, 0.03])
    B0 = np.array([2.0, 2.0, 2.0])
    result = normalize_b(B, B0)
    np.testing.assert_allclose(result, [0.005, 0.010, 0.015], atol=1e-14)


# ---------------------------------------------------------------------------
# RMP_spectrum_2d
# ---------------------------------------------------------------------------

def test_RMP_spectrum_shape():
    nS, nT, nP = 30, 64, 128
    tilde_b = np.random.default_rng(0).random((nS, nT, nP))
    spec = RMP_spectrum_2d(tilde_b)
    assert spec.shape == (nS, nT, nP // 2 + 1)


def test_RMP_spectrum_single_mode():
    """A pure (m=2, n=1) sinusoidal perturbation should produce a spike."""
    nS, nT, nP = 20, 64, 64
    theta = np.linspace(0, 2 * np.pi, nT, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, nP, endpoint=False)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")
    amplitude = 1e-3
    mode = amplitude * np.cos(2 * TT - 1 * PP)
    tilde_b = np.tile(mode[None, :, :], (nS, 1, 1))
    spec = RMP_spectrum_2d(tilde_b)
    # For cos(2θ - φ), the rfft2 packs negative-m at index nT-m=62,
    # positive n=1 in the rfft output.  Check that the mode is prominent.
    pos_m = np.abs(spec[:, 2, 1])       # positive m=2
    neg_m = np.abs(spec[:, nT - 2, 1])  # negative m �?index nT-2=62
    dominant = np.maximum(pos_m, neg_m)
    # Zero out those two contributions from all_others
    all_others = np.abs(spec.copy())
    all_others[:, 2, 1] = 0.0
    all_others[:, nT - 2, 1] = 0.0
    assert np.all(dominant > np.max(np.abs(all_others)))


# ---------------------------------------------------------------------------
# Full pipeline: synthetic equilibrium + RMP perturbation �?island widths
# ---------------------------------------------------------------------------

def _build_synthetic_RMP(eq, epsilon=1e-3, n_mode=1, m_mode=2, sigma=0.1):
    """Construct a synthetic (m, n) perturbation localized near its rational surface."""
    nS = len(eq.S)
    nT = len(eq.TET)
    nP = 64  # enough resolution for m=2, n=1
    phi = np.linspace(0, 2 * np.pi, nP, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, nT, endpoint=False)
    S = eq.S
    q = eq.q(S)
    # Find approximate S_res for q = m_mode / n_mode
    from scipy.interpolate import interp1d
    try:
        S_res = float(interp1d(q, S)(m_mode / n_mode))
    except Exception:
        S_res = 0.5
    # Gaussian envelope in S centred on rational surface
    envelope = np.exp(-((S - S_res) ** 2) / sigma**2)  # (nS,)
    # cos(m*theta - n*phi) perturbation
    TT, PP = np.meshgrid(theta, phi, indexing="ij")  # (nT, nP)
    mode = np.cos(m_mode * TT - n_mode * PP)          # (nT, nP)
    # tilde_b shape: (nS, nT, nP)
    tilde_b = epsilon * envelope[:, None, None] * mode[None, :, :]
    return tilde_b


def test_full_pipeline_island_widths(eq):
    tilde_b = _build_synthetic_RMP(eq, epsilon=1e-3, sigma=0.05)
    spec = RMP_spectrum_2d(tilde_b)
    widths = island_width_at_rational_surfaces(spec, eq, m_max=4, n_max=2)

    # For q=2/1, there should be a rational surface
    assert 2 in widths
    assert 1 in widths[2]
    w_list = widths[2][1]
    if w_list:  # surface found
        w = w_list[0]
        assert np.isfinite(w), "Island width should be finite"
        # Physically reasonable: between 0 and 30% of minor radius
        assert 0.0 < w < 0.3, f"Island width {w:.4f} out of expected range"

