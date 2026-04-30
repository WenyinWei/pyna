"""Tests for pyna.draw.resonant — RMP spectrum visualisation.

These tests use matplotlib's Agg backend so they run headless.
"""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyna.draw.resonant import plot_tilde_b_mn_spectrum, bar3d_tilde_b_mn_on_surface


@pytest.fixture()
def synthetic_spectrum():
    """Synthetic (n_modes, nS) tilde_b_mn_S array."""
    rng = np.random.default_rng(0)
    nS = 40
    modes = [(2, 1), (3, 1), (4, 1), (3, 2), (5, 2)]
    data = rng.uniform(1e-5, 1e-2, (len(modes), nS))
    m_vals = [m for m, n in modes]
    n_vals = [n for m, n in modes]
    S = np.linspace(0.01, 1.0, nS)
    return data, S, m_vals, n_vals


def test_plot_spectrum_runs(synthetic_spectrum):
    """plot_tilde_b_mn_spectrum should return fig, ax without error."""
    data, S, m_vals, n_vals = synthetic_spectrum
    fig, ax = plot_tilde_b_mn_spectrum(data, S, m_vals, n_vals)
    assert fig is not None
    assert ax is not None


def test_plot_spectrum_linear_scale(synthetic_spectrum):
    data, S, m_vals, n_vals = synthetic_spectrum
    fig, ax = plot_tilde_b_mn_spectrum(data, S, m_vals, n_vals, log_scale=False)
    assert fig is not None


def test_bar3d_runs():
    """bar3d_tilde_b_mn_on_surface should return fig, ax without error."""
    rng = np.random.default_rng(1)
    m_max, n_max = 5, 3
    data = rng.uniform(1e-5, 1e-2, (m_max + 1, n_max + 1))
    fig, ax = bar3d_tilde_b_mn_on_surface(data, m_max, n_max)
    assert fig is not None


def test_bar3d_with_S_index():
    """bar3d with 3-D input selects the correct S slice."""
    rng = np.random.default_rng(2)
    nS, m_max, n_max = 10, 4, 2
    data = rng.uniform(1e-5, 1e-2, (nS, m_max + 1, n_max + 1))
    fig, ax = bar3d_tilde_b_mn_on_surface(data, m_max, n_max, S_value=3)
    assert fig is not None
