import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyna.toroidal.perturbation_spectrum import (
    RadialPerturbationFourierSpectrum,
    ResonantIslandChain,
)
from pyna.toroidal.visual.magnetic_spectrum import (
    island_bars_on_section,
    plot_island_chains_on_section,
    plot_resonant_radial_profiles,
    plot_spectrum_heatmap,
)


def _surface():
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    radial = np.array([0.2, 0.3, 0.4])
    R0 = 3.0
    R = R0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)
    return R, Z, phi, theta, radial


def _chain():
    return ResonantIslandChain(
        m=3,
        n=1,
        radial_label=0.3,
        q=3.0,
        q_prime=2.0,
        coefficient=np.exp(0.2j) * 1.0e-4,
        b_res=2.0e-4,
        half_width=0.03,
    )


def test_island_bars_use_one_bar_per_o_point():
    R, Z, phi, theta, radial = _surface()
    bars = island_bars_on_section(R, Z, phi, theta, radial, [_chain()], phi_section=0.0)

    assert len(bars) == 3
    assert all(np.isfinite([bar.R_O, bar.Z_O, bar.R_inner, bar.Z_inner, bar.R_outer, bar.Z_outer]).all() for bar in bars)


def test_visual_plots_run_headless():
    R, Z, phi, theta, radial = _surface()
    chain = _chain()
    spectrum = RadialPerturbationFourierSpectrum(
        m=np.array([3, -3]),
        n=np.array([-1, 1]),
        dBr=np.column_stack([np.array([1.0e-4, 2.0e-4, 1.5e-4]), np.zeros(3)]),
        dBr_grid=np.zeros((3, phi.size, theta.size), dtype=complex),
        theta=theta,
        phi=phi,
        radial_labels=radial,
    )

    fig1, ax1 = plot_spectrum_heatmap(spectrum, radial_index=1, m_max=4, n_max=2, chains=[chain])
    fig2, ax2 = plot_resonant_radial_profiles(spectrum, [chain])
    fig3, ax3, bars = plot_island_chains_on_section(R, Z, phi, theta, radial, [chain], phi_section=0.0)

    assert fig1 is not None and ax1 is not None
    assert fig2 is not None and ax2 is not None
    assert fig3 is not None and ax3 is not None
    assert len(bars) == chain.m
