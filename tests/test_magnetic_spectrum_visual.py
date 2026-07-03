import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyna.toroidal.perturbation_spectrum import (
    RadialPerturbationFourierSpectrum,
    ResonantIslandChain,
    analyze_resonant_island_chains_multi_n,
)
from pyna.toroidal.visual.magnetic_spectrum import (
    apply_radial_mode_overlays,
    apply_rational_surface_overlays,
    PoincareRationalTrace,
    island_bars_on_section,
    overlay_q_profile,
    overlay_island_bars_on_section,
    plot_island_chains_on_section,
    plot_radial_mode_heatmap,
    plot_rational_surface_map,
    plot_resonant_radial_profiles,
    plot_spectrum_bar3d,
    plot_spectrum_heatmap,
    radial_mode_spectrum,
    rational_surface_markers,
    spectrum_surface_matrix,
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


def _shaped_surface():
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    radial = np.array([0.18, 0.25, 0.32, 0.40])
    R0 = 3.0
    rr = radial[None, :, None]
    th = theta[None, None, :]
    R = R0 + rr * np.cos(th) + 0.45 * rr * rr * np.cos(2.0 * th)
    Z = 0.82 * rr * np.sin(th) + 0.30 * rr * rr * rr
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


def _stack_spectrum():
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    radial = np.array([0.2, 0.3, 0.4])
    return RadialPerturbationFourierSpectrum(
        m=np.array([3, -3, 5, -5]),
        n=np.array([-1, 1, -2, 2]),
        dBr=np.column_stack(
            [
                np.array([1.0e-4, 2.0e-4, 1.5e-4]),
                np.array([1.0e-4, 2.0e-4, 1.5e-4]),
                np.array([0.5e-4, 1.2e-4, 1.0e-4]),
                np.array([0.5e-4, 1.2e-4, 1.0e-4]),
            ]
        ),
        dBr_grid=np.zeros((3, phi.size, theta.size), dtype=complex),
        theta=theta,
        phi=phi,
        radial_labels=radial,
    )


def test_island_bars_use_one_bar_per_o_point():
    R, Z, phi, theta, radial = _surface()
    bars = island_bars_on_section(R, Z, phi, theta, radial, [_chain()], phi_section=0.0)

    assert len(bars) == 3
    assert all(np.isfinite([bar.R_O, bar.Z_O, bar.R_inner, bar.Z_inner, bar.R_outer, bar.Z_outer]).all() for bar in bars)
    assert all(bar.R_path is not None and bar.R_path.size == 33 for bar in bars)


def test_island_bars_follow_constant_theta_curves_on_shaped_sections():
    import matplotlib.pyplot as plt

    R, Z, phi, theta, radial = _shaped_surface()
    bars = island_bars_on_section(
        R,
        Z,
        phi,
        theta,
        radial,
        [_chain()],
        phi_section=0.0,
        n_path=21,
    )
    bar = bars[0]

    assert bar.s_path is not None
    assert bar.R_path is not None
    assert bar.Z_path is not None
    assert bar.s_path.size == 21
    np.testing.assert_allclose([bar.R_path[0], bar.Z_path[0]], [bar.R_inner, bar.Z_inner])
    np.testing.assert_allclose([bar.R_path[-1], bar.Z_path[-1]], [bar.R_outer, bar.Z_outer])
    chord = np.array([bar.R_outer - bar.R_inner, bar.Z_outer - bar.Z_inner])
    rel = np.column_stack([bar.R_path - bar.R_inner, bar.Z_path - bar.Z_inner])
    cross = chord[0] * rel[:, 1] - chord[1] * rel[:, 0]
    assert np.nanmax(np.abs(cross)) > 1.0e-6

    fig, ax = plt.subplots()
    artists = overlay_island_bars_on_section(ax, bars, show_labels=False)
    assert artists
    curved_lines = [artist for artist in artists if hasattr(artist, "get_xdata")]
    assert any(len(line.get_xdata()) == 21 for line in curved_lines)
    plt.close(fig)


def test_visual_plots_run_headless():
    R, Z, phi, theta, radial = _surface()
    chain = _chain()
    spectrum = _stack_spectrum()

    fig1, ax1 = plot_spectrum_heatmap(spectrum, radial_index=1, m_max=4, n_max=2, chains=[chain], q_value=3.0)
    fig2, ax2 = plot_resonant_radial_profiles(spectrum, [chain])
    fig3, ax3, bars = plot_island_chains_on_section(R, Z, phi, theta, radial, [chain], phi_section=0.0)
    fig4, ax4 = plot_spectrum_heatmap(spectrum, radial_index=1, m_max=5, n_max=2, renderer="pcolormesh")
    try:
        fig5 = plot_spectrum_bar3d(spectrum, radial_index=1, m_max=5, n_max=2)
        fig5b = plot_spectrum_bar3d(spectrum, radial_index=1, m_max=5, n_max=2, range_mode="nonzero")
    except ImportError:
        fig5 = None
        fig5b = None
    fig6, ax6, radial_map = plot_radial_mode_heatmap(
        spectrum,
        fixed_n=1,
        q_profile=np.array([2.0, 3.0, 4.0]),
        chains=[chain],
    )
    fig7, ax7, _ = plot_radial_mode_heatmap(
        spectrum,
        fixed_n=1,
        q_profile=np.array([2.0, 3.0, 4.0]),
        chains=[chain],
        show_resonance_curve=False,
        show_island_bars=False,
    )
    fig8, ax8, markers = plot_rational_surface_map(
        radial,
        np.array([2.0, 3.0, 4.0]),
        n_values=[1],
        m_values=[3],
        chains=[chain],
        poincare=PoincareRationalTrace(
            ratio=np.array([2.85, 3.0, 3.15]),
            radial_label=np.array([0.28, 0.3, 0.32]),
        ),
    )
    chain_52 = ResonantIslandChain(
        m=5,
        n=2,
        radial_label=0.25,
        q=2.5,
        q_prime=2.0,
        coefficient=1.0e-4,
        b_res=2.0e-4,
        half_width=0.02,
    )
    fig9, ax9, _ = plot_radial_mode_heatmap(
        spectrum,
        fixed_m=5,
        mode_values=np.array([1, 2]),
        axis_convention="fourier",
        q_profile=np.array([2.0, 3.0, 4.0]),
        chains=[chain_52],
    )
    fig10, ax10, _ = plot_radial_mode_heatmap(
        spectrum,
        fixed_n=1,
        mode_values=np.array([-3, 3]),
        resonant_sign=1,
        q_profile=np.array([2.0, 3.0, 4.0]),
        chains=[chain],
    )

    assert fig1 is not None and ax1 is not None
    resonance_lines = [line for line in ax1.lines if "resonant branch" in line.get_label()]
    assert len(resonance_lines) == 1
    np.testing.assert_allclose(resonance_lines[0].get_ydata(), -3.0 * resonance_lines[0].get_xdata())
    assert fig2 is not None and ax2 is not None
    assert fig3 is not None and ax3 is not None
    assert fig4 is not None and ax4 is not None
    if fig5 is not None:
        assert len(fig5.data) == 2
        assert fig5.data[1].type == "scatter3d"
        assert fig5.layout.scene.aspectmode == "manual"
        assert fig5.layout.scene.camera.projection.type == "orthographic"
        assert fig5b.layout.scene.xaxis.range[0] <= -2.0
        assert fig5b.layout.scene.xaxis.range[1] >= -1.0
    assert fig6 is not None and ax6 is not None
    assert fig7 is not None and ax7 is not None
    assert not [line for line in ax7.lines if "q(s)" in line.get_label()]
    assert fig8 is not None and ax8 is not None
    assert [(marker.m, marker.n, marker.radial_label) for marker in markers] == [(3, 1, 0.3)]
    assert fig9 is not None and ax9 is not None
    fourier_curve = [line for line in ax9.lines if "q(s)" in line.get_label()]
    assert len(fourier_curve) == 1
    np.testing.assert_allclose(fourier_curve[0].get_xdata(), -5.0 / np.array([2.0, 3.0, 4.0]))
    assert any(len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [-2.0, -2.0]) for line in ax9.lines)
    assert fig10 is not None and ax10 is not None
    negative_m_curve = [line for line in ax10.lines if "q(s)" in line.get_label()]
    assert len(negative_m_curve) == 1
    np.testing.assert_allclose(negative_m_curve[0].get_xdata(), -np.array([2.0, 3.0, 4.0]))
    assert any(len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [-3.0, -3.0]) for line in ax10.lines)
    assert len(bars) == chain.m
    assert radial_map.fixed_axis == "n"
    assert radial_map.mode_axis == "m"


def test_modular_rational_and_radial_overlays():
    import matplotlib.pyplot as plt

    spectrum = _stack_spectrum()
    chain = _chain()
    radial = np.array([0.2, 0.3, 0.4])
    q_profile = np.array([2.0, 3.0, 4.0])
    trace = PoincareRationalTrace(
        ratio=np.array([2.8, 3.0, 3.2]),
        radial_label=np.array([0.28, 0.3, 0.32]),
    )

    fig, ax = plt.subplots()
    q_line = overlay_q_profile(ax, radial, q_profile)
    payload = apply_rational_surface_overlays(
        ax,
        radial,
        q_profile,
        n_values=[1],
        m_values=[3],
        chains=[chain],
        poincare=trace,
        overlays=("rationals", "points", "bars"),
    )

    assert q_line.get_label() == "q-profile"
    assert [(marker.m, marker.n) for marker in payload["markers"]] == [(3, 1)]
    assert len(payload["rational_surfaces"]) > 0
    assert len(payload["poincare"]) == 1
    assert len(payload["island_bars"]) > 0
    plt.close(fig)

    fig2, ax2, radial_map = plot_radial_mode_heatmap(
        spectrum,
        fixed_n=1,
        q_profile=q_profile,
        chains=[chain],
        poincare=trace,
        overlays=("q", "bars", "points"),
    )
    assert any("q(s)" in line.get_label() for line in ax2.lines)
    assert len(ax2.collections) >= 2  # heatmap plus Poincare scatter
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    overlay_payload = apply_radial_mode_overlays(
        ax3,
        radial_map,
        q_profile=q_profile,
        chains=[chain],
        poincare_ratio=trace.ratio,
        poincare_radial=trace.radial_label,
        overlays="all",
    )
    assert overlay_payload["q_profile"] is not None
    assert len(overlay_payload["island_bars"]) > 0
    assert len(overlay_payload["poincare"]) == 1
    plt.close(fig3)


def test_spectrum_matrix_and_radial_mode_extractors():
    spectrum = _stack_spectrum()

    matrix = spectrum_surface_matrix(spectrum, radial_index=1, m_values=[3, 5], n_values=[-2, -1])
    assert matrix.amplitude.shape == (2, 2)
    np.testing.assert_allclose(matrix.amplitude[0, 1], 2.0e-4)
    np.testing.assert_allclose(matrix.amplitude[1, 0], 1.2e-4)

    fixed_n = radial_mode_spectrum(spectrum, fixed_n=1, mode_values=[3, 5])
    assert fixed_n.coefficient.shape == (3, 2)
    np.testing.assert_allclose(fixed_n.amplitude[:, 0], [1.0e-4, 2.0e-4, 1.5e-4])
    np.testing.assert_allclose(fixed_n.amplitude[:, 1], 0.0)

    signed_n = radial_mode_spectrum(spectrum, fixed_n=1, mode_values=[-3, 3])
    assert signed_n.fourier_m.tolist() == [-3, 3]
    assert signed_n.fourier_n.tolist() == [-1, -1]
    np.testing.assert_allclose(signed_n.amplitude[:, 0], 0.0)
    np.testing.assert_allclose(signed_n.amplitude[:, 1], [1.0e-4, 2.0e-4, 1.5e-4])

    fixed_m = radial_mode_spectrum(spectrum, fixed_m=5, mode_values=[1, 2])
    assert fixed_m.coefficient.shape == (3, 2)
    np.testing.assert_allclose(fixed_m.amplitude[:, 1], [0.5e-4, 1.2e-4, 1.0e-4])

    signed_m = radial_mode_spectrum(spectrum, fixed_m=5, mode_values=[-2, 2])
    assert signed_m.fourier_m.tolist() == [5, 5]
    assert signed_m.fourier_n.tolist() == [2, -2]
    np.testing.assert_allclose(signed_m.amplitude[:, 0], 0.0)
    np.testing.assert_allclose(signed_m.amplitude[:, 1], [0.5e-4, 1.2e-4, 1.0e-4])


def test_multi_n_resonant_chain_analysis():
    spectrum = _stack_spectrum()
    q_profile = np.array([2.0, 3.0, 4.0])

    chains = analyze_resonant_island_chains_multi_n(
        spectrum,
        q_profile,
        n_values=[1, 2],
        m_values={1: [3], 2: [5]},
        min_b_res=1.0e-8,
    )

    modes = {(chain.m, chain.n) for chain in chains}
    assert (3, 1) in modes
    assert (5, 2) in modes
    assert all(chain.half_width > 0.0 for chain in chains)


def test_rational_surface_marker_scan():
    markers = rational_surface_markers(
        radial_labels=np.array([0.2, 0.3, 0.4]),
        q_profile=np.array([2.0, 3.0, 4.0]),
        n_values=[1, 2],
        m_values={1: [3], 2: [5, 6]},
    )

    assert [(marker.m, marker.n, marker.radial_label) for marker in markers] == [
        (5, 2, 0.25),
        (3, 1, 0.3),
        (6, 2, 0.3),
    ]
