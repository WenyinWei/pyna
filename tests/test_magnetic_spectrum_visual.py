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
    IsoThetaBarSpec,
    MagneticSectionOverlaySpec,
    apply_radial_mode_overlays,
    apply_rational_surface_overlays,
    PoincareRationalTrace,
    draw_magnetic_section_overlays,
    island_bars_on_section,
    magnetic_section_overlay_spec,
    magnetic_section_overlay_specs,
    overlay_q_profile,
    overlay_island_bars_on_section,
    plot_magnetic_section_overlay_grid,
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


def _nfp_stack_spectrum():
    phi = np.linspace(0.0, np.pi, 8, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    radial = np.array([0.2, 0.3, 0.4])
    resonant = np.array([1.0, 2.0, 1.5]) * 1.0e-4 * np.exp(0.2j)
    opposite = np.array([0.4, 0.8, 0.6]) * 1.0e-4 * np.exp(-0.35j)
    return RadialPerturbationFourierSpectrum(
        m=np.array([3, -3, 3, -3]),
        n=np.array([-2, 2, 2, -2]),
        dBr=np.column_stack([resonant, resonant.conjugate(), opposite, opposite.conjugate()]),
        dBr_grid=np.zeros((3, phi.size, theta.size), dtype=complex),
        theta=theta,
        phi=phi,
        radial_labels=radial,
        field_periods=2,
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


def test_magnetic_section_overlay_spec_and_renderer_are_composable():
    import matplotlib.pyplot as plt

    R, Z, phi, theta, radial = _shaped_surface()
    chain = _chain()
    spec = magnetic_section_overlay_spec(
        R,
        Z,
        phi,
        theta,
        radial,
        [chain],
        phi_section=phi[1],
        n_path=19,
    )

    assert isinstance(spec, MagneticSectionOverlaySpec)
    assert spec.phi_index == 1
    assert spec.R_section.shape == (radial.size, theta.size)
    assert len(spec.island_bars) == chain.m
    assert isinstance(spec.island_bars[0], IsoThetaBarSpec)
    assert spec.island_bars[0].R_path is not None
    assert spec.island_bars[0].R_path.size == 19

    t = np.linspace(0.0, 1.0, 8)
    poincare = np.column_stack((3.0 + 0.12 * t, 0.05 * np.sin(2.0 * np.pi * t)))
    stable = np.column_stack((3.0 + 0.03 * t, 0.025 * t))
    unstable = np.column_stack((3.0 + 0.03 * t, -0.025 * t))

    fig, ax = plt.subplots()
    payload = draw_magnetic_section_overlays(
        ax,
        spec,
        overlays=("pest", "points", "stable", "unstable", "bars", "xo"),
        poincare_points=poincare,
        stable_segments=[stable],
        unstable_segments=[unstable],
        x_points=np.array([[3.02, 0.01]]),
        o_points=np.array([[3.04, -0.01]]),
        island_bar_kwargs={"show_labels": False, "show_O": False, "show_X": False},
    )

    assert len(payload["pest_grid"]) > 0
    assert payload["poincare"] is not None
    assert len(payload["stable_manifolds"]) == 1
    assert len(payload["unstable_manifolds"]) == 1
    assert len(payload["island_bars"]) == chain.m
    assert len(payload["x_points"]) == 1
    assert len(payload["o_points"]) == 1

    empty_payload = draw_magnetic_section_overlays(
        ax,
        spec,
        overlays=("points", "xo"),
        poincare_points=[],
        x_points=[],
        o_points=[],
    )
    assert empty_payload["poincare"] is None
    assert empty_payload["x_points"] == []
    assert empty_payload["o_points"] == []
    plt.close(fig)


def test_magnetic_section_overlay_grid_builds_compact_2x2_layout():
    import matplotlib.pyplot as plt

    R, Z, phi, theta, radial = _shaped_surface()
    chain = _chain()
    phi_sections = phi[[0, 2, 4, 6]]
    t = np.linspace(0.0, 1.0, 10)
    poincare_by_section = []
    for phase in np.linspace(0.0, np.pi, phi_sections.size):
        points = np.column_stack((3.0 + 0.10 * t, 0.04 * np.sin(2.0 * np.pi * t + phase)))
        poincare_by_section.append((points[:, 0], points[:, 1]))
    x_by_section = [np.array([[3.0 + 0.01 * idx, 0.01]]) for idx in range(phi_sections.size)]
    o_by_section = [np.array([[3.0 + 0.01 * idx, -0.01]]) for idx in range(phi_sections.size)]

    specs = magnetic_section_overlay_specs(
        R,
        Z,
        phi,
        theta,
        radial,
        [chain],
        phi_sections=phi_sections,
        n_path=11,
    )
    assert [spec.phi_index for spec in specs] == [0, 2, 4, 6]

    fig, axes, payloads = plot_magnetic_section_overlay_grid(
        R,
        Z,
        phi,
        theta,
        radial,
        [chain],
        phi_sections=phi_sections,
        poincare_points=poincare_by_section,
        x_points=x_by_section,
        o_points=o_by_section,
        overlays=("pest", "points", "bars", "xo"),
        ncols=2,
        compact=True,
        island_bar_kwargs={"show_labels": False, "show_O": False, "show_X": False},
    )

    assert axes.shape == (2, 2)
    assert len(payloads) == 4
    assert fig.subplotpars.wspace == 0.0
    assert fig.subplotpars.hspace == 0.0
    assert all(payload["poincare"] is not None for payload in payloads)
    assert all(len(payload["island_bars"]) == chain.m for payload in payloads)
    assert all(len(payload["x_points"]) == 1 for payload in payloads)
    assert all(len(payload["o_points"]) == 1 for payload in payloads)
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


def test_nfp_spectrum_visuals_use_signed_nardon_modes_and_labels():
    import matplotlib.pyplot as plt

    spectrum = _nfp_stack_spectrum()
    resonant = spectrum.dBr[:, spectrum.nardon_mode_index(3, -4)]
    opposite = spectrum.dBr[:, spectrum.nardon_mode_index(3, 4)]

    default_matrix = spectrum_surface_matrix(spectrum, radial_index=1, m_values=[3])
    assert default_matrix.n_values[[0, -1]].tolist() == [-4, 4]

    matrix = spectrum_surface_matrix(
        spectrum,
        radial_index=1,
        m_values=[-3, 3],
        n_values=[-4, 4],
    )
    np.testing.assert_allclose(
        matrix.coefficient,
        [[opposite[1].conjugate(), resonant[1].conjugate()], [resonant[1], opposite[1]]],
    )

    negative_branch = radial_mode_spectrum(spectrum, fixed_n=4, mode_values=[-3, 3])
    assert negative_branch.fourier_n.tolist() == [-4, -4]
    np.testing.assert_allclose(
        negative_branch.coefficient,
        np.column_stack([opposite.conjugate(), resonant]),
    )

    positive_branch = radial_mode_spectrum(
        spectrum,
        fixed_n=4,
        mode_values=[-3, 3],
        resonant_sign=1,
    )
    assert positive_branch.fourier_n.tolist() == [4, 4]
    np.testing.assert_allclose(
        positive_branch.coefficient,
        np.column_stack([resonant.conjugate(), opposite]),
    )

    fixed_m = radial_mode_spectrum(spectrum, fixed_m=3)
    assert fixed_m.mode_values.tolist() == [4]
    assert fixed_m.fourier_n.tolist() == [-4]
    np.testing.assert_allclose(fixed_m.coefficient[:, 0], resonant)

    fig, ax = plot_spectrum_heatmap(
        spectrum,
        radial_index=1,
        m_values=[-3, 3],
        n_values=[-4, 4],
        log_scale=False,
        show_island_boxes=False,
    )
    assert ax.get_xlabel() == r"$n_N$"
    plt.close(fig)

    fig, ax, _ = plot_radial_mode_heatmap(
        spectrum,
        fixed_n=4,
        mode_values=[3],
        overlays="none",
        log_scale=False,
    )
    assert "n_0=4" in ax.get_title()
    assert "n_N=-4" in ax.get_title()
    plt.close(fig)


def test_nfp_resonant_radial_profile_uses_chain_nardon_branch():
    import matplotlib.pyplot as plt

    spectrum = _nfp_stack_spectrum()
    resonant = spectrum.dBr[:, spectrum.nardon_mode_index(3, -4)]
    chain = ResonantIslandChain(
        m=3,
        n=4,
        radial_label=0.3,
        q=0.75,
        q_prime=2.0,
        coefficient=resonant[1],
        b_res=2.0 * abs(resonant[1]),
        half_width=0.03,
        coefficient_n=-4,
    )

    fig, ax = plot_resonant_radial_profiles(spectrum, [chain])

    profile_lines = [line for line in ax.lines if line.get_label() == "(3,-4)"]
    assert len(profile_lines) == 1
    np.testing.assert_allclose(profile_lines[0].get_ydata(), 2.0 * np.abs(resonant))
    assert "n_N" in ax.get_ylabel()
    plt.close(fig)


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
