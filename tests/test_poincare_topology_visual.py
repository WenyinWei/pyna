from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyna.toroidal.visual import (
    PoincareCurvedIslandBar,
    PoincareDPKClassification,
    PoincareFixedPointValidation,
    PoincareIslandSecondaryCoordinates,
    PoincareTopologyFigureStyle,
    PoincareTopologyReportPayload,
    PoincareTraceQuality,
    circular_flux_coordinate_island_bars,
    draw_poincare_curved_island_bars,
    draw_poincare_island_secondary_contours,
    draw_poincare_island_secondary_points,
    draw_poincare_island_secondary_trace_lines,
    fixed_point_phase_comparison_markers,
    fixed_point_centered_island_bars,
    plot_poincare_topology_map,
    plot_poincare_topology_payload_map,
    plot_poincare_topology_payload_report,
    plot_poincare_topology_report,
    poincare_dpk_classification,
    poincare_fixed_point_validation,
    poincare_island_secondary_coordinates,
    poincare_topology_report_payload,
    poincare_trace_index_from_counts,
    poincare_trace_quality,
)
from pyna.toroidal.control.boundary_nonlinear_validation import HealedSurfaceSectionChart
from pyna.toroidal.visual.poincare_topology import (
    draw_poincare_phase_response_arrows,
    poincare_phase_response_radial_deltas,
    sample_fixed_s_theta_curve,
)


def _ring_points():
    theta = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    radial = np.repeat([0.30, 0.55, 0.78], theta.size)
    angle = np.tile(theta, 3)
    R = 1.0 + radial * np.cos(angle)
    Z = 0.72 * radial * np.sin(angle)
    return R, Z, radial


def _metrics():
    return [
        SimpleNamespace(eigenvalue_ftle=0.08, spectral_recurrence_min=0.002, recurrent_surface_indicator=1.0),
        SimpleNamespace(eigenvalue_ftle=0.09, spectral_recurrence_min=0.070, recurrent_surface_indicator=0.0),
        SimpleNamespace(eigenvalue_ftle=0.01, spectral_recurrence_min=0.001, recurrent_surface_indicator=1.0),
    ]


def _curved_bar():
    s_path = 0.55 + np.linspace(-0.04, 0.04, 25)
    theta = 0.35 + 0.9 * (s_path - 0.55)
    return PoincareCurvedIslandBar(
        R_path=1.0 + s_path * np.cos(theta),
        Z_path=0.72 * s_path * np.sin(theta),
        mode_m=5,
        mode_n=2,
        radial_label=0.55,
        half_width=0.04,
        amplitude=2.0e-4,
        phase=0.35,
    )


class _ResonantComponentStub:
    m = 3
    n = 1
    psi_res = 0.25
    half_width_r = 0.02
    b_mn = 2.0e-4 * np.exp(1j * 0.3)

    def fixed_points(self, phi):
        return {
            "theta_O": np.asarray([[0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]]),
            "theta_X": np.asarray([[np.pi / 3.0, np.pi, 5.0 * np.pi / 3.0]]),
        }


def _circular_healed_chart():
    def s_theta_to_x(s_theta):
        values = np.asarray(s_theta, dtype=float)
        s = values[..., 0]
        theta = values[..., 1]
        return np.stack([1.0 + s * np.cos(theta), 0.75 * s * np.sin(theta)], axis=-1)

    def x_to_s_theta(x_RZ):
        values = np.asarray(x_RZ, dtype=float)
        u = values[..., 0] - 1.0
        v = values[..., 1] / 0.75
        return np.stack([np.hypot(u, v), np.arctan2(v, u)], axis=-1)

    def jacobian(s_theta):
        s, theta = np.asarray(s_theta, dtype=float)
        return np.asarray(
            [
                [np.cos(theta), -s * np.sin(theta)],
                [0.75 * np.sin(theta), 0.75 * s * np.cos(theta)],
            ]
        )

    return HealedSurfaceSectionChart(
        s_theta_to_x,
        x_to_s_theta,
        jacobian,
        radial_phase="s",
    )


def test_fixed_s_theta_curve_is_a_chart_arc_not_an_endpoint_line():
    chart = _circular_healed_chart()
    path = sample_fixed_s_theta_curve(
        chart,
        s=0.8,
        theta0=0.0,
        delta_theta=0.5 * np.pi,
        n_points=65,
    )

    chord_midpoint = 0.5 * (path[0] + path[-1])
    curve_midpoint = path[path.shape[0] // 2]
    assert np.linalg.norm(curve_midpoint - chord_midpoint) > 0.1
    normalized_radius = np.hypot(path[:, 0] - 1.0, path[:, 1] / 0.75)
    np.testing.assert_allclose(normalized_radius, 0.8, atol=2.0e-15)


def test_phase_response_arrows_select_x_o_and_report_radial_delta_separately():
    import matplotlib.pyplot as plt

    chart = _circular_healed_chart()
    responses = [
        SimpleNamespace(
            kind="X",
            phase_space_valid=True,
            s0=0.75,
            z0=np.asarray([0.75, 0.1]),
            delta_theta_star_wrapped=0.7,
            delta_s=0.025,
        ),
        SimpleNamespace(
            kind="O",
            phase_space_valid=True,
            s0=0.55,
            z0=np.asarray([0.55, 2.0]),
            delta_theta_star_wrapped=-0.4,
            delta_s=-0.015,
        ),
    ]
    fig, ax = plt.subplots()

    x_artists = draw_poincare_phase_response_arrows(
        ax,
        responses,
        chart,
        draw_x=True,
        draw_o=False,
        n_points=41,
    )
    radial = poincare_phase_response_radial_deltas(responses)

    assert len(x_artists) == 1
    assert x_artists[0].phase_response_kind == "X"
    assert x_artists[0].radial_delta_s == pytest.approx(0.025)
    assert x_artists[0].sampled_path_RZ.shape == (41, 2)
    assert set(radial) == {"X", "O"}
    np.testing.assert_allclose(radial["X"], [0.025])
    np.testing.assert_allclose(radial["O"], [-0.015])
    plt.close(fig)


def test_poincare_dpk_classification_keeps_recurrent_surface_out_of_chaos():
    points = [0.30, 0.55, 0.78]

    classification = poincare_dpk_classification(
        points,
        dpk_radial_labels=[0.30, 0.55, 0.78],
        dpk_metrics=_metrics(),
        growth_threshold=0.05,
        recurrence_threshold=0.02,
    )

    assert isinstance(classification, PoincareDPKClassification)
    assert classification.point_chaotic_mask.tolist() == [False, True, False]
    assert classification.point_surface_mask.tolist() == [True, False, True]


def test_poincare_trace_quality_reports_clean_closed_traces():
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = np.array([0.25, 0.50])
    R = radii[:, None] * np.cos(theta)[None, :]
    Z = radii[:, None] * np.sin(theta)[None, :]
    radial = np.broadcast_to(radii[:, None], R.shape)

    quality = poincare_trace_quality(R, Z, radial_label=radial)

    assert quality.n_traces == 2
    assert quality.finite_fraction == pytest.approx(1.0)
    assert quality.lost_trace_fraction == pytest.approx(0.0)
    assert quality.radial_drift_p95 == pytest.approx(0.0)


def test_poincare_trace_quality_flags_lost_points_and_large_jumps():
    R = np.array([[0.0, 0.1, 0.2, np.nan], [0.0, 0.1, 4.0, 4.1]])
    Z = np.array([[0.0, 0.0, 0.0, np.nan], [0.0, 0.0, 0.0, 0.0]])

    quality = poincare_trace_quality(R, Z, jump_threshold=1.0)

    assert quality.finite_fraction < 1.0
    assert quality.lost_trace_fraction == pytest.approx(0.5)
    assert quality.suspicious_jump_fraction > 0.0
    assert quality.max_step > 1.0


def test_draw_poincare_curved_island_bars_runs_headless():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    artists = draw_poincare_curved_island_bars(
        ax,
        [_curved_bar()],
        label_prefix="B0 + delta B predicted",
    )

    assert len(artists) == 2
    assert artists[0].get_label().startswith("B0 + delta B predicted")
    assert len(artists[0].get_xdata()) == 25
    plt.close(fig)

    fig, ax = plt.subplots()
    same_mode_artists = draw_poincare_curved_island_bars(
        ax,
        [_curved_bar(), _curved_bar()],
        colors=["#111111", "#222222"],
        endpoint_markers=False,
    )

    assert len(same_mode_artists) == 2
    assert same_mode_artists[0].get_color() == same_mode_artists[1].get_color()
    plt.close(fig)


def test_circular_flux_coordinate_island_bars_follow_constant_theta_width():
    eq = SimpleNamespace(magnetic_axis=(3.0, 0.0), r0=0.9)

    bars = circular_flux_coordinate_island_bars(
        [_ResonantComponentStub()],
        eq,
        kinds=("O", "X"),
        n_points=5,
    )

    assert len(bars) == 6
    first = bars[0]
    assert isinstance(first, PoincareCurvedIslandBar)
    assert first.mode_m == 3
    assert first.mode_n == 1
    assert first.radial_label == pytest.approx(0.45)
    assert first.half_width == pytest.approx(0.02)
    assert first.kind == "O"
    assert first.branch == 0
    np.testing.assert_allclose(first.Z_path, np.zeros(5), atol=1.0e-14)
    assert first.R_path[0] == pytest.approx(3.0 + 0.43)
    assert first.R_path[-1] == pytest.approx(3.0 + 0.47)


def test_fixed_point_centered_island_bars_anchor_width_at_newton_o_point():
    eq = SimpleNamespace(magnetic_axis=(3.0, 0.0), r0=0.9)
    fixed_points = [
        {"R": 3.5, "Z": 0.0, "kind": "O", "mode_m": 3, "mode_n": 1, "branch": 0, "residual": 1.0e-10, "converged": True},
        {"R": 3.0, "Z": 0.5, "kind": "X", "mode_m": 3, "mode_n": 1, "branch": 0, "residual": 1.0e-10, "converged": True},
        {"R": 3.4, "Z": 0.0, "kind": "O", "mode_m": 3, "mode_n": 1, "branch": 1, "residual": 1.0e-2, "converged": False},
    ]

    bars = fixed_point_centered_island_bars(
        fixed_points,
        [_ResonantComponentStub()],
        eq=eq,
        kinds=("O",),
        n_points=5,
        residual_tol=1.0e-8,
        require_converged_fixed_points=True,
    )

    assert len(bars) == 1
    bar = bars[0]
    assert bar.kind == "O"
    assert bar.mode_m == 3
    assert bar.mode_n == 1
    assert bar.branch == 0
    assert bar.half_width == pytest.approx(0.02)
    assert bar.R_path[2] == pytest.approx(3.5)
    assert bar.Z_path[2] == pytest.approx(0.0)
    assert bar.R_path[0] == pytest.approx(3.48)
    assert bar.R_path[-1] == pytest.approx(3.52)


def test_draw_poincare_curved_island_bars_supports_mode_colormaps_and_gray():
    import matplotlib.pyplot as plt

    eq = SimpleNamespace(magnetic_axis=(3.0, 0.0), r0=0.9)
    bars = circular_flux_coordinate_island_bars(
        [_ResonantComponentStub()],
        eq,
        kinds=("O",),
        n_points=5,
        mode_colormaps={(3, 1): "Greens"},
    )

    fig, ax = plt.subplots()
    artists = draw_poincare_curved_island_bars(
        ax,
        bars[:2],
        endpoint_markers=False,
        mode_colormaps={(3, 1): "Greens"},
    )

    assert artists[0].get_color() != artists[1].get_color()
    plt.close(fig)

    fig, ax = plt.subplots()
    gray_artists = draw_poincare_curved_island_bars(
        ax,
        bars[:2],
        endpoint_markers=False,
        mode_colormaps={(3, 1): "Greens"},
        grayscale=True,
        grayscale_color="#777777",
    )

    assert gray_artists[0].get_color() == "#777777"
    assert gray_artists[1].get_color() == "#777777"
    plt.close(fig)


def test_poincare_island_secondary_coordinates_label_o_axis_and_separatrix():
    eq = SimpleNamespace(magnetic_axis=(3.0, 0.0), r0=0.9)
    component = _ResonantComponentStub()
    r_res = np.sqrt(component.psi_res) * eq.r0
    width = component.half_width_r
    theta_x = np.pi / 3.0
    R = np.asarray([
        3.0 + r_res,
        3.0 + r_res + width,
        3.0 + r_res * np.cos(theta_x),
        3.0 + r_res + 3.0 * width,
    ])
    Z = np.asarray([
        0.0,
        0.0,
        r_res * np.sin(theta_x),
        0.0,
    ])

    coords = poincare_island_secondary_coordinates(R, Z, [component], eq)

    assert isinstance(coords, PoincareIslandSecondaryCoordinates)
    assert coords.inside_island.tolist() == [True, True, True, False]
    assert coords.secondary_radius[0] == pytest.approx(0.0)
    assert coords.secondary_radius[1] == pytest.approx(1.0)
    assert coords.secondary_radius[2] == pytest.approx(1.0)
    assert np.isnan(coords.secondary_radius[3])
    assert coords.mode_m[:3].tolist() == [3, 3, 3]
    assert coords.branch[:3].tolist() == [0, 0, 0]


def test_poincare_island_secondary_coordinates_disambiguates_island_lobes():
    eq = SimpleNamespace(magnetic_axis=(3.0, 0.0), r0=0.9)
    component = _ResonantComponentStub()
    r_res = np.sqrt(component.psi_res) * eq.r0
    theta_o = component.fixed_points(0.0)["theta_O"][0]
    R = 3.0 + r_res * np.cos(theta_o)
    Z = r_res * np.sin(theta_o)

    coords = poincare_island_secondary_coordinates(R, Z, [component], eq)

    np.testing.assert_allclose(coords.secondary_radius, np.zeros(3), atol=1.0e-12)
    assert coords.branch.tolist() == [0, 1, 2]


def test_draw_poincare_island_secondary_points_runs_headless():
    import matplotlib.pyplot as plt

    R = np.asarray([1.0, 1.1, 2.0])
    Z = np.asarray([0.0, 0.1, 0.0])
    coords = PoincareIslandSecondaryCoordinates(
        secondary_radius=np.asarray([0.0, 1.0, np.nan]),
        helical_phase=np.asarray([0.0, np.pi, np.nan]),
        island_hamiltonian=np.asarray([-1.0, 1.0, np.nan]),
        inside_island=np.asarray([True, True, False]),
        mode_m=np.asarray([3, 3, -1]),
        mode_n=np.asarray([1, 1, -1]),
        branch=np.asarray([0, 0, -1]),
    )

    fig, ax = plt.subplots()
    artists = draw_poincare_island_secondary_points(
        ax,
        R,
        Z,
        coords,
        mode_colormaps={(3, 1): "Greens_r"},
    )

    assert len(artists) == 1
    assert artists[0].get_offsets().shape[0] == 2
    plt.close(fig)

    fig, ax = plt.subplots()
    gray_artists = draw_poincare_island_secondary_points(
        ax,
        R,
        Z,
        coords,
        grayscale=True,
        grayscale_color="#777777",
    )

    assert len(gray_artists) == 1
    assert gray_artists[0].get_facecolors().shape[0] >= 1
    plt.close(fig)


def test_draw_poincare_island_secondary_points_preserves_rho_by_default():
    import matplotlib.pyplot as plt

    R = np.asarray([1.0, 1.1, 1.2])
    Z = np.asarray([0.0, 0.1, 0.2])
    rho = np.asarray([0.0, 0.5, 1.0])
    coords = PoincareIslandSecondaryCoordinates(
        secondary_radius=rho,
        helical_phase=np.asarray([0.0, 0.1, 0.2]),
        island_hamiltonian=np.asarray([-1.0, 0.0, 1.0]),
        inside_island=np.ones(3, dtype=bool),
        mode_m=np.asarray([3, 3, 3]),
        mode_n=np.asarray([1, 1, 1]),
        branch=np.asarray([0, 0, 0]),
    )

    fig, ax = plt.subplots()
    artists = draw_poincare_island_secondary_points(
        ax,
        R,
        Z,
        coords,
        mode_colors={(3, 1): "#00ff00"},
        point_size=2.5,
        rho_limits=(0.0, 1.0),
    )

    assert len(artists) == 1
    np.testing.assert_allclose(artists[0].get_array(), rho)
    assert artists[0].norm.vmin == pytest.approx(0.0)
    assert artists[0].norm.vmax == pytest.approx(1.0)
    np.testing.assert_allclose(artists[0].get_sizes(), np.asarray([2.5]))
    plt.close(fig)

    fig, ax = plt.subplots()
    mode_artists = draw_poincare_island_secondary_points(
        ax,
        R,
        Z,
        coords,
        color_by="mode",
        mode_colors={(3, 1): "#00ff00"},
    )

    assert mode_artists[0].get_array() is None
    plt.close(fig)


def test_draw_poincare_island_secondary_contours_runs_headless():
    import matplotlib.pyplot as plt

    grid = np.linspace(-1.0, 1.0, 17)
    x, y = np.meshgrid(grid, grid)
    rho = np.hypot(x.ravel(), y.ravel())
    keep = rho <= 1.0
    R = 3.0 + 0.05 * x.ravel()[keep]
    Z = 0.05 * y.ravel()[keep]
    rho = rho[keep]
    n_points = rho.size
    coords = PoincareIslandSecondaryCoordinates(
        secondary_radius=rho,
        helical_phase=np.arctan2(Z, R - 3.0),
        island_hamiltonian=2.0 * rho * rho - 1.0,
        inside_island=np.ones(n_points, dtype=bool),
        mode_m=np.full(n_points, 5),
        mode_n=np.full(n_points, 2),
        branch=np.zeros(n_points, dtype=int),
    )

    fig, ax = plt.subplots()
    contours = draw_poincare_island_secondary_contours(
        ax,
        R,
        Z,
        coords,
        levels=(0.25, 0.50, 0.75),
        mode_colormaps={(5, 2): "Oranges"},
        auto_triangle_edge_factor=4.0,
    )

    assert contours
    plt.close(fig)


def test_draw_poincare_island_secondary_trace_lines_do_not_cross_traces_or_branches():
    import matplotlib.pyplot as plt

    R = np.asarray([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])
    Z = np.zeros_like(R)
    coords = PoincareIslandSecondaryCoordinates(
        secondary_radius=np.asarray([0.0, 0.4, 0.8, 0.0, 0.4, 0.8]),
        helical_phase=np.zeros(6),
        island_hamiltonian=np.asarray([-1.0, -0.2, 0.6, -1.0, -0.2, 0.6]),
        inside_island=np.ones(6, dtype=bool),
        mode_m=np.full(6, 3),
        mode_n=np.full(6, 1),
        branch=np.asarray([0, 0, 1, 0, 0, 0]),
    )
    trace_index = poincare_trace_index_from_counts([3, 3], n_points=6)

    fig, ax = plt.subplots()
    artists = draw_poincare_island_secondary_trace_lines(
        ax,
        R,
        Z,
        coords,
        trace_index=trace_index,
        max_segment_length=1.0,
    )

    assert len(artists) == 1
    segments = artists[0].get_segments()
    assert len(segments) == 3
    lengths = [float(np.linalg.norm(segment[1] - segment[0])) for segment in segments]
    assert max(lengths) < 1.0
    assert not any(np.isclose(segment[0, 0], 0.2) and np.isclose(segment[1, 0], 10.0) for segment in segments)
    plt.close(fig)


def test_plot_poincare_topology_map_accepts_secondary_layers_and_style():
    import matplotlib.pyplot as plt

    R = np.asarray([1.0, 1.1, 1.2])
    Z = np.asarray([0.0, 0.1, 0.2])
    radial = np.asarray([0.3, 0.4, 0.5])
    coords = PoincareIslandSecondaryCoordinates(
        secondary_radius=np.asarray([0.0, 0.5, 1.0]),
        helical_phase=np.asarray([0.0, 0.1, 0.2]),
        island_hamiltonian=np.asarray([-1.0, 0.0, 1.0]),
        inside_island=np.ones(3, dtype=bool),
        mode_m=np.asarray([3, 3, 3]),
        mode_n=np.asarray([1, 1, 1]),
        branch=np.asarray([0, 0, 0]),
    )

    fig, ax, _classification, _scatter = plot_poincare_topology_map(
        R,
        Z,
        style=PoincareTopologyFigureStyle(figsize=(3.0, 3.0), dpi=220, point_size=0.5),
        radial_label=radial,
        secondary_coordinates=coords,
        secondary_point_kwargs={"point_size": 1.1, "mode_colormaps": {(3, 1): "Greens"}},
        show_background_points=False,
        show_surface_points=False,
        show_chaotic_points=False,
        summary_box=False,
        legend=False,
        title=None,
    )

    assert fig is ax.figure
    assert fig.dpi == pytest.approx(220)
    assert any(collection.get_offsets().shape[0] == 3 for collection in ax.collections)
    plt.close(fig)


def test_pyna_plot_poincare_topology_facade_saves(tmp_path):
    import matplotlib.pyplot as plt
    from pyna.plot import plot_poincare_topology_map as plot_facade

    R, Z, radial = _ring_points()
    out = tmp_path / "poincare_topology.png"
    fig, ax, _classification, _scatter = plot_facade(
        R,
        Z,
        radial_label=radial,
        show_background_points=True,
        show_surface_points=False,
        show_chaotic_points=False,
        summary_box=False,
        legend=False,
        title=None,
        out_path=out,
        save_dpi=240,
    )

    assert fig is ax.figure
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close(fig)


def test_poincare_topology_report_payload_collects_plot_inputs():
    R, Z, radial = _ring_points()
    chain = SimpleNamespace(m=5, n=2, radial_label=0.55, half_width=0.03)

    payload = poincare_topology_report_payload(
        R,
        Z,
        radial_label=radial,
        dpk_radial_labels=[0.30, 0.55, 0.78],
        dpk_metrics=_metrics(),
        fixed_points=[{"R": 1.30, "Z": 0.0, "kind": "O"}],
        island_chains=[chain],
        island_bars=[_curved_bar()],
        trace_counts=[40, 40, 40],
        metadata={"case": "synthetic"},
    )

    assert isinstance(payload, PoincareTopologyReportPayload)
    assert payload.n_points == R.size
    assert payload.has_dpk_profile is True
    assert payload.metadata["case"] == "synthetic"
    assert payload.trace_index.tolist()[:3] == [0, 0, 0]
    assert payload.trace_index.tolist()[-3:] == [2, 2, 2]
    assert payload.map_kwargs()["island_bars"][0].mode_m == 5
    assert payload.report_kwargs()["island_chains"][0] is chain

    with pytest.raises(ValueError, match="dpk_metrics length"):
        poincare_topology_report_payload(
            R,
            Z,
            radial_label=radial,
            dpk_radial_labels=[0.30, 0.55],
            dpk_metrics=_metrics(),
        )
    with pytest.raises(ValueError, match="counts do not sum"):
        poincare_topology_report_payload(R, Z, trace_counts=[1, 2, 3])


def test_plot_poincare_topology_payload_helpers_run_headless(tmp_path):
    import matplotlib.pyplot as plt
    from pyna.plot import plot_poincare_topology_payload_report as plot_facade

    R, Z, radial = _ring_points()
    payload = poincare_topology_report_payload(
        R,
        Z,
        radial_label=radial,
        dpk_radial_labels=[0.30, 0.55, 0.78],
        dpk_metrics=_metrics(),
        island_chains=[SimpleNamespace(m=5, n=2, radial_label=0.55, half_width=0.03)],
        island_bars=[_curved_bar()],
    )

    fig, ax, classification, scatter = plot_poincare_topology_payload_map(
        payload,
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        summary_box=False,
        legend=False,
        title=None,
    )

    assert fig is ax.figure
    assert scatter is not None
    assert np.count_nonzero(classification.point_chaotic_mask) == 40
    plt.close(fig)

    out = tmp_path / "payload_report.png"
    fig, axes, classification = plot_facade(
        payload,
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        out_path=out,
        save_dpi=220,
        title=None,
    )

    assert axes["poincare"].figure is fig
    assert classification.chaotic_intervals
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close(fig)


def test_pyna_plot_top_level_exports_tutorial_poincare_helpers():
    import pyna.plot as plot

    assert plot.PoincareCurvedIslandBar is PoincareCurvedIslandBar
    assert plot.PoincareDPKClassification is PoincareDPKClassification
    assert plot.PoincareFixedPointValidation is PoincareFixedPointValidation
    assert plot.PoincareIslandSecondaryCoordinates is PoincareIslandSecondaryCoordinates
    assert plot.PoincareTopologyFigureStyle is PoincareTopologyFigureStyle
    assert plot.PoincareTopologyReportPayload is PoincareTopologyReportPayload
    assert plot.PoincareTraceQuality is PoincareTraceQuality
    assert plot.draw_poincare_curved_island_bars is draw_poincare_curved_island_bars
    assert plot.draw_poincare_island_secondary_contours is draw_poincare_island_secondary_contours
    assert plot.draw_poincare_island_secondary_points is draw_poincare_island_secondary_points
    assert plot.draw_poincare_island_secondary_trace_lines is draw_poincare_island_secondary_trace_lines
    assert plot.fixed_point_phase_comparison_markers is fixed_point_phase_comparison_markers
    assert plot.fixed_point_centered_island_bars is fixed_point_centered_island_bars
    assert plot.circular_flux_coordinate_island_bars is circular_flux_coordinate_island_bars
    assert callable(plot.plot_poincare_topology_payload_map)
    assert callable(plot.plot_poincare_topology_payload_report)
    assert plot.poincare_dpk_classification is poincare_dpk_classification
    assert plot.poincare_fixed_point_validation is poincare_fixed_point_validation
    assert plot.poincare_island_secondary_coordinates is poincare_island_secondary_coordinates
    assert plot.poincare_topology_report_payload is poincare_topology_report_payload
    assert plot.poincare_trace_index_from_counts is poincare_trace_index_from_counts
    assert plot.poincare_trace_quality is poincare_trace_quality


def test_fixed_point_phase_comparison_markers_carry_newton_validation_metadata():
    row = SimpleNamespace(
        m=5,
        n=2,
        predicted_kind="O",
        branch=2,
        newton_kind="X",
        newton_R=3.2,
        newton_Z=-0.1,
        residual=4.0e-11,
        converged=True,
        theta_error=0.04,
        helical_phase_error=0.12,
        radial_error=-0.003,
        map_span=6.0 * np.pi,
        phi=0.0,
    )

    marker = fixed_point_phase_comparison_markers([row])[0]

    assert marker["kind"] == "X"
    assert marker["residual"] == pytest.approx(4.0e-11)
    assert marker["converged"] is True
    assert marker["mode_m"] == 5
    assert marker["mode_n"] == 2
    assert marker["metadata"]["predicted_kind"] == "O"
    assert marker["metadata"]["mode_m"] == 5
    assert marker["metadata"]["mode_n"] == 2
    assert marker["metadata"]["branch"] == 2


def test_plot_poincare_topology_map_runs_headless():
    import matplotlib.pyplot as plt

    R, Z, radial = _ring_points()
    fig, ax, classification, scatter = plot_poincare_topology_map(
        R,
        Z,
        radial_label=radial,
        dpk_radial_labels=[0.30, 0.55, 0.78],
        dpk_metrics=_metrics(),
        fixed_points=[{"R": 1.55, "Z": 0.0, "kind": "X"}, {"R": 1.30, "Z": 0.0, "kind": "O"}],
        island_bars=[_curved_bar()],
        island_bar_kwargs={"label_prefix": "B0 + delta B predicted"},
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        title="synthetic Poincare topology map",
    )

    assert fig is ax.figure
    assert scatter is not None
    assert np.count_nonzero(classification.point_chaotic_mask) == 40
    assert np.count_nonzero(classification.point_surface_mask) == 80
    assert any(len(line.get_xdata()) == 25 for line in ax.lines)
    plt.close(fig)


def test_poincare_fixed_point_residual_gate_filters_unrefined_markers():
    import matplotlib.pyplot as plt

    R, Z, radial = _ring_points()
    fixed_points = [
        {"R": 1.55, "Z": 0.0, "kind": "X", "residual": 2.0e-10, "converged": True},
        {"R": 1.30, "Z": 0.0, "kind": "O", "residual": 4.0e-4, "converged": True},
        {"R": 1.10, "Z": 0.0, "kind": "O", "metadata": {"residual": 1.0e-10, "converged": False}},
    ]

    validation = poincare_fixed_point_validation(fixed_points, residual_tol=1.0e-8)
    assert isinstance(validation, PoincareFixedPointValidation)
    assert validation.n_total == 3
    assert validation.n_residual_pass == 2
    assert validation.n_residual_fail == 1
    assert validation.n_converged_false == 1

    fig, ax, _classification, _scatter = plot_poincare_topology_map(
        R,
        Z,
        radial_label=radial,
        dpk_radial_labels=[0.30, 0.55, 0.78],
        dpk_metrics=_metrics(),
        fixed_points=fixed_points,
        fixed_point_residual_tol=1.0e-8,
        require_converged_fixed_points=True,
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        title=None,
    )

    labels = ax.get_legend_handles_labels()[1]
    assert "X point" in labels
    assert "O point" not in labels
    plt.close(fig)


def test_plot_poincare_topology_report_runs_headless():
    import matplotlib.pyplot as plt

    R, Z, radial = _ring_points()
    fig, axes, classification = plot_poincare_topology_report(
        R,
        Z,
        radial_label=radial,
        dpk_radial_labels=[0.30, 0.55, 0.78],
        dpk_metrics=_metrics(),
        island_chains=[SimpleNamespace(m=5, n=2, radial_label=0.55, half_width=0.03)],
        island_bars=[_curved_bar()],
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        title="synthetic Poincare report",
    )

    assert axes["poincare"].figure is fig
    assert set(axes) == {"poincare", "growth", "recurrence", "surface"}
    assert classification.chaotic_intervals
    assert len(axes["growth"].lines) >= 2
    assert any(len(line.get_xdata()) == 25 for line in axes["poincare"].lines)
    plt.close(fig)
