import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyna.toroidal.control.heat_distribution import (
    FieldLineDiffusionSpec,
    diffuse_wall_heat_distribution,
    fusionsc_trace_endpoints_cylindrical,
    iterative_heat_distribution_control,
    solve_heat_distribution_control,
    wall_heat_distribution_observables,
    wall_heat_flux_metrics,
    wall_heat_flux_observables,
    wall_heat_footprint_from_fusionsc_trace,
)
from pyna.toroidal.visual.heat_distribution import (
    plot_heat_distribution_control_history,
    plot_heat_distribution_control_result,
)


def test_field_line_diffusion_proxy_spreads_heat_and_preserves_total():
    heat = np.zeros((9, 11), dtype=float)
    heat[4, 5] = 10.0
    spec = FieldLineDiffusionSpec(sigma_phi_bins=1.0, sigma_s_bins=1.5)

    diffused = diffuse_wall_heat_distribution(heat, spec)

    assert diffused.shape == heat.shape
    assert diffused[4, 5] < heat[4, 5]
    assert np.count_nonzero(diffused > 0.0) > 1
    assert float(np.sum(diffused)) == pytest.approx(float(np.sum(heat)))


def test_wall_heat_distribution_observables_coarsen_and_normalize():
    heat = np.arange(16, dtype=float).reshape(4, 4)

    rows = wall_heat_distribution_observables(heat, coarse_shape=(2, 2), normalize=True)

    assert rows.labels == (
        "heat.bin.p00.s00",
        "heat.bin.p00.s01",
        "heat.bin.p01.s00",
        "heat.bin.p01.s01",
    )
    assert rows.values.sum() == pytest.approx(1.0)
    assert rows.values[0] == pytest.approx(float(np.sum(heat[:2, :2])) / float(np.sum(heat)))


def test_wall_heat_distribution_observables_normalize_unequal_cell_power():
    heat_flux = np.ones((2, 2), dtype=float)
    cell_areas = np.array([[1.0, 2.0], [3.0, 4.0]])

    power_rows = wall_heat_distribution_observables(
        heat_flux,
        cell_areas=cell_areas,
        normalize=False,
    )
    normalized_rows = wall_heat_distribution_observables(
        heat_flux,
        cell_areas=cell_areas,
        coarse_shape=(1, 2),
        normalize=True,
    )

    np.testing.assert_allclose(power_rows.values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(normalized_rows.values, [0.4, 0.6])


def test_wall_heat_flux_metrics_measure_power_peak_centroid_and_width():
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    s = np.linspace(0.0, 1.0, 101)
    heat = np.exp(-0.5 * ((s[None, :] - 0.62) / 0.08) ** 2)
    heat = np.broadcast_to(heat, (phi.size, s.size)).copy()
    area = np.broadcast_to((2.0 * np.pi / phi.size) * np.gradient(s)[None, :], heat.shape)

    metrics = wall_heat_flux_metrics(heat, phi_values=phi, s_values=s, cell_areas=area)
    rows = wall_heat_flux_observables(
        heat,
        phi_values=phi,
        s_values=s,
        cell_areas=area,
        quantities=("total_power", "peak_flux", "centroid_s", "rms_width_s", "fwhm_s"),
    )

    assert metrics.peak_flux == pytest.approx(1.0)
    assert metrics.centroid_s == pytest.approx(0.62, abs=2.0e-5)
    assert metrics.rms_width_s == pytest.approx(0.08, rel=2.0e-4)
    assert rows.labels == (
        "heat.total_power",
        "heat.peak_flux",
        "heat.centroid_s",
        "heat.rms_width_s",
        "heat.fwhm_s",
    )
    assert rows.values[2] == pytest.approx(metrics.centroid_s)


def test_solve_heat_distribution_control_hits_linear_target():
    base = np.full((3, 3), 2.0)
    basis_a = np.zeros_like(base)
    basis_a[0, 0] = 1.0
    basis_a[2, 2] = -1.0
    basis_b = np.zeros_like(base)
    basis_b[1, :] = [0.5, -1.0, 0.5]
    target = base + 0.4 * basis_a - 0.2 * basis_b

    result = solve_heat_distribution_control(
        base,
        target,
        [base + basis_a, base + basis_b],
        [base - basis_a, base - basis_b],
        steps=1.0,
        normalize=False,
        control_labels=["trim_a", "trim_b"],
        bounds=(-1.0, 1.0),
    )

    assert result.solve.success is True
    assert result.solve.controls_by_label["trim_a"] == pytest.approx(0.4)
    assert result.solve.controls_by_label["trim_b"] == pytest.approx(-0.2)
    np.testing.assert_allclose(result.predicted_heat, target, atol=1.0e-12)


def test_iterative_heat_distribution_control_uses_callback_response():
    base = np.full((2, 3), 3.0)
    basis_a = np.array([[1.0, -1.0, 0.0], [0.0, 0.5, -0.5]])
    basis_b = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    target = base + 0.3 * basis_a + 0.6 * basis_b

    def evaluate(controls):
        return base + float(controls[0]) * basis_a + float(controls[1]) * basis_b

    history = iterative_heat_distribution_control(
        evaluate,
        [0.0, 0.0],
        target,
        control_labels=["a", "b"],
        steps=[0.1, 0.1],
        n_iterations=1,
        normalize=False,
        bounds=(-1.0, 1.0),
    )

    assert len(history) == 1
    np.testing.assert_allclose(history[0].controls_after, [0.3, 0.6], atol=1.0e-12)
    np.testing.assert_allclose(history[0].result.predicted_heat, target, atol=1.0e-12)


def test_iterative_heat_distribution_control_respects_absolute_control_bounds():
    base = np.full((2, 2), 2.0)
    basis = np.array([[1.0, -1.0], [0.5, -0.5]])
    target = base + 2.0 * basis

    def evaluate(controls):
        return base + float(controls[0]) * basis

    history = iterative_heat_distribution_control(
        evaluate,
        [0.3],
        target,
        control_labels=["trim"],
        steps=[0.25],
        n_iterations=2,
        normalize=False,
        bounds=(-1.0, 1.0),
        control_bounds={"trim": (0.0, 0.5)},
    )

    assert history[0].controls_after[0] == pytest.approx(0.5)
    assert history[0].result.solve.active_upper_bounds == ("trim",)
    assert history[1].controls_after[0] == pytest.approx(0.5)
    assert all(step.controls_after[0] <= 0.5 + 1.0e-12 for step in history)


def test_iterative_heat_distribution_control_supports_exactly_fixed_control():
    base = np.full((2, 2), 2.0)
    locked_basis = np.array([[0.5, -0.5], [0.25, -0.25]])
    free_basis = np.array([[1.0, -1.0], [0.5, -0.5]])
    target = base + 0.4 * free_basis

    def evaluate(controls):
        return base + float(controls[0]) * locked_basis + float(controls[1]) * free_basis

    history = iterative_heat_distribution_control(
        evaluate,
        [0.0, 0.0],
        target,
        control_labels=["locked", "free"],
        steps=[0.1, 0.1],
        n_iterations=1,
        normalize=False,
        bounds=(-1.0, 1.0),
        control_bounds={"locked": (0.0, 0.0), "free": (-1.0, 1.0)},
    )

    step = history[0]
    np.testing.assert_allclose(step.controls_after, [0.0, 0.4], atol=1.0e-12)
    np.testing.assert_allclose(step.result.predicted_heat, target, atol=1.0e-12)
    assert "locked" in step.result.solve.active_lower_bounds
    assert "locked" in step.result.solve.active_upper_bounds


def test_fusionsc_trace_dict_adapter_extracts_endpoints_and_bins_wall_heat():
    phi = np.array([0.0, 0.5 * np.pi, np.pi])
    trace_result = {
        "endPoints": np.array(
            [
                np.cos(phi),
                np.sin(phi),
                np.array([0.0, 0.02, -0.02]),
                np.ones(phi.size),
            ]
        )
    }

    R, Z, phi_out = fusionsc_trace_endpoints_cylindrical(trace_result)

    assert R == pytest.approx(np.ones(phi.size))
    assert Z == pytest.approx([0.0, 0.02, -0.02])
    assert phi_out == pytest.approx(phi)

    wall_phi = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    wall_R = np.broadcast_to(1.0 + 0.08 * np.cos(theta), (wall_phi.size, theta.size))
    wall_Z = np.broadcast_to(0.08 * np.sin(theta), (wall_phi.size, theta.size))

    footprint = wall_heat_footprint_from_fusionsc_trace(
        trace_result,
        wall_phi,
        wall_R,
        wall_Z,
        n_phi_bins=4,
        n_s_bins=8,
        field_period=2.0 * np.pi,
    )

    assert footprint.heat.shape == (4, 8)
    assert float(np.sum(footprint.heat)) == pytest.approx(3.0)


def test_plot_heat_distribution_control_result_runs_headless():
    import matplotlib.pyplot as plt

    base = np.full((3, 3), 2.0)
    basis = np.zeros_like(base)
    basis[0, 0] = 1.0
    basis[2, 2] = -1.0
    target = base + 0.5 * basis
    result = solve_heat_distribution_control(
        base,
        target,
        [base + basis],
        [base - basis],
        normalize=False,
        control_labels=["trim"],
    )

    fig, axes = plot_heat_distribution_control_result(result)

    assert axes.shape == (2, 2)
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_plot_heat_distribution_control_history_runs_headless():
    import matplotlib.pyplot as plt

    base = np.full((2, 3), 2.0)
    basis = np.array([[1.0, -1.0, 0.0], [0.0, 0.4, -0.4]])
    target = base + 0.5 * basis

    def evaluate(controls):
        return base + float(controls[0]) * basis

    history = iterative_heat_distribution_control(
        evaluate,
        [0.0],
        target,
        control_labels=["trim"],
        steps=[0.2],
        n_iterations=2,
        normalize=False,
        bounds=(-1.0, 1.0),
    )

    fig, axes = plot_heat_distribution_control_history(history)

    assert axes.shape == (2, 3)
    assert len(fig.axes) >= 6
    plt.close(fig)
