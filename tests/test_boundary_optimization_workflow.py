import numpy as np
import pytest

from pyna.toroidal.control.boundary_optimization_workflow import (
    BoundaryResponseOptimizationResult,
    boundary_response_residual_norm,
    iterative_boundary_response_optimization,
)
from pyna.toroidal.control.boundary_topology_design import boundary_response_observables
from pyna.toroidal.visual.boundary_optimization import (
    BoundaryResponseOptimizationObservableRow,
    BoundaryResponseOptimizationSummary,
    boundary_response_optimization_summary,
    plot_boundary_response_optimization_history,
)


def test_boundary_response_residual_norm_accepts_label_mapped_target():
    rows = boundary_response_observables(
        ["island.width", "chaos.edge"],
        [0.2, 0.4],
        weights=[4.0, 1.0],
    )

    norm = boundary_response_residual_norm(rows, {"island.width": 0.1})

    assert norm == pytest.approx(0.2)


def test_iterative_boundary_response_optimization_hits_linear_target():
    matrix = np.array([[1.0, 0.5], [0.0, 2.0]])
    base = np.array([0.2, -0.1])

    def evaluate(controls):
        return boundary_response_observables(
            ["island.width", "heat.outer"],
            base + matrix @ np.asarray(controls, dtype=float),
            weights=[2.0, 1.0],
        )

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.0, 0.0],
        {"island.width": 0.95, "heat.outer": 0.5},
        control_labels=["spectral_7_3", "strike_trim"],
        steps=[0.1, 0.1],
        n_iterations=2,
        bounds=(-2.0, 2.0),
    )

    assert isinstance(result, BoundaryResponseOptimizationResult)
    assert len(result.steps) == 1 or result.steps[-1].accepted_residual_norm < 1.0e-10
    assert result.controls_by_label["strike_trim"] == pytest.approx(0.3, abs=1.0e-10)
    assert result.controls_by_label["spectral_7_3"] == pytest.approx(0.6, abs=1.0e-10)
    assert result.steps[0].accepted_alpha == pytest.approx(1.0)
    assert result.steps[0].accepted is True


def test_boundary_response_optimization_summary_reports_controls_and_rows():
    matrix = np.array([[1.0, 0.5], [0.0, 2.0]])
    base = np.array([0.2, -0.1])

    def evaluate(controls):
        return boundary_response_observables(
            ["island.width", "heat.outer"],
            base + matrix @ np.asarray(controls, dtype=float),
            weights=[2.0, 1.0],
        )

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.0, 0.0],
        {"island.width": 0.95, "heat.outer": 0.5},
        control_labels=["spectral_7_3", "strike_trim"],
        steps=[0.1, 0.1],
        n_iterations=2,
        bounds=(-2.0, 2.0),
    )

    summary = boundary_response_optimization_summary(result)

    assert isinstance(summary, BoundaryResponseOptimizationSummary)
    assert summary.n_steps >= 1
    assert summary.n_accepted >= 1
    assert summary.n_rejected == 0
    assert summary.final_controls_by_label["spectral_7_3"] == pytest.approx(0.6, abs=1.0e-10)
    assert summary.final_controls_by_label["strike_trim"] == pytest.approx(0.3, abs=1.0e-10)
    assert summary.final_residual_norm < 1.0e-9
    assert summary.residual_reduction > 0.0
    assert summary.residual_reduction_fraction > 0.99
    assert summary.max_condition_number >= 1.0
    assert summary.min_singular_value > 0.0
    assert summary.max_column_correlation >= 0.0
    assert all(isinstance(row, BoundaryResponseOptimizationObservableRow) for row in summary.observable_rows)
    assert {row.label for row in summary.observable_rows} == {"island.width", "heat.outer"}


def test_iterative_boundary_response_optimization_line_search_reduces_nonlinear_step():
    def evaluate(controls):
        u = float(np.asarray(controls, dtype=float)[0])
        return boundary_response_observables(["chaos.layer"], [np.tanh(3.0 * u)])

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.5],
        [0.0],
        control_labels=["spectral_triplet"],
        steps=[1.0e-3],
        n_iterations=1,
        bounds=(-3.0, 3.0),
        line_search=(1.0, 0.5, 0.25),
    )

    step = result.steps[0]
    assert step.line_search_residuals[0][1] > step.current_residual_norm
    assert step.accepted is True
    assert step.accepted_alpha == pytest.approx(0.5)
    assert step.accepted_residual_norm < step.current_residual_norm


def test_iterative_boundary_response_optimization_rejects_failed_trial_and_reduces_alpha():
    def evaluate(controls):
        u = float(np.asarray(controls)[0])
        if u > 0.75:
            raise RuntimeError("Newton manifold solve failed")
        return boundary_response_observables(["chaos.layer"], [u])

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.0],
        [1.0],
        control_labels=["spectral_mode"],
        steps=[0.1],
        n_iterations=1,
        line_search=(1.0, 0.5, 0.25),
    )

    step = result.steps[0]
    assert step.accepted is True
    assert step.accepted_alpha == pytest.approx(0.5)
    assert np.isinf(step.line_search_residuals[0][1])
    assert step.line_search_failures == ((1.0, "RuntimeError: Newton manifold solve failed"),)
    assert result.final_controls[0] == pytest.approx(0.5)


def test_iterative_boundary_response_optimization_respects_absolute_control_bounds():
    def evaluate(controls):
        return boundary_response_observables(["strike.bin"], [float(np.asarray(controls)[0])])

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.4],
        [1.0],
        control_labels=["trim"],
        steps=[0.2],
        n_iterations=2,
        bounds=(-1.0, 1.0),
        control_bounds={"trim": (0.0, 0.5)},
    )

    assert result.final_controls[0] == pytest.approx(0.5)
    assert result.steps[0].accepted is True
    assert result.steps[0].solve.active_upper_bounds == ("trim",)
    assert result.steps[1].solve.controls[0] == pytest.approx(0.0)
    assert result.steps[1].accepted_alpha == pytest.approx(1.0)

    summary = boundary_response_optimization_summary(result, top_n_observables=1)
    assert summary.final_active_upper_bounds == ("trim",)
    assert len(summary.observable_rows) == 1
    assert summary.observable_rows[0].label == "strike.bin"


def test_iterative_boundary_response_optimization_preserves_unspecified_target_rows():
    def evaluate(controls):
        u = float(np.asarray(controls)[0])
        return boundary_response_observables(["drive", "preserve"], [u, 2.0 + 0.5 * u])

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.0],
        {"drive": 1.0},
        control_labels=["mode"],
        steps=[0.1],
        n_iterations=1,
        regularization=0.0,
    )

    assert result.steps[0].target_observables.values.tolist() == pytest.approx([1.0, 2.0])
    assert 0.0 < result.final_controls[0] < 1.0


def test_iterative_boundary_response_optimization_freezes_unspecified_rows_across_iterations():
    def evaluate(controls):
        u = float(np.asarray(controls)[0])
        return boundary_response_observables(["drive", "preserve"], [u, u])

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.0],
        {"drive": 1.0},
        control_labels=["mode"],
        steps=[0.1],
        n_iterations=2,
        line_search=(1.0,),
    )

    assert len(result.steps) == 2
    for step in result.steps:
        np.testing.assert_allclose(step.target_observables.values, [1.0, 0.0])
    np.testing.assert_allclose(result.target_observables.values, [1.0, 0.0])
    assert result.final_controls[0] == pytest.approx(0.5)


def test_plot_boundary_response_optimization_history_runs_headless():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def evaluate(controls):
        u = float(np.asarray(controls)[0])
        return boundary_response_observables(["chaos.layer"], [np.tanh(2.0 * u)])

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.5],
        [0.0],
        control_labels=["spectral_mode"],
        steps=[1.0e-3],
        n_iterations=1,
        bounds=(-2.0, 2.0),
    )

    fig, axes = plot_boundary_response_optimization_history(result)

    assert axes.shape == (2, 2)
    assert len(fig.axes) == 4
    assert [text.get_text() for text in axes[1, 1].texts] == ["no active bounds"]
    plt.close(fig)


def test_pyna_plot_boundary_response_optimization_history_saves(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyna.plot as plot

    def evaluate(controls):
        u = float(np.asarray(controls)[0])
        return boundary_response_observables(["chaos.layer"], [np.tanh(2.0 * u)])

    result = iterative_boundary_response_optimization(
        evaluate,
        [0.5],
        [0.0],
        control_labels=["spectral_mode"],
        steps=[1.0e-3],
        n_iterations=1,
        bounds=(-2.0, 2.0),
    )
    out = tmp_path / "optimization_history.png"

    fig, axes = plot.plot_boundary_response_optimization_history(result, out_path=out, save_dpi=160)
    summary = plot.boundary_response_optimization_summary(result)

    assert axes.shape == (2, 2)
    assert out.exists()
    assert out.stat().st_size > 0
    assert isinstance(summary, BoundaryResponseOptimizationSummary)
    assert plot.BoundaryResponseOptimizationObservableRow is BoundaryResponseOptimizationObservableRow
    assert plot.BoundaryResponseOptimizationSummary is BoundaryResponseOptimizationSummary
    plt.close(fig)
