from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pyna.toroidal.visual.boundary_topology import (
    BoundaryTopologyComparisonSummary,
    DPKRecurrenceProfile,
    boundary_dpk_recurrence_profile,
    boundary_topology_comparison_summary,
    plot_boundary_dpk_recurrence_profile,
    plot_boundary_topology_comparison,
)


def test_boundary_dpk_recurrence_profile_marks_growth_without_recurrence():
    metrics = [
        SimpleNamespace(eigenvalue_ftle=0.09, spectral_recurrence_min=0.07, recurrent_surface_indicator=0.0),
        SimpleNamespace(eigenvalue_ftle=0.00, spectral_recurrence_min=0.00, recurrent_surface_indicator=1.0),
        SimpleNamespace(eigenvalue_ftle=0.02, spectral_recurrence_min=0.00, recurrent_surface_indicator=1.0),
        SimpleNamespace(eigenvalue_ftle=0.08, spectral_recurrence_min=0.05, recurrent_surface_indicator=0.0),
    ]

    profile = boundary_dpk_recurrence_profile(
        [0.6, 0.2, 0.8, 0.4],
        metrics,
        growth_threshold=0.05,
    )

    assert isinstance(profile, DPKRecurrenceProfile)
    assert profile.radial_labels.tolist() == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert profile.chaotic_mask.tolist() == [False, True, True, False]
    np.testing.assert_allclose(np.asarray(profile.chaotic_intervals), [[0.3, 0.7]])


def test_boundary_dpk_recurrence_profile_infers_recurrent_surface_from_recurrence_threshold():
    metrics = [
        {"eigenvalue_ftle": 0.08, "spectral_recurrence_min": 0.005},
        {"eigenvalue_ftle": 0.08, "spectral_recurrence_min": 0.060},
    ]

    profile = boundary_dpk_recurrence_profile(
        [0.2, 0.4],
        metrics,
        growth_threshold=0.05,
        recurrence_threshold=0.02,
    )

    assert profile.recurrent_surface_indicator.tolist() == pytest.approx([1.0, 0.0])
    assert profile.chaotic_mask.tolist() == [False, True]


def test_plot_boundary_dpk_recurrence_profile_runs_headless():
    import matplotlib.pyplot as plt

    metrics = [
        SimpleNamespace(eigenvalue_ftle=0.00, spectral_recurrence_min=0.00, recurrent_surface_indicator=1.0),
        SimpleNamespace(eigenvalue_ftle=0.07, spectral_recurrence_min=0.06, recurrent_surface_indicator=0.0),
        SimpleNamespace(eigenvalue_ftle=0.08, spectral_recurrence_min=0.08, recurrent_surface_indicator=0.0),
    ]

    fig, axes, profile = plot_boundary_dpk_recurrence_profile(
        [0.2, 0.4, 0.6],
        metrics,
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        title="synthetic DPk recurrence profile",
    )

    assert fig is axes[0].figure
    assert profile.chaotic_mask.tolist() == [False, True, True]
    assert len(axes[0].lines) >= 2
    assert len(axes[1].lines) >= 2
    plt.close(fig)


def test_boundary_topology_comparison_summary_tracks_islands_and_chaotic_width():
    before = [
        SimpleNamespace(m=7, n=3, radial_label=0.62, half_width=0.010),
        SimpleNamespace(m=8, n=3, radial_label=0.78, half_width=0.020),
    ]
    after = [
        SimpleNamespace(m=7, n=3, radial_label=0.62, half_width=0.025),
        SimpleNamespace(m=9, n=4, radial_label=0.86, half_width=0.030),
    ]
    before_layers = [SimpleNamespace(inner=0.70, outer=0.75, max_sigma=1.2)]
    after_layers = [SimpleNamespace(inner=0.60, outer=0.72, max_sigma=1.8)]

    summary = boundary_topology_comparison_summary(
        before,
        after,
        baseline_intervals=before_layers,
        perturbed_intervals=after_layers,
        mode_tolerance=1.0e-12,
    )

    assert isinstance(summary, BoundaryTopologyComparisonSummary)
    assert summary.baseline_chain_count == 2
    assert summary.perturbed_chain_count == 2
    assert summary.delta_total_half_width == pytest.approx(0.025)
    assert summary.delta_chaotic_width == pytest.approx(0.07)
    assert summary.strengthened_modes == ((7, 3), (9, 4))
    assert summary.weakened_modes == ((8, 3),)


def test_plot_boundary_topology_comparison_runs_headless():
    import matplotlib.pyplot as plt

    before = [SimpleNamespace(m=7, n=3, radial_label=0.55, half_width=0.010)]
    after = [
        SimpleNamespace(m=7, n=3, radial_label=0.55, half_width=0.020),
        SimpleNamespace(m=8, n=3, radial_label=0.72, half_width=0.028),
    ]
    before_layers = [SimpleNamespace(inner=0.66, outer=0.70, max_sigma=1.1)]
    after_layers = [SimpleNamespace(inner=0.58, outer=0.78, max_sigma=2.1)]
    radial = [0.45, 0.60, 0.75]
    before_dpk = [
        SimpleNamespace(eigenvalue_ftle=0.00, spectral_recurrence_min=0.00, recurrent_surface_indicator=1.0),
        SimpleNamespace(eigenvalue_ftle=0.02, spectral_recurrence_min=0.01, recurrent_surface_indicator=1.0),
        SimpleNamespace(eigenvalue_ftle=0.03, spectral_recurrence_min=0.02, recurrent_surface_indicator=0.0),
    ]
    after_dpk = [
        SimpleNamespace(eigenvalue_ftle=0.01, spectral_recurrence_min=0.00, recurrent_surface_indicator=1.0),
        SimpleNamespace(eigenvalue_ftle=0.08, spectral_recurrence_min=0.06, recurrent_surface_indicator=0.0),
        SimpleNamespace(eigenvalue_ftle=0.09, spectral_recurrence_min=0.08, recurrent_surface_indicator=0.0),
    ]

    fig, axes, summary = plot_boundary_topology_comparison(
        before,
        after,
        baseline_intervals=before_layers,
        perturbed_intervals=after_layers,
        dpk_radial_labels=radial,
        baseline_dpk_metrics=before_dpk,
        perturbed_dpk_metrics=after_dpk,
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        title="synthetic boundary topology response",
    )

    assert axes.shape == (2, 2)
    assert summary.perturbed_max_sigma == pytest.approx(2.1)
    assert summary.delta_chaotic_width == pytest.approx(0.16)
    assert len(axes.ravel()[0].collections) >= 2
    assert len(axes.ravel()[2].lines) >= 3
    plt.close(fig)


def test_boundary_topology_audit_omits_sites_for_grid_field_actuators():
    import matplotlib.pyplot as plt

    from pyna.plot.boundary_topology_case import plot_boundary_topology_control_audit
    from pyna.toroidal.control.boundary_field_basis import (
        boundary_field_actuator_array_from_grid_fields,
        cylindrical_vector_field_from_array,
    )
    from pyna.toroidal.control.boundary_topology_cases import boundary_topology_case_from_arrays

    grid_R = np.linspace(1.0, 2.0, 4)
    grid_Z = np.linspace(-0.5, 0.5, 4)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    field_values = np.zeros((grid_R.size, grid_Z.size, grid_phi.size, 3))
    field_values[..., 2] = 1.0
    unit_field = cylindrical_vector_field_from_array(
        field_values,
        grid_R,
        grid_Z,
        grid_phi,
    )
    actuators = boundary_field_actuator_array_from_grid_fields(
        (unit_field,),
        labels=("arbitrary_grid_field",),
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radial = np.array([0.5, 0.9])
    PP, SS, TT = np.meshgrid(grid_phi, radial, theta, indexing="ij")
    case = boundary_topology_case_from_arrays(
        name="grid-field actuator case",
        R_surf=1.5 + 0.25 * np.sqrt(SS) * np.cos(TT),
        Z_surf=0.30 * np.sqrt(SS) * np.sin(TT),
        phi_vals=grid_phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=np.array([0.62, 0.68]),
        denominator_B3=-np.ones_like(PP),
    )
    state = SimpleNamespace(
        chains=(),
        chaotic_intervals=(),
        heat=None,
        controls=np.array([0.25]),
        metadata={},
    )

    fig, axes = plot_boundary_topology_control_audit(
        case,
        actuators,
        state,
        state,
        modes=(),
    )

    assert axes.shape == (2, 3)
    assert not axes[0, 0].collections
    assert "dipoles" not in axes[0, 0].get_title()
    plt.close(fig)


def test_pyna_plot_exports_stable_high_level_visual_facade():
    import pyna.plot as plot
    import pyna.toroidal.visual as visual

    direct_exports = (
        "BoundaryTopologyComparisonSummary",
        "DPKRecurrenceProfile",
        "boundary_dpk_recurrence_profile",
        "boundary_topology_comparison_summary",
        "plot_boundary_dpk_recurrence_profile",
        "plot_boundary_topology_comparison",
        "plot_heat_distribution_control_history",
        "plot_heat_distribution_control_result",
    )
    for name in direct_exports:
        assert getattr(plot, name) is getattr(visual, name)
        assert name in plot.__all__

    established_facade_entries = (
        "plot_boundary_response_optimization_history",
        "plot_boundary_response_matrix_audit",
        "plot_poincare_topology_payload_report",
        "plot_poincare_topology_report",
    )
    for name in established_facade_entries:
        assert callable(getattr(plot, name))
        assert name in plot.__all__
