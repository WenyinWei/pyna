import numpy as np

from pyna.toroidal.surface_coordinates import (
    circle_map_lift_iota,
    insert_axis_core_surfaces,
    periodic_shift_theta,
    rank_phase_from_axis,
    stitch_periodic,
    theta_coverage,
)


def test_circle_map_lift_iota_recovers_rotation_without_unwrap():
    iota_true = 0.1518
    n = 512
    turn = np.arange(n, dtype=np.float64)
    phase = np.mod(0.37 + 2.0 * np.pi * iota_true * turn, 2.0 * np.pi)

    iota, rms = circle_map_lift_iota(phase, max_iota=0.4)

    assert abs(iota - iota_true) < 1.0e-8
    assert rms < 1.0e-7


def test_rank_phase_from_axis_uses_curve_order():
    theta = np.array([np.pi, 0.0, 1.5 * np.pi, 0.5 * np.pi])
    R = 2.0 + np.cos(theta)
    Z = -0.1 + np.sin(theta)

    phase = rank_phase_from_axis(R, Z, axis_R=2.0, axis_Z=-0.1)

    np.testing.assert_allclose(phase, [np.pi, 0.0, 1.5 * np.pi, 0.5 * np.pi])


def test_stitch_periodic_averages_duplicate_bins_and_wraps():
    theta = np.array([0.0, 0.5 * np.pi, np.pi, np.pi, 1.5 * np.pi])
    values = np.array([1.0, 0.0, -1.0, -3.0, 0.0])
    target = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])

    out = stitch_periodic(theta, values, target, min_points=3)

    np.testing.assert_allclose(out, [1.0, 0.0, -2.0, 0.0])


def test_periodic_shift_theta_shifts_last_axis():
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    values = np.sin(theta)[np.newaxis, :]

    shifted = periodic_shift_theta(values, theta, np.pi / 2.0)

    np.testing.assert_allclose(shifted[0], np.cos(theta), atol=1.0e-14)


def test_theta_coverage_counts_occupied_bins():
    theta = np.array([0.1, 0.2, np.pi + 0.1, np.nan])

    assert theta_coverage(theta, 4) == 0.5


def test_insert_axis_core_surfaces_interpolates_to_first_reliable_surface():
    theta = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    R = np.zeros((2, 2, theta.size), dtype=np.float64)
    Z = np.zeros_like(R)
    axis_R = np.array([10.0, 20.0])
    axis_Z = np.array([1.0, -1.0])
    R[:, 0, :] = axis_R[:, None] + 2.0
    Z[:, 0, :] = axis_Z[:, None] + 4.0
    R[:, 1, :] = axis_R[:, None] + 6.0
    Z[:, 1, :] = axis_Z[:, None] + 8.0

    result = insert_axis_core_surfaces(
        R,
        Z,
        radial_labels=np.array([0.6, 0.9]),
        axis_R=axis_R,
        axis_Z=axis_Z,
        fractions=np.array([0.5, 0.25, 0.5]),
    )

    np.testing.assert_allclose(result.radial_labels, [0.15, 0.3, 0.6, 0.9])
    np.testing.assert_allclose(result.R_surf[:, 0, :], axis_R[:, None] + 0.5)
    np.testing.assert_allclose(result.Z_surf[:, 0, :], axis_Z[:, None] + 1.0)
    np.testing.assert_allclose(result.R_surf[:, 1, :], axis_R[:, None] + 1.0)
    np.testing.assert_allclose(result.Z_surf[:, 1, :], axis_Z[:, None] + 2.0)
