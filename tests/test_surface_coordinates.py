import numpy as np
import pytest

from pyna.toroidal.boozer_coords import build_Boozer_coordinates
from pyna.toroidal.perturbation_spectrum import (
    radial_perturbation_Fourier_spectrum,
    radial_perturbation_component,
)
from pyna.toroidal.pest_coords import (
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
    np.testing.assert_allclose(
        result.R_surf[:, 0, :],
        np.repeat(axis_R[:, None] + 0.5, theta.size, axis=1),
    )
    np.testing.assert_allclose(
        result.Z_surf[:, 0, :],
        np.repeat(axis_Z[:, None] + 1.0, theta.size, axis=1),
    )
    np.testing.assert_allclose(
        result.R_surf[:, 1, :],
        np.repeat(axis_R[:, None] + 1.0, theta.size, axis=1),
    )
    np.testing.assert_allclose(
        result.Z_surf[:, 1, :],
        np.repeat(axis_Z[:, None] + 2.0, theta.size, axis=1),
    )


def test_build_Boozer_coordinates_equalizes_B2_jacobian():
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radial = np.array([0.2, 0.4, 0.6])
    R0 = 3.0
    R = np.empty((phi.size, radial.size, theta.size), dtype=np.float64)
    Z = np.empty_like(R)
    for i_r, r in enumerate(radial):
        R[:, i_r, :] = R0 + r * np.cos(theta)[None, :]
        Z[:, i_r, :] = r * np.sin(theta)[None, :]

    B_abs = np.sqrt(1.0 + 0.25 * np.cos(theta))[None, None, :] * np.ones_like(R)
    boozer = build_Boozer_coordinates(
        R,
        Z,
        phi,
        theta,
        radial_labels=radial,
        B_abs=B_abs,
        n_theta=32,
    )

    assert boozer.R_surf.shape == (phi.size, radial.size, 32)
    assert boozer.Z_surf.shape == (phi.size, radial.size, 32)
    assert boozer.theta_B.shape == (32,)
    assert np.all(np.diff(boozer.theta_B_of_theta[0, 1]) > 0.0)
    weighted_jac = boozer.B2_jacobian_B[0, 1]
    assert np.nanstd(weighted_jac) / abs(np.nanmean(weighted_jac)) < 1.0e-12


def test_radial_perturbation_projection_and_Fourier_spectrum():
    phi = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    R0 = 3.0
    r = 0.5
    theta_grid = theta[None, :]
    phi_grid = phi[:, None]
    R = R0 + r * np.cos(theta_grid) * np.ones_like(phi_grid)
    Z = r * np.sin(theta_grid) * np.ones_like(phi_grid)

    m_val = 2
    n_val = -3
    phase = 0.37
    dBr_true = np.cos(m_val * theta_grid + n_val * phi_grid + phase)
    delta_B_R = dBr_true * np.cos(theta_grid)
    delta_B_Z = dBr_true * np.sin(theta_grid)

    dBr_grid = radial_perturbation_component(
        R,
        Z,
        phi,
        theta,
        delta_B_R,
        delta_B_Z,
    )
    np.testing.assert_allclose(dBr_grid, dBr_true, atol=2.0e-3)

    spec = radial_perturbation_Fourier_spectrum(
        dBr_grid,
        theta,
        phi,
        m_max=3,
        n_max=4,
        min_amplitude=1.0e-10,
    )
    idx = np.where((spec.m == m_val) & (spec.n == n_val))[0]
    assert idx.size == 1
    np.testing.assert_allclose(spec.dBr[idx[0]], 0.5 * np.exp(1j * phase), atol=2.0e-3)

    stack = dBr_grid[:, np.newaxis, :] * np.array([1.0, 2.0])[np.newaxis, :, np.newaxis]
    stack_spec = radial_perturbation_Fourier_spectrum(
        stack,
        theta,
        phi,
        m_max=3,
        n_max=4,
        min_amplitude=1.0e-10,
    )
    stack_idx = np.where((stack_spec.m == m_val) & (stack_spec.n == n_val))[0]
    assert stack_idx.size == 1
    assert stack_spec.dBr.shape[0] == 2
    np.testing.assert_allclose(
        stack_spec.dBr[:, stack_idx[0]],
        [0.5 * np.exp(1j * phase), np.exp(1j * phase)],
        atol=2.0e-3,
    )
    with pytest.raises(ValueError, match="radial_index"):
        stack_spec.split(iota=0.31)
