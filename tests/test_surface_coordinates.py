import numpy as np
import pytest

from pyna.toroidal.boozer_coords import build_Boozer_coordinates
from pyna.toroidal.perturbation_spectrum import (
    ResonantIslandChain,
    analyze_resonant_island_chains,
    chirikov_overlaps,
    nardon_radial_perturbation,
    radial_perturbation_Fourier_spectrum,
    radial_perturbation_component,
    sample_cylindrical_vector_grid_on_surfaces,
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


def test_nardon_radial_spectrum_island_phase_and_width():
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    radial = np.linspace(0.2, 0.5, 5)
    R0 = 3.0
    theta_grid = theta[None, None, :]
    phi_grid = phi[:, None, None]
    radial_grid = radial[None, :, None]
    R = R0 + radial_grid * np.cos(theta_grid) * np.ones_like(phi_grid)
    Z = radial_grid * np.sin(theta_grid) * np.ones_like(phi_grid)

    m_val = 5
    n_val = 2
    phase = 0.37
    amp_profile = 1.0e-3 * (1.0 + radial)
    tilde = amp_profile[None, :, None] * np.cos(m_val * theta_grid - n_val * phi_grid + phase)

    # For circular surfaces with s = minor radius, grad(s) is the radial unit
    # vector.  Choose B0^3 = B_phi / R = 2, so delta B^1 = 2 tilde_b^1.
    delta_B1 = 2.0 * tilde
    delta_BR = delta_B1 * np.cos(theta_grid)
    delta_BZ = delta_B1 * np.sin(theta_grid)
    B0_phi = 2.0 * R
    tilde_calc = nardon_radial_perturbation(
        R,
        Z,
        phi,
        theta,
        delta_BR,
        delta_BZ,
        None,
        radial,
        denominator_B_phi=B0_phi,
    )
    np.testing.assert_allclose(tilde_calc, tilde, atol=2.0e-12)

    spec = radial_perturbation_Fourier_spectrum(
        tilde_calc,
        theta,
        phi,
        radial_labels=radial,
        m_max=6,
        n_max=3,
        min_amplitude=1.0e-12,
    )
    idx = spec.mode_index(m_val, -n_val)
    assert idx is not None
    np.testing.assert_allclose(
        spec.dBr[:, idx],
        0.5 * amp_profile * np.exp(1j * phase),
        atol=2.0e-12,
    )

    q_profile = 2.0 + 2.0 * radial
    chains = analyze_resonant_island_chains(
        spec,
        q_profile,
        n=n_val,
        m_values=[m_val],
    )
    assert len(chains) == 1
    chain = chains[0]
    assert chain.radial_label == pytest.approx(0.25)
    assert chain.q == pytest.approx(2.5)
    assert chain.q_prime == pytest.approx(2.0)
    assert chain.coefficient_n == -n_val
    assert chain.b_res == pytest.approx(1.0e-3 * 1.25)
    expected_width = np.sqrt(4.0 * chain.q**2 * chain.b_res / (chain.q_prime * chain.m))
    assert chain.half_width == pytest.approx(expected_width)

    phase_shift = 0.6
    theta_before = chain.fixed_points(0.0)["theta_O"][0, 0]
    theta_after = chain.with_phase_shift(phase_shift).fixed_points(0.0)["theta_O"][0, 0]
    wrapped_delta = np.angle(np.exp(1j * (theta_after - theta_before)))
    assert wrapped_delta == pytest.approx(-phase_shift / m_val)


def test_resonant_chain_uses_forward_helicity_not_opposite_branch():
    radial = np.linspace(0.2, 0.5, 5)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    theta_grid = theta[None, None, :]
    phi_grid = phi[:, None, None]

    m_val = 5
    n_val = 2
    phase_forward = 0.37
    phase_opposite = -1.1
    forward = 1.0e-3 * np.cos(m_val * theta_grid - n_val * phi_grid + phase_forward)
    opposite = 8.0e-3 * np.cos(m_val * theta_grid + n_val * phi_grid + phase_opposite)
    tilde = forward + opposite
    tilde = tilde * np.ones((phi.size, radial.size, theta.size), dtype=float)

    spec = radial_perturbation_Fourier_spectrum(
        tilde,
        theta,
        phi,
        radial_labels=radial,
        m_max=6,
        n_max=3,
        min_amplitude=1.0e-12,
    )
    q_profile = 2.0 + 2.0 * radial
    chains = analyze_resonant_island_chains(
        spec,
        q_profile,
        n=n_val,
        m_values=[m_val],
    )

    assert len(chains) == 1
    chain = chains[0]
    assert chain.q == pytest.approx(float(m_val) / float(n_val))
    assert chain.coefficient_n == -n_val
    np.testing.assert_allclose(chain.coefficient, 0.5e-3 * np.exp(1j * phase_forward), atol=2.0e-12)
    assert chain.b_res == pytest.approx(1.0e-3)
    opposite_idx = spec.mode_index(m_val, n_val)
    assert opposite_idx is not None
    np.testing.assert_allclose(spec.dBr[:, opposite_idx], 4.0e-3 * np.exp(1j * phase_opposite), atol=2.0e-12)


def test_resonant_chain_uses_signed_branch_for_negative_q_profile():
    radial = np.linspace(0.2, 0.5, 5)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    theta_grid = theta[None, None, :]
    phi_grid = phi[:, None, None]

    m_val = 5
    n_val = 2
    phase_negative_q = 0.41
    phase_opposite = -0.9
    selected = 1.0e-3 * np.cos(m_val * theta_grid + n_val * phi_grid + phase_negative_q)
    opposite = 8.0e-3 * np.cos(m_val * theta_grid - n_val * phi_grid + phase_opposite)
    tilde = selected + opposite
    tilde = tilde * np.ones((phi.size, radial.size, theta.size), dtype=float)

    spec = radial_perturbation_Fourier_spectrum(
        tilde,
        theta,
        phi,
        radial_labels=radial,
        m_max=6,
        n_max=3,
        min_amplitude=1.0e-12,
    )
    q_profile = -(2.0 + 2.0 * radial)
    chains = analyze_resonant_island_chains(
        spec,
        q_profile,
        n=n_val,
        m_values=[m_val],
    )

    assert len(chains) == 1
    chain = chains[0]
    assert chain.q == pytest.approx(-float(m_val) / float(n_val))
    assert chain.coefficient_n == n_val
    np.testing.assert_allclose(chain.coefficient, 0.5e-3 * np.exp(1j * phase_negative_q), atol=2.0e-12)
    assert chain.b_res == pytest.approx(1.0e-3)
    opposite_idx = spec.mode_index(m_val, -n_val)
    assert opposite_idx is not None
    np.testing.assert_allclose(spec.dBr[:, opposite_idx], 4.0e-3 * np.exp(1j * phase_opposite), atol=2.0e-12)

    phase_shift = 0.5
    theta_before = chain.fixed_points(0.0)["theta_O"][0, 0]
    theta_after = chain.with_phase_shift(phase_shift).fixed_points(0.0)["theta_O"][0, 0]
    wrapped_delta = np.angle(np.exp(1j * (theta_after - theta_before)))
    assert wrapped_delta == pytest.approx(-phase_shift / m_val)


def test_chirikov_overlap_between_adjacent_chains():
    left = ResonantIslandChain(
        m=5,
        n=2,
        radial_label=0.25,
        q=2.5,
        q_prime=2.0,
        coefficient=1.0e-4 + 0.0j,
        b_res=2.0e-4,
        half_width=0.03,
    )
    right = ResonantIslandChain(
        m=6,
        n=2,
        radial_label=0.40,
        q=3.0,
        q_prime=2.0,
        coefficient=2.0e-4 + 0.0j,
        b_res=4.0e-4,
        half_width=0.04,
    )

    overlaps = chirikov_overlaps([right, left])

    assert len(overlaps) == 1
    assert overlaps[0].modes == ((5, 2), (6, 2))
    assert overlaps[0].separation == pytest.approx(0.15)
    assert overlaps[0].sigma == pytest.approx((0.03 + 0.04) / 0.15)


def test_sample_cylindrical_vector_grid_on_surfaces():
    grid_R = np.linspace(2.5, 3.5, 6)
    grid_Z = np.linspace(-0.4, 0.4, 5)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    phi = grid_phi.copy()
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radial = np.array([0.1, 0.2])
    R = 3.0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)

    RR, ZZ, PP = np.meshgrid(grid_R, grid_Z, grid_phi, indexing="ij")
    field_R = RR + 2.0 * ZZ + np.cos(PP)
    field_phi = 2.0 * RR - ZZ + np.sin(PP)
    field_Z = -RR + 0.5 * ZZ + np.cos(PP)

    out_R, out_phi, out_Z = sample_cylindrical_vector_grid_on_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        field_R,
        field_phi,
        field_Z,
        R,
        Z,
        phi,
        theta,
    )
    expected_phi = phi[:, None, None]
    np.testing.assert_allclose(out_R, R + 2.0 * Z + np.cos(expected_phi), atol=1.0e-12)
    np.testing.assert_allclose(out_phi, 2.0 * R - Z + np.sin(expected_phi), atol=1.0e-12)
    np.testing.assert_allclose(out_Z, -R + 0.5 * Z + np.cos(expected_phi), atol=1.0e-12)


def test_sample_cylindrical_vector_grid_on_surfaces_respects_field_periods():
    nfp = 2
    field_period = 2.0 * np.pi / nfp
    grid_R = np.linspace(0.8, 1.2, 5)
    grid_Z = np.linspace(-0.2, 0.2, 5)
    grid_phi = np.linspace(0.0, field_period, 32, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radial = np.array([0.05])
    R = 1.0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)

    RR, ZZ, PP = np.meshgrid(grid_R, grid_Z, grid_phi, indexing="ij")
    field_R = RR + ZZ + np.cos(2.0 * PP)
    field_phi = 2.0 + np.sin(2.0 * PP)
    field_Z = RR - ZZ + 0.25 * np.cos(4.0 * PP)

    out_R, out_phi, out_Z = sample_cylindrical_vector_grid_on_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        field_R,
        field_phi,
        field_Z,
        R,
        Z,
        phi,
        theta,
        field_periods=nfp,
    )

    expected_phi = np.broadcast_to(phi[:, None, None], R.shape)
    np.testing.assert_allclose(out_R, R + Z + np.cos(2.0 * expected_phi), atol=1.0e-2)
    np.testing.assert_allclose(out_phi, 2.0 + np.sin(2.0 * expected_phi), atol=1.0e-2)
    np.testing.assert_allclose(out_Z, R - Z + 0.25 * np.cos(4.0 * expected_phi), atol=1.0e-2)
