import numpy as np
import pytest

from pyna.toroidal.boozer_coords import build_Boozer_coordinates
from pyna.toroidal.perturbation_spectrum import (
    ChaoticLayerInterval,
    ChirikovOverlapBand,
    IntegrableFieldDecomposition,
    MagneticCoordinateProfile,
    ResonantIslandChain,
    ResonantSurfaceGroup,
    analyze_resonant_island_chains,
    chaotic_layer_intervals,
    chirikov_overlap_bands,
    chirikov_overlaps,
    coalesce_resonant_island_chains,
    contravariant_radial_component,
    cylindrical_field_grid_signature,
    group_resonant_island_chains,
    integrable_field_decomposition_from_grids,
    nardon_radial_perturbation,
    nardon_radial_perturbation_from_decomposition,
    nardon_radial_perturbation_from_healed_surfaces,
    periodic_orbit_geometry_distance,
    periodic_orbit_surface_alignment,
    radial_perturbation_Fourier_spectrum,
    radial_perturbation_component,
    require_matching_field_signature,
    sample_cylindrical_vector_grid_on_surfaces,
    surface_coordinate_signature,
    surface_field_alignment_diagnostics,
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


def test_nardon_decomposition_uses_background_denominator_and_keeps_provenance():
    grid_R = np.linspace(2.6, 3.4, 9)
    grid_Z = np.linspace(-0.4, 0.4, 7)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    RR, ZZ, PP = np.meshgrid(grid_R, grid_Z, grid_phi, indexing="ij")
    R0 = 3.0
    B0_R = np.zeros_like(RR)
    B0_Z = np.zeros_like(RR)
    B0_phi = 2.0 * RR
    delta_BR = RR - R0
    delta_BZ = ZZ
    delta_Bphi = np.zeros_like(RR)

    phi = grid_phi.copy()
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    radial = np.array([0.1, 0.2, 0.3])
    R = R0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)

    projection = nardon_radial_perturbation_from_decomposition(
        grid_R,
        grid_Z,
        grid_phi,
        B0_R,
        B0_phi,
        B0_Z,
        delta_BR,
        delta_Bphi,
        delta_BZ,
        R,
        Z,
        phi,
        theta,
        radial,
    )

    expected = 0.5 * radial[None, :, None] * np.ones_like(R)
    np.testing.assert_allclose(projection.tilde_b1, expected, atol=2.0e-12)
    assert projection.background_field_signature is not None
    assert projection.delta_field_signature is not None
    assert projection.surface_signature is not None

    spectrum = projection.fourier_spectrum(m_max=0, n_max=0, min_amplitude=1.0e-14)
    assert spectrum.background_field_signature == projection.background_field_signature
    assert spectrum.delta_field_signature == projection.delta_field_signature
    assert spectrum.surface_signature == projection.surface_signature
    idx = spectrum.mode_index(0, 0)
    assert idx is not None
    np.testing.assert_allclose(spectrum.dBr[:, idx], 0.5 * radial, atol=2.0e-12)


def test_surface_field_alignment_diagnostics_detects_mismatched_surfaces():
    grid_R = np.linspace(2.5, 3.5, 17)
    grid_Z = np.linspace(-0.5, 0.5, 15)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    RR, ZZ, _PP = np.meshgrid(grid_R, grid_Z, grid_phi, indexing="ij")
    R0 = 3.0
    iota = 0.37
    Bphi = RR.copy()
    BR_tangent = -iota * ZZ
    BZ_tangent = iota * (RR - R0)

    phi = grid_phi.copy()
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radial = np.array([0.1, 0.2, 0.3])
    R = R0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)

    aligned = surface_field_alignment_diagnostics(
        grid_R,
        grid_Z,
        grid_phi,
        BR_tangent,
        Bphi,
        BZ_tangent,
        R,
        Z,
        phi,
        theta,
        radial,
        iota_profile=np.full(radial.shape, iota),
    )
    assert aligned.global_radial_ratio_rms < 1.0e-11
    assert aligned.iota_profile_error_rms < 1.0e-3

    radial_leak = 0.2
    BR_leaky = BR_tangent + radial_leak * (RR - R0)
    BZ_leaky = BZ_tangent + radial_leak * ZZ
    mismatched = surface_field_alignment_diagnostics(
        grid_R,
        grid_Z,
        grid_phi,
        BR_leaky,
        Bphi,
        BZ_leaky,
        R,
        Z,
        phi,
        theta,
        radial,
        iota_profile=np.full(radial.shape, iota),
    )
    assert mismatched.global_radial_ratio_rms > 1.0e-2
    assert not mismatched.is_field_aligned


def test_healed_surface_projection_uses_total_radial_flux_not_reference_residual():
    grid_R = np.linspace(2.5, 3.5, 17)
    grid_Z = np.linspace(-0.5, 0.5, 15)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    RR, ZZ, _PP = np.meshgrid(grid_R, grid_Z, grid_phi, indexing="ij")
    R0 = 3.0
    iota = 0.37
    Bphi = RR.copy()
    BR_tangent = -iota * ZZ
    BZ_tangent = iota * (RR - R0)

    phi = grid_phi.copy()
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radial = np.array([0.1, 0.2, 0.3])
    R = R0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)

    total_leak = 0.2
    reference_leak = 0.05
    BR_total = BR_tangent + total_leak * (RR - R0)
    BZ_total = BZ_tangent + total_leak * ZZ
    BR_reference = BR_tangent + reference_leak * (RR - R0)
    BZ_reference = BZ_tangent + reference_leak * ZZ

    projection = nardon_radial_perturbation_from_healed_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        BR_total,
        Bphi,
        BZ_total,
        R,
        Z,
        phi,
        theta,
        radial,
        denominator_B_R=BR_reference,
        denominator_B_phi=Bphi,
        denominator_B_Z=BZ_reference,
    )

    total_BR_s, total_BPhi_s, total_BZ_s = sample_cylindrical_vector_grid_on_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        BR_total,
        Bphi,
        BZ_total,
        R,
        Z,
        phi,
        theta,
    )
    ref_BR_s, ref_BPhi_s, ref_BZ_s = sample_cylindrical_vector_grid_on_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        BR_reference,
        Bphi,
        BZ_reference,
        R,
        Z,
        phi,
        theta,
    )
    total_B1, _ = contravariant_radial_component(R, Z, phi, theta, total_BR_s, total_BZ_s, total_BPhi_s, radial)
    ref_B1, _ = contravariant_radial_component(R, Z, phi, theta, ref_BR_s, ref_BZ_s, ref_BPhi_s, radial)

    np.testing.assert_allclose(projection.delta_B1, total_B1, atol=2.0e-12)
    assert np.nanmean(np.abs(projection.delta_B1 - (total_B1 - ref_B1))) > 1.0e-3
    np.testing.assert_allclose(np.nanmean(projection.tilde_b1, axis=(0, 2)), total_leak * radial, atol=3.0e-3)


def test_periodic_orbit_surface_alignment_scores_field_line_on_healed_surface():
    phi = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    radial = np.array([0.2, 0.3, 0.4])
    R0 = 3.0
    R = R0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)

    target_s = 0.3
    iota = 0.25
    orbit_phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    orbit_theta = 0.4 + iota * orbit_phi
    orbit_R = R0 + target_s * np.cos(orbit_theta)
    orbit_Z = target_s * np.sin(orbit_theta)

    alignment = periodic_orbit_surface_alignment(
        orbit_R,
        orbit_Z,
        orbit_phi,
        R,
        Z,
        phi,
        theta,
        radial,
        target_radial_label=target_s,
        iota=iota,
    )

    assert alignment.n_points == orbit_phi.size
    assert alignment.radial_error_rms < 1.0e-12
    assert alignment.surface_distance_rms < 2.5e-2
    assert alignment.fieldline_phase_rms < 4.0e-2


def test_surface_and_profile_signatures_reject_mismatched_background_fields():
    grid_R = np.linspace(2.8, 3.2, 5)
    grid_Z = np.linspace(-0.2, 0.2, 4)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    RR, ZZ, PP = np.meshgrid(grid_R, grid_Z, grid_phi, indexing="ij")
    background_a = cylindrical_field_grid_signature(
        grid_R,
        grid_Z,
        grid_phi,
        np.zeros_like(RR),
        2.0 * RR,
        np.zeros_like(RR),
    )
    background_b = cylindrical_field_grid_signature(
        grid_R,
        grid_Z,
        grid_phi,
        np.zeros_like(RR),
        3.0 * RR,
        np.zeros_like(RR),
    )
    with pytest.raises(ValueError, match="sha256=") as excinfo:
        require_matching_field_signature(background_a, background_b, context="test field")
    assert "2.0" not in str(excinfo.value)
    assert "3.0" not in str(excinfo.value)

    phi = grid_phi.copy()
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radial = np.array([0.1, 0.2, 0.3])
    R = 3.0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)
    surface_a = surface_coordinate_signature(
        R,
        Z,
        phi,
        theta,
        radial,
        background_field_signature=background_a,
    )
    surface_b = surface_coordinate_signature(
        R,
        Z,
        phi,
        theta,
        radial,
        background_field_signature=background_b,
    )

    q_profile = MagneticCoordinateProfile(
        "q",
        radial,
        np.array([1.5, 2.0, 2.5]),
        surface_signature=surface_a,
        background_field_signature=background_a,
    )
    spectrum = radial_perturbation_Fourier_spectrum(
        np.ones((radial.size, phi.size, theta.size), dtype=float),
        theta,
        phi,
        radial_labels=radial,
        m_max=0,
        n_max=0,
        metadata={
            "surface_signature": surface_a,
            "background_field_signature": background_a,
        },
    )
    chains = analyze_resonant_island_chains(
        spectrum,
        q_profile,
        n=1,
        m_values=[2],
    )
    assert chains == []

    foreign_q_profile = MagneticCoordinateProfile(
        "q",
        radial,
        np.array([1.5, 2.0, 2.5]),
        surface_signature=surface_b,
        background_field_signature=background_b,
    )
    with pytest.raises(ValueError, match="q_profile surface signature mismatch"):
        analyze_resonant_island_chains(
            spectrum,
            foreign_q_profile,
            n=1,
            m_values=[2],
        )


def _synthetic_decomposition_fixture():
    grid_R = np.linspace(2.2, 3.2, 5)
    grid_Z = np.linspace(-0.45, 0.35, 7)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    assert len({grid_R.size, grid_Z.size, grid_phi.size}) == 3
    RR, ZZ, PP = np.meshgrid(grid_R, grid_Z, grid_phi, indexing="ij")

    background_R = 0.02 * ZZ * np.sin(PP)
    background_phi = 2.0 * RR + 0.05 * np.cos(PP)
    background_Z = -0.015 * ZZ * np.cos(PP)
    delta_R = 1.0e-3 * np.cos(2.0 * PP) + 2.0e-4 * ZZ
    delta_phi = 4.0e-4 * np.sin(PP)
    delta_Z = 8.0e-4 * np.sin(3.0 * PP) - 1.0e-4 * RR
    total_R = background_R + delta_R
    total_phi = background_phi + delta_phi
    total_Z = background_Z + delta_Z

    phi = np.linspace(0.0, 2.0 * np.pi, 11, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    radial = np.array([0.12, 0.21, 0.34, 0.48])
    assert len({phi.size, theta.size, radial.size}) == 3
    R = 2.7 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, phi.size, axis=0)
    Z = np.repeat(Z, phi.size, axis=0)
    return {
        "grid_R": grid_R,
        "grid_Z": grid_Z,
        "grid_phi": grid_phi,
        "background": (background_R, background_phi, background_Z),
        "delta": (delta_R, delta_phi, delta_Z),
        "total": (total_R, total_phi, total_Z),
        "surface": (R, Z, phi, theta, radial),
    }


def test_integrable_field_decomposition_contract_binds_profiles_and_spectrum():
    fixture = _synthetic_decomposition_fixture()
    background_sig = cylindrical_field_grid_signature(
        fixture["grid_R"],
        fixture["grid_Z"],
        fixture["grid_phi"],
        *fixture["background"],
    )
    R, Z, phi, theta, radial = fixture["surface"]
    surface_sig = surface_coordinate_signature(
        R,
        Z,
        phi,
        theta,
        radial,
        background_field_signature=background_sig,
    )
    q_profile = MagneticCoordinateProfile(
        "q",
        radial,
        1.8 + 1.5 * radial,
        surface_signature=surface_sig,
        background_field_signature=background_sig,
    )

    decomp = integrable_field_decomposition_from_grids(
        fixture["grid_R"],
        fixture["grid_Z"],
        fixture["grid_phi"],
        *fixture["total"],
        *fixture["background"],
        *fixture["delta"],
        surface_signature=surface_sig,
        q_profile=q_profile,
    )

    assert isinstance(decomp, IntegrableFieldDecomposition)
    assert decomp.residual_summary["max_abs"] < 1.0e-14
    assert len(decomp.digest) == 64
    decomp.require_profile(q_profile)

    tilde = np.ones((phi.size, radial.size, theta.size), dtype=float)
    spectrum = radial_perturbation_Fourier_spectrum(
        tilde,
        theta,
        phi,
        radial_labels=radial,
        m_max=0,
        n_max=0,
        metadata={
            "surface_signature": surface_sig,
            "background_field_signature": decomp.background_field_signature,
            "delta_field_signature": decomp.delta_field_signature,
        },
    )
    decomp.require_spectrum(spectrum)


def test_integrable_field_decomposition_rejects_mismatched_artifacts_without_leaking_metadata():
    fixture = _synthetic_decomposition_fixture()
    private_token = "redacted/source/path/should/not/leak"
    background_sig = cylindrical_field_grid_signature(
        fixture["grid_R"],
        fixture["grid_Z"],
        fixture["grid_phi"],
        *fixture["background"],
    )
    shifted_background = (
        fixture["background"][0],
        fixture["background"][1] + 0.1,
        fixture["background"][2],
    )
    foreign_background_sig = cylindrical_field_grid_signature(
        fixture["grid_R"],
        fixture["grid_Z"],
        fixture["grid_phi"],
        *shifted_background,
        metadata={"source": private_token},
    )
    R, Z, phi, theta, radial = fixture["surface"]
    foreign_surface_sig = surface_coordinate_signature(
        R,
        Z,
        phi,
        theta,
        radial,
        background_field_signature=foreign_background_sig,
    )

    with pytest.raises(ValueError, match="surface background field signature mismatch") as excinfo:
        integrable_field_decomposition_from_grids(
            fixture["grid_R"],
            fixture["grid_Z"],
            fixture["grid_phi"],
            *fixture["total"],
            *fixture["background"],
            *fixture["delta"],
            surface_signature=foreign_surface_sig,
        )
    assert private_token not in str(excinfo.value)

    decomp = integrable_field_decomposition_from_grids(
        fixture["grid_R"],
        fixture["grid_Z"],
        fixture["grid_phi"],
        *fixture["total"],
        *fixture["background"],
        *fixture["delta"],
    )
    good_surface_sig = surface_coordinate_signature(
        R,
        Z,
        phi,
        theta,
        radial,
        background_field_signature=background_sig,
    )
    wrong_delta = (
        fixture["delta"][0] * 1.1,
        fixture["delta"][1],
        fixture["delta"][2],
    )
    wrong_delta_sig = cylindrical_field_grid_signature(
        fixture["grid_R"],
        fixture["grid_Z"],
        fixture["grid_phi"],
        *wrong_delta,
        metadata={"source": private_token},
    )
    bad_spectrum = radial_perturbation_Fourier_spectrum(
        np.ones((phi.size, radial.size, theta.size), dtype=float),
        theta,
        phi,
        radial_labels=radial,
        m_max=0,
        n_max=0,
        metadata={
            "surface_signature": good_surface_sig,
            "background_field_signature": decomp.background_field_signature,
            "delta_field_signature": wrong_delta_sig,
        },
    )

    with pytest.raises(ValueError, match="spectrum delta field signature mismatch") as excinfo:
        decomp.require_spectrum(bad_spectrum)
    assert private_token not in str(excinfo.value)


def test_integrable_field_decomposition_validates_explicit_delta_residual_and_grid_identity():
    fixture = _synthetic_decomposition_fixture()
    bad_delta = (
        fixture["delta"][0] + 1.0e-3,
        fixture["delta"][1],
        fixture["delta"][2],
    )
    with pytest.raises(ValueError, match="explicit delta field is not consistent"):
        integrable_field_decomposition_from_grids(
            fixture["grid_R"],
            fixture["grid_Z"],
            fixture["grid_phi"],
            *fixture["total"],
            *fixture["background"],
            *bad_delta,
            residual_atol=1.0e-15,
            residual_rtol=0.0,
        )

    total_sig = cylindrical_field_grid_signature(
        fixture["grid_R"],
        fixture["grid_Z"],
        fixture["grid_phi"],
        *fixture["total"],
    )
    background_sig = cylindrical_field_grid_signature(
        fixture["grid_R"],
        fixture["grid_Z"],
        fixture["grid_phi"],
        *fixture["background"],
    )
    foreign_phi = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    RR, ZZ, PP = np.meshgrid(fixture["grid_R"], fixture["grid_Z"], foreign_phi, indexing="ij")
    delta_sig = cylindrical_field_grid_signature(
        fixture["grid_R"],
        fixture["grid_Z"],
        foreign_phi,
        1.0e-4 * RR,
        2.0e-4 * np.ones_like(RR),
        3.0e-4 * ZZ * np.cos(PP),
    )

    with pytest.raises(ValueError, match="delta field grid signature mismatch"):
        IntegrableFieldDecomposition(
            total_field_signature=total_sig,
            background_field_signature=background_sig,
            delta_field_signature=delta_sig,
        )


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
    assert spec.nardon_mode_index(m_val, -n_val) is not None
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
    assert spec.nardon_mode_index(m_val, n_val) is not None
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


def test_chirikov_overlap_bands_include_cross_toroidal_families():
    left = ResonantIslandChain(
        m=5,
        n=2,
        radial_label=0.25,
        q=2.5,
        q_prime=2.0,
        coefficient=1.0e-4 + 0.0j,
        b_res=2.0e-4,
        half_width=0.06,
    )
    middle = ResonantIslandChain(
        m=8,
        n=3,
        radial_label=0.34,
        q=8.0 / 3.0,
        q_prime=2.0,
        coefficient=1.0e-4 + 0.0j,
        b_res=2.0e-4,
        half_width=0.05,
    )
    right = ResonantIslandChain(
        m=6,
        n=2,
        radial_label=0.50,
        q=3.0,
        q_prime=2.0,
        coefficient=1.0e-4 + 0.0j,
        b_res=2.0e-4,
        half_width=0.03,
    )

    bands = chirikov_overlap_bands([right, left, middle], include_cross_n=True, radial_min=0.0, radial_max=1.0)

    assert len(bands) == 2
    assert isinstance(bands[0], ChirikovOverlapBand)
    assert bands[0].modes == ((5, 2), (8, 3))
    assert bands[0].same_toroidal_family is False
    assert bands[0].sigma == pytest.approx((0.06 + 0.05) / 0.09)
    assert bands[0].overlap_width == pytest.approx((0.25 + 0.06) - (0.34 - 0.05))
    assert bands[0].is_overlapping is True

    same_n_bands = chirikov_overlap_bands([right, left, middle], include_cross_n=False)
    assert len(same_n_bands) == 1
    assert same_n_bands[0].modes == ((5, 2), (6, 2))


def test_group_resonant_chains_combines_same_surface_harmonics_before_overlap():
    co_radial = [
        ResonantIslandChain(
            m=m,
            n=n,
            radial_label=0.50,
            q=3.0,
            q_prime=1.5,
            coefficient=complex(1.0e-4 * n, 0.0),
            b_res=2.0e-4 * n,
            half_width=width,
            coefficient_n=n,
        )
        for (m, n, width) in [(9, 3, 0.020), (12, 4, 0.030), (15, 5, 0.040)]
    ]
    outer = ResonantIslandChain(
        m=16,
        n=5,
        radial_label=0.62,
        q=3.2,
        q_prime=1.5,
        coefficient=1.0e-4 + 0.0j,
        b_res=2.0e-4,
        half_width=0.025,
        coefficient_n=5,
    )

    groups = group_resonant_island_chains(co_radial + [outer], radial_tol=1.0e-8)

    assert len(groups) == 2
    assert isinstance(groups[0], ResonantSurfaceGroup)
    assert groups[0].modes == ((9, 3), (12, 4), (15, 5))
    assert groups[0].mode_count == 3
    assert groups[0].half_width == pytest.approx(np.sqrt(0.020**2 + 0.030**2 + 0.040**2))
    assert groups[0].dominant_chain.m == 15

    raw_bands = chirikov_overlap_bands(co_radial + [outer], include_cross_n=True)
    assert len(raw_bands) == 1
    assert np.isfinite(raw_bands[0].sigma)
    assert raw_bands[0].modes == ((15, 5), (16, 5))

    effective = coalesce_resonant_island_chains(co_radial + [outer], radial_tol=1.0e-8)
    bands = chirikov_overlap_bands(effective, include_cross_n=True)

    assert len(effective) == 2
    assert len(bands) == 1
    assert bands[0].modes == ((15, 5), (16, 5))
    assert bands[0].left.half_width == pytest.approx(groups[0].half_width)


def test_chaotic_layer_intervals_merge_overlapping_bands():
    chains = [
        ResonantIslandChain(
            m=4 + idx,
            n=1 + (idx % 2),
            radial_label=radial,
            q=2.0,
            q_prime=1.0,
            coefficient=1.0e-4 + 0.0j,
            b_res=2.0e-4,
            half_width=width,
        )
        for idx, (radial, width) in enumerate([(0.20, 0.07), (0.28, 0.06), (0.37, 0.07), (0.72, 0.03)])
    ]
    bands = chirikov_overlap_bands(chains, include_cross_n=True)
    layers = chaotic_layer_intervals(bands, sigma_threshold=1.0)

    assert len(layers) == 1
    assert isinstance(layers[0], ChaoticLayerInterval)
    assert layers[0].inner == pytest.approx(0.13)
    assert layers[0].outer == pytest.approx(0.44)
    assert layers[0].max_sigma > 1.0
    assert len(layers[0].bands) == 2


def test_sample_cylindrical_vector_grid_on_surfaces():
    grid_R = np.linspace(2.5, 3.5, 6)
    grid_Z = np.linspace(-0.4, 0.4, 5)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    assert len({grid_R.size, grid_Z.size, grid_phi.size}) == 3
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


def test_sample_cylindrical_vector_grid_rejects_axis_order_swaps():
    grid_R = np.linspace(2.5, 3.5, 6)
    grid_Z = np.linspace(-0.4, 0.4, 5)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    assert len({grid_R.size, grid_Z.size, grid_phi.size}) == 3

    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radial = np.array([0.1])
    R = 3.0 + radial[None, :, None] * np.cos(theta)[None, None, :]
    Z = radial[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, grid_phi.size, axis=0)
    Z = np.repeat(Z, grid_phi.size, axis=0)

    RR, ZZ, PP = np.meshgrid(grid_R, grid_Z, grid_phi, indexing="ij")
    field_R = RR + 2.0 * ZZ + np.cos(PP)
    field_phi = 2.0 * RR - ZZ + np.sin(PP)
    field_Z = -RR + 0.5 * ZZ + np.cos(PP)

    with pytest.raises(ValueError, match="field arrays must have shape"):
        sample_cylindrical_vector_grid_on_surfaces(
            grid_R,
            grid_Z,
            grid_phi,
            np.swapaxes(field_R, 0, 1),
            field_phi,
            field_Z,
            R,
            Z,
            grid_phi,
            theta,
        )


def test_sample_cylindrical_vector_grid_on_surfaces_respects_field_periods():
    nfp = 2
    field_period = 2.0 * np.pi / nfp
    grid_R = np.linspace(0.8, 1.2, 5)
    grid_Z = np.linspace(-0.2, 0.2, 4)
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
