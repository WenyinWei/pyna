"""Candidate-field PEST angle reconstruction without magnetic healing."""

from __future__ import annotations

import numpy as np
import pytest

import pyna.toroidal.pest_coords as pest_coords_module
from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates
from pyna.toroidal.coords import (
    reparameterize_pest_on_candidate_field,
    straight_field_line_mde_adjoint_error,
)


TWOPI = 2.0 * np.pi
R0 = 5.5


def _circular_material_coordinates(*, nphi: int = 24, ntheta: int = 32) -> SmoothPestCoordinates:
    phi = np.linspace(0.0, TWOPI, nphi, endpoint=False)
    theta = np.linspace(0.0, TWOPI, ntheta, endpoint=False)
    rho = np.array([0.24, 0.37, 0.51], dtype=np.float64)
    R_section = R0 + rho[:, None] * np.cos(theta)[None, :]
    Z_section = rho[:, None] * np.sin(theta)[None, :]
    R = np.broadcast_to(R_section[None, :, :], (nphi, rho.size, ntheta)).copy()
    Z = np.broadcast_to(Z_section[None, :, :], (nphi, rho.size, ntheta)).copy()
    return SmoothPestCoordinates(
        R_surf=R,
        Z_surf=Z,
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        axis_R=np.full(nphi, R0),
        axis_Z=np.zeros(nphi),
        source="manufactured circular material surfaces",
    )


class _ManufacturedCircularField:
    def __init__(self, theta_slope, *, radial_slope=0.0, bphi_scale: float = 1.0):
        self.theta_slope = theta_slope
        self.radial_slope = radial_slope
        self.bphi_scale = float(bphi_scale)

    def interpolate_at(self, R, Z, phi):
        radius = np.hypot(R - R0, Z)
        theta = np.mod(np.arctan2(Z, R - R0), TWOPI)
        a = np.broadcast_to(np.asarray(self.theta_slope(theta, phi), dtype=np.float64), R.shape)
        radial_value = self.radial_slope(theta, phi) if callable(self.radial_slope) else self.radial_slope
        radial = np.broadcast_to(np.asarray(radial_value, dtype=np.float64), R.shape)
        # Set dphi/ds=1.  The physical toroidal component is therefore R,
        # while BR/BZ are the exact coordinate-basis combination.
        BR = radial * np.cos(theta) - a * radius * np.sin(theta)
        BZ = radial * np.sin(theta) + a * radius * np.cos(theta)
        BPhi = self.bphi_scale * R
        return BR, BZ, BPhi


def test_matrix_free_straight_field_line_operator_has_exact_discrete_adjoint():
    phi = np.linspace(0.0, TWOPI, 16, endpoint=False)[:, None]
    theta = np.linspace(0.0, TWOPI, 18, endpoint=False)[None, :]
    slope = 0.41 + 0.07 * np.cos(theta - 2.0 * phi)
    assert straight_field_line_mde_adjoint_error(slope) < 2.0e-14
    assert straight_field_line_mde_adjoint_error(slope, gauge_mode="mean") < 2.0e-14


def test_even_grid_derivative_nullspace_projection_removes_axis_nyquist_modes():
    nphi, ntheta = 16, 18
    phi_nyquist = ((-1.0) ** np.arange(nphi))[:, None]
    theta_nyquist = ((-1.0) ** np.arange(ntheta))[None, :]
    smooth = 0.07 * np.sin(
        np.linspace(0.0, TWOPI, nphi, endpoint=False)[:, None]
        + 2.0 * np.linspace(0.0, TWOPI, ntheta, endpoint=False)[None, :]
    )
    raw = smooth + 0.3 * phi_nyquist - 0.2 * theta_nyquist + 0.4 * phi_nyquist * theta_nyquist

    projected, removed = pest_coords_module._project_even_grid_derivative_nullspace(raw)

    for pattern in (phi_nyquist, theta_nyquist, phi_nyquist * theta_nyquist):
        assert abs(float(np.mean(projected * pattern))) < 2.0e-15
    np.testing.assert_allclose(projected + removed, raw, atol=2.0e-15, rtol=0.0)
    np.testing.assert_allclose(
        pest_coords_module._periodic_fft_derivative(removed, axis=0, period=TWOPI),
        0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        pest_coords_module._periodic_fft_derivative(removed, axis=1, period=TWOPI),
        0.0,
        atol=2.0e-14,
    )


def test_circular_torus_recovers_iota_and_supports_parallel_surface_selection():
    coordinates = _circular_material_coordinates()
    iota = 0.371234
    field = _ManufacturedCircularField(lambda theta, phi: iota + 0.0 * theta)
    result = reparameterize_pest_on_candidate_field(
        coordinates,
        field,
        surface_indices=[0, -1],
        workers=2,
    )

    assert result.selected_surface_indices == (0, 2)
    np.testing.assert_allclose(result.iota[[0, 2]], iota, atol=2.0e-11, rtol=0.0)
    assert np.isnan(result.iota[1])
    assert np.isnan(result.theta_slope[:, 1, :]).all()
    np.testing.assert_array_equal(result.coordinates.R_surf[:, 1, :], coordinates.R_surf[:, 1, :])
    for diagnostic in result.surface_diagnostics:
        assert diagnostic.mde_residual_relative < 1.0e-11
        assert diagnostic.theta_correction_max_abs < 1.0e-10
        assert diagnostic.radial_leakage_max < 2.0e-13
        assert diagnostic.diffeomorphism_min_jacobian > 0.999999999
    assert result.metadata["field_modified"] is False
    assert result.metadata["surface_geometry_modified"] is False
    assert result.metadata["healing_used"] is False
    assert "no healing" in result.coordinates.source


def test_manufactured_theta_correction_and_iota_are_recovered():
    coordinates = _circular_material_coordinates(nphi=24, ntheta=36)
    iota = np.sqrt(2.0) - 1.0
    epsilon = 0.08

    def phase(theta, phi):
        return theta + 2.0 * phi

    def prescribed_u(theta, phi):
        return epsilon * np.sin(phase(theta, phi))

    def theta_slope(theta, phi):
        argument = phase(theta, phi)
        u_phi = 2.0 * epsilon * np.cos(argument)
        u_theta = epsilon * np.cos(argument)
        return (iota - u_phi) / (1.0 + u_theta)

    result = reparameterize_pest_on_candidate_field(
        coordinates,
        _ManufacturedCircularField(theta_slope),
        surface_indices=1,
        lsmr_maxiter=6000,
    )
    phi_grid = coordinates.phi_vals[:, None]
    theta_grid = coordinates.theta_vals[None, :]
    expected_u = prescribed_u(theta_grid, phi_grid)
    np.testing.assert_allclose(result.theta_correction[:, 1, :], expected_u, atol=2.0e-8, rtol=0.0)
    np.testing.assert_allclose(result.iota[1], iota, atol=2.0e-9, rtol=0.0)
    diagnostic = result.surface_diagnostics[0]
    assert diagnostic.mde_residual_relative < 2.0e-9
    assert diagnostic.diffeomorphism_min_jacobian == pytest.approx(1.0 - epsilon, abs=2.0e-8)
    assert result.theta_correction[0, 1, 0] == 0.0
    # Re-sampling changes only the parameter along the same circular material surface.
    reconstructed_radius = np.hypot(result.coordinates.R_surf[:, 1, :] - R0, result.coordinates.Z_surf[:, 1, :])
    np.testing.assert_allclose(reconstructed_radius, coordinates.rho_vals[1], atol=2.0e-3, rtol=0.0)


def test_radial_and_normal_leakage_are_reported_not_projected_away():
    coordinates = _circular_material_coordinates()
    radial_amplitude = 0.025
    field = _ManufacturedCircularField(
        lambda theta, phi: 0.39 + 0.0 * theta,
        radial_slope=lambda theta, phi: radial_amplitude * (1.0 + 0.1 * np.cos(theta - phi)),
    )
    result = reparameterize_pest_on_candidate_field(coordinates, field, surface_indices=1)
    diagnostic = result.surface_diagnostics[0]
    assert diagnostic.radial_leakage_p95 == pytest.approx(1.1 * radial_amplitude, rel=0.03)
    assert diagnostic.normal_leakage_p95 > 1.0e-3
    assert result.metadata["surface_geometry_modified"] is False


def test_explicit_damping_is_orientation_preserving_and_recomputes_best_iota():
    coordinates = _circular_material_coordinates(nphi=24, ntheta=36)
    iota = 0.41421356237
    epsilon = 0.8

    def steep_slope(theta, phi):
        argument = theta + 2.0 * phi
        return (iota - 2.0 * epsilon * np.cos(argument)) / (
            1.0 + epsilon * np.cos(argument)
        )

    result = reparameterize_pest_on_candidate_field(
        coordinates,
        _ManufacturedCircularField(steep_slope),
        gauge_mode="mean",
        diffeomorphism_backtracking=True,
        min_theta_jacobian=0.25,
        max_theta_jacobian=1.75,
        max_mde_relative_residual=0.20,
        lsmr_maxiter=8000,
    )

    for diagnostic in result.surface_diagnostics:
        ir = diagnostic.surface_index
        a = result.theta_slope[:, ir, :]
        u = result.theta_correction[:, ir, :]
        transport = (
            pest_coords_module._periodic_fft_derivative(u, axis=0, period=TWOPI)
            + a * pest_coords_module._periodic_fft_derivative(u, axis=1, period=TWOPI)
        )
        expected_iota = float(np.mean(a + transport))
        residual = a + transport - expected_iota
        residual_relative = np.sqrt(np.mean(residual**2)) / max(
            np.sqrt(np.mean(a**2)), abs(expected_iota)
        )
        jacobian = 1.0 + pest_coords_module._periodic_fft_derivative(
            u, axis=1, period=TWOPI
        )
        gaps = pest_coords_module._mapped_theta_forward_gaps(u, theta_period=TWOPI)

        assert diagnostic.gauge_mode == "mean"
        assert diagnostic.diffeomorphism_backtracking_enabled is True
        assert diagnostic.damped is True
        assert diagnostic.damping_alpha == pytest.approx(0.9375, abs=8.0e-9)
        assert diagnostic.raw_diffeomorphism_min_jacobian < 0.25
        assert diagnostic.post_diffeomorphism_min_jacobian >= 0.25 - 2.0e-14
        assert diagnostic.post_diffeomorphism_max_jacobian <= 1.75 + 2.0e-14
        assert diagnostic.post_mapped_theta_min_gap > 0.0
        assert diagnostic.material_coordinate_orientation_preserved is True
        assert diagnostic.material_coordinate_min_relative_jacobian > 0.99
        assert diagnostic.iota == pytest.approx(expected_iota, abs=2.0e-12)
        assert diagnostic.post_best_fit_iota == pytest.approx(expected_iota, abs=2.0e-12)
        assert diagnostic.mde_residual_relative == pytest.approx(residual_relative, abs=2.0e-12)
        assert diagnostic.post_mde_residual_relative == pytest.approx(residual_relative, abs=2.0e-12)
        assert np.min(jacobian) >= 0.25 - 2.0e-14
        assert np.max(jacobian) <= 1.75 + 2.0e-14
        assert np.min(gaps) > 0.0
    assert result.metadata["classification"] == (
        "orientation_preserving_approximate_candidate_field_PEST_fit"
    )
    assert result.metadata["mde_nonlinear_product_dealiased"] is False
    assert result.metadata["material_coordinate_orientation_preserved"] is True
    assert result.metadata["damping_alpha_max_adjacent_jump"] < 1.0e-8


def test_alpha_zero_is_exact_identity_but_still_obeys_residual_gate():
    coordinates = _circular_material_coordinates(nphi=24, ntheta=36)
    iota = np.sqrt(2.0) - 1.0
    epsilon = 0.08

    def theta_slope(theta, phi):
        argument = theta + 2.0 * phi
        return (iota - 2.0 * epsilon * np.cos(argument)) / (
            1.0 + epsilon * np.cos(argument)
        )

    field = _ManufacturedCircularField(theta_slope)
    result = reparameterize_pest_on_candidate_field(
        coordinates,
        field,
        gauge_mode="mean",
        diffeomorphism_backtracking=True,
        min_theta_jacobian=1.0,
        max_mde_relative_residual=1.0,
        lsmr_maxiter=6000,
    )

    np.testing.assert_array_equal(result.coordinates.R_surf, coordinates.R_surf)
    np.testing.assert_array_equal(result.coordinates.Z_surf, coordinates.Z_surf)
    np.testing.assert_array_equal(result.theta_correction, 0.0)
    for diagnostic in result.surface_diagnostics:
        assert diagnostic.damping_alpha == 0.0
        assert diagnostic.damped is True
        assert diagnostic.diffeomorphism_min_jacobian == 1.0
        assert diagnostic.post_mapped_theta_min_gap == pytest.approx(TWOPI / 36.0)
        assert diagnostic.iota == pytest.approx(
            float(np.mean(result.theta_slope[:, diagnostic.surface_index, :])),
            abs=2.0e-14,
        )

    with pytest.raises(RuntimeError, match=r"best feasible damped.*no permitted result"):
        reparameterize_pest_on_candidate_field(
            coordinates,
            field,
            gauge_mode="mean",
            diffeomorphism_backtracking=True,
            min_theta_jacobian=1.0,
            max_mde_relative_residual=0.20,
            lsmr_maxiter=6000,
        )


def test_zero_bphi_and_non_diffeomorphic_angle_are_rejected():
    coordinates = _circular_material_coordinates()
    zero_bphi = _ManufacturedCircularField(lambda theta, phi: 0.4 + 0.0 * theta, bphi_scale=0.0)
    with pytest.raises(ValueError, match="Bphi"):
        reparameterize_pest_on_candidate_field(coordinates, zero_bphi, surface_indices=1)

    iota = 0.41421356237
    epsilon = 0.8

    def steep_slope(theta, phi):
        argument = theta + 2.0 * phi
        return (iota - 2.0 * epsilon * np.cos(argument)) / (1.0 + epsilon * np.cos(argument))

    with pytest.raises(RuntimeError, match="diffeomorphism"):
        reparameterize_pest_on_candidate_field(
            coordinates,
            _ManufacturedCircularField(steep_slope),
            surface_indices=1,
            min_theta_jacobian=0.25,
            max_mde_relative_residual=None,
            lsmr_maxiter=8000,
        )
