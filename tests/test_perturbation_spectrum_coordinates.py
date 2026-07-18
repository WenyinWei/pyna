import numpy as np
import pytest

from pyna.toroidal.perturbation_spectrum import (
    analyze_resonant_island_chains,
    radial_perturbation_Fourier_spectrum,
)


TWOPI = 2.0 * np.pi


def _harmonic(theta, phi, *, m, n, coefficient):
    return coefficient * np.exp(1j * (m * theta[None, :] - n * phi[:, None]))


@pytest.mark.parametrize("physical_n", [5, -4])
def test_full_torus_shifted_origins_recover_physical_helicity_and_phase(physical_n):
    theta = 0.31 + np.arange(64) * (TWOPI / 64)
    phi = -0.27 + np.arange(48) * (TWOPI / 48)
    coefficient = 1.2 * np.exp(0.73j)
    grid = _harmonic(theta, phi, m=3, n=physical_n, coefficient=coefficient)

    spectrum = radial_perturbation_Fourier_spectrum(
        grid,
        theta,
        phi,
        m_max=4,
        n_max=6,
        min_amplitude=1.0e-12,
    )

    index = spectrum.physical_mode_index(3, physical_n)
    assert index is not None
    assert spectrum.n[index] == -physical_n
    assert spectrum.field_period_harmonic[index] == -physical_n
    np.testing.assert_allclose(spectrum.dBr[index], coefficient, atol=2.0e-13)
    np.testing.assert_allclose(
        spectrum.physical_mode_coefficient(3, physical_n),
        coefficient,
        atol=2.0e-13,
    )


def test_nfp2_one_period_recovers_physical_modes_and_shifted_phase():
    theta = np.linspace(0.19, 0.19 + TWOPI, 65)
    phi = np.linspace(-0.23, -0.23 + np.pi, 33)
    modes = [
        (3, 6, 0.8 * np.exp(0.41j)),
        (-2, -4, 0.35 * np.exp(-0.62j)),
    ]
    grid = sum(
        _harmonic(theta, phi, m=m, n=n, coefficient=coefficient)
        for m, n, coefficient in modes
    )

    spectrum = radial_perturbation_Fourier_spectrum(
        grid,
        theta,
        phi,
        field_periods=2,
        m_max=4,
        n_max=6,
        min_amplitude=1.0e-12,
    )

    assert spectrum.field_periods == 2
    assert spectrum.dBr_grid.shape == (32, 64)
    assert np.all(spectrum.physical_n % 2 == 0)
    for m, physical_n, coefficient in modes:
        index = spectrum.physical_mode_index(m, physical_n)
        assert index is not None
        assert spectrum.field_period_harmonic[index] == -physical_n // 2
        np.testing.assert_allclose(spectrum.dBr[index], coefficient, atol=2.0e-13)


def test_nardon_indices_preserve_conjugates_and_distinguish_opposite_n_branches():
    theta = np.linspace(0.19, 0.19 + TWOPI, 65)
    phi = np.linspace(-0.23, -0.23 + np.pi, 33)
    m = 3
    nardon_n = 4
    coefficient_plus = 0.8 * np.exp(0.41j)
    coefficient_minus = 0.35 * np.exp(-0.62j)
    phase_plus = m * theta[None, :] + nardon_n * phi[:, None]
    phase_minus = m * theta[None, :] - nardon_n * phi[:, None]
    grid = (
        coefficient_plus * np.exp(1j * phase_plus)
        + coefficient_plus.conjugate() * np.exp(-1j * phase_plus)
        + coefficient_minus * np.exp(1j * phase_minus)
        + coefficient_minus.conjugate() * np.exp(-1j * phase_minus)
    ).real

    spectrum = radial_perturbation_Fourier_spectrum(
        grid,
        theta,
        phi,
        field_periods=2,
        m_max=4,
        n_max=6,
        min_amplitude=1.0e-12,
    )

    plus = spectrum.nardon_mode_index(m, nardon_n)
    minus = spectrum.nardon_mode_index(m, -nardon_n)
    conjugate = spectrum.nardon_mode_index(-m, -nardon_n)
    assert plus is not None and minus is not None and conjugate is not None
    assert spectrum.nardon_n[plus] == nardon_n
    assert spectrum.field_period_harmonic[plus] == nardon_n // spectrum.field_periods
    assert spectrum.physical_n[plus] == -nardon_n  # Historical compatibility API.
    assert spectrum.resonance_family_n0[plus] == nardon_n
    np.testing.assert_allclose(spectrum.dBr[plus], coefficient_plus, atol=2.0e-13)
    np.testing.assert_allclose(spectrum.dBr[conjugate], coefficient_plus.conjugate(), atol=2.0e-13)
    np.testing.assert_allclose(spectrum.dBr[minus], coefficient_minus, atol=2.0e-13)
    assert not np.isclose(spectrum.dBr[plus], spectrum.dBr[minus])


def test_wrapped_shifted_origins_infer_field_period_and_preserve_phase():
    theta = np.mod(0.41 + np.arange(48) * (TWOPI / 48), TWOPI)
    phi = np.mod(0.22 + np.arange(32) * (np.pi / 32), np.pi)
    coefficient = 0.7 * np.exp(-0.54j)
    grid = _harmonic(theta, phi, m=3, n=4, coefficient=coefficient)

    spectrum = radial_perturbation_Fourier_spectrum(
        grid,
        theta,
        phi,
        m_max=4,
        n_max=6,
        min_amplitude=1.0e-12,
    )

    assert spectrum.field_periods == 2
    np.testing.assert_allclose(
        spectrum.physical_mode_coefficient(3, 4),
        coefficient,
        atol=2.0e-13,
    )


def test_equal_phi_and_radial_lengths_require_explicit_layout():
    radial = np.linspace(0.1, 0.8, 8)
    phi = np.arange(8) * (TWOPI / 8)
    theta = np.arange(24) * (TWOPI / 24)
    coefficient = 0.6 * np.exp(-0.37j)
    profile = 1.0 + radial
    surface = _harmonic(theta, phi, m=4, n=3, coefficient=coefficient)
    phi_first = surface[:, None, :] * profile[None, :, None]
    radial_first = np.moveaxis(phi_first, 1, 0)

    with pytest.raises(ValueError, match="ambiguous.*layout"):
        radial_perturbation_Fourier_spectrum(phi_first, theta, phi, radial_labels=radial)
    with pytest.raises(ValueError, match="ambiguous.*layout"):
        radial_perturbation_Fourier_spectrum(radial_first, theta, phi, radial_labels=radial)

    phi_spectrum = radial_perturbation_Fourier_spectrum(
        phi_first,
        theta,
        phi,
        radial_labels=radial,
        layout="phi-radial-theta",
        m_max=5,
        n_max=4,
        min_amplitude=1.0e-12,
    )
    radial_spectrum = radial_perturbation_Fourier_spectrum(
        radial_first,
        theta,
        phi,
        radial_labels=radial,
        layout="radial-phi-theta",
        m_max=5,
        n_max=4,
        min_amplitude=1.0e-12,
    )

    np.testing.assert_allclose(phi_spectrum.dBr_grid, radial_spectrum.dBr_grid)
    for radial_index, amplitude in enumerate(profile):
        expected = amplitude * coefficient
        np.testing.assert_allclose(
            phi_spectrum.physical_mode_coefficient(4, 3, radial_index),
            expected,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            radial_spectrum.physical_mode_coefficient(4, 3, radial_index),
            expected,
            atol=2.0e-13,
        )


def test_unambiguous_legacy_stack_layouts_still_infer():
    radial = np.array([0.2, 0.5, 0.9])
    phi = np.arange(10) * (TWOPI / 10)
    theta = np.arange(24) * (TWOPI / 24)
    coefficient = 0.45 * np.exp(0.28j)
    profile = 1.0 + 2.0 * radial
    surface = _harmonic(theta, phi, m=3, n=-2, coefficient=coefficient)
    phi_first = surface[:, None, :] * profile[None, :, None]
    radial_first = np.moveaxis(phi_first, 1, 0)

    inferred_phi_first = radial_perturbation_Fourier_spectrum(
        phi_first,
        theta,
        phi,
        radial_labels=radial,
        m_max=4,
        n_max=3,
        min_amplitude=1.0e-12,
    )
    inferred_radial_first = radial_perturbation_Fourier_spectrum(
        radial_first,
        theta,
        phi,
        radial_labels=radial,
        m_max=4,
        n_max=3,
        min_amplitude=1.0e-12,
    )

    np.testing.assert_allclose(inferred_phi_first.dBr_grid, inferred_radial_first.dBr_grid)
    np.testing.assert_array_equal(inferred_phi_first.m, inferred_radial_first.m)
    np.testing.assert_array_equal(inferred_phi_first.n, inferred_radial_first.n)
    np.testing.assert_allclose(inferred_phi_first.dBr, inferred_radial_first.dBr)


def test_nfp2_physical_modes_feed_existing_resonance_callers():
    radial = np.linspace(0.2, 0.5, 5)
    theta = np.arange(48) * (TWOPI / 48)
    phi = np.arange(32) * (np.pi / 32)
    coefficient_profile = (1.0e-3 + 2.0e-3 * radial) * np.exp(0.35j)
    surface = _harmonic(theta, phi, m=5, n=2, coefficient=1.0 + 0.0j)
    phi_first = surface[:, None, :] * coefficient_profile[None, :, None]
    spectrum = radial_perturbation_Fourier_spectrum(
        phi_first,
        theta,
        phi,
        radial_labels=radial,
        field_periods=2,
        m_max=6,
        n_max=4,
        min_amplitude=1.0e-12,
    )

    q_profile = 2.0 + 2.0 * radial
    chains = analyze_resonant_island_chains(
        spectrum,
        q_profile,
        n=2,
        m_values=[5],
    )

    assert len(chains) == 1
    assert chains[0].coefficient_n == -2
    assert chains[0].nardon_n == -2
    assert chains[0].resonance_family_n0 == 2
    assert spectrum.nardon_mode_index(5, -2) is not None
    expected = (1.0e-3 + 2.0e-3 * chains[0].radial_label) * np.exp(0.35j)
    np.testing.assert_allclose(chains[0].coefficient, expected, atol=2.0e-13)
    split = spectrum.split(iota=2.0 / 5.0, radial_index=0)
    assert np.any((split.resonant_m == 5) & (split.resonant_n == -2))


def test_one_field_period_domain_infers_nfp_but_rejects_explicit_mismatch():
    theta = np.arange(24) * (TWOPI / 24)
    phi = np.arange(16) * (np.pi / 16)
    grid = _harmonic(theta, phi, m=2, n=4, coefficient=1.0 + 0.0j)

    inferred = radial_perturbation_Fourier_spectrum(
        grid,
        theta,
        phi,
        m_max=3,
        n_max=5,
        min_amplitude=1.0e-12,
    )
    assert inferred.field_periods == 2
    np.testing.assert_allclose(inferred.physical_mode_coefficient(2, 4), 1.0 + 0.0j)

    with pytest.raises(ValueError, match="set field_periods"):
        radial_perturbation_Fourier_spectrum(grid, theta, phi, field_periods=1)
