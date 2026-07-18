import numpy as np

from pyna.toroidal.coils.boundary_local import boundary_loop_coil_specs_from_surface
from pyna.toroidal.control.boundary_perturbation_candidates import (
    perturbation_candidate_nardon_response,
    sample_perturbation_candidate_on_surfaces,
)


def _circular_surface(n_phi=5, n_r=3, n_theta=13):
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    radial = np.linspace(0.16, 0.34, n_r)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    PP, SS, TT = np.meshgrid(phi, radial, theta, indexing="ij")
    R0 = 1.1
    R = R0 + SS * np.cos(TT)
    Z = SS * np.sin(TT)
    return R, Z, phi, radial, theta


class _KnownRadialCandidate:
    def __init__(self, *, m, n, phase, amplitude):
        self.m = int(m)
        self.n = int(n)
        self.phase = float(phase)
        self.amplitude = float(amplitude)

    def B_at(self, R, Z, phi):
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        phi = np.asarray(phi, dtype=float)
        theta = np.arctan2(Z, R - 1.1)
        radial = np.hypot(R - 1.1, Z)
        amp = self.amplitude * (1.0 + radial)
        tilde = amp * np.cos(self.m * theta - self.n * phi + self.phase)
        delta_B1 = 2.0 * tilde
        return delta_B1 * np.cos(theta), delta_B1 * np.sin(theta), np.zeros_like(R)


def test_sample_perturbation_candidate_on_surfaces_accepts_loop_specs():
    R, Z, phi, _radial, theta = _circular_surface(n_phi=5, n_r=4, n_theta=11)
    specs = boundary_loop_coil_specs_from_surface(
        R,
        Z,
        phi,
        theta,
        theta_indices=[0, 3, 7],
        radius=0.025,
        clearance=0.012,
        current=1.0,
        mode_m=2,
        mode_n=1,
    )

    BR, BZ, BPhi = sample_perturbation_candidate_on_surfaces(specs, R, Z, phi)

    assert BR.shape == R.shape
    assert BZ.shape == R.shape
    assert BPhi.shape == R.shape
    assert np.all(np.isfinite(BR + BZ + BPhi))


def test_perturbation_candidate_nardon_response_recovers_known_mode():
    R, Z, phi, radial, theta = _circular_surface(n_phi=7, n_r=3, n_theta=17)
    m_val = 4
    n_val = 2
    phase = 0.43
    amplitude = 1.0e-3
    candidate = _KnownRadialCandidate(m=m_val, n=n_val, phase=phase, amplitude=amplitude)
    denominator_B_phi = 2.0 * R

    response = perturbation_candidate_nardon_response(
        candidate,
        R,
        Z,
        phi,
        theta,
        radial,
        denominator_B_phi=denominator_B_phi,
        m_max=5,
        n_max=3,
        min_amplitude=1.0e-12,
        metadata={"case": "synthetic"},
    )

    idx = response.spectrum.mode_index(m_val, -n_val)
    assert idx is not None
    expected = 0.5 * amplitude * (1.0 + radial) * np.exp(1j * phase)
    np.testing.assert_allclose(response.spectrum.dBr[:, idx], expected, atol=2.0e-12)
    np.testing.assert_allclose(response.mode_coefficient(m_val, -n_val, radial_index=1), expected[1], atol=2.0e-12)
    assert response.metadata["candidate_kind"] == "_KnownRadialCandidate"
    assert response.spectrum.metadata["case"] == "synthetic"
