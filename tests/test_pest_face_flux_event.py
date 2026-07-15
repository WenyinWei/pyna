import multiprocessing as mp

import numpy as np
import pytest

from pyna.plot.j_streamlines import (
    PestSurfaceFaceFluxes,
    plot_j_streamline_seed_sections,
    trace_j_streamlines_on_pest,
)
from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates


def _toy_pest(*, nfp=5, n_phi=10, n_rho=2, n_theta=16):
    period = 2.0 * np.pi / int(nfp)
    phi = np.linspace(0.0, period, n_phi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rho = np.linspace(0.12, 0.24, n_rho)
    R = 1.0 + rho[None, :, None] * np.cos(theta)[None, None, :]
    Z = rho[None, :, None] * np.sin(theta)[None, None, :]
    return SmoothPestCoordinates(
        R_surf=np.repeat(R, n_phi, axis=0),
        Z_surf=np.repeat(Z, n_phi, axis=0),
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        nfp=int(nfp),
        toroidal_period=period,
    )


class _ConstantFaceFluxField:
    def __init__(
        self,
        *,
        nfp=5,
        n_phi_cells=5,
        n_theta_cells=8,
        qtheta=0.7,
        qphi=0.2,
        flux_nfp=None,
    ):
        self.nfp = int(nfp)
        self.field_period_rad = 2.0 * np.pi / self.nfp
        self.theta_faces = np.linspace(0.0, 2.0 * np.pi, n_theta_cells + 1)
        self.phi_faces = np.linspace(
            0.0, self.field_period_rad, n_phi_cells + 1
        )
        self.theta_upper = np.full(
            (n_phi_cells, n_theta_cells), float(qtheta), dtype=np.float64
        )
        self.phi_upper = np.full(
            (n_phi_cells, n_theta_cells), float(qphi), dtype=np.float64
        )
        self.flux_nfp = self.nfp if flux_nfp is None else int(flux_nfp)

    def evaluate_pest_surface_face_fluxes(self, surface_index):
        del surface_index
        return PestSurfaceFaceFluxes(
            theta_upper_face_flux=self.theta_upper,
            phi_upper_face_flux=self.phi_upper,
            theta_faces=self.theta_faces,
            phi_faces=self.phi_faces,
            nfp=self.flux_nfp,
            phi_period=self.field_period_rad,
        )


def test_face_flux_event_follows_manufactured_constant_field_across_nfp5_seams():
    pest = _toy_pest()
    field = _ConstantFaceFluxField(qtheta=0.7, qphi=0.2)
    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        integration_backend="face_flux_event",
        surface_index=-1,
        phi_values=[0.03],
        seed_count=1,
        seed_spacing="theta",
        n_turns=1.0,
        steps_per_turn=30,
        bidirectional=False,
    )

    assert lines.metadata["integration_backend"] == "face_flux_event"
    assert lines.metadata["trace_mode"] == "pest_surface_face_flux_event"
    assert lines.metadata["nfp"] == 5
    assert lines.n_points == 31
    assert np.all(np.isfinite(lines.theta))
    assert np.all(np.isfinite(lines.phi))
    np.testing.assert_allclose(
        lines.theta[0] - lines.seed_theta[0],
        3.5 * (lines.phi[0] - lines.seed_phi[0]),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.max(lines.phi[0]) > field.field_period_rad
    assert np.all(np.diff(lines.phi[0]) >= 0.0)
    assert lines.metadata["face_flux_event"]["forward_status_by_seed_line"] == [
        "max_events"
    ]


def test_face_flux_event_bidirectional_trace_retains_seed_and_unwrapped_angles():
    pest = _toy_pest()
    field = _ConstantFaceFluxField(qtheta=0.4, qphi=0.1)
    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        integration_backend="face_flux_event",
        surface_index=0,
        phi_values=[0.04],
        seed_count=1,
        seed_spacing="theta",
        n_turns=1.0,
        steps_per_turn=12,
        bidirectional=True,
    )

    assert lines.n_points == 25
    assert lines.theta[0, 12] == pytest.approx(lines.seed_theta[0])
    assert lines.phi[0, 12] == pytest.approx(lines.seed_phi[0])
    assert np.all(np.diff(lines.theta[0]) >= -2.0e-14)
    assert np.all(np.diff(lines.phi[0]) >= -2.0e-14)
    assert np.max(np.diff(lines.theta[0])) <= np.diff(field.theta_faces).max() + 2.0e-14
    assert np.max(np.diff(lines.phi[0])) <= np.diff(field.phi_faces).max() + 2.0e-14
    np.testing.assert_allclose(
        lines.theta[0] - lines.seed_theta[0],
        4.0 * (lines.phi[0] - lines.seed_phi[0]),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    evidence = lines.metadata["face_flux_event"]
    assert evidence["forward_event_count_by_seed_line"] == [12]
    assert evidence["backward_event_count_by_seed_line"] == [12]


def test_face_flux_event_crosses_exact_corners_without_chatter():
    pest = _toy_pest()
    field = _ConstantFaceFluxField(n_phi_cells=5, n_theta_cells=8)
    field.theta_upper[...] = np.diff(field.theta_faces)[0]
    field.phi_upper[...] = np.diff(field.phi_faces)[0]
    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        integration_backend="face_flux_event",
        surface_index=0,
        phi_values=[0.0],
        seed_count=1,
        seed_spacing="theta",
        n_turns=1.0,
        steps_per_turn=4,
        bidirectional=False,
    )

    dtheta = np.diff(field.theta_faces)[0]
    dphi = np.diff(field.phi_faces)[0]
    np.testing.assert_allclose(lines.theta[0], np.arange(5) * dtheta)
    np.testing.assert_allclose(lines.phi[0], np.arange(5) * dphi)
    assert lines.metadata["face_flux_event"]["forward_corner_count_by_seed_line"] == [
        4
    ]


def test_face_flux_event_uses_exact_linear_rt0_evolution_between_faces():
    pest = _toy_pest()
    field = _ConstantFaceFluxField(
        n_phi_cells=5,
        n_theta_cells=2,
        qtheta=1.0,
        qphi=1.0,
    )
    # In theta cell zero, qtheta(theta) = 1 + theta/pi.  The first event is
    # the upper phi face, so theta has a closed-form exponential update.
    field.theta_upper[:, 0] = 2.0
    field.theta_upper[:, 1] = 1.0
    phi0 = 0.03
    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        integration_backend="face_flux_event",
        surface_index=0,
        phi_values=[phi0],
        seed_count=1,
        seed_spacing="theta",
        n_turns=1.0,
        steps_per_turn=1,
        bidirectional=False,
    )

    event_time = field.phi_faces[1] - phi0
    expected_theta = np.pi * (np.exp(event_time / np.pi) - 1.0)
    assert lines.phi[0, 1] == pytest.approx(field.phi_faces[1])
    assert lines.theta[0, 1] == pytest.approx(expected_theta, rel=2.0e-14)


def test_face_flux_event_marks_stagnation_and_pads_after_seed():
    pest = _toy_pest()
    field = _ConstantFaceFluxField(qtheta=0.0, qphi=0.0)
    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        integration_backend="face_flux_event",
        surface_index=0,
        phi_values=[0.03],
        seed_count=1,
        n_turns=1.0,
        steps_per_turn=5,
        bidirectional=False,
    )

    assert np.count_nonzero(np.isfinite(lines.theta[0])) == 1
    assert np.count_nonzero(np.isfinite(lines.phi[0])) == 1
    assert lines.metadata["face_flux_event"]["forward_status_by_seed_line"] == [
        "stagnation"
    ]
    assert lines.metadata["face_flux_event"]["forward_event_count_by_seed_line"] == [
        0
    ]


def test_face_flux_event_rejects_face_data_nfp_mismatch_without_grid_inference():
    with pytest.raises(ValueError, match=r"fluxes\.nfp=1, field\.nfp=5"):
        trace_j_streamlines_on_pest(
            _ConstantFaceFluxField(nfp=5, flux_nfp=1),
            _toy_pest(nfp=5),
            integration_backend="face_flux_event",
            surface_index=0,
            phi_values=[0.03],
            seed_count=1,
        )


def test_face_flux_event_rejects_non_surface_protocol():
    class MissingExplicitPeriod:
        def evaluate_pest_surface_face_fluxes(self, surface_index):
            del surface_index

    with pytest.raises(TypeError, match="explicitly provide nfp"):
        trace_j_streamlines_on_pest(
            MissingExplicitPeriod(),
            _toy_pest(),
            integration_backend="face_flux_event",
            surface_index=0,
            phi_values=[0.03],
            seed_count=1,
        )


def test_trace_rejects_unknown_integration_backend():
    with pytest.raises(ValueError, match="integration_backend"):
        trace_j_streamlines_on_pest(
            _ConstantFaceFluxField(),
            _toy_pest(),
            integration_backend="guess_from_phi",
            surface_index=0,
            phi_values=[0.03],
            seed_count=1,
        )


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires POSIX fork")
def test_face_flux_event_parallel_surface_chunks_preserve_rows_and_evidence():
    pest = _toy_pest(n_rho=4)
    field = _ConstantFaceFluxField(qtheta=0.5, qphi=0.12)
    common = {
        "integration_backend": "face_flux_event",
        "surface_index": [0, 1, 2, 3],
        "phi_values": [0.03],
        "seed_count": 2,
        "seed_spacing": "theta",
        "n_turns": 1.0,
        "steps_per_turn": 6,
        "bidirectional": True,
    }
    serial = trace_j_streamlines_on_pest(field, pest, workers=1, **common)
    parallel = trace_j_streamlines_on_pest(field, pest, workers=3, **common)

    for name in ("R", "Z", "phi", "theta", "x", "y", "z"):
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
    assert parallel.metadata["seed_line_indices"] == serial.metadata[
        "seed_line_indices"
    ]
    assert parallel.metadata["face_flux_event"] == serial.metadata[
        "face_flux_event"
    ]
    assert parallel.metadata["parallel_trace"]["used_workers"] == 3


def test_face_flux_event_reuses_standard_seed_section_plot():
    import matplotlib.pyplot as plt

    pest = _toy_pest()
    lines = trace_j_streamlines_on_pest(
        _ConstantFaceFluxField(),
        pest,
        integration_backend="face_flux_event",
        surface_index=0,
        phi_values=[0.03],
        seed_count=2,
        n_turns=1.0,
        steps_per_turn=5,
        bidirectional=True,
    )
    figure, axes = plot_j_streamline_seed_sections(lines, pest)

    assert figure is not None
    assert axes.size == 1
    plt.close(figure)
