import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.plot.j_streamlines import (
    GriddedPestVectorField,
    VmecCurrentFourier,
    plot_j_streamline_seed_sections,
    plot_j_streamlines_on_pest_surface_plotly,
    trace_j_streamlines_on_pest,
    vmec_current_fourier_to_pest_field,
)
from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates


def _toy_pest(n_phi=8, n_rho=3, n_theta=16):
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    rho = np.linspace(0.08, 0.24, n_rho)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    R = 1.0 + rho[None, :, None] * np.cos(theta)[None, None, :]
    Z = rho[None, :, None] * np.sin(theta)[None, None, :]
    R = np.repeat(R, n_phi, axis=0)
    Z = np.repeat(Z, n_phi, axis=0)
    return SmoothPestCoordinates(R_surf=R, Z_surf=Z, rho_vals=rho, theta_vals=theta, phi_vals=phi)


def _toroidal_current_field():
    R = np.linspace(0.65, 1.35, 17)
    Z = np.linspace(-0.35, 0.35, 19)
    Phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    shape = (R.size, Z.size, Phi.size)
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=np.zeros(shape),
        BZ=np.zeros(shape),
        BPhi=np.ones(shape),
        name="J_total",
    )


def _normal_plus_toroidal_current_field():
    R = np.linspace(0.65, 1.35, 33)
    Z = np.linspace(-0.35, 0.35, 35)
    Phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    RR, ZZ, _PP = np.meshgrid(R, Z, Phi, indexing="ij")
    rho = np.sqrt((RR - 1.0) ** 2 + ZZ**2)
    BR = np.zeros_like(rho)
    BZ = np.zeros_like(rho)
    valid = rho > 1.0e-12
    BR[valid] = (RR[valid] - 1.0) / rho[valid]
    BZ[valid] = ZZ[valid] / rho[valid]
    BPhi = np.full_like(BR, 0.1)
    return VectorFieldCylind(R=R, Z=Z, Phi=Phi, BR=BR, BZ=BZ, BPhi=BPhi, name="J_total")


class _SurfaceNativeToroidalCurrent:
    nfp = 1
    field_period_rad = 2.0 * np.pi

    def evaluate_pest_surface(self, surface_index, theta, phi, *, R=None, Z=None):
        theta = np.asarray(theta, dtype=np.float64)
        return np.stack(
            [
                np.zeros_like(theta),
                np.zeros_like(theta),
                np.ones_like(theta),
            ],
            axis=-1,
        )


def test_trace_j_streamlines_on_pest_uses_vector_field_and_seed_controls():
    pest = _toy_pest()
    field = _toroidal_current_field()

    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        surface_index=-1,
        phi_indices=[0],
        seed_count=5,
        n_turns=0.08,
        steps_per_turn=40,
    )

    assert lines.n_lines == 5
    assert lines.n_points == 7
    assert lines.metadata["trace_backend"] == "pyna.plot.j_streamlines.python_rk4_pest_surface_arclength"
    assert lines.metadata["trace_mode"] == "pest_surface_constrained"
    assert lines.metadata["surface_constraint"] is True
    assert lines.metadata["seed_count"] == 5
    assert lines.metadata["nfp"] == 1
    assert lines.metadata["field_period_rad"] == pytest.approx(2.0 * np.pi)
    assert lines.metadata["finite_fraction"] == pytest.approx(1.0)
    np.testing.assert_allclose(lines.R, np.repeat(lines.seed_R[:, None], lines.n_points, axis=1), atol=2.0e-4)
    np.testing.assert_allclose(lines.Z, np.repeat(lines.seed_Z[:, None], lines.n_points, axis=1), atol=2.0e-4)
    np.testing.assert_allclose(lines.theta, np.repeat(lines.seed_theta[:, None], lines.n_points, axis=1), atol=2.0e-4)


def test_trace_j_streamlines_accepts_surface_native_evaluator():
    pest = _toy_pest()

    lines = trace_j_streamlines_on_pest(
        _SurfaceNativeToroidalCurrent(),
        pest,
        surface_index=-1,
        phi_indices=[0],
        seed_count=3,
        n_turns=0.04,
        steps_per_turn=40,
    )

    assert lines.n_lines == 3
    assert lines.metadata["trace_mode"] == "pest_surface_constrained"
    assert lines.metadata["nfp"] == 1
    assert lines.metadata["finite_fraction"] == pytest.approx(1.0)
    np.testing.assert_allclose(lines.R, np.repeat(lines.seed_R[:, None], lines.n_points, axis=1), atol=2.0e-4)
    np.testing.assert_allclose(lines.Z, np.repeat(lines.seed_Z[:, None], lines.n_points, axis=1), atol=2.0e-4)


def test_trace_j_streamlines_accepts_gridded_pest_vector_field():
    pest = _toy_pest()
    shape = pest.R_surf.shape
    field = GriddedPestVectorField.from_pest_coordinates(
        pest,
        JR=np.zeros(shape),
        JZ=np.zeros(shape),
        JPhi=np.ones(shape),
        nfp=1,
        source="toy finite-beta J",
    )

    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        surface_index=-1,
        phi_indices=[0],
        seed_count=3,
        n_turns=0.04,
        steps_per_turn=40,
    )

    assert lines.n_lines == 3
    assert lines.metadata["trace_mode"] == "pest_surface_constrained"
    assert lines.metadata["nfp"] == 1
    np.testing.assert_allclose(lines.R, np.repeat(lines.seed_R[:, None], lines.n_points, axis=1), atol=2.0e-4)
    np.testing.assert_allclose(lines.Z, np.repeat(lines.seed_Z[:, None], lines.n_points, axis=1), atol=2.0e-4)


def test_trace_j_streamlines_supports_multiple_surfaces_and_phi_sector():
    pest = _toy_pest(n_phi=12, n_rho=4, n_theta=32)
    lines = trace_j_streamlines_on_pest(
        _toroidal_current_field(),
        pest,
        surface_index=[1, 3],
        phi_range=(0.0, np.pi / 2.0),
        phi_seed_count=2,
        seed_count=3,
        seed_spacing="arclength",
        n_turns=0.16,
        steps_per_turn=40,
    )

    assert lines.n_lines == 2 * 2 * 3
    assert set(lines.seed_surface_index.tolist()) == {1, 3}
    assert lines.metadata["seed_spacing"] == "arclength"
    assert lines.metadata["phi_seed_count"] == 2
    assert lines.metadata["phi_range"] == pytest.approx([0.0, np.pi / 2.0])
    finite_phi = lines.phi[np.isfinite(lines.phi)]
    assert finite_phi.size > 0
    assert np.all(np.mod(finite_phi, 2.0 * np.pi) <= np.pi / 2.0 + 1.0e-12)


def test_trace_gridded_pest_field_keeps_pure_poloidal_current_on_seed_section():
    pest = _toy_pest(n_phi=10, n_rho=3, n_theta=32)
    shape = pest.R_surf.shape
    field = GriddedPestVectorField.from_pest_coordinates(
        pest,
        JR=np.zeros(shape),
        JZ=np.zeros(shape),
        JPhi=np.zeros(shape),
        Jtheta=np.ones(shape),
        Jphi=np.zeros(shape),
        nfp=5,
        source="toy pure poloidal finite-beta J",
    )

    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        surface_index=-1,
        phi_indices=[0],
        seed_count=2,
        n_turns=0.12,
        steps_per_turn=80,
    )

    phi_span = np.nanmax(np.unwrap(lines.phi, axis=1), axis=1) - np.nanmin(np.unwrap(lines.phi, axis=1), axis=1)
    assert np.nanmax(phi_span) < 1.0e-10
    assert np.nanmax(np.abs(lines.theta - lines.seed_theta[:, None])) > 1.0e-3


def test_vmec_current_fourier_evaluates_radially_interpolated_modes():
    current = VmecCurrentFourier(
        s=np.array([0.0, 1.0]),
        xm=np.array([0.0, 1.0]),
        xn=np.array([0.0, 0.0]),
        sqrtgJ_u_cos=np.array([[2.0, 1.0], [4.0, 3.0]]),
        sqrtgJ_v_cos=np.array([[0.5, 0.0], [1.5, 0.0]]),
        sqrtgJ_u_sin=np.array([[0.0, 5.0], [0.0, 7.0]]),
        sqrtgJ_v_sin=np.zeros((2, 2)),
        nfp=5,
        source="synthetic",
    )

    rho = np.array([0.5])
    theta = np.array([np.pi / 2.0])
    zeta = np.array([0.0])
    ju, jv = current.evaluate(rho, theta, zeta)

    assert ju[0] == pytest.approx(2.5 + 0.0 + 5.5)
    assert jv[0] == pytest.approx(0.75)


def test_vmec_current_fourier_to_pest_field_keeps_poloidal_vmec_current_closed():
    pest = _toy_pest(n_phi=10, n_rho=3, n_theta=32)
    shape = pest.R_surf.shape
    current = VmecCurrentFourier(
        s=np.array([0.0, 1.0]),
        xm=np.array([0.0]),
        xn=np.array([0.0]),
        sqrtgJ_u_cos=np.ones((2, 1)),
        sqrtgJ_v_cos=np.zeros((2, 1)),
        nfp=5,
        source="synthetic",
    )
    theta_vmec = np.broadcast_to((-pest.theta_vals)[None, None, :], shape)
    zeta = np.broadcast_to(pest.phi_vals[:, None, None], shape)
    field = vmec_current_fourier_to_pest_field(
        pest,
        current,
        theta_vmec=theta_vmec,
        zeta=zeta,
        theta_pest_t=np.ones(shape),
        theta_pest_z=np.zeros(shape),
        vmec_to_desc_theta_sign=-1.0,
    )

    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        surface_index=-1,
        phi_indices=[0],
        seed_count=2,
        n_turns=0.12,
        steps_per_turn=80,
    )

    phi_span = np.nanmax(np.unwrap(lines.phi, axis=1), axis=1) - np.nanmin(np.unwrap(lines.phi, axis=1), axis=1)
    assert np.nanmax(phi_span) < 1.0e-10
    assert np.nanmax(np.abs(lines.theta - lines.seed_theta[:, None])) > 1.0e-3


def test_surface_constrained_j_streamlines_report_normal_leakage_without_leaving_surface():
    pest = _toy_pest()
    field = _normal_plus_toroidal_current_field()

    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        surface_index=-1,
        phi_indices=[0],
        seed_count=6,
        seed_spacing="theta",
        n_turns=0.04,
        steps_per_turn=40,
    )

    assert lines.metadata["trace_mode"] == "pest_surface_constrained"
    assert lines.metadata["normal_leakage_abs_over_norm_p95"] > 0.5
    assert lines.metadata["surface_tangent_fraction_median"] < 0.5
    radius = np.sqrt((lines.R - 1.0) ** 2 + lines.Z**2)
    np.testing.assert_allclose(radius, pest.rho_vals[-1], atol=2.0e-3)


def test_trace_j_streamlines_can_run_raw_cartesian_diagnostic_mode():
    lines = trace_j_streamlines_on_pest(
        _toroidal_current_field(),
        _toy_pest(),
        phi_indices=[0],
        seed_count=2,
        n_turns=0.02,
        steps_per_turn=24,
        constrain_to_surface=False,
    )

    assert lines.metadata["trace_backend"] == "pyna.plot.j_streamlines.python_rk4_cartesian_arclength"
    assert lines.metadata["trace_mode"] == "raw_cartesian_unconstrained"
    assert lines.metadata["surface_constraint"] is False
    assert np.isnan(lines.theta).all()


def test_trace_gridded_pest_field_can_run_cartesian_surface_projected_mode():
    pest = _toy_pest()
    shape = pest.R_surf.shape
    field = GriddedPestVectorField.from_pest_coordinates(
        pest,
        JR=np.zeros(shape),
        JZ=np.zeros(shape),
        JPhi=np.ones(shape),
        nfp=1,
        source="toy finite-beta J",
    )

    lines = trace_j_streamlines_on_pest(
        field,
        pest,
        phi_indices=[0],
        seed_count=2,
        n_turns=0.03,
        steps_per_turn=32,
        constrain_to_surface=False,
    )

    assert lines.metadata["trace_backend"] == "pyna.plot.j_streamlines.python_rk4_cartesian_surface_projected_arclength"
    assert lines.metadata["trace_mode"] == "cartesian_surface_projected"
    assert lines.metadata["surface_constraint"] is False
    assert np.isfinite(lines.R).all()
    assert np.isnan(lines.theta).all()


def test_j_streamline_helpers_are_exported_from_pyna_plot():
    import pyna.plot as pplot

    assert pplot.VmecCurrentFourier is VmecCurrentFourier
    assert pplot.trace_j_streamlines_on_pest is trace_j_streamlines_on_pest
    assert pplot.plot_j_streamlines_on_pest_surface_plotly is plot_j_streamlines_on_pest_surface_plotly
    assert pplot.vmec_current_fourier_to_pest_field is vmec_current_fourier_to_pest_field


def test_trace_j_streamlines_accepts_npz_with_r_vals(tmp_path):
    pest = _toy_pest()
    path = tmp_path / "coords_pest.npz"
    np.savez(
        path,
        R_surf=pest.R_surf,
        Z_surf=pest.Z_surf,
        r_vals=pest.rho_vals,
        theta_vals=pest.theta_vals,
        phi_vals=pest.phi_vals,
        R_AX=np.ones(pest.phi_vals.shape),
        Z_AX=np.zeros(pest.phi_vals.shape),
    )

    lines = trace_j_streamlines_on_pest(
        _toroidal_current_field(),
        path,
        phi_indices=[0],
        seed_count=2,
        n_turns=0.02,
        steps_per_turn=24,
    )

    assert lines.n_lines == 2
    assert lines.metadata["pest_source"] == str(path)


def test_plot_j_streamline_seed_sections_returns_matplotlib_axes():
    pest = _toy_pest()
    lines = trace_j_streamlines_on_pest(
        _toroidal_current_field(),
        pest,
        phi_indices=[0, 2],
        seed_count=4,
        n_turns=0.04,
        steps_per_turn=32,
    )

    fig, axes = plot_j_streamline_seed_sections(lines, pest, title="synthetic J streamlines")

    assert axes.shape == (1, 2)
    assert len(axes.ravel()[0].lines) > 0
    fig.canvas.draw()


def test_plot_raw_cartesian_streamlines_projects_to_seed_pest_section():
    pest = _toy_pest()
    lines = trace_j_streamlines_on_pest(
        _toroidal_current_field(),
        pest,
        phi_indices=[0],
        seed_count=3,
        n_turns=0.04,
        steps_per_turn=32,
        constrain_to_surface=False,
    )

    fig, axes = plot_j_streamline_seed_sections(
        lines,
        pest,
        title="raw Cartesian J streamlines",
        project_cartesian_to_pest=True,
    )

    assert axes.shape == (1, 1)
    assert len(axes.ravel()[0].lines) > 0
    fig.canvas.draw()


def test_plot_j_streamlines_on_pest_surface_plotly_writes_html(tmp_path):
    pytest.importorskip("plotly")
    pest = _toy_pest()
    lines = trace_j_streamlines_on_pest(
        _toroidal_current_field(),
        pest,
        phi_indices=[0],
        seed_count=3,
        n_turns=0.04,
        steps_per_turn=32,
    )
    out = tmp_path / "j_streamlines.html"

    fig = plot_j_streamlines_on_pest_surface_plotly(
        lines,
        pest,
        html_path=out,
        include_plotlyjs=False,
        show_surface=True,
        line_width=3.0,
    )

    assert out.exists()
    assert "Plotly.newPlot" in out.read_text(encoding="utf-8")
    assert len(fig.data) == 1 + lines.n_lines
    assert fig.data[0].type == "surface"
    assert fig.data[1].type == "scatter3d"


def test_plot_j_streamlines_plotly_supports_multiple_surfaces_and_companion(tmp_path):
    pytest.importorskip("plotly")
    pest = _toy_pest(n_phi=10, n_rho=4, n_theta=24)
    j_lines = trace_j_streamlines_on_pest(
        _toroidal_current_field(),
        pest,
        surface_index=[1, 3],
        phi_range=(0.0, np.pi),
        phi_seed_count=2,
        seed_count=2,
        n_turns=0.04,
        steps_per_turn=32,
    )
    b_lines = trace_j_streamlines_on_pest(
        _toroidal_current_field(),
        pest,
        surface_index=[1, 3],
        phi_range=(0.0, np.pi),
        phi_seed_count=1,
        seed_count=1,
        theta_offset=0.2,
        n_turns=0.04,
        steps_per_turn=32,
    )
    out = tmp_path / "j_b_streamlines.html"

    fig = plot_j_streamlines_on_pest_surface_plotly(
        j_lines,
        pest,
        surface_index=[1, 3],
        phi_range=(0.0, np.pi),
        companion_streamlines=b_lines,
        companion_name="B",
        html_path=out,
        include_plotlyjs=False,
        show_surface=True,
        surface_phi_samples=14,
        show_arrows=True,
        arrow_line_stride=1,
        companion_arrow_line_stride=1,
        line_width=3.0,
    )

    assert out.exists()
    assert sum(trace.type == "surface" for trace in fig.data) == 2
    assert np.asarray(fig.data[0].x).shape[0] == 14
    assert sum(trace.type == "scatter3d" for trace in fig.data) >= j_lines.n_lines + b_lines.n_lines
    assert sum(trace.type == "cone" for trace in fig.data) == 2
    assert any(trace.name == "J streamlines" for trace in fig.data)
    assert any(trace.name == "B streamlines" for trace in fig.data)
