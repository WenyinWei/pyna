import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.plot.j_streamlines import (
    plot_j_streamline_seed_sections,
    plot_j_streamlines_on_pest_surface_plotly,
    trace_j_streamlines_on_pest,
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
    assert lines.metadata["trace_backend"] == "pyna.plot.j_streamlines.python_rk4_cartesian_arclength"
    assert lines.metadata["seed_count"] == 5
    assert lines.metadata["nfp"] == 1
    assert lines.metadata["field_period_rad"] == pytest.approx(2.0 * np.pi)
    assert lines.metadata["finite_fraction"] == pytest.approx(1.0)
    np.testing.assert_allclose(lines.R, np.repeat(lines.seed_R[:, None], lines.n_points, axis=1), atol=2.0e-4)
    np.testing.assert_allclose(lines.Z, np.repeat(lines.seed_Z[:, None], lines.n_points, axis=1), atol=2.0e-4)


def test_j_streamline_helpers_are_exported_from_pyna_plot():
    import pyna.plot as pplot

    assert pplot.trace_j_streamlines_on_pest is trace_j_streamlines_on_pest
    assert pplot.plot_j_streamlines_on_pest_surface_plotly is plot_j_streamlines_on_pest_surface_plotly


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
