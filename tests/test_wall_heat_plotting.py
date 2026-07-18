import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyna.plot.wall_heat import (
    camera_project_cylindrical,
    camera_xyz_from_cylindrical,
    plot_wall_heat_camera,
    plot_wall_heat_camera_surface,
    plot_wall_heat_camera_views,
    plot_wall_heat_footprint,
    project_strike_points_camera,
    wall_surface_heat_from_footprint,
    wall_heat_footprint_from_hits,
    write_wall_heat_surface_plotly_html,
)
import pyna.plot.wall_heat as wall_heat_module


def _toy_wall(n_phi=3, n_theta=12):
    wall_phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    wall_R = 1.0 + 0.2 * np.cos(theta)[None, :]
    wall_Z = 0.2 * np.sin(theta)[None, :]
    return wall_phi, np.repeat(wall_R, n_phi, axis=0), np.repeat(wall_Z, n_phi, axis=0)


def test_wall_heat_footprint_filters_and_bins_hits():
    wall_phi, wall_R, wall_Z = _toy_wall()
    hit_R = np.array([1.2, 1.0, 0.8, np.nan])
    hit_Z = np.array([0.0, 0.2, 0.0, 0.0])
    hit_phi = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 0.0])
    weights = np.array([2.0, 3.0, 5.0, 7.0])

    footprint = wall_heat_footprint_from_hits(
        hit_R,
        hit_Z,
        hit_phi,
        wall_phi,
        wall_R,
        wall_Z,
        weights=weights,
        n_phi_bins=3,
        n_s_bins=12,
    )

    assert footprint.heat.shape == (3, 12)
    np.testing.assert_allclose(footprint.heat.sum(), 10.0)
    np.testing.assert_allclose(footprint.hit_R, hit_R[:3])
    np.testing.assert_allclose(footprint.hit_Z, hit_Z[:3])
    np.testing.assert_allclose(footprint.hit_weight, weights[:3])
    assert np.all((footprint.hit_s >= 0.0) & (footprint.hit_s <= 1.0))
    assert np.all(footprint.hit_wall_distance < 1.0e-12)


def test_camera_projection_is_finite_for_front_facing_points():
    u, v, depth = camera_project_cylindrical(
        [1.0, 1.0],
        [0.0, 0.1],
        [0.0, 0.2],
        camera_position=(3.0, 0.0, 0.0),
    )

    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))
    assert np.all(depth > 0.0)


def test_project_strike_points_camera_keeps_front_points_visible():
    strike = {
        "R": np.array([1.0, 1.1, 1.2]),
        "Z": np.array([0.0, 0.1, -0.1]),
        "phi": np.array([0.45 * np.pi, 0.50 * np.pi, 0.55 * np.pi]),
    }

    projected = project_strike_points_camera(
        strike,
        camera_position=camera_xyz_from_cylindrical(3.0, 0.6, 0.5 * np.pi),
        camera_target=camera_xyz_from_cylindrical(1.0, 0.0, 0.5 * np.pi),
    )

    assert projected["visible"].shape == (3,)
    assert np.count_nonzero(projected["visible"]) == 3
    assert np.all(np.isfinite(projected["u"]))
    assert np.all(np.isfinite(projected["v"]))


def test_wall_heat_plot_helpers_return_artists():
    wall_phi, wall_R, wall_Z = _toy_wall()
    footprint = wall_heat_footprint_from_hits(
        [1.2, 1.0, 0.8],
        [0.0, 0.2, 0.0],
        [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0],
        wall_phi,
        wall_R,
        wall_Z,
        n_phi_bins=3,
        n_s_bins=12,
    )

    fig, (ax0, ax1) = plt.subplots(1, 2)
    mesh = plot_wall_heat_footprint(footprint, ax=ax0)
    scatter = plot_wall_heat_camera(footprint, ax=ax1, camera_position=(3.0, 0.0, 0.4), heatmap_alpha=0.41)

    assert mesh.get_array().size == 36
    assert scatter.get_offsets().shape[1] == 2
    assert scatter.get_alpha() == pytest.approx(0.41)
    plt.close(fig)


def test_wall_heat_camera_can_bin_hits():
    wall_phi, wall_R, wall_Z = _toy_wall()
    footprint = wall_heat_footprint_from_hits(
        [1.2, 1.2, 1.0],
        [0.0, 0.0, 0.2],
        [0.0, 0.01, 2.0 * np.pi / 3.0],
        wall_phi,
        wall_R,
        wall_Z,
        weights=[1.0, 2.0, 3.0],
        n_phi_bins=3,
        n_s_bins=12,
    )

    fig, ax = plt.subplots()
    image = plot_wall_heat_camera(footprint, ax=ax, camera_position=(3.0, 0.0, 0.4), bins=(8, 6))

    assert image.get_array().size == 48
    plt.close(fig)


def test_wall_heat_camera_accepts_wall_mesh_heatmap_view():
    wall_phi, wall_R, wall_Z = _toy_wall(n_phi=5, n_theta=24)
    footprint = wall_heat_footprint_from_hits(
        [1.2, 1.18, 1.0, 0.82],
        [0.0, 0.02, 0.2, 0.0],
        [0.0, 0.05, 2.0 * np.pi / 5.0, 4.0 * np.pi / 5.0],
        wall_phi,
        wall_R,
        wall_Z,
        weights=[1.0, 2.0, 3.0, 4.0],
        n_phi_bins=5,
        n_s_bins=12,
    )

    fig, ax = plt.subplots()
    image = plot_wall_heat_camera(
        footprint,
        ax=ax,
        camera_position=camera_xyz_from_cylindrical(3.0, 0.5, 0.0),
        wall_phi=wall_phi,
        wall_R=wall_R,
        wall_Z=wall_Z,
        bins=(12, 9),
        smooth_sigma=0.5,
        colorbar=False,
        background="#08090c",
    )

    assert image.get_array().size == 108
    assert len(ax.lines) > 0
    plt.close(fig)


def test_wall_surface_heat_spreads_hits_on_non_square_wall_grid():
    wall_phi, wall_R, wall_Z = _toy_wall(n_phi=5, n_theta=23)
    footprint = wall_heat_footprint_from_hits(
        [1.2, 1.18, 1.0, 0.82],
        [0.0, 0.02, 0.2, 0.0],
        [0.0, 0.05, 2.0 * np.pi / 5.0, 4.0 * np.pi / 5.0],
        wall_phi,
        wall_R,
        wall_Z,
        weights=[1.0, 2.0, 3.0, 4.0],
        n_phi_bins=5,
        n_s_bins=23,
    )

    surface = wall_surface_heat_from_footprint(
        footprint,
        wall_phi,
        wall_R,
        wall_Z,
        sigma_phi=0.2,
        sigma_s=0.04,
    )

    assert surface.heat.shape == (5, 23)
    assert surface.heat_flux.shape == (5, 23)
    assert surface.wall_xyz.shape == (5, 23, 3)
    np.testing.assert_allclose(surface.heat.sum(), footprint.hit_weight.sum(), rtol=1.0e-12, atol=1.0e-12)
    assert np.all(np.isfinite(surface.heat_flux))


def test_wall_heat_camera_surface_returns_poly_collection():
    wall_phi, wall_R, wall_Z = _toy_wall(n_phi=5, n_theta=23)
    footprint = wall_heat_footprint_from_hits(
        [1.2, 1.0, 0.8],
        [0.0, 0.2, 0.0],
        [0.0, 2.0 * np.pi / 5.0, 4.0 * np.pi / 5.0],
        wall_phi,
        wall_R,
        wall_Z,
        weights=[2.0, 3.0, 5.0],
        n_phi_bins=5,
        n_s_bins=23,
    )
    surface = wall_surface_heat_from_footprint(footprint, wall_phi, wall_R, wall_Z)

    fig, ax = plt.subplots()
    collection = plot_wall_heat_camera_surface(
        surface,
        ax=ax,
        camera_position=camera_xyz_from_cylindrical(3.0, 0.4, 0.0),
        empty_alpha=0.19,
        heat_alpha=0.63,
        rasterized=False,
        colorbar=False,
    )

    assert collection is not None
    assert collection.get_rasterized() is False
    assert collection.get_clip_on() is True
    assert collection.get_clip_box() is not None
    assert len(collection.get_paths()) > 0
    face_alpha = collection.get_facecolors()[:, 3]
    assert np.nanmin(face_alpha) == pytest.approx(0.19)
    assert np.nanmax(face_alpha) == pytest.approx(0.63)
    plt.close(fig)


def test_wall_heat_camera_views_share_physical_norm_and_overlay_hits(monkeypatch):
    wall_phi, wall_R, wall_Z = _toy_wall(n_phi=5, n_theta=23)
    footprint = wall_heat_footprint_from_hits(
        [1.2, 1.18, 1.0, 0.82],
        [0.0, 0.02, 0.2, 0.0],
        [0.0, 0.05, 2.0 * np.pi / 5.0, 4.0 * np.pi / 5.0],
        wall_phi,
        wall_R,
        wall_Z,
        weights=[1.0, 2.0, 3.0, 4.0],
        n_phi_bins=5,
        n_s_bins=23,
    )
    surface = wall_surface_heat_from_footprint(footprint, wall_phi, wall_R, wall_Z)
    received_norms = []
    original = wall_heat_module.plot_wall_heat_camera_surface

    def spy(*args, **kwargs):
        received_norms.append(kwargs["norm"])
        return original(*args, **kwargs)

    monkeypatch.setattr(wall_heat_module, "plot_wall_heat_camera_surface", spy)
    fig, axes, collections = plot_wall_heat_camera_views(
        surface,
        collision_hits=footprint,
        collision_alpha=0.37,
    )

    assert len(axes) == 4
    assert len(collections) == 4
    assert all(collection is not None for collection in collections)
    assert len({id(norm) for norm in received_norms}) == 1
    norm = received_norms[0]
    positive = surface.heat_flux[surface.heat_flux > 0.0]
    assert norm.vmin == pytest.approx(float(np.min(positive)))
    assert norm.vmax == pytest.approx(float(np.max(positive)))
    assert all(len(ax.collections) >= 2 for ax in axes)
    alpha = collections[0].get_facecolors()[:, 3]
    assert np.nanmin(alpha) == pytest.approx(0.22)
    assert np.nanmax(alpha) == pytest.approx(0.78)
    assert len(fig.axes) == 5
    plt.close(fig)


def test_wall_heat_surface_plotly_html_is_optional(tmp_path):
    pytest.importorskip("plotly")
    wall_phi, wall_R, wall_Z = _toy_wall(n_phi=5, n_theta=23)
    footprint = wall_heat_footprint_from_hits(
        [1.2, 1.0, 0.8],
        [0.0, 0.2, 0.0],
        [0.0, 2.0 * np.pi / 5.0, 4.0 * np.pi / 5.0],
        wall_phi,
        wall_R,
        wall_Z,
        weights=[2.0, 3.0, 5.0],
        n_phi_bins=5,
        n_s_bins=23,
    )
    surface = wall_surface_heat_from_footprint(footprint, wall_phi, wall_R, wall_Z)
    out = tmp_path / "wall_heat_surface.html"

    fig = write_wall_heat_surface_plotly_html(surface, out, include_plotlyjs=False, downsample=1)

    assert out.exists()
    assert "Plotly.newPlot" in out.read_text(encoding="utf-8")
    assert len(fig.data) == 1
    assert fig.data[0].opacity == pytest.approx(0.72)
