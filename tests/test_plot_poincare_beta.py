from __future__ import annotations

import numpy as np

from pyna.plot import (
    apply_section_limits,
    create_section_grid,
    orbits_for_section,
    draw_axis_point,
    draw_orbit_points,
    draw_manifold_lines,
    draw_manifold_origins,
    draw_manifold_points,
    draw_poincare_points,
    draw_wall_section,
    format_section_axis,
    manifold_lpol_max,
    manifolds_for_section,
    plot_boundary_island_sections,
    plot_poincare_beta_grid,
    save_figure,
    section_data_limits,
    trim_compact_tick_labels,
)
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import BoundaryIslandOrbit


def test_plot_poincare_beta_grid_smoke(tmp_path):
    phi_sections = [0.0, 0.25, 0.5, 0.75]
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    rows = []
    for beta in (0.0, 0.01):
        core_by_sec = []
        for sidx, _phi in enumerate(phi_sections):
            traces = []
            for rho in (0.25, 0.55):
                traces.append((
                    5.9 + 0.01 * sidx + rho * 0.2 * np.cos(theta),
                    rho * 0.2 * np.sin(theta),
                ))
            core_by_sec.append(traces)
        rows.append({
            "beta": beta,
            "core_by_sec": core_by_sec,
            "r_norm_core": np.array([0.25, 0.55]),
            "axis_by_sec": [(5.9, 0.0)] * 4,
        })

    out = tmp_path / "beta_grid.png"
    fig = plot_poincare_beta_grid(rows, phi_sections, out_path=out, vacuum_axis_R=5.9)

    assert out.exists()
    assert len(fig.axes) == 8


class _Background:
    n_seed = 2

    def section_points(self, section_index):
        theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        return (
            1.0 + 0.04 * np.cos(theta) + 0.01 * int(section_index),
            0.04 * np.sin(theta),
            np.mod(np.arange(theta.size), 2),
        )


def _section_orbit(phi, orbit_id):
    pts = []
    for i, z in enumerate((-0.02, 0.02)):
        fp = FixedPoint(phi=float(phi), R=1.1 + 0.01 * i, Z=z, kind="X", DPm=np.eye(2))
        fp.metadata.update({
            "same_orbit_key": f"chain=0:orbit={orbit_id}:kind=X",
            "map_order_index": i,
            "orbit_point_index": i,
        })
        pts.append(fp)
    return BoundaryIslandOrbit(
        points=tuple(pts),
        orbit_size=2,
        kind="X",
        map_span=np.pi,
        orbit_id=orbit_id,
        chain_id=0,
        closure_residual=1.0e-9,
        alive=True,
        metadata={"same_orbit_key": f"chain=0:orbit={orbit_id}:kind=X"},
    )


def test_plot_boundary_island_sections_compact_shared_axes(tmp_path):
    phi_sections = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    walls = [(1.0 + 0.2 * np.cos(theta), 0.2 * np.sin(theta)) for _ in phi_sections]
    orbits = {float(phi): [_section_orbit(phi, 4)] for phi in phi_sections}
    manifolds = [
        [{
            "u_R": np.asarray([1.05, 1.08, 1.12]),
            "u_Z": np.asarray([0.00, 0.03, 0.06]),
            "u_lpol": np.asarray([0.00, 0.03, 0.07]),
            "s_R": np.asarray([1.15, 1.12, 1.09]),
            "s_Z": np.asarray([0.00, -0.03, -0.06]),
            "s_lpol": np.asarray([0.00, 0.03, 0.07]),
        }]
        for _ in phi_sections
    ]

    out = tmp_path / "boundary_sections.png"
    fig, axes = plot_boundary_island_sections(
        phi_sections,
        background=_Background(),
        section_orbits=orbits,
        manifolds_by_section=manifolds,
        walls=walls,
        axis_by_section=[(1.0, 0.0)] * 4,
        out_path=out,
        compact=True,
        share_axes=True,
        aspect_ratio=1.0,
        label_orbit_ids=True,
    )

    assert out.exists()
    assert len(fig.axes) == 4
    for ax in axes.ravel():
        assert ax.get_aspect() == 1.0


def test_section_geometry_primitives_compose_core_and_edge_plot(tmp_path):
    phi_sections = [0.0, 0.5 * np.pi]
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    walls = [(1.0 + 0.25 * np.cos(theta), 0.25 * np.sin(theta)) for _ in phi_sections]
    orbits = {float(phi): [_section_orbit(phi, 2)] for phi in phi_sections}
    manifolds = {
        float(phi): [{
            "u_R": np.asarray([1.07, 1.12, 1.16]),
            "u_Z": np.asarray([0.00, 0.04, 0.08]),
            "u_lpol": np.asarray([0.00, 0.05, 0.10]),
            "u_generation": np.asarray([0, 1, 2]),
            "s_R": np.asarray([1.13, 1.09, 1.05]),
            "s_Z": np.asarray([0.00, -0.04, -0.08]),
            "s_lpol": np.asarray([0.00, 0.05, 0.10]),
            "s_generation": np.asarray([0, 1, 2]),
            "origin_R": 1.10,
            "origin_Z": 0.0,
            "orbit_id": 2,
            "map_order_index": 1,
            "same_orbit_key": "chain=0:orbit=2:kind=X",
        }]
        for phi in phi_sections
    }
    limits = section_data_limits(
        section_phis=phi_sections,
        background=_Background(),
        section_orbits=orbits,
        manifolds_by_section=manifolds,
        walls=walls,
    )
    assert limits is not None
    fig, axes = create_section_grid(
        phi_sections,
        ncols=2,
        compact=True,
        share_axes=True,
        data_limits=limits,
    )
    vmax = manifold_lpol_max(manifolds, phi_sections)
    identity_to_color = {}

    for idx, (ax, phi) in enumerate(zip(axes.ravel(), phi_sections)):
        draw_wall_section(ax, walls[idx][0], walls[idx][1])
        Rb, Zb, seed_idx = _Background().section_points(idx)
        pc = draw_poincare_points(ax, Rb, Zb, seed_idx, point_size=3.0)
        assert pc is not None
        assert draw_manifold_points(
            ax,
            manifolds_for_section(manifolds, phi, idx),
            vmax=vmax,
            max_generation=2,
        )
        assert draw_orbit_points(
            ax,
            orbits_for_section(orbits, phi, idx),
            identity_to_color=identity_to_color,
            label_orbit_ids=True,
            label_template="{orbit_id}:P{index}",
        )
        assert draw_manifold_origins(
            ax,
            manifolds_for_section(manifolds, phi, idx),
            show_labels=True,
        )
        assert draw_manifold_lines(
            ax,
            manifolds_for_section(manifolds, phi, idx),
            vmax=0.1,
            max_generation=1,
        )
        draw_axis_point(ax, 1.0, 0.0)
        format_section_axis(ax, section_phi=phi)

    apply_section_limits(axes, limits)
    trim_compact_tick_labels(axes, len(phi_sections), ncols=2)
    out = tmp_path / "section_primitives.png"
    assert save_figure(fig, out) == out
    assert out.exists()
    assert len(identity_to_color) == 1


def test_draw_manifold_lines_breaks_at_side_changes():
    import matplotlib.pyplot as plt

    manifolds = [{
        "u_R": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "u_Z": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "u_lpol": np.asarray([0.0, 0.1, 0.2, 0.3]),
        "u_point_side": np.asarray([-1.0, -1.0, 1.0, 1.0]),
        "s_R": np.asarray([], dtype=float),
        "s_Z": np.asarray([], dtype=float),
        "s_lpol": np.asarray([], dtype=float),
    }]
    fig, ax = plt.subplots()
    try:
        artists = draw_manifold_lines(ax, manifolds)
        assert len(artists) == 1
        assert len(artists[0].get_segments()) == 2
    finally:
        plt.close(fig)


def test_draw_manifold_origin_labels_prefer_map_order_index():
    import matplotlib.pyplot as plt

    manifolds = [{
        "origin_R": 1.0,
        "origin_Z": 0.0,
        "orbit_id": 2,
        "kind": "X",
        "map_power": 15,
        "map_order_index": 4,
        "u_R": np.asarray([], dtype=float),
        "u_Z": np.asarray([], dtype=float),
        "s_R": np.asarray([], dtype=float),
        "s_Z": np.asarray([], dtype=float),
    }]
    fig, ax = plt.subplots()
    try:
        artists = draw_manifold_origins(
            ax,
            manifolds,
            show_labels=True,
            draw_branch_anchors=False,
        )
        labels = [artist.get_text() for artist in artists if hasattr(artist, "get_text")]
        assert labels == ["X2:P4"]
    finally:
        plt.close(fig)
