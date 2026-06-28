from __future__ import annotations

import numpy as np

from pyna.plot import (
    apply_section_limits,
    create_section_grid,
    cycles_for_section,
    draw_axis_point,
    draw_cycle_points,
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
from pyna.toroidal.flt import BoundaryIslandCycle


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


def _section_cycle(phi, cycle_id):
    pts = []
    for i, z in enumerate((-0.02, 0.02)):
        fp = FixedPoint(phi=float(phi), R=1.1 + 0.01 * i, Z=z, kind="X", DPm=np.eye(2))
        fp.metadata.update({
            "same_cycle_key": f"chain=0:cycle={cycle_id}:kind=X",
            "map_order_index": i,
            "orbit_point_index": i,
        })
        pts.append(fp)
    return BoundaryIslandCycle(
        points=tuple(pts),
        period=2,
        kind="X",
        map_span=np.pi,
        cycle_id=cycle_id,
        chain_id=0,
        closure_residual=1.0e-9,
        alive=True,
        metadata={"same_cycle_key": f"chain=0:cycle={cycle_id}:kind=X"},
    )


def test_plot_boundary_island_sections_compact_shared_axes(tmp_path):
    phi_sections = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    walls = [(1.0 + 0.2 * np.cos(theta), 0.2 * np.sin(theta)) for _ in phi_sections]
    cycles = {float(phi): [_section_cycle(phi, 4)] for phi in phi_sections}
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
        section_cycles=cycles,
        manifolds_by_section=manifolds,
        walls=walls,
        axis_by_section=[(1.0, 0.0)] * 4,
        out_path=out,
        compact=True,
        share_axes=True,
        aspect_ratio=1.0,
        label_cycle_ids=True,
    )

    assert out.exists()
    assert len(fig.axes) == 4
    for ax in axes.ravel():
        assert ax.get_aspect() == 1.0


def test_section_geometry_primitives_compose_core_and_edge_plot(tmp_path):
    phi_sections = [0.0, 0.5 * np.pi]
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    walls = [(1.0 + 0.25 * np.cos(theta), 0.25 * np.sin(theta)) for _ in phi_sections]
    cycles = {float(phi): [_section_cycle(phi, 2)] for phi in phi_sections}
    manifolds = {
        float(phi): [{
            "u_R": np.asarray([1.07, 1.12, 1.16]),
            "u_Z": np.asarray([0.00, 0.04, 0.08]),
            "u_lpol": np.asarray([0.00, 0.05, 0.10]),
            "s_R": np.asarray([1.13, 1.09, 1.05]),
            "s_Z": np.asarray([0.00, -0.04, -0.08]),
            "s_lpol": np.asarray([0.00, 0.05, 0.10]),
            "origin_R": 1.10,
            "origin_Z": 0.0,
            "cycle_id": 2,
            "map_order_index": 1,
            "same_cycle_key": "chain=0:cycle=2:kind=X",
        }]
        for phi in phi_sections
    }
    limits = section_data_limits(
        section_phis=phi_sections,
        background=_Background(),
        section_cycles=cycles,
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
        assert draw_manifold_points(ax, manifolds_for_section(manifolds, phi, idx), vmax=vmax)
        assert draw_cycle_points(
            ax,
            cycles_for_section(cycles, phi, idx),
            identity_to_color=identity_to_color,
            label_cycle_ids=True,
            label_template="{cycle_id}:P{index}",
        )
        assert draw_manifold_origins(
            ax,
            manifolds_for_section(manifolds, phi, idx),
            show_labels=True,
        )
        draw_axis_point(ax, 1.0, 0.0)
        format_section_axis(ax, section_phi=phi)

    apply_section_limits(axes, limits)
    trim_compact_tick_labels(axes, len(phi_sections), ncols=2)
    out = tmp_path / "section_primitives.png"
    assert save_figure(fig, out) == out
    assert out.exists()
    assert len(identity_to_color) == 1
