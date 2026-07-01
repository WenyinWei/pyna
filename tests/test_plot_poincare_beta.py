from __future__ import annotations

import numpy as np

from pyna.plot import (
    apply_section_limits,
    branch_payload_point_subset,
    branch_payload_smax,
    clip_branch_payloads_by_arclength,
    contiguous_prefix_mask,
    create_section_grid,
    orbits_for_section,
    draw_axis_point,
    draw_branch_manifold_lines,
    draw_fixed_point_orbits,
    draw_orbit_points,
    draw_manifold_lines,
    draw_manifold_origins,
    draw_manifold_points,
    draw_poincare_points,
    draw_poincare_background_by_seed_value,
    draw_wall_section,
    format_section_axis,
    map_order_value,
    manifold_lpol_max,
    manifolds_for_section,
    poincare_seed_values_for_points,
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
            "u_point_side": np.asarray([1.0, 1.0, 1.0]),
            "s_R": np.asarray([1.13, 1.09, 1.05]),
            "s_Z": np.asarray([0.00, -0.04, -0.08]),
            "s_lpol": np.asarray([0.00, 0.05, 0.10]),
            "s_generation": np.asarray([0, 1, 2]),
            "s_point_side": np.asarray([-1.0, -1.0, -1.0]),
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


def test_poincare_seed_values_expand_to_section_points():
    seed_index = np.asarray([0, 2, 1, 4])
    seed_values = np.asarray([1.0, 3.0, np.inf])

    values = poincare_seed_values_for_points(seed_index, seed_values)

    assert values[0] == 1.0
    assert values[1] == np.inf
    assert values[2] == 3.0
    assert np.isnan(values[3])


def test_draw_poincare_points_explicit_color_overrides_seed_colormap():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        artist = draw_poincare_points(
            ax,
            [1.0, 1.1],
            [0.0, 0.1],
            seed_index=[0, 1],
            color="0.5",
        )
        colors = artist.get_facecolors()
        assert colors.shape[0] == 1 or np.allclose(colors[:, :3], colors[0, :3])
    finally:
        plt.close(fig)


def test_draw_poincare_background_by_seed_value_splits_finite_and_infinite():
    import matplotlib.pyplot as plt

    class Background:
        def section_points(self, section_index):
            assert section_index == 0
            return (
                np.asarray([1.0, 1.1, 1.2]),
                np.asarray([0.0, 0.1, 0.2]),
                np.asarray([0, 1, 2]),
            )

    fig, ax = plt.subplots()
    try:
        artists = draw_poincare_background_by_seed_value(
            ax,
            Background(),
            0,
            np.asarray([10.0, np.inf, 100.0]),
            transform="log10",
            nonfinite_color="0.5",
        )
        assert len(artists) == 2
        assert artists[0].get_offsets().shape[0] == 2
        assert artists[1].get_offsets().shape[0] == 1
    finally:
        plt.close(fig)


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


def test_draw_manifold_lines_requires_branch_side_metadata():
    import matplotlib.pyplot as plt

    manifolds = [{
        "u_R": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "u_Z": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "u_lpol": np.asarray([0.0, 0.1, 0.2, 0.3]),
        "s_R": np.asarray([], dtype=float),
        "s_Z": np.asarray([], dtype=float),
        "s_lpol": np.asarray([], dtype=float),
    }]
    fig, ax = plt.subplots()
    try:
        assert draw_manifold_lines(ax, manifolds) == []
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


def test_draw_orbit_point_labels_prefer_map_order_index():
    import matplotlib.pyplot as plt

    fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, kind="X", DPm=np.eye(2))
    fp.map_power = 15
    fp.metadata.update({
        "map_power": 15,
        "map_order_index": 4,
    })
    orbit = BoundaryIslandOrbit(
        points=(fp,),
        orbit_size=1,
        kind="X",
        map_span=np.pi,
        orbit_id=2,
        chain_id=0,
        closure_residual=1.0e-9,
        alive=True,
    )

    fig, ax = plt.subplots()
    try:
        artists = draw_orbit_points(
            ax,
            [orbit],
            label_orbit_ids=True,
            label_template="{orbit_id}:P{map_power}",
        )
        labels = [artist.get_text() for artist in artists if hasattr(artist, "get_text")]
        assert labels == ["2:P4"]
    finally:
        plt.close(fig)


def test_map_order_value_prefers_physical_order_metadata():
    fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, kind="X", DPm=np.eye(2))
    fp.map_order_index = 3
    fp.metadata.update({
        "physical_map_order_index": 7,
        "map_order_index": 4,
        "map_power": 15,
    })

    assert map_order_value(fp) == 7


def test_draw_fixed_point_orbits_labels_xo_by_physical_map_order():
    import matplotlib.pyplot as plt

    fp_x = FixedPoint(phi=0.0, R=1.0, Z=0.0, kind="X", DPm=np.eye(2))
    fp_x.metadata.update({"physical_map_order_index": 2})
    fp_o = FixedPoint(phi=0.0, R=1.1, Z=0.0, kind="O", DPm=np.eye(2))
    fp_o.metadata.update({"physical_map_order_index": 3})
    orbit_x = BoundaryIslandOrbit(
        points=(fp_x,),
        orbit_size=1,
        kind="X",
        map_span=np.pi,
        orbit_id=0,
        chain_id=0,
        closure_residual=1.0e-9,
        alive=True,
    )
    orbit_o = BoundaryIslandOrbit(
        points=(fp_o,),
        orbit_size=1,
        kind="O",
        map_span=np.pi,
        orbit_id=1,
        chain_id=0,
        closure_residual=1.0e-9,
        alive=True,
    )

    fig, ax = plt.subplots()
    try:
        artists = draw_fixed_point_orbits(
            ax,
            [orbit_x, orbit_o],
            show_labels=True,
            x_color="#1b4f9c",
            o_color="#c93c36",
        )
        labels = [artist.get_text() for artist in artists if hasattr(artist, "get_text")]
        assert labels == ["X0:P2", "O1:P3"]
    finally:
        plt.close(fig)


def test_draw_branch_manifold_lines_uses_branch_kind_and_side_payloads():
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    payloads = [
        {
            "R": np.asarray([1.0, 1.05, 1.11]),
            "Z": np.asarray([0.0, 0.02, 0.06]),
            "s": np.asarray([0.0, 0.1, 0.3]),
            "branch_kind": "unstable",
            "side": "+",
        },
        {
            "R": np.asarray([1.0, 0.95, 0.89]),
            "Z": np.asarray([0.0, -0.02, -0.06]),
            "s": np.asarray([0.0, 0.1, 0.3]),
            "branch_kind": "stable",
            "side": "-",
        },
    ]

    fig, ax = plt.subplots()
    try:
        assert branch_payload_smax(payloads) == 0.3
        artists = draw_branch_manifold_lines(ax, payloads, smax=0.3)
        assert len(artists) == 2
        assert all(isinstance(artist, LineCollection) for artist in artists)
        assert all(len(artist.get_segments()) == 2 for artist in artists)
    finally:
        plt.close(fig)


def test_contiguous_prefix_mask_stops_at_first_false():
    mask = np.asarray([True, True, False, True, True])

    np.testing.assert_array_equal(
        contiguous_prefix_mask(mask),
        np.asarray([True, True, False, False, False]),
    )


def test_draw_branch_manifold_lines_arclength_clip_uses_prefix():
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    payloads = [{
        "R": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        "Z": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0]),
        "s": np.asarray([0.0, 0.1, 0.4, 0.2, 0.21]),
        "branch_kind": "unstable",
        "side": "+",
    }]

    fig, ax = plt.subplots()
    try:
        artists = draw_branch_manifold_lines(ax, payloads, max_arclength=0.25)
        line_artists = [artist for artist in artists if isinstance(artist, LineCollection)]
        assert len(line_artists) == 1
        assert len(line_artists[0].get_segments()) == 1
    finally:
        plt.close(fig)


def test_clip_branch_payloads_by_arclength_returns_prefix_payload():
    payload = {
        "R": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "Z": np.asarray([0.0, 0.0, 0.0, 0.0]),
        "s": np.asarray([0.0, 0.2, 0.5, 0.1]),
        "branch_kind": "stable",
    }

    clipped = clip_branch_payloads_by_arclength([payload], 0.25)

    assert len(clipped) == 1
    np.testing.assert_allclose(clipped[0]["R"], [0.0, 1.0])
    np.testing.assert_allclose(clipped[0]["s"], [0.0, 0.2])
    assert clipped[0]["arclength_clip_mode"] == "prefix"


def test_branch_payload_point_subset_keeps_per_point_arrays_only():
    payload = {
        "R": np.asarray([0.0, 1.0, 2.0]),
        "Z": np.asarray([0.0, 0.1, 0.2]),
        "s": np.asarray([0.0, 0.1, 0.2]),
        "branch_kind": "unstable",
    }

    child = branch_payload_point_subset(payload, [True, False, True])

    assert child is not None
    np.testing.assert_allclose(child["R"], [0.0, 2.0])
    np.testing.assert_allclose(child["Z"], [0.0, 0.2])
    assert child["branch_kind"] == "unstable"


def test_manifolds_for_section_accepts_single_generic_branch_payload():
    payload = {
        "R": np.asarray([1.0, 1.1]),
        "Z": np.asarray([0.0, 0.1]),
        "s": np.asarray([0.0, 0.2]),
        "branch_kind": "unstable",
    }

    assert manifolds_for_section(payload, 0.0, 0) == [payload]


def test_section_grid_validates_aspect_ratio_and_trims_with_effective_columns():
    import matplotlib.pyplot as plt
    import pytest

    with pytest.raises(ValueError):
        create_section_grid([0.0], aspect_ratio=0.0)

    fig, axes = create_section_grid([0.0, 1.0, 2.0], ncols=0)
    try:
        trim_compact_tick_labels(axes, 3, ncols=0)
        assert axes.shape[1] == 1
        assert axes.ravel()[-1].get_visible()
    finally:
        plt.close(fig)
