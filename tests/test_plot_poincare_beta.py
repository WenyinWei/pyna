from __future__ import annotations

import numpy as np

from pyna.plot import plot_boundary_island_sections, plot_poincare_beta_grid
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
