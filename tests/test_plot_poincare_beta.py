from __future__ import annotations

import numpy as np

from pyna.plot import plot_poincare_beta_grid


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
