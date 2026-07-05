from __future__ import annotations

import numpy as np

from pyna.plot import plot_poincare, plot_poincare_data


def test_plot_poincare_data_splits_flat_fixed_length_payload(tmp_path):
    data = {
        "counts": np.asarray([3, 2]),
        "R_flat": np.asarray([
            1.00, 1.01, 1.02, 99.0,
            1.10, 1.11, 99.0, 99.0,
        ]),
        "Z_flat": np.asarray([
            0.00, 0.01, 0.02, 99.0,
            0.10, 0.11, 99.0, 99.0,
        ]),
        "turns": 4,
        "phi0": 0.25,
    }
    out = tmp_path / "poincare.png"

    fig, ax, plotted = plot_poincare_data(
        data,
        save_png=True,
        png_file=out,
        xlim=None,
        ylim=None,
    )

    assert out.exists()
    assert len(fig.axes) == 1
    assert ax.get_xlabel() == "R [m]"
    np.testing.assert_allclose(plotted["R_total"], [1.00, 1.01, 1.02, 1.10, 1.11])
    np.testing.assert_allclose(plotted["Z_total"], [0.00, 0.01, 0.02, 0.10, 0.11])
    np.testing.assert_array_equal(plotted["seed_index"], [0, 0, 0, 1, 1])


def test_plot_poincare_accepts_array_payload_and_boundary_file(tmp_path):
    boundary = tmp_path / "d1b.txt"
    boundary.write_text("R Z\n0.8 0.0\n1.2 0.0\n1.2 0.2\n0.8 0.0\n")
    R = np.asarray([[1.00, 1.01, np.nan], [1.10, 1.11, 1.12]])
    Z = np.asarray([[0.00, 0.01, np.nan], [0.10, 0.11, 0.12]])

    fig, ax, plotted = plot_poincare(
        R=R,
        Z=Z,
        plot_d1b=True,
        d1b_file=boundary,
        return_data=True,
        title="",
    )

    assert len(fig.axes) == 1
    assert len(ax.lines) == 1
    np.testing.assert_allclose(plotted["R_total"], [1.00, 1.01, 1.10, 1.11, 1.12])
    np.testing.assert_array_equal(plotted["seed_index"], [0, 0, 1, 1, 1])
