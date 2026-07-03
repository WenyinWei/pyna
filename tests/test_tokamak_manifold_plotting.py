import numpy as np
import pytest


def test_draw_manifold_segments_accepts_nested_segments():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from pyna.toroidal.visual.tokamak_manifold import draw_manifold_segments

    t = np.linspace(0.0, 1.0, 8)
    seg_a = np.column_stack((1.0 + 0.1 * t, 0.05 * np.sin(t)))
    seg_b = np.column_stack((1.0 + 0.1 * t, 0.05 * np.cos(t)))

    fig, ax = plt.subplots()
    collections = draw_manifold_segments(ax, {"arm+": seg_a, "arm-": [seg_b]}, unstable=False)

    assert len(collections) == 2
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_poincare_manifold_section_layers_are_optional():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from pyna.toroidal.visual.tokamak_manifold import plot_poincare_manifold_section

    t = np.linspace(0.0, 1.0, 10)
    stable = np.column_stack((1.0 + 0.04 * t, 0.04 * t))
    unstable = np.column_stack((1.0 + 0.04 * t, -0.04 * t))
    points = np.column_stack((1.0 + 0.02 * t, 0.02 * np.sin(4.0 * t)))

    fig, ax = plt.subplots()
    artists = plot_poincare_manifold_section(
        ax,
        poincare_points=points,
        stable_segments=[stable],
        unstable_segments=[unstable],
        x_points=np.array([[1.0, 0.0]]),
        o_points=np.array([[1.02, 0.0]]),
    )

    assert artists["poincare"] is not None
    assert len(artists["stable"]) == 1
    assert len(artists["unstable"]) == 1
    assert ax.get_aspect() == 1.0
    plt.close(fig)
