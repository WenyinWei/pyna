from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")


def _component():
    def fixed_points(phi):
        return {
            "theta_O": np.array([[0.0, np.pi]]),
            "theta_X": np.array([[0.5 * np.pi, 1.5 * np.pi]]),
        }

    return SimpleNamespace(
        m=2,
        n=1,
        harmonic_order=1,
        psi_res=0.5,
        q_res=2.0,
        half_width_r=0.02,
        fixed_points=fixed_points,
    )


def _eq():
    return SimpleNamespace(R0=3.0, r0=0.3)


def _trace():
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    return 3.0 + 0.2 * np.cos(theta), 0.2 * np.sin(theta)


def test_draw_rmp_resonance_section_smoke():
    import matplotlib.pyplot as plt

    from pyna.plot import draw_rmp_resonance_section

    R, Z = _trace()
    fig, ax = plt.subplots()
    payload = draw_rmp_resonance_section(
        ax,
        R,
        Z,
        eq=_eq(),
        components=[_component()],
    )

    assert payload["psi_values"].shape == R.shape
    assert len(payload["x_points"]) == 2
    assert len(payload["o_points"]) == 2
    assert len(payload["island_width_bars"]) == 2
    assert all(len(line.get_xdata()) == 17 for line in payload["island_width_bars"])
    assert len(ax.lines) > 0
    assert len(ax.collections) > 0
    plt.close(fig)


def test_plot_rmp_resonance_sections_smoke():
    import matplotlib.pyplot as plt

    from pyna.plot import plot_rmp_resonance_sections

    R, Z = _trace()
    fig, axes = plot_rmp_resonance_sections(
        [{"R": R, "Z": Z}, {"R": R, "Z": -Z}],
        [0.0, np.pi / 3.0],
        eq=_eq(),
        components=[_component()],
        ncols=2,
    )

    assert axes.shape == (1, 2)
    assert len(fig.axes) >= 3  # two panels plus colorbar
    plt.close(fig)


def test_rmp_section_overlay_helpers_are_composable():
    import matplotlib.pyplot as plt

    from pyna.plot.rmp import (
        create_rmp_section_layout,
        draw_rmp_poincare_points,
        draw_rmp_section_overlays,
    )

    R, Z = _trace()
    fig, ax = plt.subplots()
    points = draw_rmp_poincare_points(ax, R, Z, eq=_eq())
    payload = draw_rmp_section_overlays(
        ax,
        eq=_eq(),
        components=[_component()],
        R=R,
        Z=Z,
        overlays=("pest", "surfaces", "bars", "xo"),
    )

    assert points["psi_values"].shape == R.shape
    assert payload["poincare"] is None
    assert len(payload["pest_grid"]) > 0
    assert len(payload["resonant_surfaces"]) == 1
    assert len(payload["island_width_bars"]) == 2
    assert len(payload["x_points"]) == 2
    plt.close(fig)

    fig2, axes = create_rmp_section_layout([0.0, np.pi / 2.0], eq=_eq(), ncols=2, compact=True)
    assert axes.shape == (1, 2)
    assert fig2.subplotpars.wspace == 0.0
    assert fig2.subplotpars.hspace == 0.0
    plt.close(fig2)


def test_rmp_visual_short_wrappers_delegate_to_plot_module():
    import matplotlib.pyplot as plt

    from pyna.toroidal.visual.RMP_spectrum import plot_rmp_section, rmp_section_layout

    R, Z = _trace()
    fig, ax = plt.subplots()
    payload = plot_rmp_section(
        ax,
        R,
        Z,
        eq=_eq(),
        components=[_component()],
        overlays=("points", "xo"),
    )

    assert payload["psi_values"].shape == R.shape
    assert len(payload["x_points"]) == 2
    plt.close(fig)

    fig2, axes = rmp_section_layout([0.0], eq=_eq(), ncols=1)
    assert axes.shape == (1, 1)
    plt.close(fig2)
