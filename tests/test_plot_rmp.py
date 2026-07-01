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
