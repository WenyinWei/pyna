"""Tests for pyna.plot.tube plotting functions."""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.toroidal_invariants import Cycle, FixedPoint, MonodromyData
from pyna.topo.toroidal_tube import Tube, TubeChain


def _fp(phi=0.0, R=1.5, Z=0.0, kind='O') -> FixedPoint:
    if kind == 'X':
        DPm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])
    else:
        th = 0.4
        DPm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return FixedPoint(phi=phi, R=R, Z=Z, DPm=DPm, kind=kind)


def _make_tube(R=1.5, Z=0.0, kind='O', m=3, n=1) -> Tube:
    fp = _fp(phi=0.0, R=R, Z=Z, kind=kind)
    cycle = Cycle(winding=(m, n), sections={0.0: [fp]},
                  monodromy=fp.monodromy, ambient_dim=2)
    return Tube(o_cycle=cycle, x_cycles=[], label=f"{kind}-tube-R{R:.2f}")


def _make_tube_chain(n_tubes=3, kind='O', m=3, n=1) -> TubeChain:
    tubes = [_make_tube(R=1.50 + 0.02 * i, Z=0.0, kind=kind, m=m, n=n)
             for i in range(n_tubes)]
    tc = TubeChain(tubes=tubes)
    for t in tubes:
        t._tube_chain_ref = tc
    return tc


def test_wire_skeletons_noop():
    """wire_skeletons is a no-op for from_XO_fixed_points chains."""
    tc = _make_tube_chain(n_tubes=2, kind='O')
    tc.wire_skeletons(section_phi=0.0, proximity_tol=0.05)  # should not raise
    assert len(tc.tubes) == 2


def test_tube_chain_diagnostics():
    tc = _make_tube_chain(n_tubes=2, m=3)
    diag = tc.diagnostics([0.0])
    assert diag['m'] == 3
    assert diag['n_tubes'] == 2


def test_tube_section_view_points():
    tube = _make_tube(R=1.5, Z=0.0, kind='O')
    pts = tube.section_view_points(0.0, tube_index=0)
    assert len(pts) == 1
    assert abs(pts[0].R - 1.5) < 1e-9


@pytest.mark.parametrize("n_tubes", [1, 2, 3])
def test_tube_chain_raw_section_view(n_tubes):
    tc = _make_tube_chain(n_tubes=n_tubes, kind='O')
    view = tc.raw_section_view(0.0, kind='O')
    assert view is not None
    pts = view.unique_points()
    assert len(pts) == n_tubes


def test_plot_tube_chain_section_no_error():
    """plot_tube_chain_section runs without error on synthetic data."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyna.plot.tube import plot_tube_chain_section

    tc = _make_tube_chain(n_tubes=3, kind='O')
    fig, ax = plt.subplots()
    result_ax = plot_tube_chain_section(tc, section=0.0, ax=ax)
    assert result_ax is ax
    plt.close(fig)


def test_plot_resonance_section_from_tubechain_no_error():
    """TubeChain with both X and O points can be plotted."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyna.plot.tube import plot_resonance_section

    o_fps = [_fp(phi=0.0, R=1.50 + 0.02 * i, Z=0.0, kind='O') for i in range(3)]
    x_fps = [_fp(phi=0.0, R=1.51 + 0.02 * i, Z=0.0, kind='X') for i in range(3)]
    tc = TubeChain.from_XO_fixed_points(x_fps, o_fps, winding=(3, 1), label='test')

    fig, ax = plt.subplots()
    result_ax = plot_resonance_section(tc, section=0.0, ax=ax)
    assert result_ax is ax
    plt.close(fig)
