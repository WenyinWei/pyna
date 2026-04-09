"""Tests for pyna.plot.tube plotting functions and TubeChain.wire_skeletons.

All tests use synthetic data only.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.island_chain import IslandChainOrbit, ChainFixedPoint
from pyna.topo.tube import Tube, TubeChain


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fp(phi=0.0, R=1.5, Z=0.0, kind='O'):
    """Make a synthetic ChainFixedPoint."""
    if kind == 'X':
        DPm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])  # |Tr|=3.33 > 2 → X
    else:
        th = 0.4
        DPm = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th), np.cos(th)]])       # rotation → O
    return ChainFixedPoint(phi=phi, R=R, Z=Z, DPm=DPm, DX_pol_accum=np.eye(2))


def _make_orbit(R=1.5, Z=0.0, kind='O', m=3, n=1, Np=2):
    """Make a minimal IslandChainOrbit with one fixed point at phi=0."""
    fp = _make_fp(phi=0.0, R=R, Z=Z, kind=kind)
    return IslandChainOrbit(
        m=m, n=n, Np=Np,
        fixed_points=[fp],
        seed_phi=0.0,
        seed_RZ=(R, Z),
        section_phis=[0.0],
    )


def _make_tube_chain(n_tubes=3, kind='O', m=3, n=1, Np=2):
    """Build a TubeChain with n_tubes at different Z positions."""
    from math import pi
    orbits = []
    for i in range(n_tubes):
        angle = 2 * pi * i / n_tubes
        R = 1.5 + 0.05 * np.cos(angle)
        Z = 0.05 * np.sin(angle)
        orbits.append(_make_orbit(R=R, Z=Z, kind=kind, m=m, n=n, Np=Np))
    return TubeChain.from_orbits(orbits, label=f'{kind}-chain')


# ── wire_skeletons tests ──────────────────────────────────────────────────────

def test_wire_skeletons_x_tube_gets_o_cycle():
    """X-Tube gets its o_cycle set to the nearby O-Tube after wire_skeletons."""
    # Build an O-orbit and an X-orbit close together
    o_orbit = _make_orbit(R=1.50, Z=0.0, kind='O', m=3, n=1)
    x_orbit = _make_orbit(R=1.52, Z=0.0, kind='X', m=3, n=1)

    chain = TubeChain.from_orbits([o_orbit, x_orbit])
    chain.wire_skeletons(section_phi=0.0, proximity_tol=0.1)

    o_tube = [t for t in chain.tubes if t._seed_kind() == 'O'][0]
    x_tube = [t for t in chain.tubes if t._seed_kind() == 'X'][0]

    # After wiring, X-tube knows its O-cycle
    assert x_tube.o_cycle is not None
    assert x_tube.o_cycle is o_tube


def test_wire_skeletons_o_tube_gets_x_cycle():
    """O-Tube gets at least one x_cycle after wire_skeletons."""
    o_orbit = _make_orbit(R=1.50, Z=0.0, kind='O', m=3, n=1)
    x_orbit = _make_orbit(R=1.52, Z=0.0, kind='X', m=3, n=1)

    chain = TubeChain.from_orbits([o_orbit, x_orbit])
    chain.wire_skeletons(section_phi=0.0, proximity_tol=0.1)

    o_tube = [t for t in chain.tubes if t._seed_kind() == 'O'][0]
    assert len(o_tube.x_cycles) >= 1


def test_wire_skeletons_too_far_not_wired():
    """Tubes more than proximity_tol apart should NOT be wired."""
    o_orbit = _make_orbit(R=1.50, Z=0.0, kind='O', m=3, n=1)
    x_orbit = _make_orbit(R=2.00, Z=0.5, kind='X', m=3, n=1)  # far away

    chain = TubeChain.from_orbits([o_orbit, x_orbit])
    chain.wire_skeletons(section_phi=0.0, proximity_tol=0.05)

    x_tube = [t for t in chain.tubes if t._seed_kind() == 'X'][0]
    assert x_tube.o_cycle is None  # not wired (too far)


def test_wire_xo_refs_alias():
    """wire_xo_refs is an alias for wire_skeletons and produces same result."""
    o_orbit = _make_orbit(R=1.50, Z=0.0, kind='O', m=3, n=1)
    x_orbit = _make_orbit(R=1.52, Z=0.0, kind='X', m=3, n=1)

    chain1 = TubeChain.from_orbits([o_orbit, x_orbit])
    chain1.wire_skeletons(section_phi=0.0, proximity_tol=0.1)

    chain2 = TubeChain.from_orbits([o_orbit, x_orbit])
    chain2.wire_xo_refs(section_phi=0.0, proximity_tol=0.1)

    # Both should give same wiring result
    for t1, t2 in zip(chain1.tubes, chain2.tubes):
        assert (t1.o_cycle is None) == (t2.o_cycle is None)
        assert len(t1.x_cycles) == len(t2.x_cycles)


# ── Plotting tests ────────────────────────────────────────────────────────────

pytest.importorskip('matplotlib', reason="matplotlib not installed")


def test_plot_tube_chain_section_no_error():
    """plot_tube_chain_section runs without error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyna.plot.tube import plot_tube_chain_section

    chain = _make_tube_chain(n_tubes=3, kind='O')
    fig, ax = plt.subplots()
    result_ax = plot_tube_chain_section(chain, section=0.0, ax=ax)
    assert result_ax is ax
    plt.close(fig)


def test_plot_tube_chain_section_with_connectivity():
    """show_connectivity=True runs without error (may draw no lines if not wired)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyna.plot.tube import plot_tube_chain_section

    chain = _make_tube_chain(n_tubes=3, kind='O')
    fig, ax = plt.subplots()
    # Connectivity is not wired → should still not crash
    result_ax = plot_tube_chain_section(chain, section=0.0, ax=ax,
                                         show_connectivity=True)
    assert result_ax is ax
    plt.close(fig)


def test_plot_island_chain_by_tube_no_error():
    """plot_island_chain_by_tube works for a chain with resonance_index set."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyna.plot.tube import plot_island_chain_by_tube
    from pyna.topo.island import Island

    # Build a small synthetic IslandChain
    from pyna.topo.island import IslandChain, Island
    chain = IslandChain(m=3, n=1)

    # Add islands with resonance_index
    for i in range(3):
        angle = 2 * np.pi * i / 3
        R = 1.5 + 0.05 * np.cos(angle)
        Z = 0.05 * np.sin(angle)
        isl = Island(
            period_n=3,
            O_point=np.array([R, Z]),
            X_points=[np.array([R + 0.03, Z])],
        )
        isl.resonance_index = i
        chain.islands.append(isl)

    fig, ax = plt.subplots()
    result_ax = plot_island_chain_by_tube(chain, ax=ax)
    assert result_ax is ax
    plt.close(fig)


def test_tube_chain_legend_count():
    """tube_chain_legend adds correct number of legend entries."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyna.plot.tube import tube_chain_legend, tube_legend_handles

    chain = _make_tube_chain(n_tubes=4, kind='O')
    fig, ax = plt.subplots()
    tube_chain_legend(chain, ax)
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_lines()) == 4  # one per tube
    plt.close(fig)


def test_plot_resonance_structure_section_no_error():
    """plot_resonance_structure_section runs without error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyna.plot.tube import plot_resonance_structure_section
    from pyna.topo.tube import ResonanceStructure

    o_orbits = [_make_orbit(R=1.50 + 0.02 * i, Z=0.0, kind='O', m=3, n=1)
                for i in range(3)]
    x_orbits = [_make_orbit(R=1.51 + 0.02 * i, Z=0.0, kind='X', m=3, n=1)
                for i in range(3)]
    rs = ResonanceStructure.from_orbits(o_orbits=o_orbits, x_orbits=x_orbits,
                                         label='test RS')

    fig, ax = plt.subplots()
    result_ax = plot_resonance_structure_section(rs, section=0.0, ax=ax)
    assert result_ax is ax
    plt.close(fig)
