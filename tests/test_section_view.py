"""Tests for SectionView and SectionViewBuilder."""
from __future__ import annotations

import numpy as np

from pyna.topo.identity import IslandID, ResonanceID, TubeID
from pyna.topo.toroidal import Island, IslandChain
from pyna.topo.toroidal import Cycle, FixedPoint, MonodromyData
from pyna.topo.toroidal_section_view import SectionView, SectionViewBuilder
from pyna.topo.toroidal import Tube, TubeChain


def _fp(phi: float, R: float, Z: float, kind: str) -> FixedPoint:
    if kind == 'X':
        dpm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])
    else:
        dpm = np.eye(2)
    return FixedPoint(phi=float(phi), R=float(R), Z=float(Z), DPm=dpm, kind=kind)


def _make_tube(points, m=2, n=1, orbit_samples=None) -> Tube:
    from collections import defaultdict
    sections = defaultdict(list)
    for (phi, R, Z, kind) in points:
        sections[float(phi)].append(_fp(phi, R, Z, kind))
    fps = [fp for fps in sections.values() for fp in fps]
    mono = fps[0].monodromy if fps else None
    cycle = Cycle(winding=(m, n), sections=dict(sections), monodromy=mono, ambient_dim=2)
    orb_R = orb_Z = orb_phi = orb_alive = None
    if orbit_samples is not None:
        orb_R     = np.array([p[1] for p in orbit_samples], dtype=float)
        orb_Z     = np.array([p[2] for p in orbit_samples], dtype=float)
        orb_phi   = np.array([p[0] for p in orbit_samples], dtype=float)
        orb_alive = np.ones(len(orbit_samples), dtype=bool)
    return Tube(o_cycle=cycle, x_cycles=[], _orbit_R=orb_R, _orbit_Z=orb_Z,
                _orbit_phi=orb_phi, _orbit_alive=orb_alive)


def test_section_view_from_tubechain_basic():
    tube0 = _make_tube([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')])
    tube1 = _make_tube([(0.0, 0.90, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')])
    tc = TubeChain(tubes=[tube0, tube1])
    view = SectionViewBuilder.from_tubechain(tc, phi=0.0, kind='O')
    assert view is not None
    pts = view.unique_points()
    assert len(pts) == 2


def test_section_view_correspondence_complete():
    tube0 = _make_tube([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')])
    tube1 = _make_tube([(0.0, 0.90, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')])
    tc = TubeChain(tubes=[tube0, tube1])
    view = SectionViewBuilder.from_tubechain(tc, phi=0.0, kind='O')
    assert view.correspondence is not None
    assert view.correspondence.is_complete()


def test_section_view_reconstruct_missing():
    tube0 = _make_tube([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')])
    tube1 = _make_tube(
        [(np.pi, 0.92, 0.00, 'O')],
        orbit_samples=[(0.0, 0.90, 0.02), (np.pi, 0.92, 0.00)],
    )
    tc = TubeChain(tubes=[tube0, tube1])
    tube0._tube_chain_ref = tc
    tube1._tube_chain_ref = tc

    def finder(phi, tube, existing, reason):
        raw = tube.raw_point_near_section(phi)
        return (raw[0], raw[1]) if raw else None

    view = SectionViewBuilder.from_tubechain(
        tc, phi=0.0, kind='O', reconstruct=True, section_reconstructor=finder,
    )
    assert view.correspondence is not None
    assert view.correspondence.is_complete()
    assert len(view.correspondence.reconstructed_tube_ids) == 1


def test_section_view_to_island_chain():
    tube0 = _make_tube([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')])
    tube1 = _make_tube([(0.0, 0.90, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')])
    tc = TubeChain(tubes=[tube0, tube1])
    view = SectionViewBuilder.from_tubechain(tc, phi=0.0, kind='O')
    chain = view.to_island_chain()
    assert chain.n_islands == 2
