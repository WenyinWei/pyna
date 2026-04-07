from __future__ import annotations

import numpy as np

from pyna.topo.identity import IslandID, ResonanceID, TubeID
from pyna.topo.island import Island, IslandChain
from pyna.topo.island_chain import ChainFixedPoint, IslandChainOrbit
from pyna.topo.section_view import SectionView, SectionViewBuilder
from pyna.topo.tube import TubeChain


def _fp(phi: float, R: float, Z: float, kind: str) -> ChainFixedPoint:
    if kind == 'X':
        dpm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])
    else:
        dpm = np.eye(2)
    return ChainFixedPoint(phi=float(phi), R=float(R), Z=float(Z), DPm=dpm, DX_pol_accum=np.eye(2))


def _orbit(points, m=2, n=1, Np=1, orbit_samples=None):
    phis = [p[0] for p in points]
    if orbit_samples is None:
        orbit_samples = [(p[0], p[1], p[2]) for p in points]
    return IslandChainOrbit(
        m=m,
        n=n,
        Np=Np,
        fixed_points=[_fp(*p) for p in points],
        seed_phi=float(points[0][0]),
        seed_RZ=(float(points[0][1]), float(points[0][2])),
        section_phis=list(phis),
        orbit_R=np.array([p[1] for p in orbit_samples], dtype=float),
        orbit_Z=np.array([p[2] for p in orbit_samples], dtype=float),
        orbit_phi=np.array([p[0] for p in orbit_samples], dtype=float),
        orbit_alive=np.ones(len(orbit_samples), dtype=bool),
    )


def test_section_view_builder_from_tubechain_roundtrip_to_island_chain():
    o_orbits = [
        _orbit([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')]),
        _orbit([(0.0, 0.90, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')]),
    ]
    x_orbits = [
        _orbit([(0.0, 1.10, 0.00, 'X'), (np.pi, 1.12, 0.00, 'X')]),
        _orbit([(0.0, 0.80, 0.00, 'X'), (np.pi, 0.82, 0.00, 'X')]),
    ]
    o_chain = TubeChain.from_orbits(o_orbits, expected_kind='O', label='o-chain')
    x_chain = TubeChain.from_orbits(x_orbits, expected_kind='X', label='x-chain')

    o_view = SectionViewBuilder.from_tubechain(o_chain, 0.0, kind='O')
    x_view = SectionViewBuilder.from_tubechain(x_chain, 0.0, kind='X')

    assert o_view.correspondence is not None
    assert o_view.correspondence.is_complete()
    assert len(o_view.correspondence.tube_to_point_indices) == 2
    assert all(isinstance(pt.tube_id, TubeID) for pt in o_view.points)
    assert all(isinstance(pt.island_id, IslandID) for pt in o_view.points)

    island_chain = o_view.to_island_chain(x_section_view=x_view, proximity_tol=0.3)
    assert island_chain.n_islands == 2
    assert island_chain.expected_n_islands == 2


def test_section_view_from_island_chain_preserves_discrete_first_class_route():
    chain = IslandChain(
        m=2,
        n=1,
        islands=[
            Island(period_n=2, O_point=np.array([1.0, 0.0]), X_points=[np.array([1.1, 0.0])], label='a'),
            Island(period_n=2, O_point=np.array([0.9, 0.0]), X_points=[np.array([0.8, 0.0])], label='b'),
        ],
    )
    res = ResonanceID(m=2, n=1, Np=1, label='2/1')
    view = SectionView.from_island_chain(chain, phi=0.0, resonance_id=res, kind='O')
    assert view.correspondence is not None
    assert view.correspondence.is_complete()
    assert len(view.points) == 2
    chain2 = view.to_island_chain(proximity_tol=0.3)
    assert chain2.n_islands == 2


def test_reconstruct_section_view_marks_reconstructed_tube_ids():
    orbit0 = _orbit([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')])
    orbit1 = _orbit(
        [(np.pi, 0.92, 0.00, 'O')],
        orbit_samples=[(0.0, 0.90, 0.02), (np.pi, 0.92, 0.00)],
    )
    chain = TubeChain.from_orbits([orbit0, orbit1], expected_kind='O', label='o-chain')

    def finder(phi, tube, existing_points, reason):
        assert reason == 'missing'
        raw = tube.raw_point_near_section(phi)
        return (raw[0], raw[1])

    view = SectionViewBuilder.from_tubechain(chain, 0.0, kind='O', reconstruct=True, local_finder=finder)
    assert view.correspondence is not None
    assert view.correspondence.is_complete()
    assert len(view.correspondence.reconstructed_tube_ids) == 1
    assert view.correspondence.reconstructed_tube_ids[0].tube_index == 1
