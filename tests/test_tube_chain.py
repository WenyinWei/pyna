from __future__ import annotations

import numpy as np

from pyna.topo.island_chain import ChainFixedPoint, IslandChainOrbit
from pyna.topo.tube import ResonanceSkeleton, Tube, TubeChain, TubeCutPoint


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


def test_tube_wraps_orbit_and_maps_to_island():
    orbit = _orbit([
        (0.0, 1.00, 0.00, 'O'),
        (np.pi, 1.05, 0.02, 'O'),
    ])
    tube = Tube.from_orbit(orbit, label='o-tube')
    assert tube.kind == 'O'
    isl = tube.to_island(0.0, x_points=[np.array([1.10, 0.00])])
    assert np.allclose(isl.O_point, [1.00, 0.00])
    assert len(isl.X_points) == 1
    assert 'o-tube' in tube.summary()


def test_tube_chain_maps_to_discrete_island_chain():
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

    island_chain = o_chain.to_island_chain(0.0, x_tubechain=x_chain, proximity_tol=0.3)
    assert island_chain.n_islands == 2
    assert island_chain.expected_n_islands == 2
    assert all(len(isl.X_points) >= 1 for isl in island_chain.islands)


def test_tube_chain_diagnostics_report_incomplete_chain():
    orbit = _orbit([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.01, 0.00, 'O')], m=3, n=1, Np=1)
    chain = TubeChain.from_orbits([orbit], expected_kind='O')
    diag = chain.diagnostics([0.0, np.pi])
    assert diag['expected_n_tubes'] == 3
    assert diag['n_tubes'] == 1
    assert diag['complete'] is False
    assert diag['section_counts'][0.0] == 1


def test_tube_chain_section_cut_repairs_missing_point():
    orbit0 = _orbit([
        (0.0, 1.00, 0.00, 'O'),
        (np.pi, 1.02, 0.00, 'O'),
    ])
    # Missing exact cut at phi=0, but raw orbit contains a nearby center.
    orbit1 = _orbit(
        [(np.pi, 0.92, 0.00, 'O')],
        orbit_samples=[(0.0, 0.90, 0.02), (np.pi, 0.92, 0.00)],
    )
    chain = TubeChain.from_orbits([orbit0, orbit1], expected_kind='O')

    def finder(phi, tube, existing_points, reason):
        assert reason == 'missing'
        raw = tube.raw_point_near_section(phi)
        return (raw[0], raw[1])

    cut = chain.section_cut(0.0, repair=True, local_finder=finder)
    assert cut.is_complete()
    assert cut.missing_tube_indices == []
    assert cut.repaired_tube_indices == [1]
    assert len(cut.unique_points()) == 2


def test_tube_chain_section_cut_repairs_duplicate_point():
    orbit0 = _orbit(
        [(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')],
        orbit_samples=[(0.0, 1.00, 0.00), (np.pi, 1.02, 0.00)],
    )
    orbit1 = _orbit(
        [(0.0, 1.00, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')],
        orbit_samples=[(0.0, 0.90, 0.05), (np.pi, 0.92, 0.00)],
    )
    chain = TubeChain.from_orbits([orbit0, orbit1], expected_kind='O')

    def finder(phi, tube, existing_points, reason):
        if reason == 'duplicate':
            raw = tube.raw_point_near_section(phi)
            return (raw[0], raw[1])
        return None

    cut = chain.section_cut(0.0, dedup_tol=1e-10, repair=True, local_finder=finder)
    assert cut.is_complete()
    assert cut.duplicate_groups == []
    assert 1 in cut.repaired_tube_indices
    assert len(cut.unique_points(dedup_tol=1e-10)) == 2


def test_resonance_skeleton_provides_joint_section_view():
    o_orbits = [
        _orbit([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')]),
        _orbit([(0.0, 0.90, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')]),
    ]
    x_orbits = [
        _orbit([(0.0, 1.10, 0.00, 'X'), (np.pi, 1.12, 0.00, 'X')]),
        _orbit([(0.0, 0.80, 0.00, 'X'), (np.pi, 0.82, 0.00, 'X')]),
    ]
    skel = ResonanceSkeleton.from_orbits(o_orbits=o_orbits, x_orbits=x_orbits, label='2/1')
    cuts = skel.section_cut(0.0)
    assert cuts['O'] is not None and cuts['X'] is not None
    assert cuts['O'].is_complete()
    assert cuts['X'].is_complete()

    chains = skel.to_island_chains(0.0, proximity_tol=0.3)
    assert chains['O'] is not None
    assert chains['O'].n_islands == 2
    assert all(len(isl.X_points) >= 1 for isl in chains['O'].islands)

    anchors = skel.boundary_anchor_points(0.0)
    assert len(anchors) == 4
