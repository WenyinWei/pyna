from __future__ import annotations

import numpy as np

from pyna.topo.island import Island, IslandChain
from pyna.topo.island_chain import ChainFixedPoint, IslandChainOrbit


def _fp(phi: float, kind: str) -> ChainFixedPoint:
    if kind == 'X':
        dpm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])
    else:
        dpm = np.eye(2)
    return ChainFixedPoint(
        phi=float(phi),
        R=1.0 + 0.1 * np.cos(phi),
        Z=0.1 * np.sin(phi),
        DPm=dpm,
        kind=kind,
        DX_pol_accum=np.eye(2),
    )


def test_island_chain_orbit_diagnostics_complete_pure_chain():
    phis = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    chain = IslandChainOrbit(
        m=10,
        n=3,
        Np=2,
        fixed_points=[_fp(phi, 'X') for phi in phis],
        seed_phi=0.0,
        seed_RZ=(1.1, 0.0),
        section_phis=list(phis),
        orbit_R=np.array([1.1, 1.0, 0.9]),
        orbit_Z=np.array([0.0, 0.05, 0.0]),
        orbit_phi=np.array([0.0, 0.4, 0.8]),
        orbit_alive=np.array([True, True, True]),
    )

    diag = chain.diagnostics(phis)
    assert diag['missing_sections'] == []
    assert diag['mixed_kind'] is False
    assert diag['kind_totals']['X'] == 4
    assert chain.is_complete(phis, expected_kind='X', expected_count_per_section=1)

    xyz = chain.orbit_xyz()
    assert xyz is not None
    assert xyz.shape == (3, 3)


def test_island_chain_orbit_diagnostics_mixed_and_missing_chain():
    phis = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    chain = IslandChainOrbit(
        m=10,
        n=3,
        Np=2,
        fixed_points=[_fp(0.0, 'O'), _fp(np.pi / 4, 'X'), _fp(np.pi / 2, 'O')],
        seed_phi=0.0,
        seed_RZ=(1.0, 0.0),
        section_phis=list(phis),
    )

    diag = chain.diagnostics(phis)
    assert diag['mixed_kind'] is True
    assert len(diag['missing_sections']) == 1
    assert chain.is_complete(phis, expected_kind='O', expected_count_per_section=1) is False

    summary = chain.debug_summary(phis, expected_kind='O')
    assert 'mixed_kind=True' in summary
    assert 'missing_sections' in summary


def test_island_chain_completeness_diagnostics():
    islands = [
        Island(period_n=10, O_point=np.array([1.0, 0.0]), label='a'),
        Island(period_n=10, O_point=np.array([0.9, 0.1]), label='b'),
    ]
    chain = IslandChain(m=3, n=1, islands=islands, connected=True)
    diag = chain.completeness_diagnostics()
    assert diag['expected_n_islands'] == 3
    assert diag['n_islands'] == 2
    assert diag['complete'] is False
    assert 'islands=2/3' in chain.summary()
