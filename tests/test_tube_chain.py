from __future__ import annotations

import numpy as np

from pyna.topo.toroidal_invariants import Cycle, FixedPoint, MonodromyData
from pyna.topo.toroidal_tube import Tube, TubeChain, TubeCutPoint
from pyna.topo.section import HyperplaneSection


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fp(phi: float, R: float, Z: float, kind: str) -> FixedPoint:
    if kind == 'X':
        dpm = np.array([[3.0, 0.0], [0.0, 1.0 / 3.0]])
    else:
        dpm = np.eye(2)
    return FixedPoint(phi=float(phi), R=float(R), Z=float(Z), DPm=dpm, kind=kind)


def _cycle(points, m=2, n=1) -> Cycle:
    """Build a Cycle from (phi, R, Z, kind) tuples."""
    from collections import defaultdict
    sections = defaultdict(list)
    for (phi, R, Z, kind) in points:
        sections[float(phi)].append(_fp(phi, R, Z, kind))
    fps_all = [fp for fps in sections.values() for fp in fps]
    mono = fps_all[0].monodromy if fps_all else None
    return Cycle(winding=(m, n), sections=dict(sections), monodromy=mono, ambient_dim=2)


def _tube(points, m=2, n=1, orbit_samples=None, label=None) -> Tube:
    """Build a Tube from (phi, R, Z, kind) tuples, with optional raw trajectory."""
    cycle = _cycle(points, m=m, n=n)
    orb_R = orb_Z = orb_phi = orb_alive = None
    if orbit_samples is not None:
        orb_R     = np.array([p[1] for p in orbit_samples], dtype=float)
        orb_Z     = np.array([p[2] for p in orbit_samples], dtype=float)
        orb_phi   = np.array([p[0] for p in orbit_samples], dtype=float)
        orb_alive = np.ones(len(orbit_samples), dtype=bool)
    return Tube(
        o_cycle=cycle, x_cycles=[],
        label=label,
        _orbit_R=orb_R, _orbit_Z=orb_Z,
        _orbit_phi=orb_phi, _orbit_alive=orb_alive,
    )


def test_tube_wraps_cycle_and_maps_to_island():
    tube = _tube([
        (0.0, 1.00, 0.00, 'O'),
        (np.pi, 1.05, 0.02, 'O'),
    ], label='o-tube')
    isl = tube.to_island(0.0, x_points=[np.array([1.10, 0.00])])
    assert np.allclose(isl.O_point, [1.00, 0.00])
    assert len(isl.X_points) == 1
    assert 'o-tube' in tube.summary()


def test_tube_chain_maps_to_discrete_island_chain():
    # Build X and O FixedPoints across two sections
    o_fps = [
        _fp(0.0, 1.00, 0.00, 'O'), _fp(np.pi, 1.02, 0.00, 'O'),
        _fp(0.0, 0.90, 0.00, 'O'), _fp(np.pi, 0.92, 0.00, 'O'),
    ]
    x_fps = [
        _fp(0.0, 1.10, 0.00, 'X'), _fp(np.pi, 1.12, 0.00, 'X'),
        _fp(0.0, 0.80, 0.00, 'X'), _fp(np.pi, 0.82, 0.00, 'X'),
    ]
    tc = TubeChain.from_XO_fixed_points(x_fps, o_fps, winding=(2, 1))
    island_chain = tc.to_island_chain(0.0, proximity_tol=0.3)
    # One consolidated tube holds all O-points at phi=0 => 2 O-points => 2 islands
    # (from_XO_fixed_points builds one Cycle per type; section_cut returns per-fp islands)
    assert island_chain.n_islands >= 1


def test_tube_chain_diagnostics_report():
    o_fps = [_fp(0.0, 1.00, 0.00, 'O'), _fp(np.pi, 1.01, 0.00, 'O')]
    tc = TubeChain.from_XO_fixed_points([], o_fps, winding=(3, 1))
    diag = tc.diagnostics([0.0, np.pi])
    assert diag['m'] == 3
    assert diag['n_tubes'] == 1
    assert diag['complete'] is False   # 1 tube, expected 3
    assert diag['section_counts'][0.0] == 1


def test_tube_chain_reconstruct_section_view_recovers_missing_point():
    tube0 = _tube(
        [(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')],
    )
    # orbit1 has no exact cut at phi=0 but raw trajectory covers it
    tube1 = _tube(
        [(np.pi, 0.92, 0.00, 'O')],
        orbit_samples=[(0.0, 0.90, 0.02), (np.pi, 0.92, 0.00)],
    )
    tc = TubeChain(tubes=[tube0, tube1])
    tube0._tube_chain_ref = tc
    tube1._tube_chain_ref = tc

    def finder(phi, tube, existing_points, reason):
        assert reason == 'missing'
        raw = tube.raw_point_near_section(phi)
        return (raw[0], raw[1])

    view = tc.reconstruct_section_view(0.0, kind='O', section_reconstructor=finder)
    assert view.correspondence is not None
    assert view.correspondence.is_complete()
    assert len(view.correspondence.reconstructed_tube_ids) == 1
    assert view.correspondence.reconstructed_tube_ids[0].tube_index == 1
    assert len(view.unique_points()) == 2


def test_tube_chain_reconstruct_section_view_recovers_duplicate_point():
    tube0 = _tube(
        [(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')],
        orbit_samples=[(0.0, 1.00, 0.00), (np.pi, 1.02, 0.00)],
    )
    tube1 = _tube(
        [(0.0, 1.00, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')],
        orbit_samples=[(0.0, 0.90, 0.05), (np.pi, 0.92, 0.00)],
    )
    tc = TubeChain(tubes=[tube0, tube1])
    tube0._tube_chain_ref = tc
    tube1._tube_chain_ref = tc

    def finder(phi, tube, existing_points, reason):
        if reason == 'duplicate':
            raw = tube.raw_point_near_section(phi)
            return (raw[0], raw[1])
        return None

    view = tc.reconstruct_section_view(0.0, kind='O', dedup_tol=1e-10, section_reconstructor=finder)
    assert view.correspondence is not None
    assert view.correspondence.is_complete()
    assert view.correspondence.duplicate_tube_ids == []
    assert any(tid.tube_index == 1 for tid in view.correspondence.reconstructed_tube_ids)
    assert len(view.unique_points(dedup_tol=1e-10)) == 2


def test_tubechain_section_cut_general_section_aggregates_tubes():
    tube0 = _tube(
        [(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')],
        orbit_samples=[(0.0, 1.00, -0.02), (0.2, 1.01, 0.02)],
    )
    tube1 = _tube(
        [(0.0, 0.90, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')],
        orbit_samples=[(0.0, 0.90, -0.02), (0.2, 0.91, 0.02)],
    )
    tc = TubeChain(tubes=[tube0, tube1])
    tube0._tube_chain_ref = tc
    tube1._tube_chain_ref = tc

    sec = HyperplaneSection(normal_vec=np.array([0.0, 1.0]), offset=0.0, phase_dim=2)
    chain = tc.section_cut(sec)
    assert chain.n_islands >= 2
    assert chain.metadata['n_tubes_included'] == 2


def test_tubechain_from_XO_orbits_provides_joint_section_data():
    """TubeChain.from_XO_orbits assembles X and O data into one chain."""
    from pyna.topo.toroidal_invariants import Cycle as _Cycle

    def _orb_cycle(points, m=2, n=1):
        from collections import defaultdict
        secs = defaultdict(list)
        for (phi, R, Z, kind) in points:
            secs[float(phi)].append(_fp(phi, R, Z, kind))
        fps = [fp for fps in secs.values() for fp in fps]
        return _Cycle(winding=(m, n), sections=dict(secs),
                      monodromy=fps[0].monodromy if fps else None, ambient_dim=2)

    o_cycles = [
        _orb_cycle([(0.0, 1.00, 0.00, 'O'), (np.pi, 1.02, 0.00, 'O')]),
        _orb_cycle([(0.0, 0.90, 0.00, 'O'), (np.pi, 0.92, 0.00, 'O')]),
    ]
    x_cycles = [
        _orb_cycle([(0.0, 1.10, 0.00, 'X'), (np.pi, 1.12, 0.00, 'X')]),
        _orb_cycle([(0.0, 0.80, 0.00, 'X'), (np.pi, 0.82, 0.00, 'X')]),
    ]

    # Cycles are duck-typed: they have .sections and can be treated as orbit objects
    tc = TubeChain.from_XO_orbits(x_cycles, o_cycles, winding=(2, 1))
    assert tc.m == 2
    assert len(tc.section_opoints(0.0)) >= 1
    assert len(tc.section_xpoints(0.0)) >= 1

    # section_cut -> IslandChain with X_points populated
    island_chain = tc.to_island_chain(0.0, proximity_tol=0.3)
    assert island_chain is not None
    anchors = tc.section_xpoints(0.0) + tc.section_opoints(0.0)
    assert len(anchors) >= 2
