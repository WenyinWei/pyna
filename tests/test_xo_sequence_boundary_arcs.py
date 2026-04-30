from __future__ import annotations

from pyna.topo.healed_flux_coords import build_xo_sequence, xo_sequence_boundary_arcs


def test_xo_sequence_boundary_arcs_returns_segment_list():
    O = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    X = [(0.7, 0.7), (-0.7, 0.7), (-0.7, -0.7), (0.7, -0.7)]
    xo = build_xo_sequence(O, X, axis=(0.0, 0.0), rho_min=0.0, min_o_points=3)
    arcs = xo_sequence_boundary_arcs(xo)
    assert xo is not None
    assert len(arcs) >= 4
    assert all(a.shape == (2, 2) for a in arcs)
