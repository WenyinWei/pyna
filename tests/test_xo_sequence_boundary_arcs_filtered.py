from __future__ import annotations

import numpy as np

from pyna.topo.healed_flux_coords import build_xo_sequence, xo_sequence_boundary_arcs


def _mean_rho(arcs):
    vals = []
    for arc in arcs:
        vals.append(float(np.mean(np.hypot(arc[:, 0], arc[:, 1]))))
    return float(np.mean(vals)) if vals else 0.0


def test_xo_sequence_boundary_arcs_filters_inward_segments():
    O = [(1.4, 0.0), (0.0, 1.4), (-1.4, 0.0), (0.0, -1.4)]
    X = [(0.4, 0.4), (-0.4, 0.4), (-0.4, -0.4), (0.4, -0.4)]
    xo = build_xo_sequence(O, X, axis=(0.0, 0.0), rho_min=0.0, min_o_points=3)
    arcs_all = xo_sequence_boundary_arcs(xo, include_o_segments=True, include_x_segments=True, include_cross_segments=True, outward_quantile=0.0)
    arcs_filtered = xo_sequence_boundary_arcs(xo, include_o_segments=True, include_x_segments=False, include_cross_segments=True, outward_quantile=0.6)
    assert xo is not None
    assert len(arcs_filtered) > 0
    assert _mean_rho(arcs_filtered) >= _mean_rho(arcs_all)
