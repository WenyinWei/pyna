from __future__ import annotations

import numpy as np

from pyna.topo.healed_flux_coords import BoundaryConstraintSet, correct_boundary_with_constraints


def test_arc_constraints_pull_curve_segment_toward_arc():
    th = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    R = np.cos(th)
    Z = np.sin(th)
    arc = np.array([
        [1.15, 0.00],
        [1.10, 0.20],
        [1.00, 0.38],
        [0.86, 0.52],
    ])
    R2, Z2, ok, diag = correct_boundary_with_constraints(
        R,
        Z,
        constraints=BoundaryConstraintSet(
            attract_points=np.empty((0, 2)),
            attract_arcs=[arc],
            local_weight=0.0,
            arc_weight=1.0,
            snap_length_scale=0.5,
        ),
    )
    d_before = np.min(np.sqrt((R[None, :] - arc[:, 0][:, None])**2 + (Z[None, :] - arc[:, 1][:, None])**2), axis=1)
    d_after = np.min(np.sqrt((R2[None, :] - arc[:, 0][:, None])**2 + (Z2[None, :] - arc[:, 1][:, None])**2), axis=1)
    assert np.all(d_after <= d_before)
    assert diag['n_arc_samples_used'] >= 2
    assert np.all(ok)
