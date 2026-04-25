from __future__ import annotations

import numpy as np

from pyna.topo.healed_scaffold_3d import BoundaryConstraintSet, BoundaryFamily3D


def _identity_trace(R0, Z0, phi0, phi_span, dphi_out):
    phi1 = float(phi0 + phi_span)
    return (
        np.array([float(R0), float(R0)], dtype=float),
        np.array([float(Z0), float(Z0)], dtype=float),
        np.array([float(phi0), phi1], dtype=float),
    )


def test_constraint_points_pull_transported_boundary_locally():
    t = np.linspace(0.0, 1.0, 64, endpoint=False)
    th = 2.0 * np.pi * t
    ref_R = np.cos(th)
    ref_Z = np.sin(th)
    target = np.array([[1.25, 0.0], [0.0, 1.25], [-1.25, 0.0], [0.0, -1.25]])
    fam = BoundaryFamily3D.from_reference_curve(
        phi_ref=0.0,
        phi_samples=[0.0, 0.3],
        ref_R=ref_R,
        ref_Z=ref_Z,
        trace_func=_identity_trace,
        local_constraints=[None, BoundaryConstraintSet(attract_points=target, local_weight=1.0, snap_length_scale=0.5)],
        param_levels=t,
    )
    sec = fam.section_at(0.3)
    d_before = np.min(np.sqrt((ref_R[None, :] - target[:, 0][:, None])**2 + (ref_Z[None, :] - target[:, 1][:, None])**2), axis=1)
    d_after = np.min(np.sqrt((sec.R[None, :] - target[:, 0][:, None])**2 + (sec.Z[None, :] - target[:, 1][:, None])**2), axis=1)
    assert np.all(d_after < d_before)
