from __future__ import annotations

import numpy as np

from pyna.topo.healed_scaffold_3d import BoundaryFamily3D, BoundarySection


def test_boundary_section_splines_reject_multiwrap_curve():
    t = np.linspace(0.0, 1.0, 64, endpoint=False)
    th = 4.0 * np.pi * t
    sec = BoundarySection(
        phi=0.0,
        R=np.cos(th),
        Z=np.sin(th),
        valid=np.ones_like(t, dtype=bool),
        param=t,
        source='synthetic',
    )
    fam = BoundaryFamily3D(
        phi_ref=0.0,
        phi_samples=[0.0],
        param_levels=t,
        sections=[sec],
        ref_R=np.cos(2.0 * np.pi * t),
        ref_Z=np.sin(2.0 * np.pi * t),
    )
    sR, sZ, param, frac, source = fam.section_splines(0.0)
    assert sR is None and sZ is None
    assert 'rejected-multiwrap' in source
