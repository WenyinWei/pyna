from __future__ import annotations

import numpy as np

from pyna.topo.healed_flux_coords import BoundaryFamily3D


def _shift_trace(R0, Z0, phi0, phi_span, dphi_out):
    phi1 = float(phi0 + phi_span)
    shift = 0.1 * float(phi_span)
    return (
        np.array([float(R0), float(R0) + shift], dtype=float),
        np.array([float(Z0), float(Z0)], dtype=float),
        np.array([float(phi0), phi1], dtype=float),
    )


def test_boundary_family_transport_without_local_section():
    t = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    ref_R = np.cos(t)
    ref_Z = np.sin(t)
    fam = BoundaryFamily3D.from_reference_curve(
        phi_ref=0.0,
        phi_samples=[0.0, np.pi],
        ref_R=ref_R,
        ref_Z=ref_Z,
        trace_func=_shift_trace,
        param_levels=np.linspace(0.0, 1.0, len(t), endpoint=False),
    )
    sec = fam.section_at(np.pi)
    assert sec.source == 'transported-ref-cxo'
    assert np.mean(sec.valid) > 0.95
    expected_shift = 0.1 * (np.pi + 2.0 * 0.04)
    assert np.allclose(sec.R, ref_R + expected_shift)
    assert np.allclose(sec.Z, ref_Z)


def test_boundary_family_blends_transport_and_local():
    t = np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False)
    ref_R = np.cos(t)
    ref_Z = np.sin(t)
    local_R = ref_R + 0.4
    local_Z = ref_Z - 0.2
    fam = BoundaryFamily3D.from_reference_curve(
        phi_ref=0.0,
        phi_samples=[0.0, 1.0],
        ref_R=ref_R,
        ref_Z=ref_Z,
        trace_func=_shift_trace,
        local_sections=[None, (local_R, local_Z)],
        param_levels=np.linspace(0.0, 1.0, len(t), endpoint=False),
        blend_local_weight=0.25,
    )
    sec = fam.section_at(1.0)
    assert sec.source == 'blended-local+transport'
    delta_R = sec.R - ref_R
    delta_Z = sec.Z - ref_Z
    assert np.allclose(delta_R, 0.18)
    assert np.allclose(delta_Z, -0.05)
