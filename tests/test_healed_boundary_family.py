from __future__ import annotations

import numpy as np

from pyna.topo.healed_flux_coords import BoundaryFamily3D, build_section_scaffold_bundle


def _identity_trace(R0, Z0, phi0, phi_span, dphi_out):
    phi1 = float(phi0 + phi_span)
    return (
        np.array([float(R0), float(R0)], dtype=float),
        np.array([float(Z0), float(Z0)], dtype=float),
        np.array([float(phi0), phi1], dtype=float),
    )


def _make_coeff_family(radius: float = 1.0):
    # R = radius*cos(theta), Z = radius*sin(theta)
    class _Const:
        def __init__(self, value: float):
            self.value = float(value)
        def __call__(self, _r):
            return self.value
    cR = [_Const(0.0), _Const(radius), _Const(0.0)]
    cZ = [_Const(0.0), _Const(0.0), _Const(radius)]
    return cR, cZ


def test_boundary_family_prefers_local_section_when_available():
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    ref_R = np.cos(theta)
    ref_Z = np.sin(theta)
    local_R = 1.2 * np.cos(theta)
    local_Z = 1.2 * np.sin(theta)
    fam = BoundaryFamily3D.from_reference_curve(
        phi_ref=0.0,
        phi_samples=[0.0, np.pi / 2],
        ref_R=ref_R,
        ref_Z=ref_Z,
        trace_func=_identity_trace,
        local_sections=[None, (local_R, local_Z)],
        param_levels=np.linspace(0.0, 1.0, len(theta), endpoint=False),
        blend_local_weight=1.0,
    )
    sec = fam.section_at(np.pi / 2)
    assert sec.source in {"local-cxo", "blended-local+transport"}
    assert np.allclose(sec.R, local_R)
    assert np.allclose(sec.Z, local_Z)


def test_build_section_scaffold_bundle_uses_boundary_family():
    theta_levels = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    r_levels = [0.4, 0.8]
    cR, cZ = _make_coeff_family(radius=1.0)
    theta_b = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    ref_R = 1.1 * np.cos(theta_b)
    ref_Z = 1.1 * np.sin(theta_b)
    boundary_family = BoundaryFamily3D.from_reference_curve(
        phi_ref=0.0,
        phi_samples=[0.0, np.pi / 2],
        ref_R=ref_R,
        ref_Z=ref_Z,
        trace_func=_identity_trace,
        param_levels=np.linspace(0.0, 1.0, len(theta_b), endpoint=False),
    )
    bundle = build_section_scaffold_bundle(
        phi_samples=[0.0, np.pi / 2],
        phi_ref=0.0,
        reference_spl_R=cR,
        reference_spl_Z=cZ,
        r_levels=r_levels,
        theta_levels=theta_levels,
        trace_func=_identity_trace,
        section_axes=[(0.0, 0.0), (0.0, 0.0)],
        n_coeff=3,
        boundary_family=boundary_family,
        cxo_trace_theta=theta_b,
    )
    assert bundle.boundary_family is not None
    assert len(bundle.fits) == 2
    for fit in bundle.fits:
        assert fit.spl_R_CXO is not None
        assert fit.spl_Z_CXO is not None
        assert fit.boundary_source in {"reference", "transported-ref-cxo", "local-cxo", "blended-local+transport"}
        assert fit.boundary_valid_fraction > 0.9
