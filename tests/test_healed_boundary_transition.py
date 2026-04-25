from __future__ import annotations

import numpy as np

from pyna.topo.healed_flux_coords import BoundaryFamily3D, build_section_scaffold_bundle, eval_fourier_family


def _identity_trace(R0, Z0, phi0, phi_span, dphi_out):
    phi1 = float(phi0 + phi_span)
    return (
        np.array([float(R0), float(R0)], dtype=float),
        np.array([float(Z0), float(Z0)], dtype=float),
        np.array([float(phi0), phi1], dtype=float),
    )


class _Const:
    def __init__(self, value: float):
        self.value = float(value)
    def __call__(self, _r):
        return self.value


def _make_coeff_family(radius: float = 1.0):
    cR = [_Const(0.0), _Const(radius), _Const(0.0)]
    cZ = [_Const(0.0), _Const(0.0), _Const(radius)]
    return cR, cZ


def test_boundary_family_replaces_outer_r1_node_in_ext_family():
    theta_levels = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    r_levels = [0.4, 0.8]
    cR, cZ = _make_coeff_family(radius=1.0)
    t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    ref_R = 1.3 * np.cos(t)
    ref_Z = 1.3 * np.sin(t)
    boundary_family = BoundaryFamily3D.from_reference_curve(
        phi_ref=0.0,
        phi_samples=[0.0],
        ref_R=ref_R,
        ref_Z=ref_Z,
        trace_func=_identity_trace,
        param_levels=np.linspace(0.0, 1.0, len(t), endpoint=False),
    )
    bundle = build_section_scaffold_bundle(
        phi_samples=[0.0],
        phi_ref=0.0,
        reference_spl_R=cR,
        reference_spl_Z=cZ,
        r_levels=r_levels,
        theta_levels=theta_levels,
        trace_func=_identity_trace,
        section_axes=[(0.0, 0.0)],
        n_coeff=3,
        boundary_family=boundary_family,
        cxo_trace_theta=np.linspace(0.0, 1.0, len(t), endpoint=False),
    )
    fit = bundle.fits[0]
    R1, Z1 = eval_fourier_family(fit.spl_R_ext, fit.spl_Z_ext, 1.0, t)
    target_R = 1.3 * np.cos(t)
    target_Z = 1.3 * np.sin(t)
    assert np.max(np.abs(R1 - target_R)) < 1e-6
    assert np.max(np.abs(Z1 - target_Z)) < 1e-6
