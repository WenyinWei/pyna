import sys
sys.path.insert(0, 'D:/Repo/pyna')
import os
os.chdir('D:/Repo/pyna/notebooks/tutorials')

import numpy as np
import json, pathlib

# ---- Poincare cache ----
from pyna.MCF.equilibrium.Solovev import EquilibriumSolovev, solovev_iter_like
from pyna.topo.poincare import PoincareMap, ToroidalSection, poincare_from_fieldlines
from pyna.flt import FieldLineTracer, get_backend
from pyna.topo.island import locate_rational_surface, island_halfwidth
from pyna.topo.island_extract import extract_island_width

eq = solovev_iter_like(scale=0.3)
R_ax, Z_ax = eq.magnetic_axis

S_values = np.linspace(0.05, 0.95, 60)
psi_values = S_values**2
q_values = eq.q_profile(psi_values, n_theta=256)
S_res_list = locate_rational_surface(S_values, q_values, m=2, n=1)
S_res = S_res_list[0]
psi_res = S_res**2

delta_b = 5e-3
m_rmp, n_rmp = 2, 1

def RMP_BR(R, Z, phi):
    psi_n = eq.psi(np.atleast_1d(R), np.atleast_1d(Z))
    envelope = psi_n * (1 - psi_n)
    return delta_b * eq.B0 * envelope * np.cos(m_rmp * np.arctan2(Z - Z_ax, R - R_ax) - n_rmp * phi)

def RMP_BZ(R, Z, phi):
    psi_n = eq.psi(np.atleast_1d(R), np.atleast_1d(Z))
    envelope = psi_n * (1 - psi_n)
    return delta_b * eq.B0 * envelope * np.sin(m_rmp * np.arctan2(Z - Z_ax, R - R_ax) - n_rmp * phi)

def field_func(rzphi):
    R, Z, phi = float(rzphi[0]), float(rzphi[1]), float(rzphi[2])
    BR, BZ = eq.BR_BZ(np.array([R]), np.array([Z]))
    Bphi  = eq.Bphi(np.array([R]))
    BR_t = float(BR[0]) + RMP_BR(R, Z, phi)
    BZ_t = float(BZ[0]) + RMP_BZ(R, Z, phi)
    Bp_t = float(Bphi[0])
    B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bp_t**2) + 1e-30
    return np.array([BR_t/B_mag, BZ_t/B_mag, Bp_t/(R*B_mag)])

_CACHE = pathlib.Path("pyna_output/poincare_solovev.json")
_CACHE.parent.mkdir(exist_ok=True)

if not _CACHE.exists():
    n_lines = 5
    psi_arr = np.linspace(max(psi_res - 0.06, 0.01), min(psi_res + 0.06, 0.95), n_lines)
    start_pts = np.column_stack([
        R_ax + np.sqrt(psi_arr) * eq.a,
        np.zeros(n_lines),
        np.zeros(n_lines),
    ])
    section = ToroidalSection(phi0=0.0)
    tracer = FieldLineTracer(field_func, dt=0.05)
    print(f"Tracing {n_lines} field lines t_max=100...")
    trajs = tracer.trace_many(start_pts, t_max=100.0)
    pmap = PoincareMap([section])
    for traj in trajs:
        pmap.record_trajectory(traj)
    pts_all = pmap.crossing_array(0)
    _CACHE.write_text(json.dumps({"pts": pts_all.tolist()}))
    print(f"Poincare cache written: {len(pts_all)} pts")
else:
    print("Poincare cache already exists")

# ---- SN manifold cache ----
from pyna.MCF.equilibrium.Solovev import solovev_single_null
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold

_CACHE_SN = pathlib.Path("pyna_output/solovev_sn_manifolds.json")

if not _CACHE_SN.exists():
    eq_sn = solovev_single_null(R0=1.86, a=0.595, B0=5.3, kappa=1.8,
        delta_u=0.33, delta_l=0.40, kappa_x=1.5, q0=1.5)
    R_ax_sn, Z_ax_sn = eq_sn.magnetic_axis
    R_xpt_sn, Z_xpt_sn = eq_sn.find_xpoint()
    print(f'X-point: R={R_xpt_sn:.6f}  Z={Z_xpt_sn:.6f}')

    def field_func_sn(rzphi):
        R, Z = float(rzphi[0]), float(rzphi[1])
        BR, BZ = eq_sn.BR_BZ(np.array([R]), np.array([Z]))
        Bphi  = eq_sn.Bphi(np.array([R]))
        BR_t, BZ_t, Bp_t = float(BR[0]), float(BZ[0]), float(Bphi[0])
        B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bp_t**2) + 1e-30
        return np.array([BR_t/B_mag, BZ_t/B_mag, Bp_t/(R*B_mag)])

    def field_func_2d_sn(R, Z, phi):
        t = field_func_sn(np.array([R, Z, phi]))
        if abs(t[2]) < 1e-15:
            return np.array([0.0, 0.0])
        return np.array([t[0]/t[2], t[1]/t[2]])

    phi_span_sn = (0.0, 2.0 * np.pi)
    vq_sn = PoincareMapVariationalEquations(field_func_2d_sn, fd_eps=1e-5)
    xpt_sn = np.array([R_xpt_sn, Z_xpt_sn])
    print("Computing jacobian_matrix...")
    Jac_sn = vq_sn.jacobian_matrix(xpt_sn, phi_span_sn,
        solve_ivp_kwargs=dict(method='RK45', rtol=1e-6, atol=1e-8))
    lam_sn = np.linalg.eigvals(Jac_sn)
    lam_abs_sn = sorted(np.abs(lam_sn))
    det_sn = float(np.linalg.det(Jac_sn))
    print(f'det(J)={det_sn:.8f}')

    RZlimit_sn = (eq_sn.R0 - 1.6*eq_sn.a, eq_sn.R0 + 1.6*eq_sn.a,
                  -2.2*eq_sn.kappa*eq_sn.a, 1.8*eq_sn.kappa*eq_sn.a)
    sm_sn = StableManifold(xpt_sn, Jac_sn, field_func_2d_sn, phi_span=phi_span_sn)
    um_sn = UnstableManifold(xpt_sn, Jac_sn, field_func_2d_sn, phi_span=phi_span_sn)
    ivp_kw = dict(rtol=1e-5, atol=1e-7)
    print("Growing manifolds...")
    sm_sn.grow(n_turns=2, init_length=1e-4, n_init_pts=3, both_sides=True, RZlimit=RZlimit_sn, **ivp_kw)
    um_sn.grow(n_turns=2, init_length=1e-4, n_init_pts=3, both_sides=True, RZlimit=RZlimit_sn, **ivp_kw)
    print(f'Stable {len(sm_sn.segments)} Unstable {len(um_sn.segments)}')

    sm_segs = [s for s in sm_sn.segments if len(s) >= 2]
    um_segs = [s for s in um_sn.segments if len(s) >= 2]
    _CACHE_SN.write_text(json.dumps({
        "det_J": det_sn,
        "lam_stable": float(lam_abs_sn[0]),
        "lam_unstable": float(lam_abs_sn[1]),
        "sm_segments": [seg.tolist() for seg in sm_segs],
        "um_segments": [seg.tolist() for seg in um_segs],
    }))
    print("SN manifold cache written.")
else:
    print("SN cache already exists")

print("Done.")
