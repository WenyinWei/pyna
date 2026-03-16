import sys, json, pathlib, warnings, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
sys.path.insert(0, "D:/Repo/pyna")
import numpy as np
from pyna.MCF.equilibrium.Solovev import solovev_single_null
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold

eq_sn = solovev_single_null(R0=1.86, a=0.595, B0=5.3, kappa=1.8,
    delta_u=0.33, delta_l=0.40, kappa_x=1.5, q0=1.5)
R_xpt, Z_xpt = eq_sn.find_xpoint()
print(f"X-point: R={R_xpt:.5f}  Z={Z_xpt:.5f}", flush=True)

def ff2d(R, Z, phi):
    BR, BZ = eq_sn.BR_BZ(np.array([R]), np.array([Z]))
    Bp = eq_sn.Bphi(np.array([R]))
    brt, bzt, bpt = float(BR[0]), float(BZ[0]), float(Bp[0])
    bm = np.sqrt(brt**2 + bzt**2 + bpt**2) + 1e-30
    dphi = bpt / (R * bm)
    if abs(dphi) < 1e-15:
        return np.array([0., 0.])
    return np.array([brt/bm/dphi, bzt/bm/dphi])

phi_span = (0.0, 2.0 * np.pi)
RZlim = (eq_sn.R0-1.6*eq_sn.a, eq_sn.R0+1.6*eq_sn.a,
         -2.2*eq_sn.kappa*eq_sn.a, 1.8*eq_sn.kappa*eq_sn.a)

t0 = time.time()
J = PoincareMapVariationalEquations(ff2d, fd_eps=1e-5).jacobian_matrix(
    [R_xpt, Z_xpt], phi_span,
    solve_ivp_kwargs=dict(method="RK45", rtol=1e-5, atol=1e-7))
print(f"jac: {time.time()-t0:.2f}s  det={np.linalg.det(J):.6f}", flush=True)

lam = sorted(np.abs(np.linalg.eigvals(J)))

# Conservative grow: n_turns=1, n_init_pts=2, single side each
t0 = time.time()
sm = StableManifold([R_xpt, Z_xpt], J, ff2d, phi_span=phi_span)
sm.grow(n_turns=1, init_length=1e-3, n_init_pts=2, both_sides=False,
        RZlimit=RZlim, rtol=1e-4, atol=1e-6)
um = UnstableManifold([R_xpt, Z_xpt], J, ff2d, phi_span=phi_span)
um.grow(n_turns=1, init_length=1e-3, n_init_pts=2, both_sides=False,
        RZlimit=RZlim, rtol=1e-4, atol=1e-6)
print(f"manifolds: {time.time()-t0:.2f}s  sm={len(sm.segments)}  um={len(um.segments)}", flush=True)

CACHE = pathlib.Path("D:/Repo/pyna/notebooks/tutorials/pyna_output/solovev_sn_manifolds.json")
CACHE.parent.mkdir(exist_ok=True)
d = {
    "R_ax": float(eq_sn.magnetic_axis[0]), "Z_ax": float(eq_sn.magnetic_axis[1]),
    "R_xpt": float(R_xpt), "Z_xpt": float(Z_xpt),
    "det_J": float(np.linalg.det(J)),
    "lam_stable": float(lam[0]), "lam_unstable": float(lam[1]),
    "sm_segments": [s.tolist() for s in sm.segments if len(s) >= 2],
    "um_segments": [s.tolist() for s in um.segments if len(s) >= 2],
}
CACHE.write_text(json.dumps(d))
print(f"Saved {CACHE}  ({CACHE.stat().st_size} bytes)", flush=True)
