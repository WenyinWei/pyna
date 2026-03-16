import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
sys.path.insert(0, "D:/Repo/pyna")

from pyna.MCF.equilibrium.Solovev import solovev_single_null
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold

eq_sn = solovev_single_null(R0=1.86, a=0.595, B0=5.3, kappa=1.8,
                             delta_u=0.33, delta_l=0.40, kappa_x=1.5, q0=1.5)
R_xpt_sn, Z_xpt_sn = eq_sn.find_xpoint()
print(f"X-point: R={R_xpt_sn:.5f}  Z={Z_xpt_sn:.5f}", flush=True)

def field_func_sn(rzphi):
    R, Z = float(rzphi[0]), float(rzphi[1])
    BR, BZ = eq_sn.BR_BZ(np.array([R]), np.array([Z]))
    Bphi   = eq_sn.Bphi(np.array([R]))
    BR_t, BZ_t, Bp_t = float(BR[0]), float(BZ[0]), float(Bphi[0])
    B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bp_t**2) + 1e-30
    return np.array([BR_t/B_mag, BZ_t/B_mag, Bp_t/(R*B_mag)])

def field_func_2d(R, Z, phi):
    t = field_func_sn(np.array([R, Z, phi]))
    if abs(t[2]) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([t[0]/t[2], t[1]/t[2]])

phi_span = (0.0, 2.0 * np.pi)
xpt = np.array([R_xpt_sn, Z_xpt_sn])
RZlim = (eq_sn.R0 - 1.6*eq_sn.a, eq_sn.R0 + 1.6*eq_sn.a,
         -2.2*eq_sn.kappa*eq_sn.a, 1.8*eq_sn.kappa*eq_sn.a)

# --- Time jacobian_matrix at several tolerances ---
for rtol, atol in [(1e-5, 1e-7), (1e-4, 1e-6), (5e-4, 5e-6)]:
    t0 = time.time()
    vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=1e-5)
    J = vq.jacobian_matrix(xpt, phi_span,
            solve_ivp_kwargs=dict(method="RK45", rtol=rtol, atol=atol))
    dt = time.time() - t0
    det = float(np.linalg.det(J))
    print(f"jacobian rtol={rtol:.0e} atol={atol:.0e}: {dt:.2f}s  det={det:.6f}", flush=True)

# use last J for manifolds
t0 = time.time()
sm = StableManifold(xpt, J, field_func_2d, phi_span=phi_span)
sm.grow(n_turns=1, init_length=1e-4, n_init_pts=2, both_sides=True,
        RZlimit=RZlim, rtol=1e-4, atol=1e-6)
print(f"StableManifold n_turns=1 n_init_pts=2: {time.time()-t0:.2f}s  segs={len(sm.segments)}", flush=True)

t0 = time.time()
sm2 = StableManifold(xpt, J, field_func_2d, phi_span=phi_span)
sm2.grow(n_turns=2, init_length=1e-4, n_init_pts=3, both_sides=True,
         RZlimit=RZlim, rtol=1e-4, atol=1e-6)
print(f"StableManifold n_turns=2 n_init_pts=3: {time.time()-t0:.2f}s  segs={len(sm2.segments)}", flush=True)

print("Done", flush=True)
