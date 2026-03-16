import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Time Cell 11 equivalent (island chain X-point manifolds)
# field_func here is the RMP one using solovev_iter_like
import numpy as np, time
sys.path.insert(0, "D:/Repo/pyna")

from pyna.MCF.equilibrium.Solovev import solovev_iter_like
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold

eq = solovev_iter_like(scale=0.3)
R_ax, Z_ax = float(eq.magnetic_axis[0]), float(eq.magnetic_axis[1])

delta_b = 5e-3
m_rmp, n_rmp = 2, 1

def rmp_BR(R, Z, phi):
    psi_n = float(eq.psi(np.array([R]), np.array([Z]))[0])
    envelope = psi_n * (1.0 - psi_n)
    theta = np.arctan2(Z - Z_ax, R - R_ax)
    return delta_b * eq.B0 * envelope * np.cos(m_rmp * theta - n_rmp * phi)

def field_func(rzphi):
    R, Z, phi = float(rzphi[0]), float(rzphi[1]), float(rzphi[2])
    BR0, BZ0 = eq.BR_BZ(np.array([R]), np.array([Z]))
    Bphi0    = eq.Bphi(np.array([R]))
    dBR      = rmp_BR(R, Z, phi)
    BR_t = float(BR0[0]) + float(dBR)
    BZ_t = float(BZ0[0])
    Bp_t = float(Bphi0[0])
    B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bp_t**2) + 1e-30
    return np.array([BR_t/B_mag, BZ_t/B_mag, Bp_t/(R*B_mag)])

def field_func_2d(R, Z, phi):
    t = field_func(np.array([R, Z, phi]))
    if abs(t[2]) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([t[0]/t[2], t[1]/t[2]])

# Use a fake X-point close to q=2 surface
psi_res = float(eq.resonant_psi(2, 1)[0])
r_res = np.sqrt(psi_res) * eq.a
R_xpt = R_ax + r_res * np.cos(np.pi/4)
Z_xpt = r_res * np.sin(np.pi/4) + Z_ax

phi_span = (0.0, 2*np.pi)

for rtol, atol in [(1e-5, 1e-7), (1e-4, 1e-6)]:
    t0 = time.time()
    vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=1e-5)
    M = vq.jacobian_matrix([R_xpt, Z_xpt], phi_span,
            solve_ivp_kwargs=dict(method="RK45", rtol=rtol, atol=atol))
    t_jac = time.time()-t0

    t0 = time.time()
    sm = StableManifold([R_xpt, Z_xpt], M, field_func_2d)
    sm.grow(n_turns=3, init_length=1e-4, n_init_pts=3, both_sides=True, rtol=rtol, atol=atol)
    t_sm = time.time()-t0

    t0 = time.time()
    um = UnstableManifold([R_xpt, Z_xpt], M, field_func_2d)
    um.grow(n_turns=3, init_length=1e-4, n_init_pts=3, both_sides=True, rtol=rtol, atol=atol)
    t_um = time.time()-t0

    print(f"rtol={rtol:.0e}: jac={t_jac:.2f}s  sm={t_sm:.2f}s  um={t_um:.2f}s  total={t_jac+t_sm+t_um:.2f}s  sm_segs={len(sm.segments)}")

print("Done", flush=True)
