"""
Debug: find correct tolerance for Newton convergence at single-null X-point
"""
import matplotlib; matplotlib.use('Agg')
import numpy as np
from pyna.MCF.equilibrium.Solovev import solovev_single_null
from pyna.topo.variational import _fd_jacobian
from scipy.integrate import solve_ivp

eq_sn = solovev_single_null(
    R0=1.86, a=0.595, B0=5.3,
    kappa=1.8, delta_u=0.33, delta_l=0.40, kappa_x=1.5, q0=1.5,
)

R_xpt, Z_xpt = eq_sn.find_xpoint()
print(f'Analytic X-point: R={R_xpt:.6f}  Z={Z_xpt:.6f}')

def field_func_2d_sn(R, Z, phi):
    R_arr, Z_arr = np.array([R]), np.array([Z])
    BR, BZ = eq_sn.BR_BZ(R_arr, Z_arr)
    Bphi = eq_sn.Bphi(R_arr)
    BR_t, BZ_t, Bphi_t = float(BR[0]), float(BZ[0]), float(Bphi[0])
    B2 = BR_t**2 + BZ_t**2 + Bphi_t**2 + 1e-60
    dRds, dZds, dphis = BR_t/B2**0.5, BZ_t/B2**0.5, Bphi_t/(R*B2**0.5)
    if abs(dphis) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([dRds/dphis, dZds/dphis])

phi_span = (0.0, 2.0 * np.pi)

# Test residual at analytic X-point with different tolerances
for rtol, atol, method in [
    (1e-6, 1e-8, 'RK45'),
    (1e-8, 1e-10, 'RK45'),
    (1e-10, 1e-12, 'RK45'),
    (1e-8, 1e-10, 'DOP853'),
]:
    y0 = np.concatenate([[R_xpt, Z_xpt], np.eye(2).flatten()])
    def aug_rhs(phi, state):
        rz = state[:2]
        M = state[2:].reshape(2, 2)
        f = np.asarray(field_func_2d_sn(rz[0], rz[1], phi), dtype=float)
        A = _fd_jacobian(field_func_2d_sn, rz, phi, eps=1e-6)
        return np.concatenate([f, (A @ M).flatten()])
    import time; t0 = time.time()
    sol = solve_ivp(aug_rhs, phi_span, y0, method=method, rtol=rtol, atol=atol)
    dt = time.time()-t0
    res = np.linalg.norm(sol.y[:2,-1] - np.array([R_xpt, Z_xpt]))
    det = np.linalg.det(sol.y[2:,-1].reshape(2,2))
    print(f'{method} rtol={rtol:.0e}: |res|={res:.2e}  det(J)={det:.6f}  t={dt:.1f}s  nsteps={sol.t.shape[0]}')
