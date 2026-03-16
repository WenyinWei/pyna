import numpy as np, sys
sys.path.insert(0, 'D:/Repo/pyna')
from pyna.MCF.equilibrium.stellarator import simple_stellarator
from pyna.topo.variational import PoincareMapVariationalEquations
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

stella = simple_stellarator(R0=3.0, r0=0.30, B0=1.0, q0=1.1, q1=5.0, m_h=4, n_h=3, epsilon_h=0.08)
TARGET_M, TARGET_N = 4, 3
psi_res = stella.resonant_psi(TARGET_M, TARGET_N)[0]
r_res = np.sqrt(psi_res) * stella.r0

def field_func_2d(R, Z, phi):
    tang = stella.field_func(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

xpt_seed = np.array([stella.R0 + r_res * np.cos(np.pi/TARGET_M),
                     r_res * np.sin(np.pi/TARGET_M)])
print(f'Seed:    R={xpt_seed[0]:.5f}  Z={xpt_seed[1]:.5f}')

def pmap_n(x0):
    def rhs(phi, y): return field_func_2d(y[0], y[1], phi)
    sol = solve_ivp(rhs, [0, 2*np.pi*TARGET_N], x0, method='RK45', rtol=1e-10, atol=1e-12)
    return sol.y[:, -1]

xpt_ref, info, ier, _ = fsolve(lambda x: pmap_n(x)-x, xpt_seed, full_output=True)
res_norm = np.linalg.norm(info['fvec'])
print(f'Refined: R={xpt_ref[0]:.6f}  Z={xpt_ref[1]:.6f}  |res|={res_norm:.2e}  ok={ier==1}')

vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=1e-6)
J = vq.jacobian_matrix(xpt_ref, (0, 2*np.pi*TARGET_N),
                       solve_ivp_kwargs=dict(method='RK45', rtol=1e-8, atol=1e-10))
det_J = np.linalg.det(J)
lam = sorted(np.abs(np.linalg.eigvals(J)))
print(f'det(J) = {det_J:.6f}  (ideal 1.0)')
print(f'|lam_s|={lam[0]:.6f}  |lam_u|={lam[1]:.6f}')
print('PASS' if abs(det_J - 1.0) < 0.01 else 'FAIL')
