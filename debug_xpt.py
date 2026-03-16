import matplotlib
matplotlib.use('Agg')
import numpy as np
from pyna.MCF.equilibrium.stellarator import SimpleStellarartor
from pyna.topo.variational import _fd_jacobian
from scipy.integrate import solve_ivp

stella = SimpleStellarartor(
    R0=3.0, r0=0.30, B0=1.0,
    q0=1.1, q1=5.0,
    m_h=4, n_h=3, epsilon_h=0.08,
)

TARGET_M, TARGET_N = 4, 3

def field_func_2d(R, Z, phi):
    tang = stella.field_func(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

phi_span = (0.0, 2.0 * np.pi * TARGET_N)

psi_res_target = stella.resonant_psi(TARGET_M, TARGET_N)[0]
r_res = np.sqrt(psi_res_target) * stella.r0
import math
_theta_x = math.atan2(-0.064828, 3.056828 - 3.0)
print(f'r_res = {r_res:.4f}, theta_x0 = {math.degrees(_theta_x):.2f} deg')

# Try all 4 candidates and print results
for k in range(TARGET_M):
    theta = _theta_x + k * 2*np.pi / TARGET_M
    R_c = stella.R0 + r_res * np.cos(theta)
    Z_c = r_res * np.sin(theta)
    
    y0 = np.concatenate([[R_c, Z_c], np.eye(2).flatten()])
    def aug_rhs(phi, state):
        rz = state[:2]
        M = state[2:].reshape(2, 2)
        f = np.asarray(field_func_2d(rz[0], rz[1], phi), dtype=float)
        A = _fd_jacobian(field_func_2d, rz, phi, eps=1e-7)
        return np.concatenate([f, (A @ M).flatten()])
    
    sol = solve_ivp(aug_rhs, phi_span, y0, method='RK45', rtol=1e-6, atol=1e-8)
    x_end = sol.y[:2, -1]
    Jac = sol.y[2:, -1].reshape(2, 2)
    residual = x_end - np.array([R_c, Z_c])
    lam = np.abs(np.linalg.eigvals(Jac))
    det = np.linalg.det(Jac)
    print(f'k={k} theta={math.degrees(theta):.1f}deg  R={R_c:.4f} Z={Z_c:.4f}')
    print(f'  residual={np.linalg.norm(residual):.3e}  det(J)={det:.4f}  |lam|={sorted(lam)}')
