"""Find period-3 X-points of stellarator island chain via Poincare scan + Newton."""
import numpy as np, sys
sys.path.insert(0, 'D:/Repo/pyna')
from pyna.MCF.equilibrium.stellarator import simple_stellarator
from pyna.topo.variational import PoincareMapVariationalEquations
from scipy.optimize import root
from scipy.integrate import solve_ivp

stella = simple_stellarator(R0=3.0, r0=0.30, B0=1.0, q0=1.1, q1=5.0, m_h=4, n_h=3, epsilon_h=0.08)
TARGET_M, TARGET_N = 4, 3
psi_res = stella.resonant_psi(TARGET_M, TARGET_N)[0]
r_res = np.sqrt(psi_res) * stella.r0
print(f'r_res = {r_res:.5f}  psi_res = {psi_res:.5f}')
print(f'stella R0={stella.R0}, r0={stella.r0}')

def field_func_2d(R, Z, phi):
    tang = stella.field_func(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

def pmap_n(x0, n=TARGET_N):
    def rhs(phi, y): return field_func_2d(y[0], y[1], phi)
    sol = solve_ivp(rhs, [0, 2*np.pi*n], x0, method='DOP853', rtol=1e-11, atol=1e-13, dense_output=False)
    return sol.y[:, -1]

# Scan a ring of points at r ~ r_res and find the ones that nearly return after 3 turns
n_scan = 200
thetas = np.linspace(0, 2*np.pi, n_scan, endpoint=False)
R_ring = stella.R0 + r_res * np.cos(thetas)
Z_ring = r_res * np.sin(thetas)

residuals = []
for R0, Z0 in zip(R_ring, Z_ring):
    end = pmap_n([R0, Z0])
    residuals.append(np.linalg.norm(end - [R0, Z0]))

residuals = np.array(residuals)
# Find local minima of residual (these are near X or O points)
from scipy.signal import argrelmin
idx_mins = argrelmin(residuals, order=5)[0]
print(f'\nLocal minima of |P^3 - id| at scanned ring:')
for i in idx_mins:
    print(f'  theta={np.degrees(thetas[i]):.1f} deg  R={R_ring[i]:.4f}  Z={Z_ring[i]:.4f}  |res|={residuals[i]:.4e}')

# Try Newton from best candidates
best_seeds = [(residuals[i], R_ring[i], Z_ring[i]) for i in idx_mins]
best_seeds.sort()

print('\nNewton refinement from candidates:')
best_xpt = None
for res0, R0, Z0 in best_seeds[:8]:
    sol = root(lambda x: pmap_n(x) - x, [R0, Z0], method='hybr',
               tol=1e-12, options={'maxfev': 200})
    if sol.success:
        det_check = np.linalg.norm(pmap_n(sol.x) - sol.x)
        print(f'  R={sol.x[0]:.6f}  Z={sol.x[1]:.6f}  |res|={det_check:.2e}  ok')
        if best_xpt is None or det_check < 1e-8:
            best_xpt = sol.x.copy()
    else:
        print(f'  R={R0:.4f}  Z={Z0:.4f}  FAILED: {sol.message}')

if best_xpt is not None:
    print(f'\nBest X-point: R={best_xpt[0]:.6f}  Z={best_xpt[1]:.6f}')
    vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=1e-6)
    J = vq.jacobian_matrix(best_xpt, (0, 2*np.pi*TARGET_N),
                           solve_ivp_kwargs=dict(method='DOP853', rtol=1e-10, atol=1e-12))
    det_J = np.linalg.det(J)
    lam = sorted(np.abs(np.linalg.eigvals(J)))
    print(f'det(DP^{TARGET_N}) = {det_J:.6f}  (ideal 1.0)')
    print(f'|lam_s|={lam[0]:.6f}  |lam_u|={lam[1]:.6f}')
    print('PASS' if abs(det_J - 1.0) < 0.02 else 'FAIL')
