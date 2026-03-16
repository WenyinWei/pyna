import sys, time; sys.path.insert(0, 'D:/Repo/pyna'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pyna.MCF.equilibrium.stellarator import simple_stellarator
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
import math, warnings
warnings.filterwarnings('ignore')

stella = simple_stellarator(R0=3.0, r0=0.30, B0=1.0, q0=1.1, q1=5.0, m_h=4, n_h=3, epsilon_h=0.05)
TARGET_M, TARGET_N = 4, 3

def field_func_2d(R, Z, phi):
    tang = stella.field_func(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

psi_res = stella.resonant_psi(TARGET_M, TARGET_N)[0]
r_res = np.sqrt(psi_res) * stella.r0
theta_x = math.atan2(-0.064828, 3.056828 - 3.0)
candidates = [(stella.R0 + r_res*np.cos(theta_x + k*2*np.pi/TARGET_N),
               r_res*np.sin(theta_x + k*2*np.pi/TARGET_N)) for k in range(TARGET_N)]

phi_span = (0.0, 2.0*np.pi*TARGET_N)
vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=1e-5)
t0 = time.time()
xpt_RZ = xpt_Jac = None
for R_c, Z_c in candidates:
    M = vq.jacobian_matrix([R_c, Z_c], phi_span=phi_span)
    lam_abs = sorted(np.abs(np.linalg.eigvals(M)))
    if lam_abs[1] > 5.0:
        xpt_RZ, xpt_Jac = np.array([R_c, Z_c]), M
        print(f'X-pt: R={R_c:.5f} Z={Z_c:.5f} lam_u={lam_abs[1]:.2f}  ({time.time()-t0:.1f}s)')
        break

RZlimit = (stella.R0-stella.r0*1.05, stella.R0+stella.r0*1.05, -stella.r0*1.05, stella.r0*1.05)
t1 = time.time()
sm = StableManifold(xpt_RZ, xpt_Jac, field_func_2d, phi_span=phi_span)
um = UnstableManifold(xpt_RZ, xpt_Jac, field_func_2d, phi_span=phi_span)

# Relaxed tolerances: faster integration, still enough for visualization
sm.grow(n_turns=3, init_length=1e-4, n_init_pts=3, both_sides=True,
        RZlimit=RZlimit, rtol=1e-6, atol=1e-9)
um.grow(n_turns=3, init_length=1e-4, n_init_pts=3, both_sides=True,
        RZlimit=RZlimit, rtol=1e-6, atol=1e-9)

print(f'grow: {time.time()-t1:.1f}s  sm={len(sm.segments)} um={len(um.segments)} segs')
print(f'Total: {time.time()-t0:.1f}s')
