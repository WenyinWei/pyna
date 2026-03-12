"""Fast RMP island Poincare - standalone, minimal dependencies."""
import sys, time
sys.path.insert(0, r'D:\Repo\pyna')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor

from pyna.mag.Solovev import SolovevEquilibrium

eq = SolovevEquilibrium(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)
R_ax, Z_ax = float(eq.magnetic_axis[0]), float(eq.magnetic_axis[1])
print(f"axis R={R_ax:.3f}", flush=True)

# q profile (pre-computed, don't call inside threaded loops)
psi_v = np.linspace(0.05, 0.95, 60)
q_v = eq.q_profile(psi_v)
q_func = interp1d(psi_v, q_v)
psi_res = float(brentq(lambda p: q_func(p) - 4.0, 0.3, 0.9))
print(f"q=4/1 at psi={psi_res:.4f}", flush=True)

delta_b = 0.008
m_rmp, n_rmp = 4, 1

def field_func_arr(R, Z, phi):
    BR, BZ = eq.BR_BZ(R, Z)
    Bphi = float(eq.Bphi(R))
    psi_n = float(eq.psi(R, Z))
    theta = np.arctan2(Z - Z_ax, R - R_ax)
    env = np.clip(psi_n * (1 - psi_n) * 4, 0, 1)
    BR = BR + delta_b * eq.B0 * env * np.cos(m_rmp * theta - n_rmp * phi)
    Bm = np.sqrt(BR**2 + BZ**2 + Bphi**2) + 1e-30
    return BR/Bm, BZ/Bm, Bphi/(R*Bm)

def rk4_step(R, Z, phi, dt):
    dR1,dZ1,dp1 = field_func_arr(R,Z,phi)
    dR2,dZ2,dp2 = field_func_arr(R+.5*dt*dR1, Z+.5*dt*dZ1, phi+.5*dt*dp1)
    dR3,dZ3,dp3 = field_func_arr(R+.5*dt*dR2, Z+.5*dt*dZ2, phi+.5*dt*dp2)
    dR4,dZ4,dp4 = field_func_arr(R+dt*dR3, Z+dt*dZ3, phi+dt*dp3)
    return (R + dt*(dR1+2*dR2+2*dR3+dR4)/6,
            Z + dt*(dZ1+2*dZ2+2*dZ3+dZ4)/6,
            phi + dt*(dp1+2*dp2+2*dp3+dp4)/6)

def trace_poincare(args):
    R0, Z0, t_max, dt = args
    phi = 0.0
    R, Z = float(R0), float(Z0)
    cR, cZ = [], []
    prev_phi_mod = 0.0
    t = 0.0
    while t < t_max:
        R, Z, phi = rk4_step(R, Z, phi, dt)
        t += dt
        phi_mod = phi % (2*np.pi)
        # upward crossing through 0
        if prev_phi_mod > 5.0 and phi_mod < 1.0:
            cR.append(R); cZ.append(Z)
        prev_phi_mod = phi_mod
        if R < 0.3 or R > 4.0 or abs(Z) > 3.0:
            break
    return cR, cZ

# Start points: bracket psi_res
n_lines = 28
psi_starts = np.linspace(psi_res - 0.09, psi_res + 0.09, n_lines)
args_list = []
for ps in psi_starts:
    r_s = eq.a * float(np.sqrt(max(ps, 0.01)))
    R_s = R_ax + r_s
    args_list.append((R_s, Z_ax, 2500, 0.07))

print(f"Tracing {n_lines} lines with 16 threads...", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    res = list(ex.map(trace_poincare, args_list))
print(f"Done in {time.time()-t0:.1f}s", flush=True)

all_R = [r for rlist,_ in res for r in rlist]
all_Z = [z for _,zlist in res for z in zlist]
print(f"Total crossings: {len(all_R)}", flush=True)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6.5, 8.5))

# Flux surfaces
Rg = np.linspace(0.8, 2.8, 180)
Zg = np.linspace(-1.4, 1.4, 180)
Rm, Zm = np.meshgrid(Rg, Zg)
psi_m = np.vectorize(lambda r,z: float(eq.psi(r,z)))(Rm, Zm)
ax.contour(Rm, Zm, psi_m, levels=np.linspace(0.05,0.98,13), colors='gray', linewidths=0.4, alpha=0.6)
ax.contour(Rm, Zm, psi_m, levels=[1.0], colors='k', linewidths=1.5)
cs = ax.contour(Rm, Zm, psi_m, levels=[psi_res], colors='steelblue', linewidths=1.2, linestyles='--')
ax.clabel(cs, fmt='q=4/1', fontsize=8)

ax.scatter(all_R, all_Z, s=0.4, color='royalblue', alpha=0.4, label=f'Poincaré ({len(all_R)} pts)', rasterized=True)
ax.plot(R_ax, Z_ax, 'k+', ms=10, mew=2, label='axis')

ax.set_xlabel('R (m)', fontsize=11)
ax.set_ylabel('Z (m)', fontsize=11)
ax.set_title(f"Solov'ev + (4,1) RMP  [δ_b={delta_b}]\nq=4/1 island chain, {n_lines} field lines × t_max=2500", fontsize=10)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=8)

out = r'D:\Repo\pyna\scripts\rmp_poincare_dense.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}", flush=True)
