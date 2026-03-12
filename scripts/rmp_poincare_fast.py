"""Fast RMP island chain Poincaré plot — multi-CPU long integration.
q=4/1 in Solov'ev equilibrium + (4,1) RMP perturbation.
"""
import sys, time
sys.path.insert(0, r'D:\Repo\pyna')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor

from pyna.mag.solovev import SolovevEquilibrium

# --- Build equilibrium with q0=3 so q=4/1 is accessible ---
eq = SolovevEquilibrium(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)
R_ax, Z_ax = eq.magnetic_axis
print(f"Magnetic axis: R={R_ax:.3f}, Z={Z_ax:.4f}")

# --- q profile and find q=4/1 surface ---
psi_vals = np.linspace(0.05, 0.97, 100)
q_vals = eq.q_profile(psi_vals)
print(f"q range: {q_vals.min():.3f} -> {q_vals.max():.3f}")
q_func = interp1d(psi_vals, q_vals, fill_value='extrapolate')
psi_res = brentq(lambda p: q_func(p) - 4.0, 0.1, 0.95)
print(f"q=4/1 at psi_norm = {psi_res:.4f}")

# --- RMP perturbation field function ---
delta_b = 0.008   # RMP amplitude
m_rmp, n_rmp = 4, 1

def field_func(rzphi):
    R, Z, phi = rzphi
    BR, BZ = eq.BR_BZ(R, Z)
    Bphi = eq.Bphi(R)
    # Add (4,1) RMP: only BR perturbation
    psi_n = eq.psi(R, Z)
    theta = np.arctan2(Z - Z_ax, R - R_ax)
    envelope = psi_n * (1 - psi_n) * 4  # peaks at psi=0.5
    dBR = delta_b * eq.B0 * envelope * np.cos(m_rmp * theta - n_rmp * phi)
    BR = BR + dBR
    Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2) + 1e-30
    return np.array([BR/Bmag, BZ/Bmag, Bphi/(R*Bmag)])

# --- Field line tracer with Poincaré crossing detection ---
def trace_poincare(start_rzphi, t_max=3000, dt=0.06):
    """Trace one field line, collect phi=0 crossings."""
    y = np.array(start_rzphi, dtype=float)
    crossings_R = []
    crossings_Z = []
    phi_prev = y[2]
    t = 0.0
    # RK4
    while t < t_max:
        dy = field_func(y)
        k1 = dy
        k2 = field_func(y + 0.5*dt*k1)
        k3 = field_func(y + 0.5*dt*k2)
        k4 = field_func(y + dt*k3)
        y = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        t += dt
        # Poincaré crossing: phi crosses 0 mod 2pi (upward)
        phi_curr = y[2]
        phi_mod_prev = phi_prev % (2*np.pi)
        phi_mod_curr = phi_curr % (2*np.pi)
        if phi_mod_prev > np.pi and phi_mod_curr < np.pi:  # crossing 0 upward
            # linear interpolation
            frac = phi_mod_prev / (phi_mod_prev + (2*np.pi - phi_mod_curr))
            R_cross = y[0] - frac * (y[0] - (y[0] - dt*k1[0]))
            # simpler: just use current point
            crossings_R.append(y[0])
            crossings_Z.append(y[1])
        phi_prev = phi_curr
        # domain check
        if y[0] < 0.5 or y[0] > 3.5 or abs(y[1]) > 2.0:
            break
    return np.array(crossings_R), np.array(crossings_Z)

# --- Launch start points near q=4/1 surface ---
# Find (R, Z) points on the psi_res flux surface
theta_starts = np.linspace(0, 2*np.pi, 32, endpoint=False)
# approximate: R_s ≈ R_ax + r_s*cos(theta), Z_s ≈ Z_ax + r_s*sin(theta) where r_s from psi
# use a quick Newton to find the LCFS radius then scale
r_approx = eq.a * np.sqrt(psi_res)  # rough estimate
print(f"Approximate minor radius at resonance: {r_approx:.3f} m")

# Start lines at a few different radii bracketing psi_res
delta_psi = 0.08
psi_lines = np.concatenate([
    np.linspace(psi_res - delta_psi, psi_res + delta_psi, 24)
])
start_pts = []
for psi_target in psi_lines:
    r_s = eq.a * np.sqrt(max(psi_target, 0.01))
    # start at theta=0 (outboard midplane)
    R_s = R_ax + r_s
    Z_s = Z_ax
    start_pts.append([R_s, Z_s, 0.0])

print(f"Tracing {len(start_pts)} field lines, t_max=3000, dt=0.06...")
t0 = time.time()

def trace_one(pt):
    return trace_poincare(pt, t_max=3000, dt=0.06)

N_WORKERS = 16
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    results = list(ex.map(trace_one, start_pts))

dt_trace = time.time() - t0
print(f"Done in {dt_trace:.1f}s using {N_WORKERS} threads")

all_R = np.concatenate([r for r,z in results])
all_Z = np.concatenate([z for r,z in results])
print(f"Total Poincaré points: {len(all_R)}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 9))

# Flux surface contours
psi_grid_r = np.linspace(0.8, 2.8, 200)
psi_grid_z = np.linspace(-1.4, 1.4, 200)
Rg, Zg = np.meshgrid(psi_grid_r, psi_grid_z)
psi_grid = np.vectorize(eq.psi)(Rg, Zg)
ax.contour(Rg, Zg, psi_grid, levels=np.linspace(0.05, 0.99, 14), colors='gray', linewidths=0.5, alpha=0.5)
ax.contour(Rg, Zg, psi_grid, levels=[1.0], colors='black', linewidths=1.5)  # LCFS
ax.contour(Rg, Zg, psi_grid, levels=[psi_res], colors='steelblue', linewidths=1.2, linestyles='--')

# Poincaré scatter
ax.scatter(all_R, all_Z, s=0.3, color='steelblue', alpha=0.5, label='Poincaré', rasterized=True)

# Magnetic axis
ax.plot(R_ax, Z_ax, 'k+', ms=10, mew=2)

ax.set_xlabel('R (m)')
ax.set_ylabel('Z (m)')
ax.set_title(f"Solov'ev q=4/1 island chain\n(4,1) RMP δ_b={delta_b}, {len(all_R)} Poincaré pts")
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=8)

# Add q=4/1 label on contour
ax.text(R_ax + eq.a * np.sqrt(psi_res) + 0.03, Z_ax, 'q=4/1', color='steelblue', fontsize=8)

outpath = r'D:\Repo\pyna\scripts\rmp_poincare_dense.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
