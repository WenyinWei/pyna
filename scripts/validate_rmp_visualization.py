"""Validate q=4/1 RMP island extraction on a Solov'ev equilibrium.

Generates: scripts/rmp_island_validation.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyna.mag.solovev import solovev_iter_like
from pyna.topo.poincare import PoincareMap, ToroidalSection
from pyna.flt import FieldLineTracer
from pyna.topo.island import locate_rational_surface, island_halfwidth
from pyna.topo.island_extract import extract_island_width

# ---------------------------------------------------------------------------
# 1. Build equilibrium
# ---------------------------------------------------------------------------
eq = solovev_iter_like(scale=0.3)
print(f"R0={eq.R0:.3f} m  a={eq.a:.3f} m  κ={eq.kappa}  δ={eq.delta}")
R_ax, Z_ax = eq.magnetic_axis
print(f"Magnetic axis: R={R_ax:.4f} m  Z={Z_ax:.4f} m")

# ---------------------------------------------------------------------------
# 2. q profile and resonant surface
# ---------------------------------------------------------------------------
S_values = np.linspace(0.05, 0.95, 60)
psi_values = S_values**2
q_values = eq.q_profile(psi_values, n_theta=256)

S_res_list = locate_rational_surface(S_values, q_values, m=2, n=1)
if not S_res_list:
    raise RuntimeError("q=2/1 surface not found — check q profile range")
S_res = S_res_list[0]
psi_res = S_res**2
print(f"q=2/1 resonant surface: S_res={S_res:.4f}  psi_res={psi_res:.4f}")

# ---------------------------------------------------------------------------
# 3. RMP perturbation (analytic helical)
# ---------------------------------------------------------------------------
delta_b = 5e-3
m_rmp, n_rmp = 2, 1

def rmp_BR(R, Z, phi):
    psi_n = eq.psi(np.atleast_1d(R), np.atleast_1d(Z))
    envelope = psi_n * (1 - psi_n)
    return delta_b * eq.B0 * envelope * np.cos(m_rmp * np.arctan2(Z - Z_ax, R - R_ax) - n_rmp * phi)

def rmp_BZ(R, Z, phi):
    psi_n = eq.psi(np.atleast_1d(R), np.atleast_1d(Z))
    envelope = psi_n * (1 - psi_n)
    return delta_b * eq.B0 * envelope * np.sin(m_rmp * np.arctan2(Z - Z_ax, R - R_ax) - n_rmp * phi)

def field_func(rzphi):
    rzphi = np.asarray(rzphi)
    R, Z, phi = rzphi[0], rzphi[1], rzphi[2]
    BR0, BZ0 = eq.BR_BZ(np.array([R]), np.array([Z]))
    Bphi0 = eq.Bphi(np.array([R]))
    dBR = rmp_BR(R, Z, phi)
    dBZ = rmp_BZ(R, Z, phi)
    BR_t = float(BR0[0]) + float(np.squeeze(dBR))
    BZ_t = float(BZ0[0]) + float(np.squeeze(dBZ))
    Bphi_t = float(Bphi0[0])
    B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bphi_t**2) + 1e-30
    return np.array([BR_t/B_mag, BZ_t/B_mag, Bphi_t/(R*B_mag)])

# ---------------------------------------------------------------------------
# 4. Field-line tracing & Poincaré
# ---------------------------------------------------------------------------
n_lines = 12
psi_arr = np.linspace(max(psi_res - 0.06, 0.01), min(psi_res + 0.06, 0.95), n_lines)
start_pts = np.column_stack([
    R_ax + np.sqrt(psi_arr) * eq.a,
    np.zeros(n_lines),
    np.zeros(n_lines),
])

section = ToroidalSection(phi0=0.0)
tracer = FieldLineTracer(field_func, dt=0.05)

print(f"Tracing {n_lines} field lines (t_max=1500)…")
trajs = tracer.trace_many(start_pts, t_max=1500.0)

pmap = PoincareMap([section])
for traj in trajs:
    pmap.record_trajectory(traj)

pts_all = pmap.crossing_array(0)
print(f"Total Poincaré crossings: {len(pts_all)}")

# ---------------------------------------------------------------------------
# 5. Island extraction
# ---------------------------------------------------------------------------
r_pts = np.sqrt((pts_all[:, 0] - R_ax)**2 + pts_all[:, 1]**2)
r_res = S_res * eq.a
mask = np.abs(r_pts - r_res) < 0.2 * eq.a
pts_near = pts_all[mask] if mask.sum() >= 16 else pts_all
print(f"Points near q=4/1 surface: {mask.sum()} (total: {len(pts_all)})")

chain = None
if len(pts_near) >= 8:
    chain = extract_island_width(
        pts_near[:, :2], R_ax, Z_ax,
        mode_m=2,
        psi_func=lambda R, Z: float(eq.psi(np.array([R]), np.array([Z]))),
    )
    print(f"O-points found: {len(chain.O_points)}")
    print(f"Island half-width: w_r = {chain.half_width_r*100:.2f} cm")
    print(f"Island half-width: w_psi = {chain.half_width_psi:.4f}")

# Theoretical half-width
b_profile = delta_b * psi_values * (1 - psi_values)
w_theory = island_halfwidth(m=2, n=1, S_res=S_res, S=S_values,
                             q_profile=q_values, tilde_b_mn=b_profile)
print(f"Theoretical island half-width (Chirikov): w_S = {w_theory:.4f}")
print(f"  → w_r ≈ {w_theory * eq.a * 100:.2f} cm  (a={eq.a:.3f} m)")

if chain is not None:
    w_poincare_S = chain.half_width_r / eq.a
    ratio = w_poincare_S / w_theory if w_theory > 0 else float('nan')
    print(f"--- Validation ---")
    print(f"  Theory  : w_S = {w_theory:.4f}  → w_r = {w_theory*eq.a*100:.2f} cm")
    print(f"  Poincaré: w_r = {chain.half_width_r*100:.2f} cm  → w_S = {w_poincare_S:.4f}")
    print(f"  Ratio (Poincaré/Theory) = {ratio:.3f}")

# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
R_range = (eq.R0 - 1.4*eq.a, eq.R0 + 1.4*eq.a)
Z_range = (-1.4*eq.kappa*eq.a, 1.4*eq.kappa*eq.a)

R1d = np.linspace(*R_range, 300)
Z1d = np.linspace(*Z_range, 300)
Rg, Zg = np.meshgrid(R1d, Z1d)
psi_g = eq.psi(Rg, Zg)

fig, ax = plt.subplots(figsize=(7, 9))

ax.contour(Rg, Zg, psi_g, levels=np.linspace(0.05, 0.95, 15),
           colors='lightgray', linewidths=0.5)
ax.contour(Rg, Zg, psi_g, levels=[1.0], colors='k', linewidths=1.5)
ax.contour(Rg, Zg, psi_g, levels=[psi_res], colors='navy',
           linewidths=0.8, linestyles='--')

if len(pts_all) > 0:
    ax.scatter(pts_all[:, 0], pts_all[:, 1], s=0.8, c='steelblue',
               alpha=0.5, rasterized=True, label='Poincare')

if chain is not None and len(chain.O_points) > 0:
    ax.scatter(chain.O_points[:, 0], chain.O_points[:, 1],
               s=60, c='red', marker='o', zorder=5, label='O-point')
    ax.scatter(chain.X_points[:, 0], chain.X_points[:, 1],
               s=60, c='blue', marker='x', zorder=5, lw=2, label='X-point')

    for O_pt in chain.O_points:
        dr = O_pt[0] - R_ax
        dz = O_pt[1] - Z_ax
        dist = np.sqrt(dr**2 + dz**2) + 1e-30
        ur, uz = dr/dist, dz/dist
        w = chain.half_width_r
        ax.annotate('',
            xy=(O_pt[0] + w*ur, O_pt[1] + w*uz),
            xytext=(O_pt[0] - w*ur, O_pt[1] - w*uz),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5),
        )

ax.plot(R_ax, Z_ax, '+k', ms=10, mew=2)
ax.set_aspect('equal')
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title(f"q=2/1 island — Solov'ev + (2,1) RMP  (delta_b={delta_b:.0e})")
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(R_range); ax.set_ylim(Z_range)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), 'rmp_island_validation.png')
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
