"""CPU-based RMP island chain validation (q=4/1).

Uses FieldLineTracer + ThreadPoolExecutor (n_workers=8).
Equilibrium: EquilibriumSolovev(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)
Perturbation: (4,1) RMP with epsilon_h=0.01
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyna.MCF.equilibrium.Solovev import EquilibriumSolovev
from pyna.flt import _rk4_step
from pyna.topo.island_extract import extract_island_width

# ---------------------------------------------------------------------------
# 1. Build equilibrium
# ---------------------------------------------------------------------------
print("Building Solov'ev equilibrium (R0=1.86, q0=3.0)...")
eq = EquilibriumSolovev(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)
R_ax, Z_ax = eq.magnetic_axis
a = eq.a
B0 = eq._B0
print(f"  Magnetic axis: R={R_ax:.3f} m, Z={Z_ax:.3f} m")

# ---------------------------------------------------------------------------
# 2. Find q=4/1 resonant surface
# ---------------------------------------------------------------------------
print("Scanning q profile to find q=4 surface...")
psi_scan = np.linspace(0.05, 0.95, 300)
q_scan = eq.q_profile(psi_scan)

target_q = 4.0
psi_res = None
for i in range(len(q_scan) - 1):
    if (np.isfinite(q_scan[i]) and np.isfinite(q_scan[i+1]) and
            ((q_scan[i] - target_q) * (q_scan[i+1] - target_q) < 0)):
        frac = (target_q - q_scan[i]) / (q_scan[i+1] - q_scan[i])
        psi_res = psi_scan[i] + frac * (psi_scan[i+1] - psi_scan[i])
        print(f"  q=4 at psi_norm = {psi_res:.4f}")
        break

if psi_res is None:
    valid = np.isfinite(q_scan)
    if valid.any():
        idx = np.argmin(np.abs(q_scan[valid] - target_q))
        psi_res = psi_scan[valid][idx]
        print(f"  q=4 not crossed; nearest psi_norm = {psi_res:.4f}, q={q_scan[valid][idx]:.3f}")
    else:
        psi_res = 0.5

r_res = float(a * np.sqrt(psi_res))
R_res = float(R_ax + r_res)
print(f"  r_res = {r_res:.3f} m, R_res = {R_res:.3f} m (approx midplane)")

# ---------------------------------------------------------------------------
# 3. Build field function with (4,1) RMP perturbation
# ---------------------------------------------------------------------------
epsilon_h = 0.01
m_rmp = 4
n_rmp = 1

def field_func(rzphi):
    R, Z, phi = float(rzphi[0]), float(rzphi[1]), float(rzphi[2])
    BR, BZ = eq.BR_BZ(R, Z)
    Bphi = float(eq.Bphi(R))
    psi_n = float(np.clip(eq.psi(R, Z), 0.0, 1.0))
    theta = float(np.arctan2(Z - Z_ax, R - R_ax))
    env = psi_n * (1.0 - psi_n)
    dBR = epsilon_h * B0 * env * m_rmp * float(np.cos(m_rmp * theta - n_rmp * phi))
    BR = float(BR) + dBR
    Bmag = float(np.sqrt(BR**2 + float(BZ)**2 + Bphi**2)) + 1e-30
    return np.array([BR / Bmag, float(BZ) / Bmag, Bphi / (R * Bmag)])

# ---------------------------------------------------------------------------
# 4. Efficient tracer: collect crossings on-the-fly (no full traj storage)
# ---------------------------------------------------------------------------
T_MAX = 4000.0
DT = 0.06
N_steps = int(T_MAX / DT)

def trace_and_collect_crossings(start_pt):
    """Trace one field line and collect phi=0 crossings on-the-fly."""
    y = np.array(start_pt, dtype=float)
    phi_mod_prev = 0.0
    crossings = []
    for _ in range(N_steps):
        y = _rk4_step(field_func, y, DT)
        phi_mod = y[2] % (2 * np.pi)
        if phi_mod < phi_mod_prev - np.pi:  # wrapped (upward crossing)
            dphi = (2*np.pi - phi_mod_prev) + phi_mod
            frac = (2*np.pi - phi_mod_prev) / dphi if dphi > 1e-30 else 0.5
            crossings.append([y[0], y[1]])  # approximate with current point
        phi_mod_prev = phi_mod
    return np.array(crossings) if crossings else np.empty((0, 2))

# ---------------------------------------------------------------------------
# 5. Start points: 32 lines bracketing q=4/1
# ---------------------------------------------------------------------------
N_lines = 32
psi_targets = np.linspace(max(psi_res - 0.10, 0.02), min(psi_res + 0.10, 0.97), N_lines)
start_pts = [
    [R_ax + a * np.sqrt(p), Z_ax, 0.0]
    for p in psi_targets
]
print(f"  {N_lines} start points")

# ---------------------------------------------------------------------------
# 6. Parallel tracing with 8 workers (memory-safe)
# ---------------------------------------------------------------------------
print(f"\nTracing {N_lines} field lines (t_max={T_MAX}, dt={DT}, n_workers=8)...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as executor:
    per_line_crossings = list(executor.map(trace_and_collect_crossings, start_pts))
t_cpu = time.time() - t0
total_crossings = sum(len(c) for c in per_line_crossings)
print(f"  Done in {t_cpu:.1f} s, {total_crossings:,} total crossings")

# Collect all crossings
all_crossings_list = [c for c in per_line_crossings if len(c) > 0]
if all_crossings_list:
    poincare_pts = np.vstack(all_crossings_list)
    print(f"  Poincaré points: {len(poincare_pts):,}")
else:
    print("  WARNING: No crossings!")
    poincare_pts = np.zeros((0, 2))

# ---------------------------------------------------------------------------
# 7. Extract island chain
# ---------------------------------------------------------------------------
island_chain = None
if len(poincare_pts) > 50:
    r_from_ax = np.sqrt((poincare_pts[:,0] - R_ax)**2 + (poincare_pts[:,1] - Z_ax)**2)
    mask = np.abs(r_from_ax - r_res) < 0.3 * r_res
    pts_near = poincare_pts[mask]
    print(f"  Points near q=4 surface: {len(pts_near):,}")
    if len(pts_near) > 20:
        try:
            print("Extracting island chain (mode_m=4)...")
            island_chain = extract_island_width(
                pts_near, R_ax, Z_ax,
                mode_m=4,
                psi_func=lambda R, Z: eq.psi(R, Z)
            )
            print(f"  O-points: {island_chain.O_points}")
            print(f"  half_width_r = {island_chain.half_width_r:.4f} m")
            print(f"  half_width_psi = {island_chain.half_width_psi:.4f}")
        except Exception as exc:
            print(f"  Island extraction failed: {exc}")

# Theoretical island width estimate
try:
    dpsi = 0.01
    q_lo = float(eq.q_profile(np.array([psi_res - dpsi]))[0])
    q_hi = float(eq.q_profile(np.array([psi_res + dpsi]))[0])
    dqdpsi = (q_hi - q_lo) / (2 * dpsi) if abs(q_hi - q_lo) > 1e-10 else 1.0
    W_theory = 4.0 * np.sqrt(epsilon_h * a / abs(dqdpsi)) if abs(dqdpsi) > 1e-10 else np.nan
except Exception:
    W_theory = np.nan
print(f"  Theoretical island half-width ≈ {W_theory:.4f} m (rough)")

# ---------------------------------------------------------------------------
# 8. Build equilibrium contours
# ---------------------------------------------------------------------------
R_grid = np.linspace(R_ax - a * 1.1, R_ax + a * 1.1, 200)
Z_grid = np.linspace(-a * 1.3, a * 1.3, 200)
RR, ZZ = np.meshgrid(R_grid, Z_grid)
psi_grid = np.vectorize(eq.psi)(RR, ZZ)

# q=4/1 surface contour (just use psi_res isoline from meshgrid)

# ---------------------------------------------------------------------------
# 9. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Gray flux surface contours
ax.contour(RR, ZZ, psi_grid, levels=np.linspace(0.1, 0.95, 10),
           colors='gray', linewidths=0.5, alpha=0.5)
# Black LCFS
ax.contour(RR, ZZ, psi_grid, levels=[1.0], colors='black', linewidths=1.5)
# Blue dashed q=4/1 surface
ax.contour(RR, ZZ, psi_grid, levels=[psi_res], colors='blue', linestyles='--', linewidths=1.2)
ax.plot([], [], 'b--', lw=1.2, label=f'q=4/1 surface (ψ={psi_res:.3f})')

# Poincaré scatter (blue dots)
if len(poincare_pts) > 0:
    ax.scatter(poincare_pts[:, 0], poincare_pts[:, 1],
               s=1.0, c='blue', alpha=0.6, linewidths=0, label='Poincaré crossings', zorder=4)

# O-points and X-points
if island_chain is not None and len(island_chain.O_points) > 0:
    for op in island_chain.O_points:
        ax.plot(op[0], op[1], 'ro', ms=8, zorder=6)
        R_hat = np.array([op[0] - R_ax, op[1] - Z_ax])
        R_hat /= (np.linalg.norm(R_hat) + 1e-30)
        hw = island_chain.half_width_r
        xy1 = (op[0] - hw * R_hat[0], op[1] - hw * R_hat[1])
        xy2 = (op[0] + hw * R_hat[0], op[1] + hw * R_hat[1])
        ax.annotate('', xy=xy2, xytext=xy1,
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2), zorder=7)
    ax.plot([], [], 'ro', ms=8, label='O-points')
    ax.plot([], [], color='green', lw=2,
            label=f'Island half-width: {island_chain.half_width_r:.3f} m')
    for xp in island_chain.X_points:
        ax.plot(xp[0], xp[1], 'bx', ms=8, mew=2, zorder=6)
    ax.plot([], [], 'bx', ms=8, mew=2, label='X-points')

ax.plot(R_ax, Z_ax, 'k+', ms=12, mew=2, zorder=8, label='Magnetic axis')
ax.set_xlabel('R (m)', fontsize=12)
ax.set_ylabel('Z (m)', fontsize=12)
hw_str = f"{island_chain.half_width_r:.3f}" if island_chain is not None else "N/A"
theory_str = f"{W_theory:.3f}" if np.isfinite(W_theory) else "N/A"
ax.set_title(
    f"q=4/1 Island Chain — ε_h={epsilon_h}\n"
    f"Extracted half-width: {hw_str} m  |  Theoretical: {theory_str} m",
    fontsize=11
)
ax.legend(fontsize=8, loc='upper right')
ax.set_aspect('equal')
plt.tight_layout()

outpath = r"D:\Repo\pyna\scripts\rmp_island_fixed.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
plt.close()
