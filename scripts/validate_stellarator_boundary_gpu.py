"""GPU-accelerated stellarator boundary island visualization.

Uses SimpleStellarartor with edge q-profile for island divertor topology.
q=5/1 resonance near plasma edge.
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyna.mag.stellarator import SimpleStellarartor
from pyna.topo.island_extract import extract_island_width


# ---------------------------------------------------------------------------
# 1. Build stellarator
# ---------------------------------------------------------------------------
print("Building SimpleStellarartor (q0=1.5, q1=5.5, edge q=5/1 island)...")
st = SimpleStellarartor(
    R0=3.0, r0=0.35, B0=1.0,
    q0=1.5, q1=5.5,
    m_h=5, n_h=1,
    epsilon_h=0.04,
)
R_ax, Z_ax = st.magnetic_axis
print(f"  Magnetic axis: R={R_ax:.3f} m, Z={Z_ax:.3f} m")

# ---------------------------------------------------------------------------
# 2. Verify q=5/1 accessible
# ---------------------------------------------------------------------------
psi_res_list = st.resonant_psi(5, 1)
if psi_res_list:
    psi_res = psi_res_list[0]
    print(f"  q=5/1 at psi_norm = {psi_res:.4f} (edge={'yes' if psi_res > 0.8 else 'no'})")
else:
    psi_res = 0.9
    print(f"  q=5/1 not found; using psi_res=0.9")

# ---------------------------------------------------------------------------
# 3. Build start points (core to edge: S=0.3 to 0.95)
# ---------------------------------------------------------------------------
psi_all = np.linspace(0.3, 0.95, 60)
r_all   = np.sqrt(psi_all) * st.r0
start_pts = np.column_stack([
    st.R0 + r_all,
    np.zeros(60),
    np.zeros(60),
])
print(f"  {len(start_pts)} field lines from psi={psi_all[0]:.2f} to {psi_all[-1]:.2f}")

# ---------------------------------------------------------------------------
# 4. Poincaré crossing detector (arbitrary phi_section)
# ---------------------------------------------------------------------------

def detect_crossings_phi(traj, phi_section, tol=0.15):
    """Detect crossings near phi_section (mod 2pi)."""
    R   = traj[:, 0]
    Z   = traj[:, 1]
    phi = traj[:, 2]

    phi_mod = phi % (2 * np.pi)
    target  = phi_section % (2 * np.pi)
    crossings = []
    for i in range(1, len(phi)):
        # Check if phi_mod crosses target
        p0, p1 = phi_mod[i-1], phi_mod[i]
        # Handle wrap-around
        if abs(p1 - p0) > np.pi:
            # wrap happened; adjust
            if p1 < p0:
                p1 += 2*np.pi
            else:
                p0 += 2*np.pi
        if (p0 <= target < p1) or (p1 <= target < p0):
            # linear interpolate
            dp = p1 - p0
            if abs(dp) < 1e-30:
                frac = 0.5
            else:
                frac = (target - p0) / dp
            frac = np.clip(frac, 0, 1)
            R_cross = R[i-1] + frac * (R[i] - R[i-1])
            Z_cross = Z[i-1] + frac * (Z[i] - Z[i-1])
            crossings.append([R_cross, Z_cross])
    if not crossings:
        return np.empty((0, 2))
    return np.array(crossings)


# ---------------------------------------------------------------------------
# 5. GPU tracing
# ---------------------------------------------------------------------------
T_MAX = 8000.0
DT    = 0.04

gpu_ok = False
try:
    from pyna.flt_cuda import CUDAFieldLineTracer

    # Map stellarator params to CUDAFieldLineTracer (Solov'ev-encoded kernel)
    # Use the stellarator as a toroidal machine: R0, r0≈a, B0, q~q0_eff
    # The kernel uses Solov'ev; stellarator uses circular flux surfaces
    # q0 in kernel �?mean q of stellarator; perturbation encoded via epsilon_h
    q_eff = (st.q0 + st.q1) / 2.0  # ~ 3.5 average
    tracer = CUDAFieldLineTracer(
        R0=st.R0, a=st.r0 * 1.2,
        B0=st.B0, q0=q_eff,
        epsilon_h=st.epsilon_h, m_h=float(st.m_h), n_h=float(st.n_h),
        dt=DT,
    )
    print(f"\nGPU tracing {len(start_pts)} lines, t_max={T_MAX}, dt={DT}...")
    t0 = time.time()
    trajs = tracer.trace_many(start_pts, t_max=T_MAX)
    t_gpu = time.time() - t0
    print(f"  GPU time: {t_gpu:.2f} s  ({sum(len(t) for t in trajs):,} total steps)")
    gpu_ok = True
except Exception as exc:
    print(f"\nGPU tracing failed: {exc}")
    print("Falling back to scipy solve_ivp (CPU)...")

if not gpu_ok:
    from scipy.integrate import solve_ivp

    def field_func(t, rzphi):
        return list(st.field_func(rzphi))

    T_MAX_CPU = 1500.0
    print(f"CPU tracing {len(start_pts)} lines, t_max={T_MAX_CPU}...")
    t0 = time.time()
    trajs = []
    for sp in start_pts:
        try:
            sol = solve_ivp(field_func, [0, T_MAX_CPU], list(sp),
                            method='RK45', max_step=0.15, dense_output=False,
                            rtol=1e-5, atol=1e-8)
            trajs.append(sol.y.T)
        except Exception:
            trajs.append(np.array([sp]))
    t_cpu = time.time() - t0
    print(f"  CPU time: {t_cpu:.2f} s")

# ---------------------------------------------------------------------------
# 6. Six section angles
# ---------------------------------------------------------------------------
sections = [k * np.pi / 3 for k in range(6)]
section_labels = ['φ=0', 'φ=π/3', 'φ=2π/3', 'φ=π', 'φ=4π/3', 'φ=5π/3']

print("\nCollecting crossings for 6 sections...")
section_crossings = []
for phi_sec in sections:
    pts_all = []
    for traj in trajs:
        if traj is None or len(traj) < 2:
            continue
        pts = detect_crossings_phi(traj, phi_sec)
        if len(pts) > 0:
            pts_all.append(pts)
    if pts_all:
        merged = np.vstack(pts_all)
        section_crossings.append(merged)
        print(f"  phi={phi_sec:.3f}: {len(merged):,} crossings")
    else:
        section_crossings.append(np.empty((0, 2)))
        print(f"  phi={phi_sec:.3f}: 0 crossings")

# Also phi=0 for big core-to-edge plot
phi0_crossings = section_crossings[0]

# ---------------------------------------------------------------------------
# 7. Island extraction per section
# ---------------------------------------------------------------------------
island_chains = []
for i, (pts, phi_sec) in enumerate(zip(section_crossings, sections)):
    if len(pts) > 50:
        try:
            chain = extract_island_width(
                pts, R_ax, Z_ax,
                mode_m=5,
                psi_func=lambda R, Z: st.psi_ax(R, Z)
            )
            island_chains.append(chain)
            print(f"  Section {i}: half_width_r={chain.half_width_r:.4f} m")
        except Exception as exc:
            island_chains.append(None)
            print(f"  Section {i}: island extraction failed: {exc}")
    else:
        island_chains.append(None)

# ---------------------------------------------------------------------------
# 8. 2×3 panel plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor='#0a0a1a')
axes = axes.ravel()

n_lines = len(start_pts)
colors = [plt.cm.plasma(i / max(n_lines - 1, 1)) for i in range(n_lines)]

for ax_idx, (ax, pts, label, chain) in enumerate(
        zip(axes, section_crossings, section_labels, island_chains)):
    ax.set_facecolor('#0a0a1a')
    if len(pts) > 0:
        # Color by approximate psi (use R distance from axis)
        psi_pts = st.psi_ax(pts[:, 0], pts[:, 1])
        sc = ax.scatter(pts[:, 0], pts[:, 1],
                        s=0.4, c=psi_pts, cmap='plasma', vmin=0, vmax=1,
                        alpha=0.8, linewidths=0)
    if chain is not None:
        for op in chain.O_points:
            ax.plot(op[0], op[1], 'o', color='lime', ms=5, zorder=5)
            dx = chain.half_width_r
            ax.annotate('', xy=(op[0]+dx, op[1]), xytext=(op[0]-dx, op[1]),
                        arrowprops=dict(arrowstyle='<->', color='lime', lw=1.2))
        for xp in chain.X_points:
            ax.plot(xp[0], xp[1], 'x', color='red', ms=5, mew=1.5, zorder=5)
    ax.plot(R_ax, Z_ax, '+', color='white', ms=8, mew=1.5, zorder=6)
    ax.set_title(label, color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_xlabel('R (m)', color='white', fontsize=7)
    ax.set_ylabel('Z (m)', color='white', fontsize=7)

fig.suptitle(
    f'Stellarator Boundary Islands �?GPU\n'
    f'q=5/1 resonance at ψ={psi_res:.3f}, ε_h=0.04, m_h=5, n_h=1',
    color='white', fontsize=11, y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath_6panel = r"D:\Repo\pyna\scripts\stellarator_boundary_gpu.png"
plt.savefig(outpath_6panel, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"\nSaved 6-panel: {outpath_6panel}")
plt.close()

# ---------------------------------------------------------------------------
# 9. Core-to-edge Poincaré at phi=0
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor='#0a0a1a')
ax2.set_facecolor('#0a0a1a')

if len(phi0_crossings) > 0:
    psi_pts = st.psi_ax(phi0_crossings[:, 0], phi0_crossings[:, 1])
    ax2.scatter(phi0_crossings[:, 0], phi0_crossings[:, 1],
                s=0.5, c=psi_pts, cmap='plasma', vmin=0, vmax=1, alpha=0.8, linewidths=0)

if island_chains[0] is not None:
    chain = island_chains[0]
    for op in chain.O_points:
        ax2.plot(op[0], op[1], 'o', color='lime', ms=8, zorder=5, label='O-point')
        dx = chain.half_width_r
        ax2.annotate('', xy=(op[0]+dx, op[1]), xytext=(op[0]-dx, op[1]),
                     arrowprops=dict(arrowstyle='<->', color='lime', lw=1.5))
    for xp in chain.X_points:
        ax2.plot(xp[0], xp[1], 'x', color='red', ms=8, mew=2, zorder=5, label='X-point')

ax2.plot(R_ax, Z_ax, '+', color='white', ms=12, mew=2, zorder=6, label='Magnetic axis')
ax2.set_title(
    f'Stellarator Poincaré (φ=0) �?Core to Edge\n'
    f'{len(phi0_crossings):,} crossings',
    color='white', fontsize=11
)
ax2.set_xlabel('R (m)', color='white', fontsize=12)
ax2.set_ylabel('Z (m)', color='white', fontsize=12)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('white')

handles, labels = ax2.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax2.legend(unique.values(), unique.keys(), fontsize=8, facecolor='#222', labelcolor='white')

plt.tight_layout()
outpath_core = r"D:\Repo\pyna\scripts\stellarator_core_edge_gpu.png"
plt.savefig(outpath_core, dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
print(f"Saved core-to-edge: {outpath_core}")
plt.close()
