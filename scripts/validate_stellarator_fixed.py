"""CPU-based stellarator boundary island visualization.

Uses on-the-fly crossing detection (memory-efficient).
SimpleStellarartor: R0=3.0, r0=0.35, B0=1.0, q0=1.5, q1=5.5, m_h=5, n_h=1, epsilon_h=0.04
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

from pyna.mag.stellarator import SimpleStellarartor
from pyna.flt import _rk4_step
from pyna.topo.island_extract import extract_island_width

# ---------------------------------------------------------------------------
# 1. Build stellarator
# ---------------------------------------------------------------------------
print("Building SimpleStellarartor...")
st = SimpleStellarartor(
    R0=3.0, r0=0.35, B0=1.0,
    q0=1.5, q1=5.5,
    m_h=5, n_h=1,
    epsilon_h=0.04,
)
R_ax, Z_ax = st.magnetic_axis
print(f"  Magnetic axis: R={R_ax:.3f} m, Z={Z_ax:.3f} m")

# ---------------------------------------------------------------------------
# 2. Find q=5/1 resonant surface
# ---------------------------------------------------------------------------
psi_res_list = st.resonant_psi(5, 1)
if psi_res_list:
    psi_res = psi_res_list[0]
    print(f"  q=5/1 at psi_norm = {psi_res:.4f}")
else:
    psi_res = 0.9
    print(f"  q=5/1 not found; using psi_res=0.9")

# ---------------------------------------------------------------------------
# 3. 6 Poincaré sections
# ---------------------------------------------------------------------------
sections = [k * np.pi / 3 for k in range(6)]
section_labels = ['φ=0', 'φ=π/3', 'φ=2π/3', 'φ=π', 'φ=4π/3', 'φ=5π/3']
N_sections = len(sections)

# ---------------------------------------------------------------------------
# 4. Start points: 50 lines psi=0.3 to 0.92
# ---------------------------------------------------------------------------
N_lines = 50
psi_all = np.linspace(0.3, 0.92, N_lines)
r_all = np.sqrt(psi_all) * st.r0
start_pts = [
    [st.R0 + r_all[i], 0.0, 0.0]
    for i in range(N_lines)
]
print(f"  {N_lines} field lines, R range: {st.R0+r_all[0]:.3f}–{st.R0+r_all[-1]:.3f} m")

# ---------------------------------------------------------------------------
# 5. Trace with on-the-fly crossing detection for all 6 sections
# ---------------------------------------------------------------------------
T_MAX = 3000.0
DT = 0.07
N_steps = int(T_MAX / DT)

field_func = st.field_func

def trace_and_collect_all_sections(start_pt):
    """Trace one field line and collect crossings for all 6 sections."""
    y = np.array(start_pt, dtype=float)
    # Store (phi_shift_prev) for each section
    phi_shift_prev = np.array([(y[2] - phi_sec) % (2 * np.pi) for phi_sec in sections])
    crossings = [[] for _ in range(N_sections)]
    for _ in range(N_steps):
        y = _rk4_step(field_func, y, DT)
        for k, phi_sec in enumerate(sections):
            phi_shift = (y[2] - phi_sec) % (2 * np.pi)
            p0, p1 = phi_shift_prev[k], phi_shift
            # Upward crossing: phi_shift drops from near 2pi to near 0
            if p0 > np.pi and p1 < p0 - np.pi:
                crossings[k].append([y[0], y[1]])
            phi_shift_prev[k] = phi_shift
    return [np.array(c) if c else np.empty((0, 2)) for c in crossings]

print(f"\nTracing {N_lines} field lines (t_max={T_MAX}, dt={DT}, n_workers=8)...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(trace_and_collect_all_sections, start_pts))
t_cpu = time.time() - t0
print(f"  Done in {t_cpu:.1f} s")

# Merge crossings per section
section_crossings = []
for k in range(N_sections):
    pts_k = [results[i][k] for i in range(N_lines) if len(results[i][k]) > 0]
    if pts_k:
        merged = np.vstack(pts_k)
    else:
        merged = np.empty((0, 2))
    section_crossings.append(merged)
    print(f"  {section_labels[k]}: {len(merged):,} crossings")

# ---------------------------------------------------------------------------
# 6. Island extraction per section
# ---------------------------------------------------------------------------
island_chains = []
for i, (pts, phi_sec) in enumerate(zip(section_crossings, sections)):
    if len(pts) > 30:
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
            print(f"  Section {i}: extraction failed: {exc}")
    else:
        island_chains.append(None)

# ---------------------------------------------------------------------------
# 7. 2×3 panel plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.ravel()

for ax, pts, label, chain in zip(axes, section_crossings, section_labels, island_chains):
    if len(pts) > 0:
        psi_pts = st.psi_ax(pts[:, 0], pts[:, 1])
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=0.5, c=psi_pts, cmap='plasma', vmin=0, vmax=1,
                   alpha=0.8, linewidths=0)
    if chain is not None and len(chain.O_points) > 0:
        for op in chain.O_points:
            ax.plot(op[0], op[1], 'ro', ms=5, zorder=5)
            hw = chain.half_width_r
            R_hat = np.array([op[0] - R_ax, op[1] - Z_ax])
            R_hat /= (np.linalg.norm(R_hat) + 1e-30)
            xy1 = (op[0] - hw * R_hat[0], op[1] - hw * R_hat[1])
            xy2 = (op[0] + hw * R_hat[0], op[1] + hw * R_hat[1])
            ax.annotate('', xy=xy2, xytext=xy1,
                        arrowprops=dict(arrowstyle='<->', color='green', lw=1.2))
        for xp in chain.X_points:
            ax.plot(xp[0], xp[1], 'bx', ms=5, mew=1.5, zorder=5)
    ax.plot(R_ax, Z_ax, 'k+', ms=8, mew=1.5, zorder=6)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel('R (m)', fontsize=7)
    ax.set_ylabel('Z (m)', fontsize=7)
    ax.tick_params(labelsize=7)

fig.suptitle(
    f'Stellarator Boundary Islands �?CPU (FieldLineTracer)\n'
    f'q=5/1 at ψ={psi_res:.3f}, ε_h=0.04, m_h=5, n_h=1',
    fontsize=11
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
outpath_6panel = r"D:\Repo\pyna\scripts\stellarator_boundary_fixed.png"
plt.savefig(outpath_6panel, dpi=150, bbox_inches='tight')
print(f"\nSaved 6-panel: {outpath_6panel}")
plt.close()

# ---------------------------------------------------------------------------
# 8. Island half-width vs section
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 4))
phi_degrees = [s * 180 / np.pi for s in sections]
hw_vals = [
    chain.half_width_r if chain is not None and np.isfinite(chain.half_width_r) else np.nan
    for chain in island_chains
]
valid = [(x, y) for x, y in zip(phi_degrees, hw_vals) if np.isfinite(y)]
if valid:
    xv, yv = zip(*valid)
    ax2.plot(xv, yv, 'bo-', ms=8)
ax2.set_xlabel('Section φ (degrees)', fontsize=12)
ax2.set_ylabel('Island half-width (m)', fontsize=12)
ax2.set_title('Island half-width vs Poincaré section\nq=5/1 island chain', fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
outpath_hw = r"D:\Repo\pyna\scripts\stellarator_halfwidth_vs_section.png"
plt.savefig(outpath_hw, dpi=150, bbox_inches='tight')
print(f"Saved half-width plot: {outpath_hw}")
plt.close()
