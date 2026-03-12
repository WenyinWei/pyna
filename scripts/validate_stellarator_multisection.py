"""Multi-section Poincare validation for a simple stellarator (q=2/1).

Generates: scripts/stellarator_multisection.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyna.mag.stellarator import SimpleStellarartor
from pyna.topo.poincare import PoincareMap, ToroidalSection
from pyna.flt import FieldLineTracer
from pyna.topo.island_extract import extract_island_width

# ---------------------------------------------------------------------------
# 1. Build stellarator with q=2/1 resonance in range
# ---------------------------------------------------------------------------
# q0=1.5, q1=4.0  -> q=2 at psi=(2-1.5)/(4-1.5)=0.2, well inside
stell = SimpleStellarartor(
    R0=3.0, r0=0.3, B0=1.0,
    q0=1.5, q1=4.0,
    m_h=2, n_h=1,
    epsilon_h=0.04,
)
R_ax, Z_ax = stell.magnetic_axis
print(f"R_ax={R_ax}, r0={stell.r0}")
psi_list = stell.resonant_psi(2, 1)
print(f"q=2/1 resonant psi: {psi_list}")
if not psi_list:
    raise RuntimeError("q=2/1 surface not found")
psi_res = psi_list[0]
r_res = np.sqrt(psi_res) * stell.r0
print(f"r_res = {r_res:.4f} m")

# ---------------------------------------------------------------------------
# 2. Trace field lines on 4 toroidal sections
# ---------------------------------------------------------------------------
n_phi_sections = 4
phi_sections = [k * np.pi / 2 for k in range(n_phi_sections)]
sections = [ToroidalSection(phi0=phi) for phi in phi_sections]

start_pts = stell.start_points_near_resonance(m=2, n=1, n_lines=16, delta_psi=0.08)
print(f"Tracing {len(start_pts)} field lines (t_max=1200)...")
tracer = FieldLineTracer(stell.field_func, dt=0.06)
trajs = tracer.trace_many(start_pts, t_max=1200.0)

pmap = PoincareMap(sections)
for traj in trajs:
    pmap.record_trajectory(traj)

# ---------------------------------------------------------------------------
# 3. Extract island width per section
# ---------------------------------------------------------------------------
chains = []
for i in range(n_phi_sections):
    pts = pmap.crossing_array(i)
    print(f"Section phi={np.degrees(phi_sections[i]):.0f} deg: {len(pts)} crossings")
    if len(pts) >= 8:
        r_pts_s = np.sqrt((pts[:, 0] - R_ax)**2 + pts[:, 1]**2)
        mask = np.abs(r_pts_s - r_res) < 0.2 * stell.r0
        pts_near = pts[mask] if mask.sum() >= 8 else pts
        chain = extract_island_width(
            pts_near[:, :2], R_ax, Z_ax,
            mode_m=2,
            psi_func=lambda R, Z: float(stell.psi_ax(np.array([R]), np.array([Z]))),
        )
        chains.append(chain)
        print(f"  O-points: {len(chain.O_points)}, half_width_r = {chain.half_width_r*100:.2f} cm")
    else:
        chains.append(None)

# ---------------------------------------------------------------------------
# 4. Plot 2x2 panel
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

r_plot = 1.3 * stell.r0

for idx in range(n_phi_sections):
    ax = axes[idx]
    phi_deg = np.degrees(phi_sections[idx])
    pts = pmap.crossing_array(idx)

    # Background: flux circles
    for psi_c in np.linspace(0.1, 0.9, 8):
        r_c = np.sqrt(psi_c) * stell.r0
        theta_c = np.linspace(0, 2*np.pi, 100)
        ax.plot(R_ax + r_c*np.cos(theta_c), r_c*np.sin(theta_c),
                '-', color='lightgray', lw=0.5)
    # LCFS
    theta_c = np.linspace(0, 2*np.pi, 100)
    ax.plot(R_ax + stell.r0*np.cos(theta_c), stell.r0*np.sin(theta_c), 'k-', lw=1.2)

    # Poincare scatter
    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.8, c='steelblue', alpha=0.6, rasterized=True)

    # O/X points and island width arrows
    chain = chains[idx]
    if chain is not None and len(chain.O_points) > 0:
        ax.scatter(chain.O_points[:, 0], chain.O_points[:, 1],
                   s=50, c='red', marker='o', zorder=5, label='O-pt')
        ax.scatter(chain.X_points[:, 0], chain.X_points[:, 1],
                   s=50, c='blue', marker='x', zorder=5, lw=2, label='X-pt')
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

    ax.plot(R_ax, Z_ax, '+k', ms=8, mew=2)
    ax.set_aspect('equal')
    ax.set_xlim(R_ax - r_plot, R_ax + r_plot)
    ax.set_ylim(-r_plot, r_plot)
    ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
    ax.set_title(f"phi={phi_deg:.0f} deg")
    if chain is not None and len(chain.O_points) > 0:
        ax.legend(fontsize=7, loc='upper right')

fig.suptitle("Simple stellarator q=2/1 island chain — 4 toroidal sections", fontsize=12)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), 'stellarator_multisection.png')
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
