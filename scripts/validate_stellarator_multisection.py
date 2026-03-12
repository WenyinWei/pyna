"""Multi-section Poincaré validation for a simple stellarator (q=4/1).

Generates: scripts/stellarator_multisection_v2.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyna.MCF.equilibrium.stellarator import simple_stellarator
from pyna.topo.poincare import PoincareMap, ToroidalSection
from pyna.flt import FieldLineTracer
from pyna.topo.island_extract import extract_island_width

# ---------------------------------------------------------------------------
# 1. Build stellarator: q=4/1 at psi=(4-1.5)/(5.0-1.5)=2.5/3.5?.714
#    with m_h=4, n_h=1 helical ripple
# ---------------------------------------------------------------------------
st = simple_stellarator(R0=3.0, r0=0.3, B0=1.0, q0=1.5, q1=5.0,
                         m_h=4, n_h=1, epsilon_h=0.04)
R_ax, Z_ax = st.magnetic_axis
print(f"R_ax={R_ax:.3f}  r0={st.r0:.3f}")
print(f"q range: [{st.q0:.2f}, {st.q1:.2f}]")

# Verify q=4/1 resonance
psi_list = st.resonant_psi(4, 1)
print(f"q=4/1 resonant psi: {psi_list}")
if not psi_list:
    raise RuntimeError("q=4/1 surface not found in [0,1]. Check q0,q1 parameters.")
psi_res = psi_list[0]
r_res = np.sqrt(psi_res) * st.r0
print(f"psi_res={psi_res:.4f}  r_res={r_res:.4f} m")

# ---------------------------------------------------------------------------
# 2. Start points near q=4/1 surface and trace
# ---------------------------------------------------------------------------
start_pts = st.start_points_near_resonance(4, 1, n_lines=24, delta_psi=0.06)
print(f"Tracing {len(start_pts)} field lines (t_max=600, dt=0.1)?)

n_phi_sections = 4
phi_sections = [k * np.pi / 2 for k in range(n_phi_sections)]
sections = [ToroidalSection(phi0=phi) for phi in phi_sections]

tracer = FieldLineTracer(st.field_func, dt=0.1)
trajs = tracer.trace_many(start_pts, t_max=600.0)

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
        # Filter to points within 25% of r0 from q=4/1 surface
        r_pts_s = np.sqrt((pts[:, 0] - R_ax)**2 + pts[:, 1]**2)
        mask = np.abs(r_pts_s - r_res) < 0.25 * st.r0
        pts_filt = pts[mask] if mask.sum() >= 8 else pts
        print(f"  Near-resonance: {mask.sum()} pts used for island extraction")
        chain = extract_island_width(
            pts_filt[:, :2], R_ax, Z_ax,
            mode_m=4,
            psi_func=lambda R, Z: float(st.psi_ax(np.array([R]), np.array([Z]))),
        )
        chains.append(chain)
        print(f"  O-points: {len(chain.O_points)},  half_width_r={chain.half_width_r*100:.2f} cm")
    else:
        chains.append(None)

# ---------------------------------------------------------------------------
# 4. 2x2 panel + 5th subplot for island width vs section
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 12))
# 2x2 for sections, 1 narrow at bottom for width comparison
gs = fig.add_gridspec(3, 2, height_ratios=[5, 5, 2], hspace=0.35, wspace=0.3)

r_plot = 1.3 * st.r0

# Build psi grid for contour backgrounds
R1d = np.linspace(R_ax - r_plot, R_ax + r_plot, 200)
Z1d = np.linspace(-r_plot, r_plot, 200)
Rg, Zg = np.meshgrid(R1d, Z1d)
psi_g = st.psi_ax(Rg, Zg)

for idx in range(n_phi_sections):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col])
    phi_deg = np.degrees(phi_sections[idx])
    pts = pmap.crossing_array(idx)

    # Background: psi contours (gray)
    ax.contour(Rg, Zg, psi_g, levels=np.linspace(0.1, 0.9, 8),
               colors='lightgray', linewidths=0.5)
    # LCFS
    ax.contour(Rg, Zg, psi_g, levels=[1.0], colors='k', linewidths=1.2)
    # q=4/1 resonant surface contour (blue dashed)
    cs = ax.contour(Rg, Zg, psi_g, levels=[psi_res], colors='royalblue',
                    linewidths=1.0, linestyles='--')
    ax.clabel(cs, fmt='q=4/1', fontsize=7)

    # Full Poincaré scatter
    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.8, c='lightblue',
                   alpha=0.7, rasterized=True, label='Poincaré')

    # O/X points and width arrows
    chain = chains[idx]
    if chain is not None and len(chain.O_points) > 0:
        ax.scatter(chain.O_points[:, 0], chain.O_points[:, 1],
                   s=50, c='red', marker='o', zorder=5, label='O-pt')
        if len(chain.X_points) > 0:
            ax.scatter(chain.X_points[:, 0], chain.X_points[:, 1],
                       s=50, c='blue', marker='x', zorder=5, lw=2, label='X-pt')
        w = chain.half_width_r if not np.isnan(chain.half_width_r) else 0.0
        if w > 0:
            for O_pt in chain.O_points:
                dr = O_pt[0] - R_ax
                dz = O_pt[1] - Z_ax
                dist = np.sqrt(dr**2 + dz**2) + 1e-30
                ur, uz = dr/dist, dz/dist
                ax.annotate('',
                    xy=(O_pt[0] + w*ur, O_pt[1] + w*uz),
                    xytext=(O_pt[0] - w*ur, O_pt[1] - w*uz),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.2),
                )

    ax.plot(R_ax, Z_ax, '+k', ms=8, mew=2)
    ax.set_aspect('equal')
    ax.set_xlim(R_ax - r_plot, R_ax + r_plot)
    ax.set_ylim(-r_plot, r_plot)
    ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
    ax.set_title(f"φ = {phi_deg:.0f}°")
    if chain is not None and len(chain.O_points) > 0:
        ax.legend(fontsize=7, loc='upper right', markerscale=3)

# 5th subplot: island half_width_r vs toroidal section
ax5 = fig.add_subplot(gs[2, :])
phi_degs = [np.degrees(p) for p in phi_sections]
hw_vals = []
for chain in chains:
    if chain is not None and not np.isnan(chain.half_width_r):
        hw_vals.append(chain.half_width_r * 100)  # cm
    else:
        hw_vals.append(np.nan)

ax5.bar(phi_degs, hw_vals, width=30, color='steelblue', alpha=0.8)
ax5.set_xlabel('Toroidal angle φ (deg)')
ax5.set_ylabel('Island half-width (cm)')
ax5.set_title('Island half-width vs toroidal section')
ax5.set_xticks(phi_degs)
ax5.set_xticklabels([f'{p:.0f}°' for p in phi_degs])
for x, y in zip(phi_degs, hw_vals):
    if not np.isnan(y):
        ax5.text(x, y + 0.02, f'{y:.2f} cm', ha='center', va='bottom', fontsize=8)

fig.suptitle(
    f"Simple stellarator q=4/1 island chain ?4 toroidal sections\n"
    f"R0={st.R0} m, r0={st.r0} m, m_h={st.m_h}, n_h={st.n_h}, ε_h={st.epsilon_h}",
    fontsize=11
)

out = os.path.join(os.path.dirname(__file__), 'stellarator_multisection_v2.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
