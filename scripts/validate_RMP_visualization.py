"""Validate q=4/1 (or q=2/1 fallback) RMP island extraction on a Solov'ev equilibrium.

Generates: scripts/rmp_island_validation_v2.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyna.MCF.equilibrium.Solovev import SolovevEquilibrium, solovev_iter_like
from pyna.topo.poincare import PoincareMap, ToroidalSection
from pyna.flt import FieldLineTracer
from pyna.topo.island import locate_rational_surface, island_halfwidth
from pyna.topo.island_extract import extract_island_width

# ---------------------------------------------------------------------------
# 1. Build equilibrium — try to get a higher-q profile by increasing q0
#    so that q reaches 4 somewhere in the domain.
# ---------------------------------------------------------------------------
# Use q0=2.0; with Solov'ev the edge q is typically ~2-3x the axis value.
# We'll check empirically below.
eq = SolovevEquilibrium(
    R0=6.2 * 0.3,
    a=2.0 * 0.3,
    B0=5.3,
    kappa=1.7,
    delta=0.33,
    q0=3.0,   # q0=3.0 gives q_max~4.75, so q=4/1 is reachable
)
print(f"R0={eq.R0:.3f} m  a={eq.a:.3f} m  κ={eq.kappa}  δ={eq.delta}")
R_ax, Z_ax = eq.magnetic_axis
print(f"Magnetic axis: R={R_ax:.4f} m  Z={Z_ax:.4f} m")

# ---------------------------------------------------------------------------
# 2. q profile and resonant surface
# ---------------------------------------------------------------------------
S_values = np.linspace(0.05, 0.97, 80)
psi_values = S_values**2
q_values = eq.q_profile(psi_values, n_theta=256)

# Debug print q range
for s_check in [0.1, 0.5, 0.9, 0.95]:
    idx = np.argmin(np.abs(S_values - s_check))
    print(f"  q(S={s_check}) = {q_values[idx]:.3f}")

q_max = float(np.nanmax(q_values))
q_min_v = float(np.nanmin(q_values))
print(f"q range: [{q_min_v:.3f}, {q_max:.3f}]")

# Decide mode
if q_max >= 3.9:
    m_mode, n_mode = 4, 1
    print("Using q=4/1 mode.")
else:
    m_mode, n_mode = 2, 1
    print(f"WARNING: q_max={q_max:.3f} < 4. Falling back to q=2/1 mode.")

S_res_list = locate_rational_surface(S_values, q_values, m=m_mode, n=n_mode)
if not S_res_list:
    # Last resort: use q=2/1
    m_mode, n_mode = 2, 1
    print(f"WARNING: q={m_mode}/{n_mode} surface not found. Trying q=2/1.")
    S_res_list = locate_rational_surface(S_values, q_values, m=2, n=1)
    if not S_res_list:
        raise RuntimeError("No resonant surface found — check q profile.")

S_res = S_res_list[0]
psi_res = S_res**2
print(f"q={m_mode}/{n_mode} resonant surface: S_res={S_res:.4f}  psi_res={psi_res:.4f}")
r_res = S_res * eq.a

# ---------------------------------------------------------------------------
# 3. RMP perturbation (analytic helical)
# ---------------------------------------------------------------------------
delta_b = 5e-3
m_rmp, n_rmp = m_mode, n_mode

def RMP_BR(R, Z, phi):
    psi_n = eq.psi(np.atleast_1d(R), np.atleast_1d(Z))
    envelope = psi_n * (1 - psi_n)
    return delta_b * eq.B0 * envelope * np.cos(m_rmp * np.arctan2(Z - Z_ax, R - R_ax) - n_rmp * phi)

def RMP_BZ(R, Z, phi):
    psi_n = eq.psi(np.atleast_1d(R), np.atleast_1d(Z))
    envelope = psi_n * (1 - psi_n)
    return delta_b * eq.B0 * envelope * np.sin(m_rmp * np.arctan2(Z - Z_ax, R - R_ax) - n_rmp * phi)

def field_func(rzphi):
    rzphi = np.asarray(rzphi)
    R, Z, phi = rzphi[0], rzphi[1], rzphi[2]
    BR0, BZ0 = eq.BR_BZ(np.array([R]), np.array([Z]))
    Bphi0 = eq.Bphi(np.array([R]))
    dBR = RMP_BR(R, Z, phi)
    dBZ = RMP_BZ(R, Z, phi)
    BR_t = float(BR0[0]) + float(np.squeeze(dBR))
    BZ_t = float(BZ0[0]) + float(np.squeeze(dBZ))
    Bphi_t = float(Bphi0[0])
    B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bphi_t**2) + 1e-30
    return np.array([BR_t/B_mag, BZ_t/B_mag, Bphi_t/(R*B_mag)])

# ---------------------------------------------------------------------------
# 4. Field-line tracing — start points bracketing resonant surface
# ---------------------------------------------------------------------------
n_lines = 24
delta_S = 0.08
S_arr = np.linspace(max(S_res - delta_S, 0.02), min(S_res + delta_S, 0.97), n_lines)
psi_start = S_arr**2
start_pts = np.column_stack([
    R_ax + S_arr * eq.a,
    np.zeros(n_lines),
    np.zeros(n_lines),
])

section = ToroidalSection(phi0=0.0)
tracer = FieldLineTracer(field_func, dt=0.1)

print(f"Tracing {n_lines} field lines near q={m_mode}/{n_mode} (t_max=500)…")
trajs = tracer.trace_many(start_pts, t_max=500.0)

pmap = PoincareMap([section])
for traj in trajs:
    pmap.record_trajectory(traj)

pts_all = pmap.crossing_array(0)
print(f"Total Poincaré crossings: {len(pts_all)}")

# ---------------------------------------------------------------------------
# 5. Filter crossings near resonant surface (±30% of r_res from axis)
# ---------------------------------------------------------------------------
if len(pts_all) > 0:
    r_pts = np.sqrt((pts_all[:, 0] - R_ax)**2 + pts_all[:, 1]**2)
    r_res_val = S_res * eq.a
    mask = np.abs(r_pts - r_res_val) < 0.30 * r_res_val
    pts_near = pts_all[mask] if mask.sum() >= 12 else pts_all
    print(f"Points near resonant surface: {mask.sum()} / {len(pts_all)}")
else:
    pts_near = pts_all

# ---------------------------------------------------------------------------
# 6. Island extraction
# ---------------------------------------------------------------------------
chain = None
if len(pts_near) >= 8:
    chain = extract_island_width(
        pts_near[:, :2], R_ax, Z_ax,
        mode_m=m_mode,
        psi_func=lambda R, Z: float(eq.psi(np.array([R]), np.array([Z]))),
    )
    print(f"O-points found: {len(chain.O_points)}")
    if not np.isnan(chain.half_width_r):
        print(f"Island half-width: w_r = {chain.half_width_r*100:.2f} cm")
    print(f"O-points: {chain.O_points}")
    print(f"X-points: {chain.X_points}")
else:
    print(f"Not enough near-resonance points ({len(pts_near)}) for island extraction.")

# Theoretical half-width
b_profile = delta_b * psi_values * (1 - psi_values)
w_theory = island_halfwidth(m=m_mode, n=n_mode, S_res=S_res, S=S_values,
                             q_profile=q_values, tilde_b_mn=b_profile)
print(f"Theoretical island half-width (Chirikov): w_S = {w_theory:.4f}")
print(f"  → w_r ≈ {w_theory * eq.a * 100:.2f} cm  (a={eq.a:.3f} m)")

# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
R_range = (eq.R0 - 1.4*eq.a, eq.R0 + 1.4*eq.a)
Z_range = (-1.4*eq.kappa*eq.a, 1.4*eq.kappa*eq.a)

R1d = np.linspace(*R_range, 300)
Z1d = np.linspace(*Z_range, 300)
Rg, Zg = np.meshgrid(R1d, Z1d)
psi_g = eq.psi(Rg, Zg)

fig, ax = plt.subplots(figsize=(7, 9))

# (a) Background flux surface contours
ax.contour(Rg, Zg, psi_g, levels=np.linspace(0.05, 0.95, 15),
           colors='lightgray', linewidths=0.5)
ax.contour(Rg, Zg, psi_g, levels=[1.0], colors='k', linewidths=1.5)

# (b) Resonant surface contour (blue dashed)
cs = ax.contour(Rg, Zg, psi_g, levels=[psi_res], colors='royalblue',
                linewidths=1.2, linestyles='--')
ax.clabel(cs, fmt=f'q={m_mode}/{n_mode}', fontsize=8)

# (c) Poincaré scatter
if len(pts_all) > 0:
    ax.scatter(pts_all[:, 0], pts_all[:, 1], s=0.8, c='lightblue',
               alpha=0.6, rasterized=True, label='Poincaré')

# (d-f) O/X points and width arrows
if chain is not None and len(chain.O_points) > 0:
    ax.scatter(chain.O_points[:, 0], chain.O_points[:, 1],
               s=60, c='red', marker='o', zorder=5, label='O-point')
    if len(chain.X_points) > 0:
        ax.scatter(chain.X_points[:, 0], chain.X_points[:, 1],
                   s=60, c='blue', marker='x', zorder=5, lw=2, label='X-point')

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
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5),
            )

ax.plot(R_ax, Z_ax, '+k', ms=10, mew=2)
ax.set_aspect('equal')
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
mode_label = f"q={m_mode}/{n_mode}"
ax.set_title(
    f"{mode_label} island chain — Solov'ev + ({m_rmp},{n_rmp}) RMP\n"
    f"delta_b={delta_b:.0e},  w_theory={w_theory*eq.a*100:.1f} cm"
)
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(R_range); ax.set_ylim(Z_range)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), 'rmp_island_validation_v2.png')
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
