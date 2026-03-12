"""
RMP Stellarator Resonance — Multi-Section Verification
=======================================================
6-panel figure at φ = 0°, 60°, 120°, 180°, 240°, 300°
Each panel: Poincaré scatter + island O/X markers at predicted poloidal angles.
"""
import sys
sys.path.insert(0, r'D:\Repo\pyna')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'figure.dpi': 150,
    'text.usetex': False,
    'axes.linewidth': 0.7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

from pyna.MCF.equilibrium.stellarator import SimpleStellarartor, simple_stellarator
from pyna.MCF.visual.RMP_spectrum import (
    find_resonant_components_analytic, island_fixed_points, ISLAND_CMAPS
)
from pyna.topo.poincare import poincare_from_fieldlines, ToroidalSection, PoincareMap

# ── 1. Build equilibrium ──────────────────────────────────────────────────
print("[1] Building stellarator equilibrium...")
eq = simple_stellarator(
    R0=3.0, r0=0.3, B0=2.5,
    q0=1.5, q1=4.5,
    m_h=3, n_h=3, epsilon_h=0.03,
)
print(f"  eq: R0={eq.R0}, r0={eq.r0}, B0={eq.B0}, q=[{eq.q0},{eq.q1}]")

# ── 2. RMP perturbation ────────────────────────────────────────────────────
base_m, base_n = 2, 1
B_rmp = 1e-3   # 1 mT
# Rename: use RMP (capital) as per naming convention
R0_eq = eq.R0
r0_eq = eq.r0

def delta_B_RMP(R, Z, phi, m=base_m, n=base_n, B_amp=B_rmp):
    theta_pol = np.arctan2(Z, R - R0_eq)
    phase = m * theta_pol - n * phi
    dBR = B_amp * np.cos(phase) * np.cos(theta_pol)
    dBZ = B_amp * np.cos(phase) * np.sin(theta_pol)
    return np.array([dBR, dBZ, 0.0])

# ── 3. Find resonant components ────────────────────────────────────────────
print("[2] Finding resonant components...")
components = find_resonant_components_analytic(
    eq, delta_B_RMP, base_m=base_m, base_n=base_n,
    max_harmonic=3, n_theta=64, n_phi=32,
)
print(f"  Found {len(components)} resonant components")

# ── 4. Build perturbed field_func (arc-length parameterized) ─────────────
# eq.field_func(rzphi) returns (dR/ds, dZ/ds, dphi/ds) unit tangent
# We need to add the RMP perturbation to BR and BZ before normalizing

def field_func_with_rmp(rzphi_1d):
    """field_func = unit tangent dRZphi/ds, with RMP added to BR, BZ."""
    rzphi_1d = np.asarray(rzphi_1d, dtype=float)
    R, Z, phi = rzphi_1d[0], rzphi_1d[1], rzphi_1d[2]

    theta = np.arctan2(Z, R - R0_eq)
    psi   = eq.psi_ax(R, Z)
    q     = float(eq.q_of_psi(psi))

    r_minor = np.sqrt((R - R0_eq)**2 + Z**2)

    # Equilibrium fields (same as eq.field_func)
    B_phi = eq.B0 * eq.R0 / R
    B_pol = B_phi * r_minor / (R * max(abs(q), 1e-3))

    if r_minor > 1e-10:
        BR0 = -B_pol * np.sin(theta)
        BZ0 =  B_pol * np.cos(theta)
    else:
        BR0 = 0.0
        BZ0 = 0.0

    # Helical ripple from equilibrium
    delta_BR_eq = eq.epsilon_h * eq.B0 * psi * np.cos(eq.m_h * theta - eq.n_h * phi)

    # RMP perturbation
    db = delta_B_RMP(R, Z, phi)

    BR_tot = BR0 + delta_BR_eq + db[0]
    BZ_tot = BZ0                + db[1]

    B_mag = np.sqrt(BR_tot**2 + BZ_tot**2 + B_phi**2) + 1e-30

    return np.array([BR_tot / B_mag, BZ_tot / B_mag, B_phi / (R * B_mag)])

# ── 5. Trace Poincaré maps at 6 sections ──────────────────────────────────
phi_sections = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0

# Start points: scan radially from near axis to LCFS
R_starts = np.linspace(eq.R0 + 0.04*eq.r0, eq.R0 + 0.92*eq.r0, 22)
Z_starts = np.zeros_like(R_starts)

# Start all lines at phi=0 section
start_pts = np.zeros((len(R_starts), 3))
start_pts[:, 0] = R_starts
start_pts[:, 1] = Z_starts
start_pts[:, 2] = 0.0

# Build sections
sections = [ToroidalSection(phi0=ph) for ph in phi_sections]

# Estimate t_max: one toroidal turn ≈ 2*pi*R0 (arc length ~ circumference)
# 150 turns × 2*pi*3 m/turn ≈ 2827 m
n_turns = 120
t_max = n_turns * 2 * np.pi * eq.R0
dt    = 0.06   # step size in arc length

print(f"[3] Tracing Poincaré maps: {len(R_starts)} field lines × {n_turns} turns...")
print(f"    t_max={t_max:.1f} m, dt={dt}, steps/line={int(t_max/dt)}")

pmap = poincare_from_fieldlines(
    field_func_with_rmp,
    start_pts,
    sections,
    t_max=t_max,
    dt=dt,
)
print("    Done.")

# Collect crossings per section
all_crossings = []  # list of (N,3) arrays, one per section
for i_sec in range(len(sections)):
    arr = pmap.crossing_array(i_sec)   # shape (N, 3): R, Z, phi
    all_crossings.append(arr)
    print(f"    Section φ={np.degrees(phi_sections[i_sec]):.0f}°: {len(arr)} crossings")

# ── 6. Build figure ────────────────────────────────────────────────────────
print("[4] Building figure...")
fig = plt.figure(figsize=(15, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

R_min = eq.R0 - 1.15 * eq.r0
R_max = eq.R0 + 1.15 * eq.r0
Z_lim = 1.15 * eq.r0

# ψ_norm colormap: each starting field line gets a color based on its R (proxy for psi)
cmap_poincare = cm.get_cmap('plasma')
psi_starts = ((R_starts - eq.R0)**2) / eq.r0**2   # normalized psi at start
psi_norm_starts = psi_starts / psi_starts.max()

# We need to know which crossing belongs to which starting field line.
# poincare_from_fieldlines traces each start_pt independently, so crossings
# are interleaved in the order they occur. We can't easily map back to start_pt
# from the flattened crossing_array without grouping by phi proximity.
# Instead, let's color by R value directly.
for idx, phi_s in enumerate(phi_sections):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])

    arr = all_crossings[idx]   # (N, 3)
    if len(arr) > 0:
        R_pts = arr[:, 0]
        Z_pts = arr[:, 1]
        # Color by R-position (proxy for ψ_norm)
        psi_pts = ((R_pts - eq.R0)**2 + Z_pts**2) / eq.r0**2
        psi_pts_norm = np.clip(psi_pts, 0, 1)
        colors_scatter = cmap_poincare(psi_pts_norm * 0.87 + 0.05)
        ax.scatter(R_pts, Z_pts, s=0.8, c=colors_scatter, rasterized=True, alpha=0.6, zorder=2)

    # Draw resonant surface circles and O/X markers
    for comp in components:
        color = ISLAND_CMAPS[(comp.harmonic_order - 1) % len(ISLAND_CMAPS)]
        r_res = np.sqrt(comp.psi_res) * eq.r0

        # Resonant surface circle (thin dashed)
        theta_circ = np.linspace(0, 2*np.pi, 200)
        ax.plot(eq.R0 + r_res * np.cos(theta_circ),
                r_res * np.sin(theta_circ),
                '--', color=color, linewidth=0.7, alpha=0.5, zorder=3)

        # O/X point angles at this φ section
        pts_fp = island_fixed_points(
            comp.m, comp.n, comp.b_mn, phi_s,
            getattr(comp, 'q_prime_sign', 1)
        )
        theta_O_arr = pts_fp['theta_O'][0]   # shape (m,)
        theta_X_arr = pts_fp['theta_X'][0]   # shape (m,)

        for theta_op in theta_O_arr:
            R_O = eq.R0 + r_res * np.cos(theta_op)
            Z_O =          r_res * np.sin(theta_op)
            # Island width bar
            r_in  = max(0.005, r_res - comp.half_width_r)
            r_out = r_res + comp.half_width_r
            ax.plot([eq.R0 + r_in  * np.cos(theta_op), eq.R0 + r_out * np.cos(theta_op)],
                    [         r_in  * np.sin(theta_op),           r_out * np.sin(theta_op)],
                    '-', color=color, linewidth=3.0, alpha=0.85,
                    solid_capstyle='round', zorder=5)
            ax.plot(R_O, Z_O, 'o', color=color, markersize=5, zorder=6)

        for theta_xp in theta_X_arr:
            R_X = eq.R0 + r_res * np.cos(theta_xp)
            Z_X =          r_res * np.sin(theta_xp)
            ax.plot(R_X, Z_X, 'x', color=color, markersize=6,
                    markeredgewidth=1.5, zorder=6, alpha=0.75)

    ax.set_xlim(R_min, R_max)
    ax.set_ylim(-Z_lim, Z_lim)
    ax.set_aspect('equal')
    ax.set_title(f'$\\varphi = {int(round(np.degrees(phi_s)))}°$', fontsize=11, pad=4)
    ax.set_xlabel('$R$ (m)', fontsize=9)
    if col == 0:
        ax.set_ylabel('$Z$ (m)', fontsize=9)
    # white background set in rcParams
    ax.set_facecolor('#0a0a12')   # dark background for Poincaré

# Colorbar for ψ_norm
sm = plt.cm.ScalarMappable(cmap='plasma', norm=Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label(r'$\psi_\mathrm{norm}$', fontsize=11)
cb.ax.tick_params(labelsize=9)

# Mode legend
legend_patches = [
    mpatches.Patch(
        color=ISLAND_CMAPS[(c.harmonic_order - 1) % len(ISLAND_CMAPS)],
        label=f'$({c.m},{c.n})$ q={c.q_res:.1f}, w={c.half_width_r*100:.1f} cm'
    )
    for c in components
]
legend_patches += [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='O-point (predicted)', linestyle='None'),
    plt.Line2D([0], [0], marker='x', color='gray', markersize=6,
               markeredgewidth=1.5, label='X-point (predicted)', linestyle='None'),
]
fig.legend(handles=legend_patches, loc='lower center',
           ncol=len(components) + 2,
           fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.46, -0.02))

fig.suptitle(
    f'Stellarator RMP Resonance Analysis — Multi-Section Verification\n'
    f'$R_0={eq.R0}$ m,  $a={eq.r0}$ m,  $B_0={eq.B0}$ T,  '
    f'$q \\in [{eq.q0},{eq.q1}]$,  '
    f'base mode $({base_m},{base_n})$,  '
    f'$\\delta B/B_0={B_rmp/eq.B0*100:.2f}\\%$',
    fontsize=12, y=1.02,
)

out_path = r'D:\Repo\pyna\scripts\RMP_stellarator_resonance.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_path}")
plt.close()
print("[Done]")
