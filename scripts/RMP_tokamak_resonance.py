"""
RMP Tokamak Resonance — Multi-Section Verification (Solov'ev Equilibrium)
==========================================================================
6-panel Poincaré figure at φ = 0°, 60°, 120°, 180°, 240°, 300°

Physics:
  Axisymmetric Solov'ev equilibrium + helical (m=4,n=1) RMP perturbation.
  Island O/X points rotate: θ_O(φ) = [nφ - π/2 - arg(b_{m,-n})] / m + 2πk/m
"""
import sys
sys.path.insert(0, r'D:\Repo\pyna')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.integrate import solve_ivp

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

from pyna.MCF.equilibrium.Solovev import EquilibriumSolovev
from pyna.MCF.visual.RMP_spectrum import island_fixed_points

# ── 1. Equilibrium ────────────────────────────────────────────────────────
print("[1] Building Solov'ev equilibrium (EAST-like)...", flush=True)
eq = EquilibriumSolovev(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)
R0_eq = eq.R0
base_m, base_n = 4, 1
B_rmp = 5e-4   # 0.5 mT
print(f"  R0={eq.R0} m, a={eq.a} m, B0={eq.B0} T, kappa={eq.kappa}", flush=True)

# ── 2. RMP perturbation ───────────────────────────────────────────────────
def delta_B_RMP(R, Z, phi, m=base_m, n=base_n, B_amp=B_rmp):
    """Helical RMP: δBψ ~ B_amp * cos(mθ_pol - nφ)"""
    theta_pol = np.arctan2(Z, R - R0_eq)
    phase = m * theta_pol - n * phi
    dBR = B_amp * np.cos(phase) * np.cos(theta_pol)
    dBZ = B_amp * np.cos(phase) * np.sin(theta_pol)
    return dBR, dBZ

# ── 3. Find resonant surface ──────────────────────────────────────────────
print(f"[2] Finding q={base_m}/{base_n} resonant surface...", flush=True)
psi_res_list = eq.resonant_psi(base_m, base_n)
if not psi_res_list:
    raise RuntimeError(f"No q={base_m}/{base_n} resonant surface found!")
psi_res = psi_res_list[0]
print(f"  psi_res = {psi_res:.4f}", flush=True)

# ── 4. Compute b_mn on resonant surface ──────────────────────────────────
print("[3] Computing b_mn on resonant surface...", flush=True)
R_surf, Z_surf = eq.flux_surface(psi_res, n_pts=256)
theta_arr = np.arctan2(Z_surf, R_surf - R0_eq)
sort_idx = np.argsort(theta_arr)
R_sorted = R_surf[sort_idx]
Z_sorted = Z_surf[sort_idx]
theta_sorted = theta_arr[sort_idx]

dBpsi_1d = np.array([
    delta_B_RMP(R, Z, 0.0)[0] * np.cos(th) + delta_B_RMP(R, Z, 0.0)[1] * np.sin(th)
    for R, Z, th in zip(R_sorted, Z_sorted, theta_sorted)
])
b_fft = np.fft.fft(dBpsi_1d) / len(dBpsi_1d)
m_freq = np.fft.fftfreq(len(dBpsi_1d), 1.0 / len(dBpsi_1d)).astype(int)
m_idx = np.where(m_freq == base_m)[0]
b_mn = complex(b_fft[m_idx[0]]) if len(m_idx) > 0 else complex(B_rmp * 0.5)
print(f"  |b_mn| = {abs(b_mn):.3e}, arg = {np.degrees(np.angle(b_mn)):.1f} deg", flush=True)

r_res_approx = eq.a * float(np.sqrt(psi_res))
half_width_r  = 0.03 * eq.a

# ── 5. Field line RHS (φ as independent variable) ────────────────────────
def rhs_with_RMP(phi, state):
    R, Z = state
    BR_eq, BZ_eq = eq.BR_BZ(R, Z)
    Bph = eq.Bphi(R)
    dBR, dBZ = delta_B_RMP(R, Z, phi)
    return [R * (BR_eq + dBR) / Bph, R * (BZ_eq + dBZ) / Bph]

# ── 6. Trace Poincaré — one integration per field line ────────────────────
#  Integrate each field line from phi=0 to phi=n_turns*2π.
#  Crossings at each section (phi_s + k*2π) are found via dense output.
phi_sections = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
R_starts = eq.R0 + np.linspace(0.05, 0.88, 20) * eq.a
Z_starts = np.zeros_like(R_starts)
n_turns = 100

print(f"[4] Tracing {len(R_starts)} field lines × {n_turns} turns...", flush=True)

# For each section, we collect crossings per field line
all_poincare = {ph: ([], []) for ph in phi_sections}

for li, (R0s, Z0s) in enumerate(zip(R_starts, Z_starts)):
    # For each section, we need crossings at phi = phi_s + k*2π, k=0..n_turns-1
    # We trace turn-by-turn, collecting at all 6 sections simultaneously per turn
    # Each 2π step is split at the 6 section angles

    # Sort section angles within [0, 2π)
    phi_mod = phi_sections % (2 * np.pi)
    sort_order = np.argsort(phi_mod)
    phi_sorted = phi_mod[sort_order]

    state = [R0s, Z0s]
    # per-section accumulation
    sec_R = {ph: [] for ph in phi_sections}
    sec_Z = {ph: [] for ph in phi_sections}

    for turn in range(n_turns):
        phi_base = turn * 2 * np.pi
        # Integrate each 2π turn in sub-intervals between section angles
        # Subsegments: [phi_base, phi_base+phi_sorted[0]], ... , [phi_base+phi_sorted[-1], phi_base+2π]
        breakpoints = list(phi_base + np.append(phi_sorted, 2 * np.pi))
        phi_cur = phi_base
        ok = True
        for ib, phi_next in enumerate(breakpoints):
            if phi_next <= phi_cur + 1e-12:
                continue
            sol = solve_ivp(rhs_with_RMP, [phi_cur, phi_next], state,
                            method='RK45', rtol=1e-7, atol=1e-9,
                            max_step=0.1)
            if not sol.success or not np.isfinite(sol.y[:, -1]).all():
                ok = False
                break
            state = [sol.y[0, -1], sol.y[1, -1]]
            phi_cur = phi_next
            # Record crossing at the section that ends this sub-interval
            if ib < len(phi_sorted):
                ph_key = phi_sections[sort_order[ib]]
                sec_R[ph_key].append(state[0])
                sec_Z[ph_key].append(state[1])
        if not ok:
            break

    for ph in phi_sections:
        all_poincare[ph][0].append(np.array(sec_R[ph]))
        all_poincare[ph][1].append(np.array(sec_Z[ph]))

    if (li + 1) % 5 == 0 or li == 0:
        total = sum(len(v) for v in sec_R.values())
        print(f"  Line {li+1}/{len(R_starts)}: {total} total crossings", flush=True)

# ── 7. ψ_norm colormap ────────────────────────────────────────────────────
psi_axis = float(eq.psi(eq.R0, 0.0))
psi_bdy  = float(eq.psi(eq.R0 + eq.a, 0.0))
psi_norm_starts = np.clip(
    np.array([(float(eq.psi(R, 0.0)) - psi_axis) / (psi_bdy - psi_axis)
               for R in R_starts]), 0, 1
)

# ── 8. Plot ───────────────────────────────────────────────────────────────
print("[5] Plotting...", flush=True)
fig = plt.figure(figsize=(16, 11))
gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)
cmap_p = plt.colormaps.get_cmap('viridis')

try:
    R_fc, Z_fc = eq.flux_surface(psi_res, n_pts=300)
except Exception:
    R_fc, Z_fc = None, None

for idx, phi_s in enumerate(phi_sections):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])
    pts_R, pts_Z = all_poincare[phi_s]

    for i, (Rpts, Zpts) in enumerate(zip(pts_R, pts_Z)):
        if len(Rpts) > 0:
            c = cmap_p(psi_norm_starts[i])
            ax.scatter(Rpts, Zpts, s=0.6, color=c, rasterized=True, alpha=0.7, zorder=2)

    if R_fc is not None:
        ax.plot(R_fc, Z_fc, '--', color='#E53935', lw=0.7, alpha=0.5, zorder=3)

    pts_fp = island_fixed_points(base_m, base_n, b_mn, phi_s, q_prime_sign=1)
    theta_O_arr = pts_fp['theta_O'][0]
    theta_X_arr = pts_fp['theta_X'][0]

    for theta_op in theta_O_arr:
        R_O = R0_eq + r_res_approx * np.cos(theta_op)
        Z_O = r_res_approx * eq.kappa * np.sin(theta_op)
        r_in  = max(0.01, r_res_approx - half_width_r)
        r_out = r_res_approx + half_width_r
        ax.plot(
            [R0_eq + r_in * np.cos(theta_op), R0_eq + r_out * np.cos(theta_op)],
            [eq.kappa * r_in * np.sin(theta_op), eq.kappa * r_out * np.sin(theta_op)],
            '-', color='#1E88E5', lw=3.0, alpha=0.85, solid_capstyle='round', zorder=5,
        )
        ax.plot(R_O, Z_O, 'o', color='#1E88E5', ms=5, zorder=6)

    for theta_xp in theta_X_arr:
        R_X = R0_eq + r_res_approx * np.cos(theta_xp)
        Z_X = r_res_approx * eq.kappa * np.sin(theta_xp)
        ax.plot(R_X, Z_X, 'x', color='#1E88E5', ms=6, markeredgewidth=1.5,
                zorder=6, alpha=0.8)

    R_lo = R0_eq - 1.15 * eq.a
    R_hi = R0_eq + 1.15 * eq.a
    Z_lm = 1.15 * eq.a * eq.kappa
    ax.set_xlim(R_lo, R_hi)
    ax.set_ylim(-Z_lm, Z_lm)
    ax.set_aspect('equal')
    ax.set_title(f'$\\varphi = {int(round(np.degrees(phi_s)))}\\degree$', fontsize=11, pad=4)
    ax.set_xlabel('$R$ (m)', fontsize=9)
    if col == 0:
        ax.set_ylabel('$Z$ (m)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.4)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label(r'$\psi_\mathrm{norm}$', fontsize=11)
cb.ax.tick_params(labelsize=9)

legend_handles = [
    plt.Line2D([0], [0], linestyle='--', color='#E53935', lw=1.0,
               label=f'$q={base_m}/{base_n}$ resonant surface ($\\psi_{{\\rm res}}={psi_res:.3f}$)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5',
               ms=6, lw=0, label='O-point (analytic)'),
    plt.Line2D([0], [0], marker='x', color='#1E88E5', ms=6,
               markeredgewidth=1.5, lw=0, label='X-point (analytic)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.46, -0.02))

fig.suptitle(
    "Tokamak RMP Resonance \u2014 Multi-Section Verification (Solov'ev)\n"
    f"$R_0={eq.R0}$ m,  $a={eq.a}$ m,  $B_0={eq.B0}$ T,  $\\kappa={eq.kappa}$,  "
    f"mode $({base_m},{base_n})$,  $\\delta B/B_0={B_rmp/eq.B0*100:.3f}\\%$,  "
    f"$q_{{4/1}}$ surface at $\\psi_{{\\rm norm}}={psi_res:.3f}$",
    fontsize=12, y=1.02,
)

print("\nO-point positions at each section:", flush=True)
for phi_s in phi_sections:
    pts_fp = island_fixed_points(base_m, base_n, b_mn, phi_s, 1)
    print(f"  phi={np.degrees(phi_s):.0f}deg: theta_O = {np.degrees(pts_fp['theta_O'][0]).round(1)}", flush=True)

out_path = r'D:\Repo\pyna\scripts\RMP_tokamak_resonance.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_path}", flush=True)
plt.close()
print("[Done]", flush=True)
