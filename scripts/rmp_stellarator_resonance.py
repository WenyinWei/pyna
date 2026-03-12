"""RMP Resonance Analysis: Stellarator 3D Magnetic Topology
==========================================================

Publication-quality figure showing:
  1. Poincare cross-section of nested flux surfaces (colored by psi_norm)
  2. Multiple resonant components at different rational surfaces
  3. Island width bars at O-point positions for each resonant surface
  4. Chirikov overlap parameter (sigma > 1 -> chaos)
  5. RMP Fourier spectrum bar chart with island half-width overlay

Physics
-------
For an RMP with multiple Fourier components b_{mn}, each component
resonates at the surface where q(psi_res) = m/n.

The island half-width (Rutherford formula in normalized psi):
    w_psi = 4 * sqrt(|b_mn| / (m * |dq/dpsi|))

The O-point poloidal angle at phi=0:
    theta_O = arg(b_mn) / m  [mod 2pi/m]

Chirikov overlap criterion:
    sigma = (w1 + w2) / |psi2 - psi1| > 1  -->  overlapping islands (chaos)
"""
import sys
sys.path.insert(0, r'D:\Repo\pyna')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Academic matplotlib style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'text.usetex': False,
    'axes.linewidth': 0.8,
})

from pyna.MCF.equilibrium.stellarator import SimpleStellarartor, simple_stellarator
from pyna.MCF.visual.equilibrium import plot_nested_flux_surfaces, ISLAND_CMAPS
from pyna.MCF.visual.RMP_spectrum import (
    find_resonant_components_analytic,
    plot_island_width_bars,
    ResonantComponent,
)

# ── 1. Build stellarator equilibrium ──────────────────────────────────────
print("=" * 60)
print("RMP Resonance Analysis -- Stellarator 3D Topology")
print("=" * 60)
print("\n[1] Building stellarator equilibrium...")

eq = simple_stellarator(
    R0=3.0,
    r0=0.3,
    B0=2.5,
    q0=1.5,
    q1=4.5,
    m_h=3,
    n_h=3,
    epsilon_h=0.03,
)

q_at_01 = float(eq.q_of_psi(0.0))
q_at_1  = float(eq.q_of_psi(1.0))
print(f"  Stellarator: R0={eq.R0} m, r0={eq.r0} m, B0={eq.B0} T")
print(f"  q profile: q(psi=0)={q_at_01:.2f}, q(psi=1)={q_at_1:.2f}")

# Show accessible rational surfaces
print("  Accessible rational surfaces:")
for m_test, n_test in [(2, 1), (3, 2), (3, 1), (4, 1), (5, 2), (7, 2)]:
    psi_list = eq.resonant_psi(m_test, n_test)
    if psi_list:
        print(f"    q={m_test}/{n_test}={m_test/n_test:.3f} -> psi_res={psi_list[0]:.3f}")

# ── 2. Define multi-mode RMP ──────────────────────────────────────────────
print("\n[2] Defining multi-mode RMP field...")

B_rmp = 5e-3
R0_eq = eq.R0
r0_eq = eq.r0

# Modes: (m,n) pairs at different q surfaces for non-trivial Chirikov analysis
# q=2/1=2.00  -> psi=0.167  (m=2,n=1)
# q=3/2=1.50  -> psi=0.000 (edge, skip) ... try (5/2=2.5 -> psi=0.333)
# q=3/1=3.00  -> psi=0.500
# Amplitudes designed to show clear Chirikov analysis
rmp_modes = [
    (2, 1, 1.00),    # (m, n, relative_amplitude)
    (5, 2, 0.60),    # q=2.5, psi~0.333
    (3, 1, 0.35),    # q=3.0, psi=0.500
]

print("  RMP mode spectrum:")
for m_r, n_r, amp_r in rmp_modes:
    psi_list = eq.resonant_psi(m_r, n_r)
    psi_str = f"{psi_list[0]:.3f}" if psi_list else "outside"
    print(f"    ({m_r},{n_r}): q={m_r/n_r:.3f}  psi_res={psi_str}  |b|={amp_r*B_rmp*1e3:.2f} mT")

def delta_B_rmp(R, Z, phi):
    """Multi-mode RMP spanning several rational surfaces."""
    theta_pol = np.arctan2(Z, R - R0_eq)
    dBR = 0.0
    dBZ = 0.0
    for (m_r, n_r, amp_r) in rmp_modes:
        phase = m_r * theta_pol - n_r * phi
        dBR += B_rmp * amp_r * np.cos(phase) * np.cos(theta_pol)
        dBZ += B_rmp * amp_r * np.cos(phase) * np.sin(theta_pol)
    return np.array([dBR, dBZ, 0.0])

# ── 3. Find resonant components ───────────────────────────────────────────
print("\n[3] Computing resonant components...")

# Use a custom multi-mode approach: find each mode independently
def find_all_components(eq, rmp_modes, B_rmp, n_theta=128, n_phi=64):
    """Find resonant components for each (m,n) in rmp_modes."""
    components = []
    dq_dpsi = eq.q1 - eq.q0   # constant slope for linear q profile

    for idx, (m_k, n_k, amp_k) in enumerate(rmp_modes):
        psi_list = eq.resonant_psi(m_k, n_k)
        if not psi_list:
            print(f"  ({m_k},{n_k}): no resonant surface, skipping")
            continue

        psi_res = float(psi_list[0])
        r_res   = np.sqrt(psi_res) * eq.r0
        q_res   = float(eq.q_of_psi(psi_res))

        # Sample single-mode RMP on this surface
        theta_arr = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        phi_arr   = np.linspace(0, 2*np.pi, n_phi,   endpoint=False)

        R_surf = eq.R0 + r_res * np.cos(theta_arr)
        Z_surf =         r_res * np.sin(theta_arr)

        # Build 2D field array (using only this mode's component)
        dBpsi = np.zeros((n_theta, n_phi), dtype=complex)
        for j, phi in enumerate(phi_arr):
            for i in range(n_theta):
                phase = m_k * theta_arr[i] - n_k * phi
                val   = B_rmp * amp_k * np.cos(phase)
                # radial projection
                dBpsi[i, j] = val  # simplified: already in radial direction

        b_fft = np.fft.fft2(dBpsi) / (n_theta * n_phi)
        m_freq = np.fft.fftfreq(n_theta, 1/n_theta).astype(int)
        n_freq = np.fft.fftfreq(n_phi,   1/n_phi).astype(int)

        m_idx_arr = np.where(m_freq == m_k)[0]
        n_idx_arr = np.where(n_freq == -n_k)[0]

        if len(m_idx_arr) == 0 or len(n_idx_arr) == 0:
            # Try positive n
            n_idx_arr = np.where(n_freq == n_k)[0]
            if len(n_idx_arr) == 0:
                print(f"  ({m_k},{n_k}): FFT index not found, using analytic value")
                b_mn = B_rmp * amp_k / 2.0  # analytic: cos(phase)/2 in complex FFT
            else:
                b_mn = b_fft[m_idx_arr[0], n_idx_arr[0]]
        else:
            b_mn = b_fft[m_idx_arr[0], n_idx_arr[0]]

        # Ensure non-trivial amplitude
        if abs(b_mn) < 1e-10:
            b_mn = B_rmp * amp_k / 2.0  # use analytic value

        # Rutherford formula
        half_width_psi = 4.0 * np.sqrt(abs(b_mn) / (m_k * abs(dq_dpsi) + 1e-30))
        half_width_r   = half_width_psi * eq.r0 / (2.0 * np.sqrt(max(psi_res, 0.01)))

        # O-point phase
        phi_mn = np.angle(b_mn)
        opoint_theta = (phi_mn / m_k) % (2*np.pi / m_k)
        xpoint_theta = opoint_theta + np.pi / m_k

        comp = ResonantComponent(
            m=m_k, n=n_k,
            harmonic_order=idx+1,
            b_mn=complex(abs(b_mn), 0),  # use real amplitude
            psi_res=psi_res,
            q_res=q_res,
            half_width_psi=half_width_psi,
            half_width_r=half_width_r,
            opoint_theta=opoint_theta,
            xpoint_theta=xpoint_theta,
        )
        components.append(comp)
        print(f"  ({m_k},{n_k}): psi_res={psi_res:.3f}  q={q_res:.3f}  "
              f"|b_mn|={abs(b_mn):.3e}  "
              f"w_psi={half_width_psi:.4f}  ({half_width_r*100:.2f} cm)  "
              f"theta_O={np.degrees(opoint_theta):.1f} deg")

    return components

components = find_all_components(eq, rmp_modes, B_rmp)

if not components:
    print("  ERROR: No components found!")

# ── 4. Chirikov overlap analysis ──────────────────────────────────────────
print("\n[4] Chirikov overlap analysis:")
components_sorted = sorted(components, key=lambda c: c.psi_res)
for i, c in enumerate(components_sorted):
    if i < len(components_sorted) - 1:
        c_next = components_sorted[i+1]
        gap    = abs(c_next.psi_res - c.psi_res)
        sigma  = (c.half_width_psi + c_next.half_width_psi) / (gap + 1e-10)
        status = "OVERLAPPING -> chaos!" if sigma > 1.0 else "separated"
        print(f"  ({c.m},{c.n})<->({c_next.m},{c_next.n}): "
              f"gap={gap:.3f}  sigma_Chirikov = {sigma:.3f}  [{status}]")

# ── 5. Publication figure ─────────────────────────────────────────────────
print("\n[5] Generating publication figure...")

fig = plt.figure(figsize=(14, 6.5))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.35], wspace=0.4)

ax_eq   = fig.add_subplot(gs[0])
ax_spec = fig.add_subplot(gs[1])

# Left panel: Poincare section + island bars
print("    Tracing Poincare map (n_turns=150)...")
fig_tmp, ax_eq = plot_nested_flux_surfaces(
    eq, ax=ax_eq,
    n_fieldlines=24,
    n_turns=150,
    cmap='plasma',
    show_colorbar=True,
)

# Highlight resonant surfaces (approximate circles)
mode_colors = ISLAND_CMAPS
for ci, comp in enumerate(components_sorted):
    r_s = np.sqrt(comp.psi_res) * eq.r0
    theta_c = np.linspace(0, 2*np.pi, 300)
    R_s = eq.R0 + r_s * np.cos(theta_c)
    Z_s =         r_s * np.sin(theta_c)
    color = mode_colors[ci % len(mode_colors)]
    ax_eq.plot(R_s, Z_s, '--', color=color, linewidth=1.0, alpha=0.75, zorder=4)

# Island width bars (use sorted list with harmonic_order = sorted index)
for ci, comp in enumerate(components_sorted):
    comp.harmonic_order = ci + 1  # reassign for color consistency
plot_island_width_bars(ax_eq, components_sorted, eq, label_harmonics=True)

# Legend
legend_patches = []
for ci, c in enumerate(components_sorted):
    col = mode_colors[ci % len(mode_colors)]
    legend_patches.append(
        mpatches.Patch(color=col,
                       label=f'$({c.m},{c.n})$  $q={c.q_res:.2f}$  '
                             f'$\\psi={c.psi_res:.2f}$')
    )
if legend_patches:
    ax_eq.legend(handles=legend_patches, loc='upper right', fontsize=8,
                 framealpha=0.88, edgecolor='none', title='Mode $(m,n)$',
                 title_fontsize=8)

ax_eq.set_title('Poincare Cross-Section\n& RMP Island Widths', fontsize=12, pad=8)
ax_eq.spines['top'].set_visible(True)
ax_eq.spines['right'].set_visible(True)

# Right panel: spectrum bar chart
if components_sorted:
    x_pos      = np.arange(len(components_sorted))
    bar_colors = [mode_colors[ci % len(mode_colors)] for ci in range(len(components_sorted))]
    bar_heights = [abs(c.b_mn) for c in components_sorted]

    bars = ax_spec.bar(x_pos, bar_heights, color=bar_colors, alpha=0.82,
                       edgecolor='white', linewidth=0.6, zorder=3)

    ax_spec.set_xticks(x_pos)
    ax_spec.set_xticklabels(
        [f'$({c.m},{c.n})$\n$q={c.q_res:.2f}$\n$\\psi={c.psi_res:.2f}$'
         for c in components_sorted],
        fontsize=9,
    )
    ax_spec.set_ylabel(r'$|b_{mn}|\ (\mathrm{T})$', fontsize=12)
    ax_spec.set_yscale('log')
    ax_spec.set_title('RMP Fourier Spectrum\nat Resonant Surfaces', fontsize=12, pad=8)
    ax_spec.grid(axis='y', alpha=0.3, linewidth=0.5, zorder=0)

    # Annotate O-point phase on each bar
    for i, comp in enumerate(components_sorted):
        ax_spec.annotate(
            f'$\\theta_O={np.degrees(comp.opoint_theta):.0f}°$',
            xy=(i, bar_heights[i]),
            xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=9, color='#111111',
        )

    # Secondary y-axis: island half-width
    ax2 = ax_spec.twinx()
    ax2.plot(x_pos, [c.half_width_r * 100 for c in components_sorted],
             'D--', color='#E91E63', markersize=9, linewidth=2.0,
             label=r'$w_{1/2}\ (\mathrm{cm})$', zorder=5)
    ax2.set_ylabel(r'Island half-width $w_{1/2}\ (\mathrm{cm})$',
                   fontsize=11, color='#E91E63')
    ax2.tick_params(axis='y', labelcolor='#E91E63', labelsize=9)
    ax2.legend(loc='lower right', fontsize=10)

    # Chirikov sigma annotations between bars
    for i in range(len(components_sorted) - 1):
        c1, c2 = components_sorted[i], components_sorted[i+1]
        gap    = abs(c2.psi_res - c1.psi_res)
        sigma  = (c1.half_width_psi + c2.half_width_psi) / (gap + 1e-10)
        x_mid  = (i + i + 1) / 2.0
        y_ann  = min(bar_heights) * 0.5
        ann_color = '#C62828' if sigma > 1 else '#2E7D32'
        ax_spec.annotate(
            f'$\\sigma={sigma:.2f}$',
            xy=(x_mid, y_ann), ha='center', fontsize=9,
            color=ann_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=ann_color, alpha=0.85),
        )

# Suptitle
fig.suptitle(
    f'Stellarator RMP Resonance Analysis  |  '
    f'$N_\\mathrm{{fp}}=3$  '
    f'$R_0={eq.R0}$ m, $a={eq.r0}$ m, $B_0={eq.B0}$ T\n'
    f'$\\delta B_\\mathrm{{max}}/B_0 = {B_rmp/eq.B0*100:.2f}\\%$  '
    f'$q$ range: {q_at_01:.2f}--{q_at_1:.2f}  '
    f'($q_0+({eq.q1}-{eq.q0})\\psi$ linear profile)',
    fontsize=11.5, y=1.02,
)

out_path = r'D:\Repo\pyna\scripts\rmp_stellarator_resonance.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n  Saved -> {out_path}")
plt.close()
print("\n[Done]")
