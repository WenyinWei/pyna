import json

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: add rcParams after existing imports
cell1_new = '''import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path

from pyna.MCF.equilibrium.stellarator import SimpleStellarartor, simple_stellarator
from pyna.MCF.coils.coil_system import StellaratorControlCoils, CoilSet, biot_savart_field
from pyna.MCF.control.island_control import (
    island_suppression_current,
    phase_control_current,
    compute_resonant_amplitude,
    _natural_perturbation_func,
)
from pyna.topo.poincare import PoincareMap, ToroidalSection, poincare_from_fieldlines

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.dpi': 150,
    'text.usetex': False,
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

print('pyna loaded successfully')
'''

# Cell 5: more start points, more transits
cell5_new = '''# Target island: q = 4/3 (exists in the plasma)
TARGET_M, TARGET_N = 4, 3

N_TRANSITS = 150

# Radial scan: covers whole plasma cross-section
R_starts = np.linspace(stella.R0 + 0.02*stella.r0, stella.R0 + 0.93*stella.r0, 24)
start_pts_radial = np.column_stack([R_starts, np.zeros(len(R_starts)), np.zeros(len(R_starts))])
# Near-resonance detail
start_pts_resonance = stella.start_points_near_resonance(TARGET_M, TARGET_N, n_lines=12, delta_psi=0.06)
start_pts = np.vstack([start_pts_radial, start_pts_resonance])

print(f\'Tracing {len(start_pts)} field lines near q={TARGET_M}/{TARGET_N}...\')

section = ToroidalSection(0.0)
t_max = N_TRANSITS * 2 * np.pi * stella.R0

pmap_natural = poincare_from_fieldlines(
    stella.field_func,
    start_pts,
    sections=[section],
    t_max=t_max,
    dt=0.04,
)
results_natural = pmap_natural.crossing_array(0)  # shape (N, 3): R, Z, phi
print(f\'Done. {len(results_natural)} crossings recorded.\')
'''

# Cell 6: full viz rewrite
cell6_new = '''fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor(\'white\')

# Color scatter by psi_norm
if len(results_natural) > 0:
    R_pts, Z_pts = results_natural[:, 0], results_natural[:, 1]
    psi_pts = ((R_pts - stella.R0)**2 + Z_pts**2) / stella.r0**2
    psi_norm = np.clip(psi_pts, 0, 1.0)
    cmap = cm.get_cmap(\'plasma\')
    colors = cmap(psi_norm * 0.85 + 0.05)
    ax.scatter(R_pts, Z_pts, s=0.8, c=colors, rasterized=True, alpha=0.7, zorder=2)

# Resonant surface circle
psi_res_target = stella.resonant_psi(TARGET_M, TARGET_N)[0]
r_res = np.sqrt(psi_res_target) * stella.r0
theta_circ = np.linspace(0, 2*np.pi, 300)
ax.plot(stella.R0 + r_res*np.cos(theta_circ), r_res*np.sin(theta_circ),
        \'--\', color=\'tomato\', lw=0.8, alpha=0.7, label=f\'q={TARGET_M}/{TARGET_N} surface\')

# O/X points at phi=0 plane
hw = 0.05 * stella.r0
for k in range(TARGET_M):
    theta_O = 2*np.pi*k/TARGET_M
    theta_X = 2*np.pi*k/TARGET_M + np.pi/TARGET_M
    R_O = stella.R0 + r_res * np.cos(theta_O)
    Z_O = r_res * np.sin(theta_O)
    R_X = stella.R0 + r_res * np.cos(theta_X)
    Z_X = r_res * np.sin(theta_X)
    ax.plot([stella.R0+(r_res-hw)*np.cos(theta_O), stella.R0+(r_res+hw)*np.cos(theta_O)],
            [(r_res-hw)*np.sin(theta_O), (r_res+hw)*np.sin(theta_O)],
            \'-\', color=\'tomato\', lw=3, alpha=0.85, solid_capstyle=\'round\', zorder=5)
    ax.plot(R_O, Z_O, \'o\', color=\'tomato\', ms=6, zorder=6, label=\'O-pt\' if k==0 else \'\')
    ax.plot(R_X, Z_X, \'x\', color=\'tomato\', ms=7, mew=1.8, zorder=6, label=\'X-pt\' if k==0 else \'\')

# LCFS
ax.plot(stella.R0 + stella.r0*np.cos(theta_circ), stella.r0*np.sin(theta_circ),
        \'k-\', lw=1.2, label=\'LCFS\')

# Colorbar
sm = plt.cm.ScalarMappable(cmap=\'plasma\', norm=Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r\'$\\psi_\\mathrm{norm}$\', fontsize=11)

ax.set_xlim(stella.R0 - 1.15*stella.r0, stella.R0 + 1.15*stella.r0)
ax.set_ylim(-1.15*stella.r0, 1.15*stella.r0)
ax.set_xlabel(\'$R$ (m)\', fontsize=11)
ax.set_ylabel(\'$Z$ (m)\', fontsize=11)
ax.set_title(f\'Natural Poincar\\u00e9 map \\u2014 $q={TARGET_M}/{TARGET_N}$ island chain\\n\'
             f\'(stellarator, $\\\\phi=0$)\', fontsize=12)
ax.set_aspect(\'equal\')
ax.legend(loc=\'upper right\', fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(\'natural_island_poincare.png\', dpi=150, bbox_inches=\'tight\')
plt.show()
'''

# Cell 11: improved suppression scan plot
cell11_new = '''# Build a helper that adds coil field at unit current
class _UnitCoilSet:
    def __init__(self, coils, I0):
        self.coils = [(pts, I / I0) for pts, I in coils]

unit_coils = _UnitCoilSet(control_coils.coils, I0_COIL)

def coil_field_func(rzphi_3d):
    R, Z, phi = float(rzphi_3d[0]), float(rzphi_3d[1]), float(rzphi_3d[2])
    R_arr = np.array([[R]]); Z_arr = np.array([[Z]]); phi_arr = np.array([[phi]])
    br = bz = bp = 0.0
    for pts, I in unit_coils.coils:
        _br, _bz, _bp = biot_savart_field(pts, I, R_arr, Z_arr, phi_arr)
        br += float(_br); bz += float(_bz); bp += float(_bp)
    return br, bz, bp

b_coil_unit = compute_resonant_amplitude(
    coil_field_func, psi_res_target, TARGET_M, TARGET_N, stella, n_theta=20, n_phi=20
)

I0_scan = np.linspace(0, 1500, 12)
b_total_scan = [abs(b_nat + b_coil_unit * I0) for I0 in I0_scan]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(I0_scan, np.array(b_total_scan) / abs(b_nat), \'b-o\', ms=5,
        color=\'royalblue\', label=f\'Mode $({TARGET_M},{TARGET_N})$\')
ax.axhline(1, color=\'gray\', ls=\':\', lw=0.8, label=\'Natural level\')
ax.axhline(0, color=\'tomato\', ls=\'--\', lw=0.8, alpha=0.6, label=\'Full suppression\')
ax.set_xlabel(\'Control coil current $I_0$ (A)\', fontsize=11)
ax.set_ylabel(r\'$|\\tilde{b}_{mn}| / |\\tilde{b}_{mn}^{\\rm nat}|$\', fontsize=11)
ax.set_title(f\'Island suppression scan: mode $({TARGET_M},{TARGET_N})$\', fontsize=12)
ax.grid(True, alpha=0.3, lw=0.5)
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(\'island_suppression_scan.png\', dpi=150, bbox_inches=\'tight\')
plt.show()
'''

# Cell 13: improved gourd problem plot
cell13_new = '''MONITOR_M, MONITOR_N = 4, 2
psi_res_monitor = stella.resonant_psi(MONITOR_M, MONITOR_N)

if psi_res_monitor:
    psi_res_mon = psi_res_monitor[0]
    b_nat_mon = compute_resonant_amplitude(
        nat_func, psi_res_mon, MONITOR_M, MONITOR_N, stella, n_theta=20, n_phi=20)
    b_coil_mon = compute_resonant_amplitude(
        coil_field_func, psi_res_mon, MONITOR_M, MONITOR_N, stella, n_theta=20, n_phi=20
    )
    b_target_scan = [abs(b_nat + b_coil_unit * I0) for I0 in I0_scan]
    b_monitor_scan = [abs(b_nat_mon + b_coil_mon * I0) for I0 in I0_scan]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(I0_scan, np.array(b_target_scan)/abs(b_nat), \'-o\', ms=5,
             color=\'royalblue\', label=f\'Target $({TARGET_M},{TARGET_N})$\')
    ax1.axhline(1, color=\'gray\', ls=\':\', lw=0.8)
    ax1.set_ylabel(r\'Norm. $|\\tilde{b}_{mn}|$\', fontsize=11)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, lw=0.5)
    ax1.set_title(\'Press-Down-Gourd: Island control side effects\', fontsize=12)

    ax2.plot(I0_scan, np.array(b_monitor_scan)/abs(b_nat_mon), \'-s\', ms=5,
             color=\'tomato\', label=f\'Monitor $({MONITOR_M},{MONITOR_N})$\')
    ax2.axhline(1, color=\'gray\', ls=\':\', lw=0.8)
    ax2.set_xlabel(\'Control coil current $I_0$ (A)\', fontsize=11)
    ax2.set_ylabel(r\'Norm. $|\\tilde{b}_{mn}|$\', fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, lw=0.5)

    plt.tight_layout()
    plt.savefig(\'gourd_problem.png\', dpi=150, bbox_inches=\'tight\')
    plt.show()
else:
    print(f\'Monitor surface q={MONITOR_M}/{MONITOR_N} not in plasma\')
'''

# Cell 15: phase control with better style
cell15_new = '''# Phase control: compute optimal currents for different desired phase shifts
# (Poincare tracing with phase-shifted coils would take too long for a notebook demo;
#  here we show the linear-model prediction of how the resonant amplitude responds)

phase_shifts = np.linspace(0, 2*np.pi, 17)
b_phase = []

for dphase in phase_shifts:
    cc_p = StellaratorControlCoils(
        R0=stella.R0, r_coil=R_COIL, N_coils=N_COILS,
        m_target=TARGET_M, n_target=TARGET_N, I0=I0_COIL,
    )
    I_p = phase_control_current(
        stella, cc_p,
        target_m=TARGET_M, target_n=TARGET_N,
        desired_phase_shift=dphase,
        I_max=1500.0, n_theta=20, n_phi=20,
    )
    b_p = compute_resonant_amplitude(
        coil_field_func, psi_res_target, TARGET_M, TARGET_N, stella,
        n_theta=16, n_phi=16
    )
    b_phase.append(abs(b_nat + b_p * float(np.mean(np.abs(I_p)))))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.degrees(phase_shifts), np.array(b_phase)/abs(b_nat), \'-o\', ms=5,
        color=\'seagreen\', label=f\'Mode $({TARGET_M},{TARGET_N})$\')
ax.axhline(1, color=\'gray\', ls=\':\', lw=0.8, label=\'Natural level\')
ax.set_xlabel(\'Desired phase shift (degrees)\', fontsize=11)
ax.set_ylabel(r\'$|\\tilde{b}_{mn}| / |\\tilde{b}_{mn}^{\\rm nat}|$ (linear model)\', fontsize=11)
ax.set_title(f\'Phase control: mode $({TARGET_M},{TARGET_N})$ response vs. phase shift\', fontsize=12)
ax.grid(True, alpha=0.3, lw=0.5)
ax.legend(fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(\'phase_control.png\', dpi=150, bbox_inches=\'tight\')
plt.show()
print(\'Phase control scan complete.\')
'''

# Apply changes
nb['cells'][1]['source'] = cell1_new
nb['cells'][5]['source'] = cell5_new
nb['cells'][6]['source'] = cell6_new
nb['cells'][11]['source'] = cell11_new
nb['cells'][13]['source'] = cell13_new
nb['cells'][15]['source'] = cell15_new

# Clear all outputs before re-execution
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook updated successfully.')
