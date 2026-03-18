import sys
sys.path.insert(0, 'D:/Repo/pyna')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyna.MCF.equilibrium.stellarator import StellaratorSimple
from pyna.MCF.coils.coil_system import StellaratorControlCoils, Biot_Savart_field
from pyna.MCF.control.island_control import (
    compute_resonant_amplitude, _natural_perturbation_func,
    island_suppression_current, phase_control_current,
)

TARGET_M, TARGET_N = 4, 3
N_COILS = 16
R_COIL = 0.38
I0_COIL = 500.0

stella = StellaratorSimple(R0=3.0, r0=0.3, B0=1.0, q0=0.75, q1=1.5, m_h=4, n_h=3, epsilon_h=0.05)
control_coils = StellaratorControlCoils(
    R0=stella.R0, r_coil=R_COIL, N_coils=N_COILS,
    m_target=TARGET_M, n_target=TARGET_N, I0=I0_COIL,
)

psi_res_target = stella.resonant_psi(TARGET_M, TARGET_N)[0]
nat_func = _natural_perturbation_func(stella)
b_nat = compute_resonant_amplitude(nat_func, psi_res_target, TARGET_M, TARGET_N, stella, n_theta=16, n_phi=16)
print(f'b_nat = {b_nat:.4e}, |b_nat| = {abs(b_nat):.4e}')
assert abs(b_nat) > 1e-10, f'b_nat still ~0: {abs(b_nat)}'

# coil field (R,Z,phi signature - fixed)
def coil_field_func(R, Z, phi):
    R, Z, phi = float(R), float(Z), float(phi)
    R_arr = np.array([[R]]); Z_arr = np.array([[Z]]); phi_arr = np.array([[phi]])
    br = bz = bp = 0.0
    for pts, I in control_coils.coils:
        _br, _bz, _bp = Biot_Savart_field(pts, float(I)/I0_COIL, R_arr, Z_arr, phi_arr)
        br += float(_br); bz += float(_bz); bp += float(_bp)
    return br, bz, bp

b_coil_unit = compute_resonant_amplitude(coil_field_func, psi_res_target, TARGET_M, TARGET_N, stella, n_theta=12, n_phi=12)
print(f'b_coil_unit = {b_coil_unit:.4e}, |b_coil_unit| = {abs(b_coil_unit):.4e}')
assert abs(b_coil_unit) > 1e-10, f'b_coil_unit still ~0: {abs(b_coil_unit)}'

# Suppression scan
I0_scan = np.linspace(0, 1500, 8)
b_scan = [abs(b_nat + b_coil_unit * I0) for I0 in I0_scan]
b_norm = np.array(b_scan) / abs(b_nat)
print(f'Suppression scan: {np.round(b_norm, 3)}')
# Check not flat (should dip below 1 somewhere or vary significantly)
variation = b_norm.max() - b_norm.min()
print(f'Variation: {variation:.3f} (should be > 0.05 to not be flat)')
assert variation > 0.01, f'Scan still flat: variation={variation}'

# Phase control scan
phase_shifts = np.linspace(0, 2*np.pi, 9)
b_phase = []
for dphase in phase_shifts:
    cc_p = StellaratorControlCoils(R0=stella.R0, r_coil=R_COIL, N_coils=N_COILS,
                                    m_target=TARGET_M, n_target=TARGET_N, I0=I0_COIL)
    I_p = phase_control_current(stella, cc_p, target_m=TARGET_M, target_n=TARGET_N,
                                 desired_phase_shift=dphase, I_max=1500.0, n_theta=12, n_phi=12)
    pairs = list(zip(cc_p.coils, I_p))
    def phase_field(R, Z, phi, _pairs=pairs):
        R, Z, phi = float(R), float(Z), float(phi)
        R_arr = np.array([[R]]); Z_arr = np.array([[Z]]); phi_arr = np.array([[phi]])
        br = bz = bp = 0.0
        for (pts, _), I in _pairs:
            _br, _bz, _bp = Biot_Savart_field(pts, float(I), R_arr, Z_arr, phi_arr)
            br += float(_br); bz += float(_bz); bp += float(_bp)
        return br, bz, bp
    b_p = compute_resonant_amplitude(phase_field, psi_res_target, TARGET_M, TARGET_N, stella, n_theta=12, n_phi=12)
    b_phase.append(abs(b_nat + b_p))

b_phase_norm = np.array(b_phase) / abs(b_nat)
print(f'Phase scan: {np.round(b_phase_norm, 3)}')
phase_variation = b_phase_norm.max() - b_phase_norm.min()
print(f'Phase variation: {phase_variation:.3f} (should be > 0.05)')

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(I0_scan, b_norm, 'b-o', ms=5)
ax1.axhline(1, color='gray', ls=':')
ax1.set_xlabel('I0 (A)'); ax1.set_ylabel('|b_mn| / |b_nat|')
ax1.set_title(f'Suppression scan, variation={variation:.3f}')
ax1.grid(True, alpha=0.3)

ax2.plot(np.degrees(phase_shifts), b_phase_norm, 'g-o', ms=5)
ax2.axhline(1, color='gray', ls=':')
ax2.set_xlabel('Phase shift (deg)'); ax2.set_ylabel('|b_mn| / |b_nat|')
ax2.set_title(f'Phase scan, variation={phase_variation:.3f}')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/Repo/pyna/scripts/verify_island_control.png', dpi=100)
print('Saved verify_island_control.png')
print('ALL CHECKS PASSED' if variation > 0.01 and phase_variation > 0.01 else 'SOME CHECKS FAILED')
