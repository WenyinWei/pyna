import sys, json, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

# First find all 3 X-points with targeted seeds
import numpy as np
sys.path.insert(0, 'D:/Repo/pyna')
from pyna.MCF.equilibrium.stellarator import simple_stellarator
from pyna.topo.cycle import find_cycle

stella = simple_stellarator(R0=3.0, r0=0.30, B0=1.0, q0=1.1, q1=5.0, m_h=4, n_h=3, epsilon_h=0.05)
TARGET_M, TARGET_N = 4, 3
psi_res = stella.resonant_psi(TARGET_M, TARGET_N)[0]
r_res = np.sqrt(psi_res) * stella.r0
RZlimit = (stella.R0-stella.r0*1.1, stella.R0+stella.r0*1.1, -stella.r0*1.1, stella.r0*1.1)

# For q=4/3, period-3 X-points: try theta offsets that are hyperbolic
# Known: theta~5pi/3 gives X-pt. Try theta ~ pi/3 + 2pi/3*k for k=0,1,2 shifted by pi
xpts = []
seen = set()
for theta in np.linspace(0, 2*np.pi, 48, endpoint=False):
    R0 = stella.R0 + r_res * np.cos(theta)
    Z0 = r_res * np.sin(theta)
    try:
        orb = find_cycle(stella.field_func, np.array([R0, Z0, 0.0]),
                         n_turns=TARGET_N, dt=0.05, RZlimit=RZlimit, max_iter=20, tol=1e-7)
        if orb is not None and not orb.is_stable:
            key = (round(orb.rzphi0[0], 3), round(orb.rzphi0[1], 3))
            if key not in seen:
                seen.add(key)
                xpts.append(orb)
                lam = sorted(np.abs(np.linalg.eigvals(orb.Jac)))
                print(f'X-pt: R={orb.rzphi0[0]:.6f} Z={orb.rzphi0[1]:.6f} lam={lam}')
    except: pass

print(f'\nTotal X-pts found: {len(xpts)}')
for i, orb in enumerate(xpts):
    print(f'  xpt{i}: rzphi0={orb.rzphi0.tolist()}')
    print(f'  Jac={orb.Jac.tolist()}')
