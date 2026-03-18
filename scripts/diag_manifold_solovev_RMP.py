"""
Diagnostic: manifold grow() ordering test on Solovev tokamak + RMP.

We use a simple q=2/1 resonant surface in Solovev equilibrium + analytic RMP.
The test checks whether the grown segments are ordered along the manifold
(monotone distance from X-point) or zigzag between generations.
"""
import sys
sys.path.insert(0, 'D:/Repo/pyna')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyna.MCF.equilibrium.Solovev import solovev_iter_like
from pyna.topo.cycle import find_cycle
from pyna.topo.manifold_improve import StableManifold, UnstableManifold

eq = solovev_iter_like(scale=0.5)   # slightly bigger so q=2/1 is in range
Rmaxis, Zmaxis = eq.magnetic_axis
print(f"Solovev: R0={eq.R0:.3f}, a={eq.a:.3f}, B0={eq.B0:.1f}")
print(f"Axis: ({Rmaxis:.3f}, {Zmaxis:.3f})")

# RMP perturbation: m=2, n=1
RMP_amp = 0.05

def field_func_2d(R, Z, phi):
    BR, BZ = eq.BR_BZ(np.array([R]), np.array([Z]))
    Bphi = eq.Bphi(np.array([R]))
    BR = float(BR[0]); BZ = float(BZ[0]); Bphi = float(Bphi[0])
    theta = np.arctan2(Z - Zmaxis, R - Rmaxis)
    dBR = RMP_amp * np.cos(2*theta - phi)
    dBZ = RMP_amp * np.sin(2*theta - phi) * 0.5
    return np.array([(BR + dBR)/Bphi, (BZ + dBZ)/Bphi])

def field_func_1d(rzphi):
    R, Z, phi = rzphi
    v = field_func_2d(R, Z, phi)
    norm = np.sqrt(v[0]**2 + v[1]**2 + 1.0)
    return np.array([v[0]/norm, v[1]/norm, 1.0/norm])

# Find q=2/1 resonant radius
psi_vals = np.linspace(0.05, 0.95, 50)
q_vals = eq.q_profile(psi_vals, n_theta=64)
# Interpolate to find psi where q=2
from scipy.interpolate import interp1d
valid = np.isfinite(q_vals)
if valid.sum() < 3:
    print("ERROR: q_profile failed")
    sys.exit(1)
q_interp = interp1d(q_vals[valid], psi_vals[valid], bounds_error=False, fill_value='extrapolate')
psi_21 = float(q_interp(2.0))
r_21 = np.sqrt(psi_21) * eq.a
print(f"q=2/1 surface: psi={psi_21:.4f}, r={r_21:.4f} m")

# Find X-point via find_cycle (period-1 for q=2/1 RMP with n=1)
RZlimit = (eq.R0 - eq.a*1.1, eq.R0 + eq.a*1.1, -eq.a*1.5, eq.a*1.5)
found = []
for theta_seed in np.linspace(0, 2*np.pi, 16, endpoint=False):
    R_seed = Rmaxis + r_21 * np.cos(theta_seed)
    Z_seed = Zmaxis + r_21 * np.sin(theta_seed)
    orb = find_cycle(field_func_1d, np.array([R_seed, Z_seed, 0.0]),
                     n_turns=1, dt=0.05, RZlimit=RZlimit, max_iter=40, tol=1e-7)
    if orb is None: continue
    dup = any(np.linalg.norm(orb.rzphi0[:2] - f.rzphi0[:2]) < 5e-3 for f in found)
    if not dup: found.append(orb)

xpts = [o for o in found if not o.is_stable]
opts = [o for o in found if o.is_stable]
print(f"Found {len(opts)} O-points, {len(xpts)} X-points")
for xp in xpts:
    print(f"  X: ({xp.rzphi0[0]:.4f}, {xp.rzphi0[1]:.4f}), eigvals={np.round(xp.eigenvalues, 3)}")

if not xpts:
    print("No X-points found — increase RMP_amp or check q profile")
    sys.exit(0)

xpoint = xpts[0]
R_xpt, Z_xpt = xpoint.rzphi0[0], xpoint.rzphi0[1]
DP_1 = xpoint.Jac

print(f"\nUsing X-point ({R_xpt:.4f}, {Z_xpt:.4f})")
print(f"DP_1 eigenvalues: {np.round(np.linalg.eigvals(DP_1), 4)}")
print()

# Grow manifolds with small n_turns to inspect ordering
print("=== Growing UNSTABLE manifold (5 turns) ===")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    um = UnstableManifold([R_xpt, Z_xpt], DP_1, field_func_2d, phi_span=(0.0, 2*np.pi))
    um.grow(n_turns=5, init_length=1e-5, n_init_pts=6, both_sides=False)
    for warning in w:
        print(f"  WARNING: {warning.message}")

if um.segments:
    seg = um.segments[0]
    print(f"Segment shape: {seg.shape}")
    dists = np.linalg.norm(seg - np.array([R_xpt, Z_xpt]), axis=1)
    print(f"Distance from X-point: first 12 values = {np.round(dists[:12], 6)}")
    
    # Check monotonicity
    back_steps = np.sum(np.diff(dists) < -1e-7)
    print(f"Non-monotone distance steps: {back_steps} out of {len(dists)-1}")
    print()
    
    # Show what the correct ordering should be
    print("=== EXPECTED STRUCTURE (n_init_pts=6, n_turns=5) ===")
    print("Correct: distance should increase monotonically")
    print("Bug:     each generation starts over near X-point (zigzag)")
    print()
    print("Generation boundaries (n_init_pts=6):")
    for t in range(6):
        start = t * 6
        end = start + 6
        if end <= len(dists):
            print(f"  Gen {t}: indices {start}-{end-1}, dist range [{dists[start]:.5f}, {dists[end-1]:.5f}]")

# Trace Poincare for background
print("\nTracing Poincare background...")
from scipy.integrate import solve_ivp
poincare_R, poincare_Z = [], []
for R0_seed in np.linspace(Rmaxis + 0.02, Rmaxis + 0.85*eq.a, 8):
    y = np.array([R0_seed, Zmaxis])
    for _ in range(150):
        sol = solve_ivp(lambda phi, y: field_func_2d(y[0], y[1], phi),
                       [0, 2*np.pi], y, method='DOP853', rtol=1e-9, atol=1e-12)
        if not sol.success: break
        y = sol.y[:, -1]
        psi = float(eq.psi(np.array([y[0]]), np.array([y[1]])))
        if psi > 1.05 or y[0] < 0.1: break
        poincare_R.append(y[0]); poincare_Z.append(y[1])

# Plot
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(poincare_R, poincare_Z, s=0.5, c='lightblue', alpha=0.4)
for seg in um.segments:
    if len(seg) > 2:
        ax.plot(seg[:,0], seg[:,1], 'r-', lw=1.0)
for seg in (getattr(um, '_sm', None) or []):
    pass
ax.plot(R_xpt, Z_xpt, 'r+', ms=12, mew=2)
for op in opts:
    ax.plot(op.rzphi0[0], op.rzphi0[1], 'go', ms=6)
ax.set_aspect('equal')
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title(f'Solovev + RMP: Unstable manifold (5 turns)\nX: ({R_xpt:.3f}, {Z_xpt:.3f})')
plt.tight_layout()
plt.savefig('D:/Repo/pyna/scripts/diag_solovev_rmp.png', dpi=120)
print("Saved diag_solovev_rmp.png")
