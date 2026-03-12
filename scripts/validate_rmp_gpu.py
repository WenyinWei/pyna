"""GPU-accelerated RMP island chain validation.

Uses CUDAFieldLineTracer for dense Poincaré coverage.
Target: q=4/1 island chain in Solov'ev equilibrium + (4,1) RMP perturbation.
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyna.mag.solovev import SolovevEquilibrium
from pyna.topo.island_extract import extract_island_width


# ---------------------------------------------------------------------------
# 1. Build equilibrium with q0=3.0 (so q=4 is accessible inside LCFS)
# ---------------------------------------------------------------------------
print("Building Solov'ev equilibrium (q0=3.0)...")
eq = SolovevEquilibrium(R0=6.2, a=2.0, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)
R_ax, Z_ax = eq.magnetic_axis
print(f"  Magnetic axis: R={R_ax:.3f} m, Z={Z_ax:.3f} m")

# ---------------------------------------------------------------------------
# 2. Find q=4/1 resonant surface
# ---------------------------------------------------------------------------
print("Scanning q profile to find q=4 surface...")
psi_scan = np.linspace(0.05, 0.95, 200)
q_scan = eq.q_profile(psi_scan)

# Find psi where q crosses 4
target_q = 4.0
psi_res = None
for i in range(len(q_scan) - 1):
    if (np.isfinite(q_scan[i]) and np.isfinite(q_scan[i+1]) and
            ((q_scan[i] - target_q) * (q_scan[i+1] - target_q) < 0)):
        # Linear interpolation
        frac = (target_q - q_scan[i]) / (q_scan[i+1] - q_scan[i])
        psi_res = psi_scan[i] + frac * (psi_scan[i+1] - psi_scan[i])
        print(f"  q=4 at psi_norm = {psi_res:.4f}")
        break

if psi_res is None:
    # Fallback: use psi where q is closest to 4
    valid = np.isfinite(q_scan)
    if valid.any():
        idx = np.argmin(np.abs(q_scan[valid] - target_q))
        psi_res = psi_scan[valid][idx]
        print(f"  q=4 not crossed; using nearest psi_norm = {psi_res:.4f}, q={q_scan[valid][idx]:.3f}")
    else:
        psi_res = 0.5
        print(f"  Warning: no valid q values; using psi_res=0.5")

# Find R on midplane corresponding to psi_res
# psi_norm(R, 0) = psi_res  => scan R outboard from axis
R_outer = np.linspace(R_ax + 0.01, R_ax + 1.95, 1000)
psi_outer = eq.psi(R_outer, np.zeros_like(R_outer))
R_res = float(np.interp(psi_res, psi_outer, R_outer))
print(f"  Resonant surface R_res = {R_res:.3f} m (midplane)")

# ---------------------------------------------------------------------------
# 3. Start points: 20 inside + 20 outside (within ±0.05 in psi)
# ---------------------------------------------------------------------------
dpsi = 0.04
psi_in  = np.linspace(max(psi_res - dpsi, 0.02), psi_res - 0.002, 20)
psi_out = np.linspace(psi_res + 0.002, min(psi_res + dpsi, 0.97), 20)
psi_all = np.concatenate([psi_in, psi_out])

R_starts = np.array([float(np.interp(p, psi_outer, R_outer)) for p in psi_all])
start_pts = np.column_stack([R_starts,
                              np.zeros(len(R_starts)),
                              np.zeros(len(R_starts))])
print(f"  Start points: {len(start_pts)} lines (R range {R_starts.min():.3f}–{R_starts.max():.3f} m)")


# ---------------------------------------------------------------------------
# 4. Poincaré crossing detector
# ---------------------------------------------------------------------------

def detect_crossings_phi0(traj):
    """Detect phi=0 (mod 2pi) crossings in trajectory.

    Parameters
    ----------
    traj : ndarray (n, 3) with columns R, Z, phi
    Returns ndarray (m, 2) of crossing (R, Z) points.
    """
    R   = traj[:, 0]
    Z   = traj[:, 1]
    phi = traj[:, 2]

    # phi is monotonically increasing; find where phi crosses k*2pi for integer k>=1
    phi_mod = phi % (2 * np.pi)
    # crossing when phi_mod wraps (value drops from near 2pi to near 0)
    crossings = []
    for i in range(1, len(phi)):
        if phi_mod[i] < phi_mod[i-1] - np.pi:  # wrapped
            # linear interpolate
            dphi_mod = (2*np.pi - phi_mod[i-1]) + phi_mod[i]
            if dphi_mod < 1e-30:
                frac = 0.5
            else:
                frac = (2*np.pi - phi_mod[i-1]) / dphi_mod
            R_cross = R[i-1] + frac * (R[i] - R[i-1])
            Z_cross = Z[i-1] + frac * (Z[i] - Z[i-1])
            crossings.append([R_cross, Z_cross])

    if not crossings:
        return np.empty((0, 2))
    return np.array(crossings)


# ---------------------------------------------------------------------------
# 5. GPU tracing
# ---------------------------------------------------------------------------
T_MAX = 5000.0
DT    = 0.05

gpu_ok = False
try:
    from pyna.flt_cuda import CUDAFieldLineTracer
    tracer = CUDAFieldLineTracer(
        R0=eq.R0, a=eq.a, B0=eq._B0, q0=eq.q0,
        epsilon_h=0.01, m_h=4.0, n_h=1.0,
        dt=DT,
    )
    print(f"\nGPU tracing {len(start_pts)} lines, t_max={T_MAX}, dt={DT}...")
    t0 = time.time()
    trajs = tracer.trace_many(start_pts, t_max=T_MAX)
    t_gpu = time.time() - t0
    print(f"  GPU time: {t_gpu:.2f} s  ({sum(len(t) for t in trajs):,} total steps)")
    gpu_ok = True
except Exception as exc:
    print(f"\nGPU tracing failed: {exc}")
    print("Falling back to CPU tracer...")

if not gpu_ok:
    try:
        from pyna.flt import FieldLineTracer
        from pyna.topo.poincare import poincare_section
        T_MAX_CPU = 2000.0
        tracer_cpu = FieldLineTracer(
            field_func=None,   # will build below
            dt=DT,
            n_workers=16,
        )
        # Build field function from equilibrium
        def field_func_rmp(rzphi):
            R, Z, phi = rzphi
            BR, BZ = eq.BR_BZ(R, Z)
            Bphi = eq.Bphi(R)
            # RMP perturbation
            theta = np.arctan2(Z - Z_ax, R - R_ax)
            psi_n = float(eq.psi(R, Z))
            env = psi_n * (1.0 - psi_n)
            BR += 0.01 * eq._B0 * env * np.cos(4 * theta - 1 * phi)
            BZ += 0.01 * eq._B0 * env * np.sin(4 * theta - 1 * phi)
            Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2) + 1e-30
            return np.array([BR/Bmag, BZ/Bmag, Bphi/(R*Bmag)])

        tracer_cpu.field_func = field_func_rmp
        print(f"CPU tracing {len(start_pts)} lines, t_max={T_MAX_CPU}...")
        t0 = time.time()
        trajs = [tracer_cpu.trace(sp, T_MAX_CPU) for sp in start_pts]
        t_cpu = time.time() - t0
        print(f"  CPU time: {t_cpu:.2f} s")
    except Exception as exc2:
        print(f"CPU fallback also failed: {exc2}")
        # Build simple CPU tracer manually using scipy
        from scipy.integrate import solve_ivp

        def field_func_rmp(t, rzphi):
            R, Z, phi = rzphi
            BR, BZ = eq.BR_BZ(R, Z)
            Bphi = float(eq.Bphi(R))
            theta = np.arctan2(Z - Z_ax, R - R_ax)
            psi_n = float(eq.psi(R, Z))
            env = psi_n * (1.0 - psi_n)
            BR += 0.01 * eq._B0 * env * np.cos(4 * theta - 1 * phi)
            BZ += 0.01 * eq._B0 * env * np.sin(4 * theta - 1 * phi)
            Bmag = np.sqrt(float(BR)**2 + float(BZ)**2 + Bphi**2) + 1e-30
            return [float(BR)/Bmag, float(BZ)/Bmag, Bphi/(R*Bmag)]

        print("  Using scipy solve_ivp for CPU fallback...")
        T_MAX_CPU = 500.0
        t0 = time.time()
        trajs = []
        for sp in start_pts[:20]:  # limit for speed
            sol = solve_ivp(field_func_rmp, [0, T_MAX_CPU], sp,
                            method='RK45', max_step=0.1, dense_output=False)
            traj = sol.y.T  # (n, 3)
            trajs.append(traj)
        t_cpu = time.time() - t0
        print(f"  SciPy CPU time: {t_cpu:.2f} s")

# ---------------------------------------------------------------------------
# 6. Collect crossings
# ---------------------------------------------------------------------------
print("\nCollecting phi=0 crossings...")
all_crossings = []
for traj in trajs:
    if traj is None or len(traj) < 2:
        continue
    pts = detect_crossings_phi0(traj)
    if len(pts) > 0:
        all_crossings.append(pts)

if all_crossings:
    poincare_pts = np.vstack(all_crossings)
    print(f"  Total crossings: {len(poincare_pts):,}")
else:
    print("  No crossings detected! Using raw trajectory midplane points.")
    # Fallback: use Z≈0 points from trajectories
    fallback = []
    for traj in trajs:
        if traj is None or len(traj) < 2:
            continue
        Z = traj[:, 1]
        R = traj[:, 0]
        mask = np.abs(Z) < 0.5
        if mask.any():
            fallback.append(np.column_stack([R[mask], Z[mask]]))
    if fallback:
        poincare_pts = np.vstack(fallback)
        print(f"  Fallback points: {len(poincare_pts):,}")
    else:
        poincare_pts = np.zeros((1, 2))

# ---------------------------------------------------------------------------
# 7. Extract island chain
# ---------------------------------------------------------------------------
island_chain = None
if len(poincare_pts) > 50:
    try:
        print("Extracting island chain (mode_m=4)...")
        island_chain = extract_island_width(
            poincare_pts, R_ax, Z_ax,
            mode_m=4,
            psi_func=lambda R, Z: eq.psi(R, Z)
        )
        print(f"  O-points: {island_chain.O_points}")
        print(f"  half_width_r = {island_chain.half_width_r:.4f} m")
        print(f"  half_width_psi = {island_chain.half_width_psi:.4f}")
    except Exception as exc:
        print(f"  Island extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('#0a0a1a')
fig.patch.set_facecolor('#0a0a1a')

# Color by field line index
n_lines = len(trajs)
for i, crossings in enumerate(all_crossings):
    color = plt.cm.plasma(i / max(n_lines - 1, 1))
    ax.scatter(crossings[:, 0], crossings[:, 1],
               s=0.5, c=[color], alpha=0.7, linewidths=0)

# Mark O and X points
if island_chain is not None:
    for op in island_chain.O_points:
        ax.plot(op[0], op[1], 'o', color='lime', ms=8, zorder=5, label='O-point')
        # Double-headed arrow for half-width
        dx = island_chain.half_width_r
        ax.annotate('', xy=(op[0]+dx, op[1]), xytext=(op[0]-dx, op[1]),
                    arrowprops=dict(arrowstyle='<->', color='lime', lw=1.5))
    for xp in island_chain.X_points:
        ax.plot(xp[0], xp[1], 'x', color='red', ms=8, zorder=5, mew=2, label='X-point')

# Mark resonant surface
ax.axvline(R_res, color='cyan', ls='--', lw=0.8, alpha=0.6, label=f'q=4 surface (R={R_res:.2f} m)')
# Mark axis
ax.plot(R_ax, Z_ax, '+', color='white', ms=12, mew=2, zorder=6, label='Magnetic axis')

ax.set_xlabel('R (m)', color='white', fontsize=12)
ax.set_ylabel('Z (m)', color='white', fontsize=12)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

title = (f"RMP Island Chain Validation — GPU\n"
         f"q=4/1 resonance, ε_h=0.01, {len(poincare_pts):,} crossings")
ax.set_title(title, color='white', fontsize=11)

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), fontsize=8, facecolor='#222', labelcolor='white')

plt.tight_layout()
outpath = r"D:\Repo\pyna\scripts\rmp_island_gpu.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {outpath}")
plt.close()
