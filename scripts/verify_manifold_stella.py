import sys
sys.path.insert(0, 'D:/Repo/pyna')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyna.MCF.equilibrium.stellarator import SimpleStellarartor, simple_stellarator
from pyna.topo.cycle import find_cycle
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.MCF.visual.tokamak_manifold import _manifold_line_collection, manifold_legend_handles
from matplotlib.colors import Normalize
import matplotlib.cm as cm

TARGET_M, TARGET_N = 4, 1

# Build stellarator (same as notebook)
stella = simple_stellarator(R0=3.0, r0=0.3, B0=1.0, q0=1.1, q1=5.0, m_h=4, n_h=4, epsilon_h=0.05)
print(stella)

field_func_1d = stella.field_func

def field_func_2d(R, Z, phi):
    tang = field_func_1d(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

psi_res_list = stella.resonant_psi(TARGET_M, TARGET_N)
print(f"Resonant psi: {psi_res_list}")
psi_res_target = psi_res_list[0]
r_res = np.sqrt(psi_res_target) * stella.r0

RZlimit = (stella.R0 - stella.r0*1.05, stella.R0 + stella.r0*1.05,
           -stella.r0*1.05, stella.r0*1.05)

# Find X-points
found_orbits = []
for theta_seed in np.linspace(0, 2*np.pi, 24, endpoint=False):
    R_seed = stella.R0 + r_res * np.cos(theta_seed)
    Z_seed = r_res * np.sin(theta_seed)
    orbit = find_cycle(field_func_1d, np.array([R_seed, Z_seed, 0.0]),
                       n_turns=TARGET_N, dt=0.1, RZlimit=RZlimit,
                       max_iter=40, tol=1e-8)
    if orbit is None: continue
    if np.sqrt((orbit.rzphi0[0]-stella.R0)**2 + orbit.rzphi0[1]**2) < 0.02: continue
    dup = any(np.linalg.norm(orbit.rzphi0[:2] - fo.rzphi0[:2]) < 1e-3 for fo in found_orbits)
    if not dup: found_orbits.append(orbit)

x_points = [o for o in found_orbits if not o.is_stable]
o_points = [o for o in found_orbits if o.is_stable]
print(f"Found {len(o_points)} O-points, {len(x_points)} X-points")

hyperbolic = [o for o in x_points if o.stability_index > 1.0]
if not hyperbolic:
    print("No hyperbolic X-points found!")
    sys.exit(1)

hyperbolic.sort(key=lambda o: o.stability_index, reverse=True)
xpoint = hyperbolic[0]
R_xpt, Z_xpt = xpoint.rzphi0[0], xpoint.rzphi0[1]
DP_m = xpoint.Jac
print(f"X-point: ({R_xpt:.4f}, {Z_xpt:.4f}), stability_index={xpoint.stability_index:.3f}")
print(f"DP_m eigenvalues: {np.round(xpoint.eigenvalues, 4)}")

# Grow manifolds with n_turns=6
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sm_mfld = StableManifold([R_xpt, Z_xpt], DP_m, field_func_2d,
                              phi_span=(0.0, 2*np.pi*TARGET_N))
    um_mfld = UnstableManifold([R_xpt, Z_xpt], DP_m, field_func_2d,
                                phi_span=(0.0, 2*np.pi*TARGET_N))
    sm_mfld.grow(n_turns=6, init_length=5e-5, n_init_pts=8, both_sides=True)
    um_mfld.grow(n_turns=6, init_length=5e-5, n_init_pts=8, both_sides=True)
    large_jump = [x for x in w if 'large jumps' in str(x.message)]

print(f"Stable segments: {len(sm_mfld.segments)}, Unstable: {len(um_mfld.segments)}")
print(f"Expected: {2*(6+1)}={2*7} each")
print(f"Large-jump warnings: {len(large_jump)}")

# Plot
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor('#f8f8f8')

for seg in sm_mfld.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='GnBu')
        lc.set_linewidth(1.2); lc.set_alpha(0.85); lc.set_zorder(6)
        ax.add_collection(lc)
for seg in um_mfld.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='Oranges')
        lc.set_linewidth(1.2); lc.set_alpha(0.85); lc.set_zorder(6)
        ax.add_collection(lc)

for xp in x_points:
    ax.plot(xp.rzphi0[0], xp.rzphi0[1], 'r+', ms=12, mew=2.5, zorder=8)
for op in o_points:
    ax.plot(op.rzphi0[0], op.rzphi0[1], 'go', ms=7, zorder=7)

theta_c = np.linspace(0, 2*np.pi, 300)
ax.plot(stella.R0 + r_res*np.cos(theta_c), r_res*np.sin(theta_c),
        '--', color='tomato', lw=0.8, alpha=0.6)
ax.plot(stella.R0 + stella.r0*np.cos(theta_c), stella.r0*np.sin(theta_c), 'k-', lw=1.2)

ax.set_xlim(stella.R0 - 1.2*stella.r0, stella.R0 + 1.2*stella.r0)
ax.set_ylim(-1.2*stella.r0, 1.2*stella.r0)
ax.set_aspect('equal')
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title(f'Manifold fix verification: q={TARGET_M}/{TARGET_N}, n_turns=6\n'
             f'Segments: {len(sm_mfld.segments)}S + {len(um_mfld.segments)}U, '
             f'large-jump warnings: {len(large_jump)}')
plt.tight_layout()
outpath = 'D:/Repo/pyna/scripts/verify_manifold_stella.png'
plt.savefig(outpath, dpi=120)
print(f"Saved {outpath}")
print("DONE")
