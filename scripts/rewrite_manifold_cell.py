import sys, json
sys.stdout.reconfigure(encoding='utf-8')

NEW_CELL_SRC = """\
# === Manifold Visualization ===
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.MCF.visual.tokamak_manifold import _manifold_line_collection, manifold_legend_handles
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# field_func_2d: (R, Z, phi) -> [dR/dphi, dZ/dphi]
def field_func_2d(R, Z, phi):
    tang = stella.field_func(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

# X-point coordinates (analytic estimate for q=4/3 island in this stellarator)
# For m_h=4, n_h=3: X-pt at R=3.0236, Z=0.0695 (lambda_u ~ 102)
psi_res_target = stella.resonant_psi(TARGET_M, TARGET_N)[0]
r_res = np.sqrt(psi_res_target) * stella.r0
import math
_theta_x = math.atan2(-0.064828, 3.056828 - 3.0)
_xpt_candidates = [
    (stella.R0 + r_res * np.cos(_theta_x + k * 2*np.pi/TARGET_N),
     r_res * np.sin(_theta_x + k * 2*np.pi/TARGET_N))
    for k in range(TARGET_N)
]

# Compute Jacobian for each candidate, pick the hyperbolic one (|lam_max| >> 1)
vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=1e-5)
phi_span = (0.0, 2.0 * np.pi * TARGET_N)

xpt_RZ, xpt_Jac = None, None
for R_c, Z_c in _xpt_candidates:
    M = vq.jacobian_matrix([R_c, Z_c], phi_span=phi_span)
    lam_abs = sorted(np.abs(np.linalg.eigvals(M)))
    if lam_abs[1] > 5.0:          # truly hyperbolic
        xpt_RZ, xpt_Jac = np.array([R_c, Z_c]), M
        print(f'X-point: R={R_c:.5f}  Z={Z_c:.5f}  lambda_u={lam_abs[1]:.2f}')
        break

if xpt_RZ is None:
    print('Warning: no hyperbolic X-point found; using first candidate')
    xpt_RZ = np.array(_xpt_candidates[0])
    xpt_Jac = vq.jacobian_matrix(xpt_RZ, phi_span=phi_span)

RZlimit = (stella.R0 - stella.r0*1.05, stella.R0 + stella.r0*1.05,
           -stella.r0*1.05, stella.r0*1.05)

sm_mfld = StableManifold(xpt_RZ, xpt_Jac, field_func_2d, phi_span=phi_span)
um_mfld = UnstableManifold(xpt_RZ, xpt_Jac, field_func_2d, phi_span=phi_span)

sm_mfld.grow(n_turns=6, init_length=1e-4, n_init_pts=5, both_sides=True, RZlimit=RZlimit)
um_mfld.grow(n_turns=6, init_length=1e-4, n_init_pts=5, both_sides=True, RZlimit=RZlimit)

print(f'Stable   manifold: {len(sm_mfld.segments)} segments')
print(f'Unstable manifold: {len(um_mfld.segments)} segments')

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 6))

# Poincaré background (already computed as pmap_natural)
rz = pmap_natural[section]
psi_vals = ((rz[:, 0] - stella.R0)**2 + rz[:, 1]**2) / stella.r0**2
sc = ax.scatter(rz[:, 0], rz[:, 1], c=psi_vals, s=0.6, cmap='plasma',
                vmin=0, vmax=1, rasterized=True, alpha=0.55, zorder=2)

# Stable manifold (cool, teal)
s_ref_s = max((np.ptp(seg[:, 0]) for seg in sm_mfld.segments if len(seg) > 1), default=1.0)
for seg in sm_mfld.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='GnBu', s_ref=s_ref_s, lw=1.2)
        ax.add_collection(lc)

# Unstable manifold (warm, orange-red)
s_ref_u = max((np.ptp(seg[:, 0]) for seg in um_mfld.segments if len(seg) > 1), default=1.0)
for seg in um_mfld.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='Oranges', s_ref=s_ref_u, lw=1.2)
        ax.add_collection(lc)

# X-point marker
ax.plot(*xpt_RZ, 'kx', ms=10, mew=2, zorder=10, label='X-point')

ax.set_xlabel('R (m)')
ax.set_ylabel('Z (m)')
ax.set_title(f'Poincaré + $W^s$/$W^u$ Manifolds, $q={TARGET_M}/{TARGET_N}$ X-point, $\\phi=0$',
             fontsize=12)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('poincare_manifolds.png', dpi=150, bbox_inches='tight')
plt.show()
"""

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
for c in code_cells:
    src = ''.join(c['source'])
    if 'find_cycle' in src or ('StableManifold' in src and 'jacobian_matrix' in src):
        c['source'] = [NEW_CELL_SRC]
        c['outputs'] = []
        c['execution_count'] = None
        print('Cell rewritten (removed find_cycle, kept jacobian_matrix approach)')
        break

with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
