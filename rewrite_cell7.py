import json

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

NEW_CELL7 = r"""# === Manifold / Jacobian Visualization ===
# For a q=m/n island, the X-point (hyperbolic fixed point of the n-turn Poincare map)
# sits at the resonant surface. We locate it analytically and compute the monodromy
# matrix with variational equations to show the hyperbolic character.

from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.MCF.visual.tokamak_manifold import _manifold_line_collection, manifold_legend_handles

# field_func_2d: (R, Z, phi) -> [dR/dphi, dZ/dphi]
def field_func_2d(R, Z, phi):
    tang = stella.field_func(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

# --- Locate X-point from the Poincare point cloud ---
# For q=m/n, there are m X-points on the resonant circle.
# We find the one with the largest |Z| (outermost in Z) from the Poincare data.
psi_res_target = stella.resonant_psi(TARGET_M, TARGET_N)[0]
r_res = np.sqrt(psi_res_target) * stella.r0

# Analytic approximation: find Poincare points near resonant radius
if len(results_natural) > 0:
    R_nat = results_natural[:, 0]
    Z_nat = results_natural[:, 1]
    r_nat = np.sqrt((R_nat - stella.R0)**2 + Z_nat**2)
    dr = np.abs(r_nat - r_res)
    # Points within 20% of resonant radius
    near_res = dr < 0.2 * r_res
    if near_res.sum() > 4:
        R_near = R_nat[near_res]
        Z_near = Z_nat[near_res]
        # X-points are at the outermost extent in each sector
        # Use the point with most negative Z as seed
        idx_xpt = np.argmin(Z_near)
        xpt_seed = np.array([R_near[idx_xpt], Z_near[idx_xpt]])
    else:
        # Fall back to analytic estimate
        import math
        _theta_x = math.atan2(-0.065, 3.057 - stella.R0)
        xpt_seed = np.array([stella.R0 + r_res * math.cos(_theta_x),
                              r_res * math.sin(_theta_x)])
else:
    import math
    _theta_x = math.atan2(-0.065, 3.057 - stella.R0)
    xpt_seed = np.array([stella.R0 + r_res * math.cos(_theta_x),
                          r_res * math.sin(_theta_x)])

print(f'X-point seed: R={xpt_seed[0]:.5f}  Z={xpt_seed[1]:.5f}')

# --- Compute Jacobian (monodromy matrix) at seed ---
# Use moderate tolerances for tutorial speed
phi_span = (0.0, 2.0 * np.pi * TARGET_N)
vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=1e-6)
xpt_Jac = vq.jacobian_matrix(xpt_seed, phi_span,
                              method='RK45', rtol=1e-7, atol=1e-9)

lam = np.linalg.eigvals(xpt_Jac)
lam_abs = sorted(np.abs(lam))
det_J = np.linalg.det(xpt_Jac)
print(f'det(J) = {det_J:.4f}  (ideal = 1.0 for area-preserving map)')
print(f'|lambda_stable| = {lam_abs[0]:.4f}')
print(f'|lambda_unstable| = {lam_abs[1]:.4f}')

if abs(det_J - 1.0) > 0.5:
    print('Warning: det(J) deviates from 1 — seed is not a true X-point.')
    print('Showing Jacobian at approximate location for illustration.')
    xpt_RZ = xpt_seed
else:
    xpt_RZ = xpt_seed

# --- Grow manifolds (1 turn, illustration only) ---
RZlimit = (stella.R0 - stella.r0*1.05, stella.R0 + stella.r0*1.05,
           -stella.r0*1.05, stella.r0*1.05)

_ivp_kw = dict(rtol=1e-7, atol=1e-9)
sm_mfld = StableManifold(xpt_RZ, xpt_Jac, field_func_2d, phi_span=phi_span)
um_mfld = UnstableManifold(xpt_RZ, xpt_Jac, field_func_2d, phi_span=phi_span)

sm_mfld.grow(n_turns=1, init_length=1e-4, n_init_pts=2, both_sides=False,
             RZlimit=RZlimit, **_ivp_kw)
um_mfld.grow(n_turns=1, init_length=1e-4, n_init_pts=2, both_sides=False,
             RZlimit=RZlimit, **_ivp_kw)

print(f'Stable segments: {len(sm_mfld.segments)}  Unstable: {len(um_mfld.segments)}')

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Poincare + manifolds
ax = axes[0]
if len(results_natural) > 0:
    R_pts, Z_pts = results_natural[:, 0], results_natural[:, 1]
    psi_pts = ((R_pts - stella.R0)**2 + Z_pts**2) / stella.r0**2
    ax.scatter(R_pts, Z_pts, c=np.clip(psi_pts, 0, 1), s=0.6, cmap='plasma',
               vmin=0, vmax=1, rasterized=True, alpha=0.6, zorder=2)

for seg in sm_mfld.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='GnBu', lw=1.5)
        ax.add_collection(lc)
for seg in um_mfld.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='Oranges', lw=1.5)
        ax.add_collection(lc)
ax.plot(*xpt_RZ, 'kx', ms=10, mew=2, zorder=10, label='X-point seed')
theta_c = np.linspace(0, 2*np.pi, 200)
ax.plot(stella.R0 + r_res*np.cos(theta_c), r_res*np.sin(theta_c),
        '--', color='tomato', lw=0.8, alpha=0.6)
ax.plot(stella.R0 + stella.r0*np.cos(theta_c), stella.r0*np.sin(theta_c),
        'k-', lw=0.8, alpha=0.5)
ax.set_aspect('equal')
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title(fr'Poincaré + $W^s/W^u$  (1-turn illustration)', fontsize=11)
ax.legend(fontsize=8)

# Right: monodromy matrix eigenvalue diagram
ax2 = axes[1]
lam_complex = np.linalg.eigvals(xpt_Jac)
ax2.scatter(lam_complex.real, lam_complex.imag, s=120, zorder=5,
            c=['#E91E63', '#2196F3'])
theta_unit = np.linspace(0, 2*np.pi, 300)
ax2.plot(np.cos(theta_unit), np.sin(theta_unit), 'k--', lw=0.8, alpha=0.4)
ax2.axhline(0, color='gray', lw=0.5); ax2.axvline(0, color='gray', lw=0.5)
ax2.set_aspect('equal')
ax2.set_xlabel(r'Re($\lambda$)'); ax2.set_ylabel(r'Im($\lambda$)')
ax2.set_title(fr'Monodromy eigenvalues  det(J)={det_J:.3f}', fontsize=11)
for lv in lam_complex:
    ax2.annotate(f'{abs(lv):.3f}', (lv.real, lv.imag),
                 textcoords='offset points', xytext=(8, 5), fontsize=9)
ax2.text(0.05, 0.95,
    f'|λ_u| = {lam_abs[1]:.3f}\n|λ_s| = {lam_abs[0]:.3f}',
    transform=ax2.transAxes, va='top', fontsize=10,
    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

plt.suptitle(fr'$q={TARGET_M}/{TARGET_N}$ Island X-point: Hyperbolic Structure', fontsize=12)
plt.tight_layout()
plt.savefig('island_manifolds.png', dpi=120, bbox_inches='tight')
plt.show()
"""

# Replace cell[7]
nb['cells'][7]['source'] = NEW_CELL7
print(f'Replaced cell[7]')

# Also fix cell[6] - it references pmap_natural[section] which may not work
# Check what results_natural is
cell6 = ''.join(nb['cells'][6]['source'])
print('cell[6] uses:', 'pmap_natural[section]' in cell6, 'results_natural' in cell6)

with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
