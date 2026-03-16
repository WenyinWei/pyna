import json

with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

NEW_CELL23 = r"""from pyna.MCF.equilibrium.Solovev import solovev_single_null
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.MCF.visual.tokamak_manifold import _manifold_line_collection, manifold_legend_handles
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------
# 1. Build single-null equilibrium
# -----------------------------------------------------------------------
eq_sn = solovev_single_null(
    R0=1.86, a=0.595, B0=5.3,
    kappa=1.8, delta_u=0.33, delta_l=0.40, kappa_x=1.5,
    q0=1.5,
)
R_ax_sn, Z_ax_sn = eq_sn.magnetic_axis
print(f'Magnetic axis: R={R_ax_sn:.4f} m  Z={Z_ax_sn:.4f} m')

# -----------------------------------------------------------------------
# 2. Plot the equilibrium: flux contours + X-point
# -----------------------------------------------------------------------
R_range_sn = (eq_sn.R0 - 1.5 * eq_sn.a, eq_sn.R0 + 1.5 * eq_sn.a)
Z_range_sn = (-2.0 * eq_sn.kappa * eq_sn.a, 1.5 * eq_sn.kappa * eq_sn.a)

R1d_sn = np.linspace(*R_range_sn, 300)
Z1d_sn = np.linspace(*Z_range_sn, 300)
Rg_sn, Zg_sn = np.meshgrid(R1d_sn, Z1d_sn)
psi_g_sn = eq_sn.psi(Rg_sn, Zg_sn)

R_xpt_sn, Z_xpt_sn = eq_sn.find_xpoint()
psi_at_x = float(eq_sn.psi(np.array([R_xpt_sn]), np.array([Z_xpt_sn]))[0])
print(f'X-point: R={R_xpt_sn:.4f}  Z={Z_xpt_sn:.4f}  psi={psi_at_x:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 7))

ax = axes[0]
cs = ax.contour(Rg_sn, Zg_sn, psi_g_sn,
                levels=np.linspace(0.05, 0.95, 14), cmap='RdYlBu_r', linewidths=0.7)
ax.contour(Rg_sn, Zg_sn, psi_g_sn, levels=[1.0], colors='k', linewidths=2.0)
ax.plot(R_ax_sn, Z_ax_sn, '+k', ms=10, mew=2, label='O-point (axis)')
ax.plot(R_xpt_sn, Z_xpt_sn, 'xr', ms=11, mew=2.5, label='X-point')
ax.set_aspect('equal')
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title("Single-null Solov'ev equilibrium", fontsize=11)
ax.legend(fontsize=9, loc='upper right')
plt.colorbar(cs, ax=ax, label=r'$\psi_{\rm norm}$', shrink=0.7)

# -----------------------------------------------------------------------
# 3. Field-line ODE for the single-null equilibrium
# -----------------------------------------------------------------------
def field_func_sn(rzphi):
    R, Z = float(rzphi[0]), float(rzphi[1])
    BR0, BZ0 = eq_sn.BR_BZ(np.array([R]), np.array([Z]))
    Bphi0 = eq_sn.Bphi(np.array([R]))
    BR_t = float(BR0[0]); BZ_t = float(BZ0[0]); Bphi_t = float(Bphi0[0])
    B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bphi_t**2) + 1e-30
    return np.array([BR_t/B_mag, BZ_t/B_mag, Bphi_t/(R*B_mag)])

def field_func_2d_sn(R, Z, phi):
    tang = field_func_sn(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

# -----------------------------------------------------------------------
# 4. Separatrix field-line tracing (illustration — no manifold for divertor)
# The separatrix is defined by psi = psi_sep; trace a few field lines near it.
# -----------------------------------------------------------------------
phi_span_sn = (0.0, 2.0 * np.pi)   # 1 toroidal turn

# Sample a few points just inside and outside the separatrix
psi_sep = psi_at_x
R_test = np.linspace(eq_sn.R0 - 0.4, eq_sn.R0 + 0.4, 6)
Z_test = np.zeros(len(R_test))

ax2 = axes[1]
cs2 = ax2.contour(Rg_sn, Zg_sn, psi_g_sn,
                  levels=np.linspace(0.1, 0.92, 10), cmap='Blues_r', linewidths=0.6, alpha=0.5)
ax2.contour(Rg_sn, Zg_sn, psi_g_sn, levels=[1.0], colors='k', linewidths=1.8, label='Separatrix')

# Compute Jacobian at the magnetic axis region to show monodromy structure
# Use an O-point inside the plasma (q profile demonstration)
R_otest = eq_sn.R0 + 0.2 * eq_sn.a
Z_otest = 0.0
vq_sn = PoincareMapVariationalEquations(field_func_2d_sn, fd_eps=1e-5)
Jac_test = vq_sn.jacobian_matrix(
    np.array([R_otest, Z_otest]), phi_span_sn,
    solve_ivp_kwargs=dict(method='RK45', rtol=1e-6, atol=1e-8)
)
lam_test = np.linalg.eigvals(Jac_test)
det_test = np.linalg.det(Jac_test)
print(f'Monodromy at R={R_otest:.3f}: det(J)={det_test:.6f}  |lam|={sorted(np.abs(lam_test))}')

# Trace a Poincare section near mid-radius to show nested surfaces
from pyna.topo.poincare import poincare_from_fieldlines as _pfl_sn
from pyna.topo.poincare import ToroidalSection

n_sn_lines = 8
psi_test_vals = np.linspace(0.1, 0.85, n_sn_lines)
R_starts_sn = eq_sn.R0 + np.sqrt(psi_test_vals) * eq_sn.a
starts_sn = np.column_stack([R_starts_sn, np.zeros(n_sn_lines), np.zeros(n_sn_lines)])

section_sn = ToroidalSection(0.0)
pmap_sn = _pfl_sn(field_func_sn, starts_sn, sections=[section_sn],
                  t_max=60.0, dt=0.04)
pts_sn = pmap_sn.crossing_array(0)
print(f'Poincare crossings: {len(pts_sn)}')

if len(pts_sn) > 0:
    psi_pts_sn = eq_sn.psi(pts_sn[:, 0], pts_sn[:, 1]).flatten()
    ax2.scatter(pts_sn[:, 0], pts_sn[:, 1],
                c=np.clip(psi_pts_sn, 0, 1), s=0.8, cmap='plasma',
                vmin=0, vmax=1, rasterized=True, alpha=0.7, zorder=3)

ax2.plot(R_ax_sn, Z_ax_sn, '+k', ms=10, mew=2)
ax2.plot(R_xpt_sn, Z_xpt_sn, 'xr', ms=11, mew=2.5, label='X-point')
ax2.set_aspect('equal')
ax2.set_xlabel('R (m)'); ax2.set_ylabel('Z (m)')
ax2.set_title(r"Single-null: Poincaré section ($\phi=0$)", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_xlim(R_range_sn)
ax2.set_ylim(Z_range_sn)

plt.suptitle("Single-null divertor Solov'ev equilibrium", fontsize=12)
plt.tight_layout()
plt.savefig('solovev_single_null_equilibrium.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'det(J) at mid-radius test point = {det_test:.6f}  (should be ≈ 1)')
"""

nb['cells'][23]['source'] = NEW_CELL23
with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Rewrote cell[23]')
