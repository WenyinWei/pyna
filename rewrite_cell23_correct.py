import json

with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

NEW_CELL23 = r"""# =========================================================================
# Single-null divertor equilibrium: separatrix X-point & stable/unstable manifolds
#
# The divertor X-point IS a fixed point of the 1-turn Poincaré map (phi: 0→2π).
# A field line starting on the X-point returns to the same (R,Z) after one
# toroidal transit.  The monodromy matrix J has eigenvalues (λ_u, λ_s) with
# λ_u·λ_s = 1 (area-preserving), |λ_u|>1, |λ_s|<1  →  hyperbolic fixed point.
# =========================================================================

from pyna.MCF.equilibrium.Solovev import solovev_single_null
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.MCF.visual.tokamak_manifold import _manifold_line_collection

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
# 2. Locate X-point (analytic from equilibrium)
# -----------------------------------------------------------------------
R_xpt_sn, Z_xpt_sn = eq_sn.find_xpoint()
psi_at_x = float(eq_sn.psi(np.array([R_xpt_sn]), np.array([Z_xpt_sn]))[0])
print(f'X-point: R={R_xpt_sn:.6f}  Z={Z_xpt_sn:.6f}  psi={psi_at_x:.6f}')

# -----------------------------------------------------------------------
# 3. Define 2-D field-line ODE  dR/dphi, dZ/dphi
# -----------------------------------------------------------------------
def field_func_sn(rzphi):
    R, Z = float(rzphi[0]), float(rzphi[1])
    BR, BZ = eq_sn.BR_BZ(np.array([R]), np.array([Z]))
    Bphi  = eq_sn.Bphi(np.array([R]))
    BR_t, BZ_t, Bp_t = float(BR[0]), float(BZ[0]), float(Bphi[0])
    B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bp_t**2) + 1e-30
    return np.array([BR_t/B_mag, BZ_t/B_mag, Bp_t/(R*B_mag)])

def field_func_2d_sn(R, Z, phi):
    t = field_func_sn(np.array([R, Z, phi]))
    if abs(t[2]) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([t[0]/t[2], t[1]/t[2]])

# -----------------------------------------------------------------------
# 4. Compute monodromy matrix J at the X-point (1 toroidal turn)
#    rtol=1e-8 gives det(J)≈1 in <0.3 s; no Newton refinement needed
#    because eq_sn.find_xpoint() already returns the exact analytic location.
# -----------------------------------------------------------------------
phi_span_sn = (0.0, 2.0 * np.pi)

vq_sn = PoincareMapVariationalEquations(field_func_2d_sn, fd_eps=1e-6)
xpt_sn = np.array([R_xpt_sn, Z_xpt_sn])
Jac_sn = vq_sn.jacobian_matrix(
    xpt_sn, phi_span_sn,
    solve_ivp_kwargs=dict(method='RK45', rtol=1e-8, atol=1e-10),
)

lam_sn = np.linalg.eigvals(Jac_sn)
lam_abs_sn = sorted(np.abs(lam_sn))
det_sn = np.linalg.det(Jac_sn)
print(f'det(J) = {det_sn:.8f}  (ideal = 1.0 for area-preserving)')
print(f'|lambda_stable|   = {lam_abs_sn[0]:.6f}')
print(f'|lambda_unstable| = {lam_abs_sn[1]:.6f}')

# -----------------------------------------------------------------------
# 5. Grow stable / unstable manifolds (1 turn for tutorial speed)
# -----------------------------------------------------------------------
RZlimit_sn = (eq_sn.R0 - 1.6*eq_sn.a, eq_sn.R0 + 1.6*eq_sn.a,
              -2.2*eq_sn.kappa*eq_sn.a, 1.8*eq_sn.kappa*eq_sn.a)

sm_sn = StableManifold(xpt_sn, Jac_sn, field_func_2d_sn, phi_span=phi_span_sn)
um_sn = UnstableManifold(xpt_sn, Jac_sn, field_func_2d_sn, phi_span=phi_span_sn)

ivp_kw = dict(rtol=1e-7, atol=1e-9)
sm_sn.grow(n_turns=2, init_length=1e-4, n_init_pts=3, both_sides=True,
           RZlimit=RZlimit_sn, **ivp_kw)
um_sn.grow(n_turns=2, init_length=1e-4, n_init_pts=3, both_sides=True,
           RZlimit=RZlimit_sn, **ivp_kw)

print(f'Stable   segments: {len(sm_sn.segments)}')
print(f'Unstable segments: {len(um_sn.segments)}')

# -----------------------------------------------------------------------
# 6. Plot: equilibrium + X-point manifolds
# -----------------------------------------------------------------------
R_range_sn = (eq_sn.R0 - 1.6*eq_sn.a, eq_sn.R0 + 1.6*eq_sn.a)
Z_range_sn = (-2.2*eq_sn.kappa*eq_sn.a, 1.8*eq_sn.kappa*eq_sn.a)
R1d_sn = np.linspace(*R_range_sn, 300)
Z1d_sn = np.linspace(*Z_range_sn, 300)
Rg_sn, Zg_sn = np.meshgrid(R1d_sn, Z1d_sn)
psi_g_sn = eq_sn.psi(Rg_sn, Zg_sn)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Left: flux contours
ax = axes[0]
cs = ax.contour(Rg_sn, Zg_sn, psi_g_sn,
                levels=np.linspace(0.05, 0.95, 14), cmap='RdYlBu_r', linewidths=0.7)
ax.contour(Rg_sn, Zg_sn, psi_g_sn, levels=[1.0], colors='k', linewidths=2.0)
ax.plot(R_ax_sn, Z_ax_sn,  '+k', ms=10, mew=2, label='O-point (axis)')
ax.plot(R_xpt_sn, Z_xpt_sn, 'xr', ms=11, mew=2.5, label='X-point')
ax.set_aspect('equal'); ax.set_xlim(R_range_sn); ax.set_ylim(Z_range_sn)
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title("Single-null Solov'ev: flux surfaces", fontsize=11)
ax.legend(fontsize=9); plt.colorbar(cs, ax=ax, label=r'$\psi_{\rm norm}$', shrink=0.7)

# Right: manifolds
ax2 = axes[1]
ax2.contour(Rg_sn, Zg_sn, psi_g_sn,
            levels=np.linspace(0.1, 0.9, 10), colors='lightgray', linewidths=0.5)
ax2.contour(Rg_sn, Zg_sn, psi_g_sn, levels=[1.0], colors='k', linewidths=1.5)

s_ref_s = max((np.ptp(seg[:, 0]) for seg in sm_sn.segments if len(seg) > 1), default=1.0)
for seg in sm_sn.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='GnBu', s_ref=s_ref_s, lw=1.5)
        ax2.add_collection(lc)

s_ref_u = max((np.ptp(seg[:, 0]) for seg in um_sn.segments if len(seg) > 1), default=1.0)
for seg in um_sn.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='Oranges', s_ref=s_ref_u, lw=1.5)
        ax2.add_collection(lc)

ax2.plot(R_xpt_sn, Z_xpt_sn, 'kx', ms=12, mew=2.5, zorder=10, label='X-point')
# dummy lines for legend
ax2.plot([], [], color='steelblue', lw=2, label=r'$W^s$ (stable)')
ax2.plot([], [], color='darkorange', lw=2, label=r'$W^u$ (unstable)')
ax2.set_aspect('equal'); ax2.set_xlim(R_range_sn); ax2.set_ylim(Z_range_sn)
ax2.set_xlabel('R (m)'); ax2.set_ylabel('Z (m)')
ax2.set_title(fr'Separatrix manifolds  det(J)={det_sn:.6f}', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')

plt.suptitle("Single-null X-point: stable/unstable manifolds of Poincaré map", fontsize=12)
plt.tight_layout()
plt.savefig('solovev_single_null_manifolds.png', dpi=120, bbox_inches='tight')
plt.show()
"""

nb['cells'][23]['source'] = NEW_CELL23
with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Rewrote cell[23] with correct X-point manifold computation')
