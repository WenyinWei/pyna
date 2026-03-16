import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pyna.MCF.equilibrium.Solovev import solovev_single_null
from pyna.topo.variational import PoincareMapVariationalEquations, _fd_jacobian
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
# 2. Plot the separatrix
# -----------------------------------------------------------------------
R_range_sn = (eq_sn.R0 - 1.5 * eq_sn.a, eq_sn.R0 + 1.5 * eq_sn.a)
Z_range_sn = (-2.0 * eq_sn.kappa * eq_sn.a, 1.5 * eq_sn.kappa * eq_sn.a)

R1d_sn = np.linspace(*R_range_sn, 400)
Z1d_sn = np.linspace(*Z_range_sn, 400)
Rg_sn, Zg_sn = np.meshgrid(R1d_sn, Z1d_sn)
psi_g_sn = eq_sn.psi(Rg_sn, Zg_sn)

fig, ax = plt.subplots(figsize=(5, 8))
cs = ax.contour(Rg_sn, Zg_sn, psi_g_sn,
                levels=np.linspace(0.05, 0.95, 18), cmap='RdYlBu_r', linewidths=0.6)
ax.contour(Rg_sn, Zg_sn, psi_g_sn, levels=[1.0], colors='k', linewidths=2)
# Lower separatrix (psi just above 1 traces the X legs)
ax.contour(Rg_sn, Zg_sn, psi_g_sn, levels=[1.02], colors='gray',
           linewidths=0.8, linestyles='--')
ax.plot(R_ax_sn, Z_ax_sn, '+k', ms=10, mew=2, label='O-point (axis)')
R_xpt_sn, Z_xpt_sn = eq_sn.find_xpoint()
ax.plot(R_xpt_sn, Z_xpt_sn, 'xr', ms=10, mew=2, label='X-point')
print(f'find_xpoint -> R={R_xpt_sn:.4f}  Z={Z_xpt_sn:.4f}')
psi_at_x = float(eq_sn.psi(np.array([R_xpt_sn]), np.array([Z_xpt_sn]))[0])
print(f'psi at X-point = {psi_at_x:.4f}  (LCFS = 1.0)')
ax.set_aspect('equal'); ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title("Single-null Solov'ev equilibrium")
ax.legend(fontsize=9, loc='upper right')
plt.colorbar(cs, ax=ax, label=r'$\psi_{\rm norm}$')
plt.tight_layout()
plt.savefig('solovev_single_null_equilibrium.png', dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------
# 3. Build field_func_2d for the single-null equilibrium
# -----------------------------------------------------------------------
def field_func_sn(rzphi):
    """Unit tangent [dR/ds, dZ/ds, dphi/ds] for field-line ODE."""
    R, Z = float(rzphi[0]), float(rzphi[1])
    BR0, BZ0 = eq_sn.BR_BZ(np.array([R]), np.array([Z]))
    Bphi0 = eq_sn.Bphi(np.array([R]))
    BR_t = float(BR0[0]); BZ_t = float(BZ0[0]); Bphi_t = float(Bphi0[0])
    B_mag = np.sqrt(BR_t**2 + BZ_t**2 + Bphi_t**2) + 1e-30
    return np.array([BR_t/B_mag, BZ_t/B_mag, Bphi_t/(R*B_mag)])

def field_func_2d_sn(R, Z, phi):
    """(dR/dphi, dZ/dphi) for variational equations."""
    tang = field_func_sn(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

# -----------------------------------------------------------------------
# 4. Newton-refine the separatrix X-point to machine precision
# -----------------------------------------------------------------------
def refine_xpoint_sn(x0, ff2d, phi_span, n_iter=12, tol=1e-11):
    """Newton iteration to pin the separatrix X-point."""
    x = np.asarray(x0, dtype=float)
    for i in range(n_iter):
        y0 = np.concatenate([x, np.eye(2).flatten()])
        def aug_rhs(phi, state, _ff=ff2d):
            rz = state[:2]; M = state[2:].reshape(2, 2)
            f = np.asarray(_ff(rz[0], rz[1], phi), dtype=float)
            A = _fd_jacobian(_ff, rz, phi, eps=1e-7)
            return np.concatenate([f, (A @ M).flatten()])
        sol = solve_ivp(aug_rhs, phi_span, y0,
                        method='DOP853', rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise RuntimeError(f'Newton iter {i}: {sol.message}')
        x_end = sol.y[:2, -1]; Jac = sol.y[2:, -1].reshape(2, 2)
        res = x_end - x
        print(f'  Newton iter {i}: |res|={np.linalg.norm(res):.2e}  det(J)={np.linalg.det(Jac):.8f}')
        if np.linalg.norm(res) < tol:
            break
        try:
            x = x + np.linalg.solve(Jac - np.eye(2), -res)
        except np.linalg.LinAlgError:
            break
    return x, Jac

# The separatrix X-point is a fixed point of the 1-turn map (q~1.5, so
# we need multiple turns; the true hyperbolic fixed point of the Poincare map
# requires n_turns such that n_turns * 2pi covers the X-point period.
# Near the separatrix q -> infinity, so use the actual Poincare map directly.
# For the divertor X-point, field lines near it wind many times - treat as
# 1-turn map and look for the fixed point of 1 full turn.
phi_span_sn = (0.0, 2.0 * np.pi)   # 1 toroidal turn

print('Refining separatrix X-point with Newton iteration...')
try:
    xpt_sn, Jac_sn = refine_xpoint_sn(
        [R_xpt_sn, Z_xpt_sn], field_func_2d_sn, phi_span_sn
    )
    lam_abs = sorted(np.abs(np.linalg.eigvals(Jac_sn)))
    det_err = abs(np.linalg.det(Jac_sn) - 1.0)
    print(f'Refined X-point: R={xpt_sn[0]:.6f}  Z={xpt_sn[1]:.6f}')
    print(f'lambda_u={lam_abs[1]:.4f}  lambda_s={lam_abs[0]:.4f}  det_err={det_err:.2e}')
    xpt_refined = True
except Exception as e:
    print(f'Newton refinement failed: {e}')
    xpt_sn = np.array([R_xpt_sn, Z_xpt_sn])
    vq_sn = PoincareMapVariationalEquations(field_func_2d_sn, fd_eps=1e-6)
    Jac_sn = vq_sn.jacobian_matrix(xpt_sn, phi_span_sn)
    xpt_refined = False

# -----------------------------------------------------------------------
# 5. Grow stable and unstable manifolds
# -----------------------------------------------------------------------
_ivp_kw = dict(rtol=1e-9, atol=1e-12)
RZlimit_sn = (
    eq_sn.R0 - 1.6 * eq_sn.a, eq_sn.R0 + 1.6 * eq_sn.a,
    -2.2 * eq_sn.kappa * eq_sn.a, 1.6 * eq_sn.kappa * eq_sn.a,
)

sm_sn = StableManifold(xpt_sn, Jac_sn, field_func_2d_sn, phi_span=phi_span_sn)
um_sn = UnstableManifold(xpt_sn, Jac_sn, field_func_2d_sn, phi_span=phi_span_sn)

sm_sn.grow(n_turns=6, init_length=5e-5, n_init_pts=5, both_sides=True,
           RZlimit=RZlimit_sn, **_ivp_kw)
um_sn.grow(n_turns=6, init_length=5e-5, n_init_pts=5, both_sides=True,
           RZlimit=RZlimit_sn, **_ivp_kw)

print(f'Stable manifold: {len(sm_sn.segments)} segments')
print(f'Unstable manifold: {len(um_sn.segments)} segments')

# -----------------------------------------------------------------------
# 6. Final figure: separatrix + manifolds
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 9))

# psi contours
ax.contour(Rg_sn, Zg_sn, psi_g_sn,
           levels=np.linspace(0.05, 0.92, 16), colors='lightgray', linewidths=0.5)
ax.contour(Rg_sn, Zg_sn, psi_g_sn, levels=[1.0], colors='k', linewidths=1.8)

# Stable manifold (teal)
for seg in sm_sn.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='GnBu', lw=1.4)
        lc.set_alpha(0.92); lc.set_zorder(6)
        ax.add_collection(lc)

# Unstable manifold (orange)
for seg in um_sn.segments:
    if len(seg) >= 2:
        lc, _ = _manifold_line_collection(seg, cmap='Oranges', lw=1.4)
        lc.set_alpha(0.92); lc.set_zorder(6)
        ax.add_collection(lc)

# X-point and O-point
ax.plot(*xpt_sn, 'xr', ms=12, mew=2.5, zorder=10, label='Separatrix X-point')
ax.plot(R_ax_sn, Z_ax_sn, '+k', ms=10, mew=2, zorder=10, label='O-point (axis)')

# Legend
mfld_handles = manifold_legend_handles('Oranges', 'GnBu')
handles_all = [plt.Line2D([0],[0], marker='x', color='r', ms=9, mew=2, ls='none', label='Separatrix X-point'),
               plt.Line2D([0],[0], marker='+', color='k', ms=9, mew=2, ls='none', label='O-point (axis)')] + mfld_handles
ax.legend(handles=handles_all, fontsize=8, loc='upper right')

ax.set_xlim(R_range_sn)
ax.set_ylim(Z_range_sn)
ax.set_aspect('equal')
ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')
ax.set_title("Single-null divertor: separatrix $W^s$/$W^u$ manifolds", fontsize=12)
plt.tight_layout()
plt.savefig('solovev_single_null_manifolds.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: solovev_single_null_manifolds.png')
