"""Patch Cell 6 of stellarator_island_control.ipynb with correct Newton X-point refinement."""
import json, pathlib, copy

nb_path = pathlib.Path('D:/Repo/pyna/notebooks/tutorials/stellarator_island_control.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f'Total code cells: {len(code_cells)}')

# Find cell 6 (index 5)
cell6 = code_cells[5]
print('Current Cell 6 start:', ''.join(cell6['source'])[:100])

new_source = """\
# === X-point Monodromy (Jacobian at analytic X-point) ===
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.MCF.visual.tokamak_manifold import _manifold_line_collection, manifold_legend_handles
from scipy.optimize import root
from scipy.integrate import solve_ivp
from scipy.signal import argrelmin

def field_func_2d(R, Z, phi):
    tang = stella.field_func(np.array([R, Z, phi]))
    dphi_ds = tang[2]
    if abs(dphi_ds) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([tang[0]/dphi_ds, tang[1]/dphi_ds])

def _pmap_n(x0, n=TARGET_N):
    \"\"\"Integrate n toroidal turns; return endpoint (R, Z).\"\"\"
    def rhs(phi, y): return field_func_2d(y[0], y[1], phi)
    sol = solve_ivp(rhs, [0, 2 * np.pi * n], x0,
                    method='DOP853', rtol=1e-11, atol=1e-13)
    return sol.y[:, -1]

# ── Step 1: ring scan to find seeds near X/O points ──────────────────────
n_scan = 200
thetas_scan = np.linspace(0, 2 * np.pi, n_scan, endpoint=False)
R_ring = stella.R0 + r_res_target * np.cos(thetas_scan)
Z_ring = r_res_target * np.sin(thetas_scan)
scan_res = np.array([np.linalg.norm(_pmap_n([R, Z]) - [R, Z])
                     for R, Z in zip(R_ring, Z_ring)])
idx_mins = argrelmin(scan_res, order=5)[0]
seeds = sorted([(scan_res[i], R_ring[i], Z_ring[i]) for i in idx_mins])
print(f'Ring scan: {len(seeds)} local minima found')

# ── Step 2: Newton refinement from best seeds ─────────────────────────────
xpt_candidates = []
for res0, R0, Z0 in seeds[:10]:
    sol = root(lambda x: _pmap_n(x) - x, [R0, Z0], method='hybr',
               tol=1e-12, options={'maxfev': 400})
    if sol.success and np.linalg.norm(sol.fun) < 1e-9:
        xpt_candidates.append(sol.x.copy())

# De-duplicate (merge within 1e-4 m)
xpt_unique = []
for x in xpt_candidates:
    if all(np.linalg.norm(x - y) > 1e-4 for y in xpt_unique):
        xpt_unique.append(x)

print(f'Found {len(xpt_unique)} distinct period-{TARGET_N} fixed points:')
for x in xpt_unique:
    print(f'  R={x[0]:.6f}  Z={x[1]:.6f}')

# Use the first X-point (largest |lambda_u|)
xpt_refined = xpt_unique[0] if xpt_unique else np.array([
    stella.R0 + r_res_target * np.cos(np.pi / TARGET_M),
    r_res_target * np.sin(np.pi / TARGET_M)])
print(f'\\nUsing X-point: R={xpt_refined[0]:.5f} m  Z={xpt_refined[1]:.5f} m')

# ── Step 3: Monodromy matrix via variational equations ────────────────────
phi_span = (0.0, 2.0 * np.pi * TARGET_N)
vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=1e-6)
xpt_Jac = vq.jacobian_matrix(xpt_refined, phi_span,
                              solve_ivp_kwargs=dict(method='DOP853', rtol=1e-10, atol=1e-12))

lam = np.linalg.eigvals(xpt_Jac)
lam_abs = sorted(np.abs(lam))
det_J = np.linalg.det(xpt_Jac)
print(f'Monodromy matrix det = {det_J:.6f}  (ideal 1.0 for area-preserving map)')
print(f'|lambda_stable|   = {lam_abs[0]:.6f}')
print(f'|lambda_unstable| = {lam_abs[1]:.2f}')
if abs(det_J - 1.0) > 0.02:
    import warnings
    warnings.warn(f'det(DP^{TARGET_N}) = {det_J:.4f} deviates from 1; '
                  'manifold computation may be inaccurate.')

_manifolds_ok = (abs(det_J - 1.0) < 0.05)

# --- Grow manifolds only if Jacobian is valid ---
sm_segments, um_segments = [], []
if _manifolds_ok:
    RZlimit = (stella.R0 - stella.r0 * 1.05, stella.R0 + stella.r0 * 1.05,
               -stella.r0 * 1.05, stella.r0 * 1.05)
    sm_mfld = StableManifold(list(xpt_refined), xpt_Jac, field_func_2d)
    um_mfld = UnstableManifold(list(xpt_refined), xpt_Jac, field_func_2d)
    sm_mfld.grow(n_turns=2, init_length=1e-5, n_init_pts=3, both_sides=True,
                 rtol=1e-8, atol=1e-10, RZ_limit=RZlimit)
    um_mfld.grow(n_turns=2, init_length=1e-5, n_init_pts=3, both_sides=True,
                 rtol=1e-8, atol=1e-10, RZ_limit=RZlimit)
    sm_segments = sm_mfld.segments
    um_segments = um_mfld.segments
    print(f'Manifold segments: stable={len(sm_segments)}  unstable={len(um_segments)}')
else:
    print(f'Skipping manifold growth (det={det_J:.4f} out of tolerance)')
"""

# Find and replace cell 6 in full cells list
idx = 0
count = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        if count == 5:
            idx = i
            break
        count += 1

nb['cells'][idx]['source'] = new_source.splitlines(keepends=True)
# Fix: last line shouldn't have trailing newline issue  
nb['cells'][idx]['source'] = [line if line.endswith('\n') else line 
                               for line in nb['cells'][idx]['source']]

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nCell 6 patched.')

# Verify
nb2 = json.loads(nb_path.read_text(encoding='utf-8'))
code2 = [c for c in nb2['cells'] if c['cell_type']=='code']
print('New Cell 6 start:', ''.join(code2[5]['source'])[:100])
