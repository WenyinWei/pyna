import json
import numpy as np

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
cell8 = cells[8]
src = ''.join(cell8['source'])

newton_code = (
    "from scipy.optimize import fsolve\n"
    "\n"
    "def poincare_map_n(x0, n, field_func_2d, phi_per_turn=2*np.pi):\n"
    "    from scipy.integrate import solve_ivp\n"
    "    def rhs(phi, y): return field_func_2d(y[0], y[1], phi)\n"
    "    sol = solve_ivp(rhs, [0, phi_per_turn * n], x0,\n"
    "                    method='RK45', rtol=1e-10, atol=1e-12)\n"
    "    return sol.y[:, -1]\n"
    "\n"
    "def fixed_point_residual(x0):\n"
    "    return poincare_map_n(x0, TARGET_N, field_func_2d) - x0\n"
    "\n"
    'xpt_refined, info, ier, _ = fsolve(fixed_point_residual, xpt_seed, full_output=True)\n'
    'print(f\'Newton converged: {ier==1}, |residual|={np.linalg.norm(info["fvec"]):.2e}\')\n'
    'print(f\'Refined X-point: R={xpt_refined[0]:.6f}  Z={xpt_refined[1]:.6f}\')\n'
    "\n"
)

target = 'phi_span = (0.0, 2.0 * np.pi * TARGET_N)'
new_src = src.replace(target, newton_code + target)

new_src = new_src.replace(
    'xpt_Jac = vq.jacobian_matrix(xpt_seed, phi_span,',
    'xpt_Jac = vq.jacobian_matrix(xpt_refined, phi_span,'
)
new_src = new_src.replace(
    'sm_mfld = StableManifold(xpt_seed, xpt_Jac,',
    'sm_mfld = StableManifold(xpt_refined, xpt_Jac,'
)
new_src = new_src.replace(
    'um_mfld = UnstableManifold(xpt_seed, xpt_Jac,',
    'um_mfld = UnstableManifold(xpt_refined, xpt_Jac,'
)
new_src = new_src.replace(
    'ax.plot(*xpt_seed,',
    'ax.plot(*xpt_refined,'
)

cell8['source'] = [new_src]

with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Cell 8 updated')
print('Replacements check:')
print('  newton code inserted:', 'fixed_point_residual' in new_src)
print('  xpt_refined in jacobian_matrix:', 'jacobian_matrix(xpt_refined' in new_src)
