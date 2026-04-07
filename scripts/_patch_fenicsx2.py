content = open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\MCF\equilibrium\fenicsx_corrector.py', encoding='utf-8').read()

# Add petsc_solver param to solve_force_balance_correction
old_sig = '''def solve_force_balance_correction(
    B_pred_RZPhi_2d,
    p_2d,
    R_arr,
    Z_arr,
    mu0: float = MU0_DEFAULT,
    eps_reg: float = 1e-6,
    lambda_div: float = 1.0,
    max_iter: int = 3,
    anderson_depth: int = 5,
    anderson_beta: float = 0.8,
    use_line_search: bool = True,
    diverge_tol: float = 2.0,
):'''
new_sig = '''def solve_force_balance_correction(
    B_pred_RZPhi_2d,
    p_2d,
    R_arr,
    Z_arr,
    mu0: float = MU0_DEFAULT,
    eps_reg: float = 1e-6,
    lambda_div: float = 1.0,
    max_iter: int = 3,
    anderson_depth: int = 5,
    anderson_beta: float = 0.8,
    use_line_search: bool = True,
    diverge_tol: float = 2.0,
    petsc_solver: str = "gamg",
):'''
assert old_sig in content, 'old_sig not found'
content = content.replace(old_sig, new_sig, 1)

# Pass petsc_solver to solve_linearised_fb
old_call = '''        delta = solve_linearised_fb(B_curr, J_curr, r_R, r_Z, R_arr, Z_arr,
                                    eps_reg, lambda_div=lambda_div, mu0=mu0)'''
new_call = '''        delta = solve_linearised_fb(B_curr, J_curr, r_R, r_Z, R_arr, Z_arr,
                                    eps_reg, lambda_div=lambda_div, mu0=mu0,
                                    petsc_solver=petsc_solver)'''
assert old_call in content, 'old_call not found'
content = content.replace(old_call, new_call, 1)

open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\MCF\equilibrium\fenicsx_corrector.py', 'w', encoding='utf-8').write(content)
print('OK: petsc_solver param propagated')
