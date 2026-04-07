content = open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\MCF\equilibrium\fenicsx_corrector.py', encoding='utf-8').read()

old_sig = 'def solve_linearised_fb(B_curr, J_curr, r_R, r_Z, R_arr, Z_arr, eps_reg,\n                        lambda_div=1.0, mu0=MU0_DEFAULT):'
new_sig = 'def solve_linearised_fb(B_curr, J_curr, r_R, r_Z, R_arr, Z_arr, eps_reg,\n                        lambda_div=1.0, mu0=MU0_DEFAULT,\n                        petsc_solver: str = "gamg"):'
assert old_sig in content, f'sig not found'
content = content.replace(old_sig, new_sig, 1)

old_petsc = '    problem = LinearProblem(\n        a_form, L_form,\n        bcs=[],\n        petsc_options_prefix="fenicsx_mhd_",\n        petsc_options={"ksp_type": "gmres", "pc_type": "ilu",\n                       "ksp_rtol": 1e-8, "ksp_max_it": 500},\n    )'
new_petsc = '''    # Build PETSc solver options (gamg=algebraic multigrid, ilu=fallback)
    if petsc_solver == "gamg":
        _petsc_opts = {
            "ksp_type": "gmres",
            "pc_type": "gamg",
            "pc_gamg_type": "agg",
            "pc_gamg_agg_nsmooths": 1,
            "ksp_rtol": 1e-8,
            "ksp_max_it": 500,
            "ksp_gmres_restart": 50,
        }
    elif petsc_solver == "hypre":
        _petsc_opts = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 500,
            "ksp_gmres_restart": 50,
        }
    else:
        # "ilu" or unknown -- fallback
        if petsc_solver not in ("ilu", "gamg", "hypre"):
            import warnings
            warnings.warn(
                f"Unknown petsc_solver={petsc_solver!r}; falling back to ILU.",
                stacklevel=3,
            )
        _petsc_opts = {
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 500,
        }

    problem = LinearProblem(
        a_form, L_form,
        bcs=[],
        petsc_options_prefix="fenicsx_mhd_",
        petsc_options=_petsc_opts,
    )'''

assert old_petsc in content, 'old_petsc not found'
content = content.replace(old_petsc, new_petsc, 1)

open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\MCF\equilibrium\fenicsx_corrector.py', 'w', encoding='utf-8').write(content)
print('fenicsx_corrector.py: GAMG preconditioner added OK')
