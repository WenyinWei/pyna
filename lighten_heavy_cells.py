import json

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    new = src

    if i == 5:
        # Reduce field lines: 24 radial -> 10, n_lines=12 -> 6, N_TRANSITS 20 -> 8
        new = new.replace('N_TRANSITS = 20  # reduced for CI', 'N_TRANSITS = 8  # tutorial: 8 turns is enough to show island structure')
        new = new.replace(
            'R_starts = np.linspace(stella.R0 + 0.02*stella.r0, stella.R0 + 0.93*stella.r0, 24)',
            'R_starts = np.linspace(stella.R0 + 0.02*stella.r0, stella.R0 + 0.93*stella.r0, 10)  # 10 radial lines'
        )
        new = new.replace('n_lines=12,', 'n_lines=6,')

    if i == 7:
        # Reduce Newton refinement iterations, use faster RK45 with looser tol
        new = new.replace(
            'def refine_xpoint(x0, field_func_2d, phi_span, n_iter=4, tol=1e-7):',
            'def refine_xpoint(x0, field_func_2d, phi_span, n_iter=2, tol=1e-5):'
        )
        new = new.replace(
            "method='RK45', rtol=1e-8, atol=1e-10, dense_output=False)",
            "method='RK45', rtol=1e-6, atol=1e-8, dense_output=False)"
        )
        # Only try 1 candidate (the first one), not all 4
        new = new.replace(
            'for R_c, Z_c in _xpt_candidates:',
            'for R_c, Z_c in _xpt_candidates[:1]:  # tutorial: try 1 seed only'
        )

    if new != src:
        cell['source'] = new
        print(f'Patched cell[{i}]')

with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
