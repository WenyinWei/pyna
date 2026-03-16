import json, glob, os

nbs = [
    'notebooks/tutorials/rmp_resonance_analysis.ipynb',
    'notebooks/tutorials/magnetic_coordinates_comparison.ipynb',
    'notebooks/tutorials/stellarator_island_control.ipynb',
    'notebooks/tutorials/rmp_island_validation_solovev.ipynb',
    'notebooks/tutorials/island_jacobian_analysis.ipynb',
    'notebooks/tutorials/stellarator_multiSection_q41.ipynb',
]

for nb_path in nbs:
    if not os.path.exists(nb_path):
        print(f'MISSING: {nb_path}')
        continue
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    exec_count = sum(1 for c in code_cells if c.get('execution_count'))
    errors = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code': continue
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                errors.append(f"  cell[{i}] {out.get('ename')}: {out.get('evalue','')[:80]}")
    status = 'OK' if exec_count == len(code_cells) and not errors else ('ERRORS' if errors else f'PARTIAL {exec_count}/{len(code_cells)}')
    print(f'{status:30s} {nb_path}')
    for e in errors:
        print(e)
