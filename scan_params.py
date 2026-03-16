import json

nbs = {
    'island_control': 'notebooks/tutorials/stellarator_island_control.ipynb',
    'solovev': 'notebooks/tutorials/rmp_island_validation_solovev.ipynb',
}

for name, path in nbs.items():
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    print(f'\n=== {name} ===')
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code': continue
        src = ''.join(cell['source'])
        # Look for computation-heavy lines
        lines = src.split('\n')
        for ln in lines:
            ln_s = ln.strip()
            if any(k in ln_s for k in ['N_TRANSITS', 'n_turns=', 'n_restarts=', 'n_weights=',
                                        'n_theta=', 'n_phi=', 'N_COILS', 'n_lines=', 'n_grid=',
                                        '_build_response', 'grow(', 'pareto_scan', 'poincare_from']):
                print(f'  cell[{i}]: {ln_s[:100]}')
