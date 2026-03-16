import json

# Analyze the heavy cells in each notebook
nbs = {
    'stellarator_island_control': 'notebooks/tutorials/stellarator_island_control.ipynb',
    'rmp_island_validation_solovev': 'notebooks/tutorials/rmp_island_validation_solovev.ipynb',
    'rmp_resonance_analysis': 'notebooks/tutorials/rmp_resonance_analysis.ipynb',
}

HEAVY_KWS = ['N_TRANSITS', 'n_turns', 'n_restarts', 'n_weights', 'n_theta', 'n_phi',
             'poincare_from_fieldlines', '_build_response', 'grow(', 'pareto_scan',
             'nbconvert', 'N_COILS', 'range(', 'for ', 'n_lines', 'n_grid']

for name, path in nbs.items():
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    print(f'\n=== {name} ===')
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code': continue
        src = ''.join(cell['source'])
        heavy = []
        for kw in HEAVY_KWS:
            if kw in src:
                import re
                m = re.findall(rf'{kw}\s*[=:(]\s*[\w\.]+', src)
                if m:
                    heavy.extend(m[:2])
        if heavy:
            print(f'  cell[{i}]: {", ".join(set(heavy))}')
