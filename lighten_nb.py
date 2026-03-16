import json

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        patched.append(cell)
        continue
    src = ''.join(cell['source'])

    # 1. Reduce N_COILS from 16 to 4 for CI speed
    if 'N_COILS = 16' in src:
        src = src.replace('N_COILS = 16', 'N_COILS = 4  # reduced for CI speed')
        print(f'Patched N_COILS in cell[{i}]')

    # 2. Reduce N_TRANSITS for Poincare (already 60 but reduce to 30 for CI)
    if 'N_TRANSITS = 60' in src:
        src = src.replace('N_TRANSITS = 60', 'N_TRANSITS = 30  # reduced for CI')
        print(f'Patched N_TRANSITS in cell[{i}]')

    # 3. Manifold n_turns 5 -> 3
    if 'n_turns=5' in src and 'StableManifold' in src or 'grow(' in src:
        src = src.replace('n_turns=5', 'n_turns=3')
        print(f'Patched manifold n_turns in cell[{i}]')

    # 4. n_theta/n_phi in mn_spectrum cells: 24->12
    if 'n_theta=24, n_phi=24' in src:
        src = src.replace('n_theta=24, n_phi=24', 'n_theta=12, n_phi=12')
        print(f'Patched mn_spectrum resolution in cell[{i}]')

    cell = dict(cell)
    cell['source'] = src
    patched.append(cell)

nb['cells'] = patched
with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
