import json, re

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    new = src.replace(
        'n_lines=4,  # fewer near-resonance lines to keep total time low delta_psi=0.06)',
        'n_lines=4, delta_psi=0.06)'
    )
    if new != src:
        cell['source'] = new
        print(f'Fixed syntax in cell[{i}]')

with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
