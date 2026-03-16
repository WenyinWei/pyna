import json

with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    if 't_max=150.0  # ~24 turns' in src:
        new = src.replace('t_max=150.0  # ~24 turns, enough for Poincare pattern', 't_max=150.0')
        cell['source'] = new
        print(f'Fixed syntax in cell[{i}]')

with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
