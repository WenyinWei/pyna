import json

# Fix the heavy cells in rmp_island_validation_solovev
with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changes = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    new = src

    # cell[11]: t_max=1500 is 240 toroidal turns - way too heavy for tutorial
    # Replace with Poincare section based approach at much lower turn count
    if 't_max=1500' in new:
        new = new.replace('n_lines = 10', 'n_lines = 6  # reduced for tutorial speed')
        new = new.replace('t_max=1500.0', 't_max=150.0  # ~24 turns, enough for Poincare pattern')
        changes.append(i)

    # cell[13]: filter to points near resonant surface - no change needed

    if new != src:
        cell['source'] = new

print(f'Fixed cells: {changes}')

with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
