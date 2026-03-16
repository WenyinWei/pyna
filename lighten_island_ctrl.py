import json

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changes = []

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    new = src

    # cell[11]: n_theta=20 -> 8 for b_nat computation  
    new = new.replace('n_theta=20, n_phi=20)', 'n_theta=8, n_phi=8)')

    # cell[12]: coil response n_theta=20 -> 8
    new = new.replace('n_theta=20, n_phi=20\n', 'n_theta=8, n_phi=8\n')

    # cell[16]: phase scan - reduce from 17 points to 9, and n_theta/n_phi
    new = new.replace('np.linspace(0, 2*np.pi, 17)', 'np.linspace(0, 2*np.pi, 9)')
    new = new.replace('I_max=1500.0, n_theta=20, n_phi=20,', 'I_max=1500.0, n_theta=8, n_phi=8,')
    new = new.replace('n_theta=16, n_phi=16', 'n_theta=8, n_phi=8')

    # cell[21]: n_restarts=2 -> 1
    new = new.replace('n_restarts=2,', 'n_restarts=1,')

    # cell[23]: n_weights=5 -> 3  
    new = new.replace('n_weights=5,', 'n_weights=3,')

    if new != src:
        changes.append(i)
        cell['source'] = new

print(f'Patched cells: {changes}')

with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
