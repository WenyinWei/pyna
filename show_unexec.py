import json

with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
# Find the unexecuted cell
for i, cell in code_cells:
    if cell.get('execution_count') is None:
        print(f'Unexecuted cell[{i}]:')
        print(''.join(cell['source']).encode('ascii', 'replace').decode())
