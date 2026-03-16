import json

# Check rmp_island_validation_solovev - which cell is missing
with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print('rmp_island_validation_solovev cell statuses:')
for i, cell in code_cells:
    ec = cell.get('execution_count')
    n_out = len(cell.get('outputs', []))
    src_preview = ''.join(cell['source'])[:60].replace('\n',' ')
    print(f'  cell[{i}] exec={ec} outs={n_out} | {src_preview}')
