import json
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])[:400]
    print(f'=== Cell {i} ({ct}) ===')
    print(src)
    print()
