import json
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells_of_interest = [1, 5, 6, 10, 11, 12, 13, 14, 15]
for i in cells_of_interest:
    cell = nb['cells'][i]
    ct = cell['cell_type']
    src = ''.join(cell['source'])
    print(f'=== Cell {i} ({ct}) ===')
    print(src)
    print()
