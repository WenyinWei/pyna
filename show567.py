import json
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for i in [5, 6, 7]:
    cell = nb['cells'][i]
    print(f'=== cell[{i}] ===')
    print(''.join(cell['source']).encode('ascii','replace').decode())
    print()
