import json, re

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i in [5, 11, 12, 14, 16, 20, 21, 23, 25, 26]:
    cell = nb['cells'][i]
    src = ''.join(cell['source'])
    print(f'\n=== cell[{i}] ===')
    print(src[:1200])
    print('...' if len(src) > 1200 else '')
