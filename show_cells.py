import json
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for i in range(10, len(nb['cells'])):
    cell = nb['cells'][i]
    src = ''.join(cell['source'])
    print(f'=== cell[{i}] type={cell["cell_type"]} ===')
    print(src[:250].encode('ascii','replace').decode())
    print()
