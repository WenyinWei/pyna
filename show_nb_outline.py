import json
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print(f'Total code cells: {len(code_cells)}')
for i, cell in code_cells:
    src = ''.join(cell['source'])
    # Print first line and key keywords
    first_line = src.split('\n')[0][:80]
    kw = []
    for k in ['X-point','O-point','island','manifold','phase','optimizer','pareto','response','control']:
        if k.lower() in src.lower(): kw.append(k)
    print(f'  cell[{i}]: {first_line!r}  [{",".join(kw)}]')
