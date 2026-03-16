import json
with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'error':
            errors.append(f"  cell[{i}] {out.get('ename')}: {out.get('evalue','')[:80]}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
exec_count = sum(1 for c in code_cells if c.get('execution_count'))
print(f'Executed {exec_count}/{len(code_cells)}')
for e in errors:
    print(e)
