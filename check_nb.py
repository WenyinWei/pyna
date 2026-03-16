import json
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'error':
            tb_last = out.get('traceback', [''])[-1][:300]
            errors.append((i, out.get('ename',''), out.get('evalue',''), tb_last))
if errors:
    for e in errors:
        print(f'cell[{e[0]}] {e[1]}: {e[2]}')
        print(e[3].encode('ascii','replace').decode())
else:
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    exec_count = sum(1 for c in code_cells if c.get('execution_count'))
    print(f'No errors. Executed {exec_count}/{len(code_cells)} code cells')
