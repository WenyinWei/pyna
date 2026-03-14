import json
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
img_count = 0
errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'display_data' or out.get('output_type') == 'execute_result':
                if 'image/png' in out.get('data', {}):
                    img_count += 1
                    print(f'Cell {i}: image/png found')
            if out.get('output_type') == 'error':
                errors.append(f'Cell {i}: {out.get("ename")}: {out.get("evalue")}')
print(f'Total images: {img_count}')
if errors:
    print('ERRORS:')
    for e in errors:
        print(e)
