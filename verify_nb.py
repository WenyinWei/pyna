import json
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'plot_mn_heatmap' in src and 'b_mn_nat' in src:
            print(f'Heatmap cell found at [{i}]')
print('JSON OK')
