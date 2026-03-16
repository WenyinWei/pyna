import sys, json
sys.stdout.reconfigure(encoding='utf-8')
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
for c in code_cells:
    src = ''.join(c['source'])
    if 'find_cycle' in src and 'np.linspace(0, 2*np.pi, 24' in src:
        new_src = src.replace(
            'np.linspace(0, 2*np.pi, 24, endpoint=False)',
            'np.linspace(0, 2*np.pi, 12, endpoint=False)'
        )
        c['source'] = [new_src]
        print('Reduced seeds 24->12 in manifold cell')
        break
with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('saved')
