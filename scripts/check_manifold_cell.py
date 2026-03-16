import sys, json
sys.stdout.reconfigure(encoding='utf-8')
with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
for i, c in enumerate(code_cells):
    src = ''.join(c['source'])
    if 'find_cycle' in src or 'StableManifold' in src:
        print(f'Code cell {i}:')
        print(src[:1500])
        print('--- outputs:', len(c.get('outputs',[])))
        print()
