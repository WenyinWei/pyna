import json
with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][23]
src = ''.join(cell['source'])
print('=== CURRENT cell[23] manifold lines ===')
for ln in src.split('\n'):
    if any(k in ln for k in ['grow(', 'n_turns', 'n_init_pts', 'Newton', 'n_iter', 'refine', 'phi_span_sn']):
        print(' ', repr(ln))
