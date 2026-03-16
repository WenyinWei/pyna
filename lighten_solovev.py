import json

# ---- 1. rmp_island_validation_solovev: reduce manifold n_turns ----
with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    new = src
    # cell[21]: n_turns=12 -> 3, n_init_pts=6 -> 3
    new = new.replace('n_turns=12, init_length=5e-5, n_init_pts=6', 'n_turns=3, init_length=1e-4, n_init_pts=3')
    # cell[23]: n_turns=6 -> 3, n_init_pts=5 -> 3
    new = new.replace('n_turns=6, init_length=5e-5, n_init_pts=5', 'n_turns=3, init_length=1e-4, n_init_pts=3')
    if new != src:
        cell['source'] = new
        patched += 1
        print(f'  Patched cell[{i}]')

with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'solovev: patched {patched} cells')
