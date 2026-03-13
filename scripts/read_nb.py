import json
nbs = [
    r'D:\Repo\MHDpy\nb\field_calc_visual\visual_equilibrium_in_EAST_Bifurcated_Manifold_py313.ipynb',
    r'D:\Repo\MHDpy\nb\field_calc_visual\visual_equilibrium_in_EAST_Jac_Manifold.ipynb',
    r'D:\Repo\MHDpy\nb\field_calc_visual\visual_equilibrium_in_EAST_divertor_Lc.ipynb',
]
for nb in nbs:
    with open(nb, encoding='utf-8') as f:
        data = json.load(f)
    cells = data['cells']
    print(f'\n=== {nb.split(chr(92))[-1]}: {len(cells)} cells ===')
    for i, c in enumerate(cells):
        src = ''.join(c['source'])
        ctype = c['cell_type']
        print(f'[{i}] {ctype}: {src[:400]}')
        print()
