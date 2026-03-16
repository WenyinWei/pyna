import json
with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cell = nb['cells'][23]
print(''.join(cell['source']))
