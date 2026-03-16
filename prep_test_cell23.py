import json
import subprocess, sys

with open('notebooks/tutorials/rmp_island_validation_solovev.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Get cell[23] source
cell = nb['cells'][23]
src = ''.join(cell['source'])
print('Source preview:')
print(src[:200].encode('ascii','replace').decode())
print('...')

# Write to a test script
with open('_test_cell23.py', 'w', encoding='utf-8') as f:
    # Add matplotlib non-interactive backend
    f.write("import matplotlib\nmatplotlib.use('Agg')\n")
    f.write("import numpy as np\nimport matplotlib.pyplot as plt\n")
    f.write(src)

print('Written to _test_cell23.py')
