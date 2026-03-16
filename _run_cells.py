"""Execute cells 0-8 of stellarator_island_control.ipynb and report det(J)."""
import json, sys, os, io
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt

nb_path = 'notebooks/tutorials/stellarator_island_control.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Change working dir to notebook dir so relative file saves work
os.chdir('notebooks/tutorials')

# Execute cells 0-8 in shared namespace
ns = {}
for i, cell in enumerate(cells[:9]):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    print(f'\n=== Executing cell {i} ===')
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception as e:
        print(f'ERROR in cell {i}: {e}')
        import traceback; traceback.print_exc()
        sys.exit(1)

# Report
det_J = ns.get('det_J', None)
if det_J is not None:
    print(f'\n=== RESULT: det(DP^m) = {det_J:.6f} ===')
    assert abs(det_J - 1.0) < 0.01, f'det(J) = {det_J:.4f} is not close to 1.0!'
    print('PASS: det is within 0.01 of 1.0')
else:
    print('WARNING: det_J not found in namespace')
    print('Available keys:', [k for k in ns if not k.startswith('_')])
