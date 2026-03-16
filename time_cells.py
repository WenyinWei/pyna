import subprocess, time, json, sys, os

nb_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/tutorials/stellarator_island_control.ipynb'

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print(f'Total code cells: {len(code_cells)} in {nb_path}')

header = "import matplotlib\nmatplotlib.use('Agg')\n"
cumulative = header
total_time = 0.0

for idx, (cell_idx, cell) in enumerate(code_cells):
    src = ''.join(cell['source'])
    cumulative_new = cumulative + '\n# === CELL ' + str(cell_idx) + ' ===\n' + src + '\n'

    script_path = f'_tc_{idx}.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(cumulative_new)

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=90,
            cwd=os.getcwd(),
            encoding='utf-8', errors='replace',
        )
        dt = time.time() - t0
        total_time += dt
        status = 'OK' if result.returncode == 0 else 'FAIL'
        print(f'cell[{cell_idx}] ({idx+1}/{len(code_cells)}) {dt:.1f}s {status}  cumulative={total_time:.0f}s')
        if result.returncode != 0:
            stderr_clean = result.stderr.replace('\r','')
            # Show last real traceback lines
            lines = [l for l in stderr_clean.split('\n') if l.strip() and 
                     not l.startswith('py :') and not l.startswith('~') and
                     not 'CategoryInfo' in l and not 'FullyQualified' in l]
            print('  ERROR:', '\n  '.join(lines[-8:]))
            print('  STOPPING.')
            break
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        total_time += dt
        print(f'cell[{cell_idx}] ({idx+1}/{len(code_cells)}) TIMEOUT (>90s)  cumulative={total_time:.0f}s')
        print('  STOPPING - cell too slow.')
        break

    cumulative = cumulative_new

# Cleanup
for idx2 in range(idx + 1):
    try:
        os.remove(f'_tc_{idx2}.py')
    except:
        pass

print(f'\nTotal measured time: {total_time:.1f}s')
