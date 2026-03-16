import json, os, glob

nbs = [
    'notebooks/tutorials/rmp_resonance_analysis.ipynb',
    'notebooks/tutorials/magnetic_coordinates_comparison.ipynb',
    'notebooks/tutorials/stellarator_island_control.ipynb',
    'notebooks/tutorials/rmp_island_validation_solovev.ipynb',
    'notebooks/tutorials/island_jacobian_analysis.ipynb',
    'notebooks/tutorials/stellarator_multiSection_q41.ipynb',
]

for nb_path in nbs:
    if not os.path.exists(nb_path):
        print(f'MISSING: {nb_path}')
        continue
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    exec_count = sum(1 for c in code_cells if c.get('execution_count'))
    
    # Sum execution times from metadata
    total_time = 0.0
    times = []
    for cell in code_cells:
        t = cell.get('metadata', {}).get('execution', {})
        if t:
            # jupyter stores iopub_execute_input -> shell_execute_reply
            import datetime
            try:
                start = t.get('iopub_execute_input') or t.get('shell_execute_reply')
                end = t.get('shell_execute_reply') or t.get('iopub_execute_input')
                # just show what's there
                times.append(str(t))
            except:
                pass
        # Also check nbformat timing
        timing = cell.get('metadata', {}).get('execution')
        if timing:
            try:
                s = timing.get('iopub_execute_input','')
                e = timing.get('shell_execute_reply','')
                if s and e:
                    from datetime import datetime, timezone
                    t0 = datetime.fromisoformat(s.replace('Z','+00:00'))
                    t1 = datetime.fromisoformat(e.replace('Z','+00:00'))
                    dt = (t1 - t0).total_seconds()
                    total_time += dt
            except:
                pass
    
    errors = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code': continue
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                errors.append(f"  cell[{i}] {out.get('ename')}: {out.get('evalue','')[:60]}")
    
    status = 'ERRORS' if errors else ('OK' if exec_count == len(code_cells) else f'PARTIAL {exec_count}/{len(code_cells)}')
    print(f'{status:25s} total={total_time:.0f}s  {nb_path}')
    for e in errors:
        print(e)
