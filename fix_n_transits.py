import json

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix N_TRANSITS in cell[3] or cell[5]
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    new = src

    # N_TRANSITS: increase to 60, but reduce field lines to compensate
    if 'N_TRANSITS' in new:
        import re
        new = re.sub(r'N_TRANSITS\s*=\s*\d+', 'N_TRANSITS = 60  # 60 turns needed to see island structure', new)
        print(f'cell[{i}]: set N_TRANSITS=60')

    # Reduce radial lines: fewer lines × more turns = same total computation
    if 'np.linspace(stella.R0' in new and '10)' in new:
        new = new.replace(
            'R_starts = np.linspace(stella.R0 + 0.02*stella.r0, stella.R0 + 0.93*stella.r0, 10)  # 10 radial lines',
            'R_starts = np.linspace(stella.R0 + 0.02*stella.r0, stella.R0 + 0.93*stella.r0, 8)  # 8 radial lines × 60 turns'
        )
        print(f'cell[{i}]: reduced radial lines to 8')

    if 'n_lines=6' in new:
        new = new.replace('n_lines=6,', 'n_lines=4,  # fewer near-resonance lines to keep total time low')
        print(f'cell[{i}]: reduced resonance lines to 4')

    if new != src:
        cell['source'] = new

with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
