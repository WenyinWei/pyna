with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
          encoding='utf-8', errors='replace') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

# Find and remove the old fp_by_section loop that was added earlier
idx = content.find('    # ---- Apply X/O cycle constraint')
if idx < 0:
    idx = content.find('    # r=1 constraint already applied')
    print(f'Already cleaned marker at {idx}')
    
# Find the apply constraint section
idx2 = content.find('    # r=1 constraint already applied')
if idx2 >= 0:
    print('Already done')
else:
    # Find the old loop
    idx3 = content.find('    if fp_by_section is not None:\n        # Use ALL X/O cycle')
    if idx3 >= 0:
        # Find where it ends (next top-level comment)
        end_markers = [
            '    # ── Build anchors',
            '    # ── Load X/O ring',
            '    phi_sec_arr',
        ]
        end_idx = len(content)
        for marker in end_markers:
            m = content.find(marker, idx3)
            if m > idx3 and m < end_idx:
                end_idx = m
        old_loop = content[idx3:end_idx]
        print('Found old loop, length:', len(old_loop))
        print('First 200 chars:', repr(old_loop[:200]))
        content = content[:idx3] + '    # r=1 already applied in from_full_grid_with_cycle_constraint\n\n' + content[end_idx:]
        with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
                  'w', encoding='utf-8') as f:
            f.write(content)
        print('Fixed')
    else:
        print('No old loop found')
        print(repr(content[45600:46000]))
