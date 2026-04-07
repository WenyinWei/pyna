with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
          encoding='utf-8', errors='replace') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

# Find and remove the orphaned with_all_cycle_points body
# It starts with "        self," right after the new classmethod's closing )
# and ends before "    # Keep the old 2-point method"
orphan_start = '        self,\n        R_cycle: np.ndarray,\n        Z_cycle: np.ndarray,\n        r_island: float = 1.0,\n        r_geom_min: float = 0.05,\n    ) -> \'InnerFourierSection\':'
keep_marker = '    # Keep the old 2-point method'

idx_start = content.find(orphan_start)
idx_end   = content.find(keep_marker)

if idx_start >= 0 and idx_end > idx_start:
    # Find the actual end of the orphaned docstring+body
    # It ends just before "    # Keep the old 2-point method"
    removed_len = idx_end - idx_start
    print(f'Removing orphaned body: chars {idx_start}..{idx_end} ({removed_len} chars)')
    content = content[:idx_start] + content[idx_end:]
    with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
              'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed')
else:
    print(f'orphan_start found: {idx_start}, keep_marker found: {idx_end}')
    if idx_start >= 0:
        print(repr(content[idx_start:idx_start+100]))
