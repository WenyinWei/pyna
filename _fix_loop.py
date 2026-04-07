with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
          encoding='utf-8', errors='replace') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

old_constraint_loop = '''    # ── Apply X/O cycle constraint to each section ────────────────────────
    # At each phi section, r=r_island is anchored to the X-cycle and O-cycle
    # positions AT THAT SPECIFIC phi section (from the continuous trajectory).
    #
    # Physical rationale: the X-cycle is a 3D trajectory that crosses each
    # phi section exactly ONCE. That single crossing point defines r=1 in
    # the theta_X direction for that section. Similarly for the O-cycle.
    # Other X-points visible at phi=0 belong to different phi sections of
    # the same trajectory and should NOT be used as r=1 constraints here.
    for ip in range(n_phi):
        phi_v = float(phi_vals[ip])
        sections[ip] = sections[ip].with_island_constraint(
            float(_Rx_itp(phi_v % (2*np.pi))),
            float(_Zx_itp(phi_v % (2*np.pi))),
            float(_Ro_itp(phi_v % (2*np.pi))),
            float(_Zo_itp(phi_v % (2*np.pi))),
            r_island=r_island,
        )'''

# Remove this block (the new from_full_grid_with_cycle_constraint handles it)
if old_constraint_loop in content:
    content = content.replace(old_constraint_loop, '    # r=1 constraint already applied in from_full_grid_with_cycle_constraint', 1)
    print('Old constraint loop removed')
else:
    # Find what's there
    idx = content.find('    # ── Apply X/O cycle constraint')
    print(f'Old loop at {idx}')
    if idx >= 0:
        print(repr(content[idx:idx+300]))

with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
          'w', encoding='utf-8') as f:
    f.write(content)
print('Written')
