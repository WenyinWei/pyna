with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
          encoding='utf-8', errors='replace') as f:
    content = f.read()

content = content.replace('\r\n', '\n')

# Find the section-building loop and replace it
old_block = '''    n_phi = len(phi_vals)
    mask  = r_vals <= r_inner_fit
    r_fit = r_vals[mask]

    sections = []
    for ip in range(n_phi):
        sec = InnerFourierSection.from_poincare_surfaces(
            phi_ref=float(phi_vals[ip]),
            R_ax=float(R_AX[ip]),
            Z_ax=float(Z_AX[ip]),
            r_norms=r_fit,
            R_surf=R_surf_3d[mask, :, ip],
            Z_surf=Z_surf_3d[mask, :, ip],
            theta_arr=theta_arr,
            n_fourier=n_fourier,
        )
        sections.append(sec)'''

new_block = '''    n_phi = len(phi_vals)

    # ── Build sections using unified Fourier representation ───────────────
    # When fp_by_section is provided: use all X/O cycle points per section
    # to anchor r=1. Otherwise: 2-point fallback from trajectory.
    sections = []
    for ip in range(n_phi):
        # Collect X/O cycle points for this section
        R_cyc_ip = np.array([], dtype=float)
        Z_cyc_ip = np.array([], dtype=float)
        if fp_by_section is not None:
            phi_v = float(phi_vals[ip])
            best_phi = min(fp_by_section.keys(),
                           key=lambda k: abs(k % np.pi - phi_v % np.pi))
            sec_data = fp_by_section[best_phi]
            R_list, Z_list = [], []
            for xpt in sec_data.get('xpts', []):
                R_p = xpt[0] if not hasattr(xpt, 'R') else xpt.R
                Z_p = xpt[1] if not hasattr(xpt, 'R') else xpt.Z
                R_list.append(R_p); Z_list.append(Z_p)
            for opt in sec_data.get('opts', []):
                R_p = opt[0] if not hasattr(opt, 'R') else opt.R
                Z_p = opt[1] if not hasattr(opt, 'R') else opt.Z
                R_list.append(R_p); Z_list.append(Z_p)
            R_cyc_ip = np.array(R_list)
            Z_cyc_ip = np.array(Z_list)
        else:
            # 2-point fallback from trajectory
            phi_v = float(phi_vals[ip]) % (2 * np.pi)
            R_cyc_ip = np.array([float(_Rx_itp(phi_v)), float(_Ro_itp(phi_v))])
            Z_cyc_ip = np.array([float(_Zx_itp(phi_v)), float(_Zo_itp(phi_v))])

        sec = InnerFourierSection.from_full_grid_with_cycle_constraint(
            phi_ref=float(phi_vals[ip]),
            R_ax=float(R_AX[ip]),
            Z_ax=float(Z_AX[ip]),
            R_surf_row=R_surf_3d[:, :, ip],   # (n_r, n_theta)
            Z_surf_row=Z_surf_3d[:, :, ip],
            r_vals=r_vals,
            theta_arr=theta_arr,
            R_cycle=R_cyc_ip,
            Z_cycle=Z_cyc_ip,
            n_fourier=n_fourier,
            cycle_weight=cycle_weight,
            r_geom_min=r_geom_min,
        )
        sections.append(sec)'''

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    print('Block replaced')
else:
    # Try to find where it is
    idx = content.find('    n_phi = len(phi_vals)\n    mask')
    print(f'Old block at index {idx}')
    if idx >= 0:
        print(repr(content[idx:idx+400]))

with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
          'w', encoding='utf-8') as f:
    f.write(content)
print('Written')
