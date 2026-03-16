import json

new_cell_source = [
    "# --- (m,n) Fourier spectrum heatmap: before and after control ---\n",
    "from pyna.MCF.visual.RMP_spectrum import compute_mn_spectrum, plot_mn_heatmap\n",
    "from pyna.MCF.control.island_optimizer import _make_coil_field_func\n",
    "from pyna.MCF.control.island_control import _natural_perturbation_func\n",
    "\n",
    "# Natural (background) field perturbation function\n",
    "nat_func_bg = _natural_perturbation_func(stella)\n",
    "\n",
    "# Coil field at optimal currents\n",
    "coil_func_after = _make_coil_field_func(coils)  # coils already set to optimal\n",
    "\n",
    "def total_pert_func(R, Z, phi):\n",
    '    """Natural + coil perturbation."""\n',
    "    db_nat = nat_func_bg(R, Z, phi)\n",
    "    try:\n",
    "        db_coil = coil_func_after(R, Z, phi)\n",
    "        return [db_nat[0] + db_coil[0], db_nat[1] + db_coil[1], 0.0]\n",
    "    except Exception:\n",
    "        return db_nat\n",
    "\n",
    "# Compute spectra at q=4/3 resonant surface\n",
    "S_res = stella.resonant_psi(TARGET_M, TARGET_N)[0]\n",
    "M_MAX, N_MAX = 5, 3\n",
    "\n",
    "b_mn_nat   = compute_mn_spectrum(nat_func_bg,   S_res, stella, m_max=M_MAX, n_max=N_MAX, n_theta=24, n_phi=24)\n",
    "b_mn_total = compute_mn_spectrum(total_pert_func, S_res, stella, m_max=M_MAX, n_max=N_MAX, n_theta=24, n_phi=24)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
    "\n",
    "plot_mn_heatmap(b_mn_nat, m_max=M_MAX, n_max=N_MAX, ax=axes[0],\n",
    "                log_scale=True, cmap='hot_r',\n",
    r"                title=r'$|\tilde{b}_{mn}|$ \u2014 Natural field  (S=' + f'{S_res:.3f})'," + "\n",
    "                highlight_modes=[(TARGET_M, TARGET_N)])\n",
    "\n",
    "plot_mn_heatmap(b_mn_total, m_max=M_MAX, n_max=N_MAX, ax=axes[1],\n",
    "                log_scale=True, cmap='hot_r',\n",
    r"                title=r'$|\tilde{b}_{mn}|$ \u2014 After optimisation  (S=' + f'{S_res:.3f}')" + ",\n",
    "                highlight_modes=[(TARGET_M, TARGET_N)])\n",
    "\n",
    "fig.suptitle(f'(m,n) Fourier spectrum: target mode ({TARGET_M},{TARGET_N}) highlighted (red box)', fontsize=11)\n",
    "plt.tight_layout()\n",
    "plt.savefig('mn_spectrum_before_after.png', dpi=120, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# Print the change in (4,3) amplitude\n",
    "amp_nat   = abs(b_mn_nat  [M_MAX + TARGET_M, N_MAX + TARGET_N])\n",
    "amp_after = abs(b_mn_total[M_MAX + TARGET_M, N_MAX + TARGET_N])\n",
    'print(f"b_({TARGET_M},{TARGET_N}) before: {amp_nat:.4e}")\n',
    'print(f"b_({TARGET_M},{TARGET_N}) after:  {amp_after:.4e}")\n',
    'print(f"Suppression ratio: {amp_after/(amp_nat+1e-30):.4f}")',
]

nb_path = r'notebooks/tutorials/stellarator_island_control.ipynb'

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

target_idx = None
bar_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'eps_before = epsilon_eff_proxy' in src:
            target_idx = i
        if 'plot_island_width_bars' in src or 'bar(x,' in src or ('b_nat' in src and 'bar' in src):
            bar_idx = i

print(f'target_idx={target_idx}, bar_idx={bar_idx}')
assert target_idx is not None, "Target cell not found!"

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': new_cell_source
}
nb['cells'].insert(target_idx + 1, new_cell)

if bar_idx is not None:
    actual_bar_idx = bar_idx if bar_idx <= target_idx else bar_idx + 1
    cell = nb['cells'][actual_bar_idx]
    if isinstance(cell['source'], list):
        if cell['source'] and not cell['source'][-1].endswith('\n'):
            cell['source'][-1] += '\n'
        cell['source'].append('# (m,n) heatmap: see next cell')
    else:
        cell['source'] += '\n# (m,n) heatmap: see next cell'

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Saved OK')
