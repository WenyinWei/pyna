with open('scripts/test_fenicsx_corrector.py', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = [
    '    # Non-trivial external perturbation\n',
    '    dBext_R_2d   = np.zeros((nR, nZ))\n',
    '    dBext_Z_2d   = 1e-3 * np.sin(np.pi * (RR - R_arr[0]) / (R_arr[-1] - R_arr[0]))\n',
    '    dBext_Phi_2d = np.zeros((nR, nZ))\n',
    '    dBext_field  = _make_field(dBext_R_2d, dBext_Z_2d, dBext_Phi_2d, "dBext")\n',
]
lines[295:297] = new_lines

with open('scripts/test_fenicsx_corrector.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('done, lines now:', len(lines))
for i in range(293, 303):
    print(i, repr(lines[i]))
