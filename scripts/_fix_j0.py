with open('scripts/test_fenicsx_corrector.py', encoding='utf-8') as f:
    lines = f.readlines()

# Replace zero J0 with computed J0 from diamagnetic current
new_j0_lines = [
    '    # J0 from numerical curl of B0 (non-zero because B0Phi varies with R)\n',
    '    # For B0Phi = B_val*R0/R: (curl B)_Z = B_Phi/R + dB_Phi/dR = B_val*R0/R^2 - B_val*R0/R^2 = 0\n',
    '    # Use diamagnetic current estimate instead for a non-trivial J0\n',
    '    mu0_loc = 4e-7 * 3.14159265358979\n',
    '    # J_dia = (B x grad_p) / B^2\n',
    '    grad_p_R = np.gradient(p0_2d, R_arr, axis=0)\n',
    '    grad_p_Z = np.gradient(p0_2d, Z_arr, axis=1)\n',
    '    B2 = B0R_2d**2 + B0Z_2d**2 + B0Phi_2d**2 + 1e-30\n',
    '    J0R_2d   = (B0Z_2d * 0.0 - B0Phi_2d * grad_p_Z) / B2\n',
    '    J0Z_2d   = (B0Phi_2d * grad_p_R - B0R_2d * 0.0) / B2\n',
    '    J0Phi_2d = (B0R_2d * grad_p_Z - B0Z_2d * grad_p_R) / B2\n',
]
lines[276:279] = new_j0_lines

with open('scripts/test_fenicsx_corrector.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('done')
