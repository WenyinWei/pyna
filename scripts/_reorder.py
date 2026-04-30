with open('scripts/test_fenicsx_corrector.py', encoding='utf-8') as f:
    lines = f.readlines()

# Remove lines 271-288 (old J0 comment + code)
# and replace with: p0 block first, then J0 code
p0_block = lines[289:293]  # r_loc, a_eff, psi_norm, p0_2d

new_j0 = [
    '    # J0 from diamagnetic current: J_dia = (B x grad_p) / B^2 (non-trivial)\n',
    '    grad_p_R = np.gradient(p0_2d, R_arr, axis=0)\n',
    '    grad_p_Z = np.gradient(p0_2d, Z_arr, axis=1)\n',
    '    B2       = B0R_2d**2 + B0Z_2d**2 + B0Phi_2d**2 + 1e-30\n',
    '    J0R_2d   = (B0Z_2d * 0.0 - B0Phi_2d * grad_p_Z) / B2\n',
    '    J0Z_2d   = (B0Phi_2d * grad_p_R - B0R_2d * 0.0) / B2\n',
    '    J0Phi_2d = (B0R_2d * grad_p_Z - B0Z_2d * grad_p_R) / B2\n',
    '\n',
]

# Lines 271-288 is the old J0 block (including the extended new code we inserted earlier)
# Lines 289-292 is p0 block
# We want: after B0 setup (end at line 270), insert: p0 block, then new J0
# B0 ends at line 270 (B0Phi_2d line + blank)
# Replace lines 271-292 with: p0 block + new_j0

replacement = p0_block + ['\n'] + new_j0
lines[271:293] = replacement

with open('scripts/test_fenicsx_corrector.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('done, total lines:', len(lines))
# Verify
for i in range(270, 295):
    print(i, repr(lines[i]))
