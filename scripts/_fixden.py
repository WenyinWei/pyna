with open('scripts/test_fenicsx_corrector.py', encoding='utf-8') as f:
    lines = f.readlines()

# Fix den to use max of mu0_dJ and curl(dB) norms
lines[330] = '    norm_ref = max(np.sqrt(np.mean(mu0_dJR**2 + mu0_dJZ**2 + mu0_dJPhi**2)),\n'
lines[330] += '                  np.sqrt(np.mean(curl_R_num**2 + curl_Z_num**2 + curl_Phi_num**2))) + 1e-30\n'
lines[331] = '    ampere_residual = num / norm_ref\n'

lines[354] = '    norm_ref_old = max(np.sqrt(np.mean(mu0_dJR_old**2 + mu0_dJZ_old**2 + mu0_dJPhi_old**2)),\n'
lines[354] += '                      np.sqrt(np.mean(curl_R_o**2 + curl_Z_o**2 + curl_Phi_o**2))) + 1e-30\n'
lines[355] = '    ampere_residual_old = num_old / norm_ref_old\n'

with open('scripts/test_fenicsx_corrector.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('done')
