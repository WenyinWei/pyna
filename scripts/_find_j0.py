with open('scripts/test_fenicsx_corrector.py', encoding='utf-8') as f:
    lines = f.readlines()

# Find where J0 is set (all-zero lines)
for i, l in enumerate(lines):
    if 'J0R_2d' in l or 'J0Z_2d' in l or 'J0Phi_2d' in l:
        print(i, repr(l))
