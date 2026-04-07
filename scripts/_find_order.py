with open('scripts/test_fenicsx_corrector.py', encoding='utf-8') as f:
    lines = f.readlines()

# Move p0_2d computation before J0 (reorder so p0 is computed first)
# Find r_loc line (start of pressure)
for i, l in enumerate(lines):
    if 'r_loc' in l and 'a_eff' not in l and 'sqrt' in l:
        print('r_loc at', i, repr(l))
    if 'psi_norm' in l and 'clip' in l:
        print('psi_norm at', i, repr(l))
    if 'p0_2d' in l:
        print('p0_2d at', i, repr(l))
