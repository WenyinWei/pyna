with open('scripts/test_fenicsx_corrector.py', encoding='utf-8') as f:
    lines = f.readlines()

# Find the two den= lines and fix them
for i, l in enumerate(lines):
    if 'den =' in l and 'sqrt' in l and 'mean' in l:
        print(i, repr(l))
    if 'den_old' in l and 'sqrt' in l:
        print(i, repr(l))
    if 'ampere_residual =' in l and '/' in l:
        print(i, repr(l))
    if 'ampere_residual_old =' in l and '/' in l:
        print(i, repr(l))
