with open('scripts/test_fenicsx_corrector.py', encoding='utf-8') as f:
    lines = f.readlines()

# p0 computed at lines 289-292; J0 uses p0 at lines 281-287
# Move the p0 block (289-292) to before J0 block (the new J0 code at 276+)
# Find the exact positions
j0_start = 271  # first comment for J0
j0_end = 289    # start of r_loc (p0 block)

# Extract p0 block lines (289-292 inclusive = indices 289,290,291,292)
p0_block = lines[289:293]  # lines for r_loc, a_eff already defined elsewhere? check
print('p0 block:')
for l in p0_block:
    print(repr(l))

print('\nJ0 block:')
for i in range(j0_start, 289):
    print(i, repr(lines[i]))
