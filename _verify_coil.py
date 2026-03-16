import numpy as np
from pyna.MCF.coils.coil import BRBZ_induced_by_current_loop, CoilFieldAnalyticCircular

# Horizontal loop at origin, radius=1.0, current=1000 A
a = 1.0
I = 1000.0
coil = CoilFieldAnalyticCircular(radius=a, center_xyz=[0, 0, 0], normal_xyz=[0, 0, 1], current=I)

# Test points (R, Z, phi)
Rs = np.array([0.5, 1.5, 0.3, 2.0])
Zs = np.array([0.2, -0.3, 0.5, 0.1])
phis = np.array([0.0, 1.0, 2.0, 3.0])

BR_analytic, BZ_analytic = BRBZ_induced_by_current_loop(a, 0.0, I, Rs, Zs)
BR_class, BZ_class, Bp_class = coil.B_at(Rs, Zs, phis)

err_R = np.max(np.abs(BR_class - BR_analytic))
err_Z = np.max(np.abs(BZ_class - BZ_analytic))
err_phi = np.max(np.abs(Bp_class))  # should be zero for horizontal loop

print(f'Max |BR error| = {err_R:.2e}')
print(f'Max |BZ error| = {err_Z:.2e}')
print(f'Max |Bphi|     = {err_phi:.2e}  (should be ~0)')

assert err_R < 1e-10, f'BR error too large: {err_R}'
assert err_Z < 1e-10, f'BZ error too large: {err_Z}'
assert err_phi < 1e-10, f'Bphi should be zero: {err_phi}'
print('All checks passed!')
