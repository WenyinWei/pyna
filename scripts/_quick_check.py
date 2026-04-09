"""Quick field strength check."""
import numpy as np
import glob

files = sorted(glob.glob("/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/dipole_coil_*.npz"))[:30]
BR = BPhi = BZ = None
for fp in files:
    d = np.load(fp)
    if BR is None:
        BR = d["BR_resp"].astype(np.float64).copy()
        BPhi = d["BPhi_resp"].astype(np.float64).copy()
        BZ = d["BZ_resp"].astype(np.float64).copy()
    else:
        BR += d["BR_resp"].astype(np.float64)
        BPhi += d["BPhi_resp"].astype(np.float64)
        BZ += d["BZ_resp"].astype(np.float64)

B_mag = np.sqrt(BR**2 + BPhi**2 + BZ**2)
print(f"30 coils: <B>={np.mean(B_mag):.6e} T, max|B|={np.max(B_mag):.6e} T")

MU0 = 4e-7*np.pi
B_avg = np.mean(B_mag)
p0 = 0.02 * B_avg**2 / (2*MU0) * 3
print(f"  p0 (beta=0.02, alpha=2) = {p0:.2f} Pa")
print(f"  Expected <beta> ~ 0.02")
