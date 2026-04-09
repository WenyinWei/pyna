"""Check all 332 coils field strength."""
import numpy as np
import glob

files = sorted(glob.glob("/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/dipole_coil_*.npz"))
print(f"Total files: {len(files)}")

# Sum all
BR = BPhi = BZ = None
for i, fp in enumerate(files):
    d = np.load(fp)
    if BR is None:
        BR = d["BR_resp"].astype(np.float64).copy()
        BPhi = d["BPhi_resp"].astype(np.float64).copy()
        BZ = d["BZ_resp"].astype(np.float64).copy()
    else:
        BR += d["BR_resp"].astype(np.float64)
        BPhi += d["BPhi_resp"].astype(np.float64)
        BZ += d["BZ_resp"].astype(np.float64)
    if (i+1) % 100 == 0:
        B_mag_partial = np.sqrt(BR**2 + BPhi**2 + BZ**2)
        print(f"  {i+1} coils: <B>={np.mean(B_mag_partial):.6e} T")

B_mag = np.sqrt(BR**2 + BPhi**2 + BZ**2)
print(f"\nAll {len(files)} coils: <B>={np.mean(B_mag):.6e} T, max|B|={np.max(B_mag):.6e} T")

# Check individual coil field strength
print("\nIndividual coil field strengths (first 10):")
for fp in files[:10]:
    d = np.load(fp)
    B_mag_c = np.sqrt(d["BR_resp"]**2 + d["BPhi_resp"]**2 + d["BZ_resp"]**2)
    print(f"  Coil {d['coil_index']}: <B>={np.mean(B_mag_c):.6e}, max={np.max(B_mag_c):.6e}")
