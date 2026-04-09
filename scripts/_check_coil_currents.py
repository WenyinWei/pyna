"""Check coil currents."""
import numpy as np
import glob

files = sorted(glob.glob("/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/dipole_coil_*.npz"))
print(f"Total files: {len(files)}")

currents = []
for fp in files[:20]:
    dd = np.load(fp)
    I = float(dd.get("coil_current", 0))
    currents.append(I)
    print(f"  {dd['coil_index']:3d}: I = {I:10.0f} A, |B| max = {np.max(np.abs(dd['BR_resp'])):.6e}")

print(f"\nCurrents range: {min(currents):.0f} to {max(currents):.0f} A")
print(f"Sum of first 20: {sum(currents):.0f} A")

# Check ALL coils
all_currents = []
for fp in files:
    dd = np.load(fp)
    I = float(dd.get("coil_current", 0))
    all_currents.append(I)

print(f"\nAll {len(all_currents)} coils:")
print(f"  Sum: {sum(all_currents):.0f} A")
print(f"  Abs sum: {sum(abs(c) for c in all_currents):.0f} A")
print(f"  Positive: {sum(1 for c in all_currents if c > 0)}")
print(f"  Negative: {sum(1 for c in all_currents if c < 0)}")
print(f"  Mean: {np.mean(all_currents):.0f} A")
print(f"  Std: {np.std(all_currents):.0f} A")

# Check field magnitude per coil
print("\nField magnitudes (first 5 coils):")
for fp in files[:5]:
    dd = np.load(fp)
    I = float(dd.get("coil_current", 0))
    BR = dd["BR_resp"].astype(np.float64) * I
    BPhi = dd["BPhi_resp"].astype(np.float64) * I
    BZ = dd["BZ_resp"].astype(np.float64) * I
    B_mag = np.sqrt(BR**2 + BPhi**2 + BZ**2)
    print(f"  Coil {dd['coil_index']}: I={I:.0f}A, <B>={np.mean(B_mag):.6e}, max|B|={np.max(B_mag):.6e}")
