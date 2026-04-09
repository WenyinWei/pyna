"""Check if resp fields are already scaled by current."""
import numpy as np
import glob
import re

files = sorted(glob.glob("/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/dipole_coil_*.npz"))

# Load two coils with different currents
f1 = files[0]  # -87600 A
f2 = files[1]  # 75600 A

d1 = np.load(f1)
d2 = np.load(f2)

m1 = re.search(r"current([-0-9.]+)A", f1)
m2 = re.search(r"current([-0-9.]+)A", f2)
I1 = float(m1.group(1))
I2 = float(m2.group(1))
print(f"Coil {d1['coil_index']}: I = {I1:.0f} A, max|BR| = {np.max(np.abs(d1['BR_resp'])):.6e}")
print(f"Coil {d2['coil_index']}: I = {I2:.0f} A, max|BR| = {np.max(np.abs(d2['BR_resp'])):.6e}")

# If resp is unit-current response, then BR/I should be similar
print(f"\nBR_resp / I:")
print(f"  Coil 1: {np.max(np.abs(d1['BR_resp']))/abs(I1):.6e}")
print(f"  Coil 2: {np.max(np.abs(d2['BR_resp']))/abs(I2):.6e}")

# Check if already scaled: BR/I should be very different if already scaled
# vs similar if unit response
print(f"\nBR_resp (raw):")
print(f"  Coil 1: {np.max(np.abs(d1['BR_resp'])):.6e}")
print(f"  Coil 2: {np.max(np.abs(d2['BR_resp'])):.6e}")
