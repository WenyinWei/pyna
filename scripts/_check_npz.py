"""Check npz file content and extract current from filename."""
import numpy as np
import glob
import re

files = sorted(glob.glob("/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/dipole_coil_*.npz"))
f = files[0]
d = np.load(f)
print("Keys:", list(d.keys()))
for k in d.keys():
    val = d[k]
    if val.ndim == 0:
        print(f"  {k}: {val}")
    else:
        print(f"  {k}: shape={val.shape}, dtype={val.dtype}")

m = re.search(r"current([-0-9.]+)A", f)
if m:
    print(f"\nCurrent from filename: {m.group(1)} A")
