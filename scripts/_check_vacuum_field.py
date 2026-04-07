"""Check vacuum field data format."""
import numpy as np
import glob
import os

VACUUM_DIR = "/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields"

# Find one file
files = glob.glob(os.path.join(VACUUM_DIR, "dipole_coil_*.npz"))
print(f"Found {len(files)} vacuum field files")

# Load one file
f0 = files[0]
print(f"\nLoading: {os.path.basename(f0)}")
f = np.load(f0)
print("Keys:", list(f.keys()))
for k in f.keys():
    arr = f[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.ndim > 0:
        print(f"    min={arr.min():.4e}, max={arr.max():.4e}")
    else:
        print(f"    value={arr}")

# Check coil current info
print("\n--- First 5 files ---")
for fp in sorted(files)[:5]:
    print(f"  {os.path.basename(fp)}")

# Check if there's a response matrix file
resp_files = glob.glob(os.path.join(VACUUM_DIR, "*response*.npz"))
print(f"\nFound {len(resp_files)} response matrix files")
for rf in resp_files[:3]:
    print(f"  {os.path.basename(rf)}")
    d = np.load(rf)
    print(f"    keys: {list(d.keys())}")
    for k in d.keys():
        arr = d[k]
        print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")
