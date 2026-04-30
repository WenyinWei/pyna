"""test_topology_eval.py
=======================
Quick integration test for evaluate_topology() using HAO starting config.
"""
import sys
import time
from pathlib import Path

TOPOQUEST = Path(r"C:\Users\Legion\Nutstore\1\Repo\topoquest")
PYNA = Path(r"C:\Users\Legion\Nutstore\1\Repo\pyna")
sys.path.insert(0, str(TOPOQUEST))
sys.path.insert(0, str(PYNA))

from explore_hao_divertor_configs import load_field_cache
from pyna.topo.topology_eval import evaluate_topology

print("=" * 60)
print("test_topology_eval.py -- HAO starting config")
print("=" * 60)

print("\nLoading field cache...")
t_load = time.time()
fc = load_field_cache()
print(f"  Loaded in {time.time()-t_load:.1f}s")
print(f"  BR shape: {fc['BR'].shape}")

print("\nRunning evaluate_topology()...")
result = evaluate_topology(
    field_cache=fc,
    n_core=20,
    n_core_turns=150,
    n_lcfs_turns=200,
    n_iota=15,
    n_iota_turns=100,
)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Magnetic axis:  R={result.R_ax:.5f} m  Z={result.Z_ax:.6f} m")
print(f"  V_lcfs:         {result.V_lcfs*1e3:.3f} L  ({result.V_lcfs:.5f} m3)")
print(f"  Config type:    {result.config_type}")
print(f"  LCFS points:    {len(result.lcfs_R)} (at phi=0)")
print(f"  Elapsed time:   {result.elapsed_s:.1f}s")

r_norm_arr, iota_arr = result.iota_profile
print(f"\n  Iota profile ({len(r_norm_arr)} pts):")
for rn, iota in zip(r_norm_arr, iota_arr):
    print(f"    r_norm={rn:.2f}  iota={iota:.4f}")

print(f"\n  X-point DPm results ({len(result.xpt_DPm_list)} pts):")
for entry in result.xpt_DPm_list:
    print(f"    R={entry['R']:.4f}  Z={entry['Z']:.4f}  "
          f"lu={entry['lambda_u']:.4f}  "
          f"SI={entry['stability_index']:.4f}  "
          f"GR={entry['greene_residue']:.4f}")
    print(f"    DPm = {entry['DPm'].tolist()}")

# Sanity checks
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

ok = True

V = result.V_lcfs
if 0.3 < V < 1.0:
    print(f"  [OK] V_lcfs={V:.4f} m3 in [0.3, 1.0]")
else:
    print(f"  [WARN] V_lcfs={V:.4f} m3 outside [0.3, 1.0]")
    ok = False

if 0.80 < result.R_ax < 0.90:
    print(f"  [OK] R_ax={result.R_ax:.4f} in [0.80, 0.90]")
else:
    print(f"  [WARN] R_ax={result.R_ax:.4f} outside [0.80, 0.90]")
    ok = False

if result.config_type in ("divertor", "limiter"):
    print(f"  [OK] config_type='{result.config_type}'")
else:
    print(f"  [FAIL] config_type='{result.config_type}' invalid")
    ok = False

if len(r_norm_arr) > 0:
    print(f"  [OK] iota profile has {len(r_norm_arr)} pts")
else:
    print(f"  [WARN] iota profile is empty")
    ok = False

for entry in result.xpt_DPm_list:
    lu = abs(entry["lambda_u"])
    if lu > 1.0:
        print(f"  [OK] lambda_u={lu:.4f} > 1 (hyperbolic)")
    else:
        print(f"  [WARN] lambda_u={lu:.4f} <= 1 (not hyperbolic?)")
        ok = False

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
