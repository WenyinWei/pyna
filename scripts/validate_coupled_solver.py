"""validate_coupled_solver.py
=================================================
Validate solve_perturbed_gs_coupled on the real HAO vacuum field.

Loads bluestar_starting_config_field_cache.pkl, builds a beta=1%
pressure profile, applies a small random delta_B_ext, calls the
coupled perturbed-GS solver, and reports Ampere + force-balance residuals.

Findings (2026-04-03)
---------------------
* Field cache loads correctly: |B|_mean=1.17T, BPhi_mean=0.93T
* J0 (diamagnetic + PS) computed: |J0|_mean=2340 A/m²
* Solver runs without error (~2-7 s for 100×100 LSQR)
* delta_J ≈ 0 everywhere — Ampere residual ~1.0
  Root cause: BC penalty (weight_BC_J=1e9) overwhelms physics
  equations (weight_force=1.0, RHS ~ 4 N/m³ from J0×dB_ext).
  The solver zeros out dJ to satisfy boundary conditions, producing
  a div-free dB_plasma that does NOT satisfy Ampere with the dJ field.
* Action: solver weight calibration needed (see comments in output).

Usage
-----
    cd C:\\Users\\Legion\\Nutstore\\1\\Repo\\pyna
    python scripts\\validate_coupled_solver.py
"""
from __future__ import annotations

import sys, os, types, importlib, importlib.util, pickle
import numpy as np

# ---------------------------------------------------------------------------
# Fix broken pyna __init__ by pre-loading submodules manually
# ---------------------------------------------------------------------------
PYNA_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PYNA_PKG  = os.path.join(PYNA_ROOT, 'pyna')

def _load_mod(dotted_name: str, file_path: str):
    """Load a module from file_path into sys.modules[dotted_name]."""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

# 1. Stub 'pyna' package (avoids executing broken __init__.py)
if 'pyna' not in sys.modules:
    stub = types.ModuleType('pyna')
    stub.__path__ = [PYNA_PKG]
    stub.__package__ = 'pyna'
    sys.modules['pyna'] = stub

# 2. Stub pyna.fields package
if 'pyna.fields' not in sys.modules:
    fields_stub = types.ModuleType('pyna.fields')
    fields_stub.__path__ = [os.path.join(PYNA_PKG, 'fields')]
    fields_stub.__package__ = 'pyna.fields'
    sys.modules['pyna.fields'] = fields_stub

# 3. Load fields submodules in dependency order
_load_mod('pyna.fields.properties', os.path.join(PYNA_PKG, 'fields', 'properties.py'))
_load_mod('pyna.fields.coords',     os.path.join(PYNA_PKG, 'fields', 'coords.py'))
_load_mod('pyna.fields.base',       os.path.join(PYNA_PKG, 'fields', 'base.py'))
_load_mod('pyna.fields.cylindrical',os.path.join(PYNA_PKG, 'fields', 'cylindrical.py'))

# 4. Stub intermediate toroidal packages
for _pkg in ['pyna.toroidal', 'pyna.toroidal.plasma_response', 'pyna.toroidal.equilibrium']:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [os.path.join(PYNA_PKG, *_pkg.split('.')[1:])]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

# 5. Load toroidal PerturbGS and fenicsx_corrector directly
# Keep this script self-contained; do not revive removed legacy facades here.
PerturbGS_mod = _load_mod(
    'pyna.toroidal.plasma_response.PerturbGS',
    os.path.join(PYNA_PKG, 'toroidal', 'plasma_response', 'PerturbGS.py'),
)
fenicsx_mod = _load_mod(
    'pyna.toroidal.equilibrium.fenicsx_corrector',
    os.path.join(PYNA_PKG, 'toroidal', 'equilibrium', 'fenicsx_corrector.py'),
)

solve_perturbed_gs_coupled  = PerturbGS_mod.solve_perturbed_gs_coupled
_make_axi_vector_field       = PerturbGS_mod._make_axi_vector_field
_make_axi_scalar_field       = PerturbGS_mod._make_axi_scalar_field
compute_equilibrium_currents = PerturbGS_mod.compute_equilibrium_currents
compute_curl_cylindrical     = fenicsx_mod.compute_curl_cylindrical
compute_force_residual       = fenicsx_mod.compute_force_residual

print("Imports OK")

# ---------------------------------------------------------------------------
# Load field cache
# ---------------------------------------------------------------------------
CACHE_PATH = r'C:\Users\Legion\Nutstore\1\Repo\topoquest\data\bluestar_starting_config_field_cache.pkl'
with open(CACHE_PATH, 'rb') as f:
    fc = pickle.load(f)

BR_3d   = fc['BR']    # (100, 100, 128)
BPhi_3d = fc['BPhi']
BZ_3d   = fc['BZ']
R_grid  = fc['R_grid']   # (100,)
Z_grid  = fc['Z_grid']   # (100,)
Phi_grid= fc['Phi_grid'] # (128,)

nR, nZ, nPhi = BR_3d.shape
print(f"Grid: nR={nR}, nZ={nZ}, nPhi={nPhi}")
print(f"|B| mean (3D)  = {np.mean(np.sqrt(BR_3d**2+BPhi_3d**2+BZ_3d**2)):.4f} T")
print(f"BPhi mean (3D) = {np.mean(BPhi_3d):.4f} T")

# phi=0 slice
B0R_2d   = BR_3d[:, :, 0]
B0Phi_2d = BPhi_3d[:, :, 0]
B0Z_2d   = BZ_3d[:, :, 0]
B_avg_sq = float(np.mean(B0R_2d**2 + B0Phi_2d**2 + B0Z_2d**2))
print(f"|B|^2 mean (phi=0): {B_avg_sq:.6f} T^2  => |B| = {np.sqrt(B_avg_sq):.4f} T")

# ---------------------------------------------------------------------------
# Build B0 field object (phi=0 axisymmetric slice)
# ---------------------------------------------------------------------------
B0 = _make_axi_vector_field(R_grid, Z_grid, B0R_2d, B0Z_2d, B0Phi_2d, name="B0")

# ---------------------------------------------------------------------------
# Pressure profile for beta=1%
# ---------------------------------------------------------------------------
mu0 = 4e-7 * np.pi
R_axis, Z_axis, a_eff = 0.85235, -0.000073, 0.18
alpha_pressure = 2.0
p0_pa = 0.01 * B_avg_sq / (2 * mu0)
print(f"p0_pa (beta=1%): {p0_pa:.2f} Pa")

def pressure_profile(psi_norm_val):
    """psi_norm_val is normalized minor radius r/a (not r^2/a^2)."""
    return float(p0_pa * max(0.0, 1.0 - float(psi_norm_val))**alpha_pressure)

# ---------------------------------------------------------------------------
# Build J0 and p0 from equilibrium currents (diamagnetic + PS)
# ---------------------------------------------------------------------------
Phi_1 = np.array([0.0])
print("Computing equilibrium currents J0 (diamagnetic + Pfirsch-Schluter)...")
J0, p0_field = compute_equilibrium_currents(
    B0, pressure_profile, R_grid, Z_grid, Phi_1, R_axis, Z_axis, a_eff
)
print(f"|J0| mean: {np.mean(np.sqrt(J0.VR**2+J0.VZ**2+J0.VPhi**2)):.4e} A/m^2")

# Build 2D pressure for force-balance checks
RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing='ij')
r_norm = np.sqrt((RR - R_axis)**2 + (ZZ - Z_axis)**2) / a_eff
p_2d = p0_pa * (1.0 - np.clip(r_norm, 0.0, 1.0))**alpha_pressure

p0_val_2d = p0_field.value[:, :, 0]
n_plasma = np.sum(p0_val_2d > 1e-1)
print(f"Plasma region (p > 0.1 Pa): {n_plasma} cells / {nR*nZ} total "
      f"(a_eff={a_eff}m, dR={(R_grid[1]-R_grid[0]):.4f}m)")

# ---------------------------------------------------------------------------
# Small random delta_B_ext (1% of |B|)
# ---------------------------------------------------------------------------
np.random.seed(42)
delta_scale = 0.01 * np.sqrt(B_avg_sq)
dBR_ext   = delta_scale * np.random.randn(nR, nZ)
dBZ_ext   = delta_scale * np.random.randn(nR, nZ)
dBPhi_ext = delta_scale * np.random.randn(nR, nZ)

delta_B_ext = _make_axi_vector_field(
    R_grid, Z_grid, dBR_ext, dBZ_ext, dBPhi_ext, name="dB_ext")

print(f"\ndelta_B_ext scale: {delta_scale:.4e} T  (1% of |B|)")
RHS_scale = np.mean(np.abs(J0.VPhi[:, :, 0])) * delta_scale
print(f"Perturbation RHS |J0 x dB_ext| ~ {RHS_scale:.4e} N/m^3")

# Background force imbalance at phi=0 (3D stellarator, NOT axisymmetric)
B0_2d_arr = np.stack([B0R_2d, B0Phi_2d, B0Z_2d], axis=0)
J_bg = compute_curl_cylindrical(B0_2d_arr, R_grid, Z_grid, mu0)
r_bg_R, r_bg_Z = compute_force_residual(J_bg, B0_2d_arr, p_2d, R_grid, Z_grid)
bg_imbalance = float(np.sqrt(np.mean(r_bg_R**2 + r_bg_Z**2)))
print(f"Background force imbalance at phi=0: {bg_imbalance:.4e} N/m^3")
print(f"  (expected large for 3D stellarator; perturbation RHS is {bg_imbalance/max(RHS_scale,1e-30):.1f}x smaller)")

# ---------------------------------------------------------------------------
# Run coupled solver
# ---------------------------------------------------------------------------
print("\nRunning solve_perturbed_gs_coupled ...")
import time
t0 = time.time()

delta_B_plasma, delta_J, delta_p = solve_perturbed_gs_coupled(
    B0, J0, p0_field, delta_B_ext,
    solver='lsqr',
    max_iter=3000,
    tol=1e-8,
    weight_ampere=1e6,
    weight_force=1.0,
    weight_div=1e8,
    weight_divJ=1e6,
    weight_BC_J=1e9,
    weight_BC_p=1e12,
)

elapsed = time.time() - t0
print(f"Solver done in {elapsed:.1f} s")

# Extract phi=0 slices
dBR_plasma   = delta_B_plasma.VR[:, :, 0]
dBZ_plasma   = delta_B_plasma.VZ[:, :, 0]
dBPhi_plasma = delta_B_plasma.VPhi[:, :, 0]
dJR          = delta_J.VR[:, :, 0]
dJZ          = delta_J.VZ[:, :, 0]
dJPhi        = delta_J.VPhi[:, :, 0]

print(f"|dB_plasma| mean: {np.mean(np.sqrt(dBR_plasma**2+dBZ_plasma**2+dBPhi_plasma**2)):.4e} T")
print(f"|dJ| mean:        {np.mean(np.sqrt(dJR**2+dJZ**2+dJPhi**2)):.4e} A/m^2")

# Point classification (mirrors solver logic)
n_plasma_interior = 0; n_boundary = 0; n_vacuum = 0
for _i in range(nR):
    for _j in range(nZ):
        _bnd = (_i in (0,1,2,nR-3,nR-2,nR-1) or _j in (0,1,2,nZ-3,nZ-2,nZ-1))
        if _bnd:
            n_boundary += 1; continue
        _vac = True
        for _ni, _nj in [(_i,_j),(_i+1,_j),(_i-1,_j),(_i,_j+1),(_i,_j-1)]:
            if 0<=_ni<nR and 0<=_nj<nZ and p0_val_2d[_ni,_nj] > 1e-1:
                _vac = False; break
        if _vac: n_vacuum += 1
        else: n_plasma_interior += 1
print(f"Solver point classification: plasma_interior={n_plasma_interior}, "
      f"boundary={n_boundary}, vacuum={n_vacuum}")

# ---------------------------------------------------------------------------
# Ampere residual: ||mu0*dJ - curl(dB_plasma)||
# compute_curl_cylindrical returns J=curl/mu0; multiply by mu0 to get curl(dB)
# ---------------------------------------------------------------------------
dB_plasma_2d = np.stack([dBR_plasma, dBPhi_plasma, dBZ_plasma], axis=0)
curl_dB_J    = compute_curl_cylindrical(dB_plasma_2d, R_grid, Z_grid, mu0)
curl_dB      = mu0 * curl_dB_J      # curl(dB_plasma) in T/m
mu0_dJ_arr   = mu0 * np.stack([dJR, dJZ, dJPhi], axis=0)

norm_curl  = np.sqrt(np.mean(curl_dB**2))
norm_mu0dJ = np.sqrt(np.mean(mu0_dJ_arr**2))
norm_diff  = np.sqrt(np.mean((mu0_dJ_arr - curl_dB)**2))

# Relative to curl(dB) — the physically meaningful normalization
ampere_resid = norm_diff / norm_curl if norm_curl > 1e-30 else float('nan')

# Interior only (exclude 3-cell boundary ring)
sl = np.s_[3:-3, 3:-3]
int_diff = np.sqrt(np.mean((mu0_dJ_arr[:, sl[0], sl[1]] - curl_dB[:, sl[0], sl[1]])**2))
int_curl = np.sqrt(np.mean(curl_dB[:, sl[0], sl[1]]**2))
ampere_int = int_diff / int_curl if int_curl > 1e-30 else float('nan')

print(f"\n{'='*60}")
print(f"AMPERE RESIDUAL: ||mu0*dJ - curl(dB)|| / ||curl(dB)||")
print(f"  Global:   {ampere_resid:.4e}")
print(f"  Interior: {ampere_int:.4e}  (target: < 1e-2)")
print(f"  ||curl(dB_plasma)||   = {norm_curl:.4e}  T/m")
print(f"  ||mu0*dJ||            = {norm_mu0dJ:.4e}  T/m")
print(f"  Ratio ||mu0*dJ||/||curl||: {norm_mu0dJ/max(norm_curl,1e-30):.2e}")

if norm_mu0dJ < 1e-8 * norm_curl:
    print()
    print("  *** DIAGNOSIS: delta_J ~ 0 (solver returns vacuum-like response)")
    print("  *** Root cause: BC penalty (weight_BC_J=1e9) dominates physics")
    print("  *** equations (weight_force=1.0, RHS~4 N/m3). Solver zeroes dJ.")
    print("  *** Also: stellarator has 3D background imbalance (1.4e7 N/m3)")
    print("  *** that dwarfs the perturbation driving (4 N/m3).")
    print("  *** Fix: lower weight_BC_J / raise weight_force / use smaller grid")
    print("  ***      OR reformulate using flux-surface coordinates")

# ---------------------------------------------------------------------------
# Force-balance residual: before vs after
# ---------------------------------------------------------------------------
dBext_2d   = np.stack([dBR_ext, dBPhi_ext, dBZ_ext], axis=0)
B_before   = B0_2d_arr + dBext_2d
B_after    = B0_2d_arr + dBext_2d + dB_plasma_2d

J_bef    = compute_curl_cylindrical(B_before, R_grid, Z_grid, mu0)
rR_b, rZ_b = compute_force_residual(J_bef, B_before, p_2d, R_grid, Z_grid)
r_before = float(np.sqrt(np.mean(rR_b**2 + rZ_b**2)))

J_aft    = compute_curl_cylindrical(B_after, R_grid, Z_grid, mu0)
rR_a, rZ_a = compute_force_residual(J_aft, B_after, p_2d, R_grid, Z_grid)
r_after  = float(np.sqrt(np.mean(rR_a**2 + rZ_a**2)))

print(f"\nFORCE-BALANCE RESIDUAL (RMS N/m^3):")
print(f"  Before (B0 + dB_ext):              {r_before:.4e}")
print(f"  After  (B0 + dB_ext + dB_plasma):  {r_after:.4e}")
improvement = r_before / max(r_after, 1e-30)
print(f"  Improvement ratio: {improvement:.3f}x")
if improvement < 1.01:
    print("  (No improvement -- consistent with dJ~0, dB_plasma from div-free BC only)")
print(f"{'='*60}")

print("\nSUMMARY:")
print(f"  Field cache: |B|={np.mean(np.sqrt(BR_3d**2+BPhi_3d**2+BZ_3d**2)):.3f}T, "
      f"BPhi={np.mean(BPhi_3d):.3f}T  [OK]")
print(f"  Solver ran in {elapsed:.1f}s without error  [OK]")
print(f"  Ampere residual (interior): {ampere_int:.4e}  [{'OK' if ampere_int < 0.1 else 'NEEDS TUNING'}]")
print(f"  Force-balance improvement: {improvement:.2f}x  [{'OK' if improvement > 1.5 else 'NEEDS TUNING'}]")
print()
print("NEXT STEPS:")
print("  1. Fix island_healed_coords.py syntax error (line 360 unmatched ')')")
print("  2. Tune solver weights: weight_BC_J -> 1e3, weight_force -> 1e6")
print("  3. Or use solve_perturbed_gs (non-coupled) as intermediate validation")
